use super::db_healthcheck::DbHealthChecker;
use super::env_pool::SharedEnv;
use crate::error::{Error, Result};
use crate::file_picker::FFFMode;
use crate::git::is_modified_status;
use crate::lmdb::{DbHealth, LmdbStore, WRITE_TXN_CHUNK, is_map_full};
use heed::Database;
use heed::types::{Bytes, SerdeBincode};
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{collections::VecDeque, path::Path};

const DECAY_CONSTANT: f64 = 0.0693; // ln(2)/10 for 10-day half-life
const SECONDS_PER_DAY: f64 = 86400.0;
const MAX_HISTORY_DAYS: f64 = 30.0; // Only consider accesses within 30 days
const MAX_TIMESTAMPS_PER_FILE: usize = 128;

// AI mode: faster decay since AI sessions are shorter and more intense
const AI_DECAY_CONSTANT: f64 = 0.231; // ln(2)/3 for 3-day half-life
const AI_MAX_HISTORY_DAYS: f64 = 7.0; // Only consider accesses within 7 days

#[derive(Debug)]
pub struct FrecencyTracker {
    env: SharedEnv,
    db: Database<Bytes, SerdeBincode<VecDeque<u64>>>,
    health: DbHealth,
    // 2 = unknown, 1 = non-empty, 0 = empty. Avoids a read txn per scan.
    // Stale-1 after a GC purge only costs one wasted hydration pass; writes
    // always set 1, so stale-0 (skipped hydration with data) is impossible.
    empty_hint: AtomicU8,
}

const MODIFICATION_THRESHOLDS: [(i64, u64); 5] = [
    (16, 60 * 2),          // 2 minutes
    (8, 60 * 15),          // 15 minutes
    (4, 60 * 60),          // 1 hour
    (2, 60 * 60 * 24),     // 1 day
    (1, 60 * 60 * 24 * 7), // 1 week
];

// AI mode: compressed thresholds since AI edits happen in rapid bursts
const AI_MODIFICATION_THRESHOLDS: [(i64, u64); 5] = [
    (16, 30),         // 30 seconds
    (8, 60 * 5),      // 5 minutes
    (4, 60 * 15),     // 15 minutes
    (2, 60 * 60),     // 1 hour
    (1, 60 * 60 * 4), // 4 hours
];

impl DbHealthChecker for FrecencyTracker {
    fn get_env(&self) -> &heed::Env<heed::WithoutTls> {
        &self.env
    }

    fn is_healthy(&self) -> bool {
        self.health.is_healthy()
    }

    fn count_entries(&self) -> Result<Vec<(&'static str, u64)>> {
        let rtxn = self
            .env
            .read_txn()
            .map_err(|source| Error::DbStartReadTxn {
                db: Self::LABEL,
                source,
            })?;
        let count = self.db.len(&rtxn).map_err(|source| Error::DbRead {
            db: Self::LABEL,
            source,
        })?;

        Ok(vec![("absolute_frecency_entries", count)])
    }
}

impl LmdbStore for FrecencyTracker {
    const LABEL: &'static str = "frecency";
    // 10 MiB hard ceiling. Owner's db after years of use is ~560 KiB, so this
    // leaves ~18× headroom while capping runaway growth (see GH issue #437).
    const MAP_SIZE: usize = 10 * 1024 * 1024;
    const MAX_DBS: u32 = 0;
    // Erase on disk when over 8 MiB — under MAP_SIZE so the open-time erase
    // fires before MDB_MAP_FULL can.
    const SIZE_CAP_BYTES: u64 = 8 * 1024 * 1024;

    fn shared_env(&self) -> &SharedEnv {
        &self.env
    }

    fn health(&self) -> &DbHealth {
        &self.health
    }

    fn purge_stale_data(env: &SharedEnv) -> Result<()> {
        let (deleted, pruned) = Self::purge_stale_entries(env)?;
        if deleted > 0 || pruned > 0 {
            tracing::info!(deleted, pruned, "Frecency GC purged entries");
        }

        Ok(())
    }
}

impl FrecencyTracker {
    /// Returns the on-disk path of the LMDB environment directory.
    pub fn db_path(&self) -> &Path {
        self.env.path()
    }

    pub fn open(db_path: impl AsRef<Path>) -> Result<Self> {
        let db_path = db_path.as_ref();
        let (env, health) = Self::open_env(db_path)?;

        let db = Self::open_database_safe(&env, None)?;
        Ok(FrecencyTracker {
            db,
            env,
            health,
            empty_hint: AtomicU8::new(2),
        })
    }

    #[deprecated(
        since = "0.7.0",
        note = "LMDB unsafe no-lock mode is no longer supported; use `FrecencyTracker::open` instead. \
                The `_use_unsafe_no_lock` argument is ignored."
    )]
    pub fn new(db_path: impl AsRef<Path>, _use_unsafe_no_lock: bool) -> Result<Self> {
        Self::open(db_path)
    }

    /// Removes entries where all timestamps are older than MAX_HISTORY_DAYS,
    /// and prunes stale timestamps from entries that still have recent ones.
    /// Returns (deleted_count, pruned_count).
    fn purge_stale_entries(env: &SharedEnv) -> Result<(usize, usize)> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let cutoff_time = now.saturating_sub((MAX_HISTORY_DAYS * SECONDS_PER_DAY) as u64);

        let db: Database<Bytes, SerdeBincode<VecDeque<u64>>> = Self::open_database_safe(env, None)?;

        let rtxn = env.read_txn().map_err(|source| Error::DbStartReadTxn {
            db: Self::LABEL,
            source,
        })?;
        let mut to_delete: Vec<Vec<u8>> = Vec::new();
        let mut to_update: Vec<(Vec<u8>, VecDeque<u64>)> = Vec::new();

        let iter = db.iter(&rtxn).map_err(|source| Error::DbRead {
            db: Self::LABEL,
            source,
        })?;
        // Cursor order: single forward pass, then chunked write txns so a
        // large purge never holds one huge txn (MDB_MAP_FULL/abort risk).
        for result in iter {
            let (key, accesses) = result.map_err(|source| Error::DbRead {
                db: Self::LABEL,
                source,
            })?;

            // Timestamps chronologically ordered (oldest at front).
            let fresh_start = accesses.iter().position(|&ts| ts >= cutoff_time);
            match fresh_start {
                None => to_delete.push(key.to_vec()),
                Some(0) => {}
                Some(start) => {
                    let pruned: VecDeque<u64> = accesses.iter().skip(start).copied().collect();
                    to_update.push((key.to_vec(), pruned));
                }
            }
        }
        drop(rtxn);

        if to_delete.is_empty() && to_update.is_empty() {
            return Ok((0, 0));
        }

        // Chunked commits bound each write txn; values drain as we go.
        const PURGE_CHUNK: usize = 1_024;
        let n_delete = to_delete.len();
        let n_update = to_update.len();
        for chunk in to_delete.chunks(PURGE_CHUNK) {
            let mut wtxn = env.write_txn().map_err(|source| Error::DbStartWriteTxn {
                db: Self::LABEL,
                source,
            })?;
            for key in chunk {
                db.delete(&mut wtxn, key).map_err(|source| Error::DbWrite {
                    db: Self::LABEL,
                    source,
                })?;
            }
            wtxn.commit().map_err(|source| Error::DbCommit {
                db: Self::LABEL,
                source,
            })?;
        }
        for chunk in to_update.chunks(PURGE_CHUNK) {
            let mut wtxn = env.write_txn().map_err(|source| Error::DbStartWriteTxn {
                db: Self::LABEL,
                source,
            })?;
            for (key, accesses) in chunk {
                db.put(&mut wtxn, key, accesses)
                    .map_err(|source| Error::DbWrite {
                        db: Self::LABEL,
                        source,
                    })?;
            }
            wtxn.commit().map_err(|source| Error::DbCommit {
                db: Self::LABEL,
                source,
            })?;
        }

        Ok((n_delete, n_update))
    }

    fn get_accesses(&self, path: &Path) -> Result<Option<VecDeque<u64>>> {
        let key_hash = Self::path_to_hash_bytes(path)?;

        let rtxn = self
            .env
            .read_txn()
            .map_err(|source| Error::DbStartReadTxn {
                db: Self::LABEL,
                source,
            })?;
        let result = self
            .db
            .get(&rtxn, &key_hash)
            .map_err(|source| Error::DbRead {
                db: Self::LABEL,
                source,
            })?;
        drop(rtxn);

        Ok(result)
    }

    pub fn is_empty(&self) -> bool {
        if self.empty_hint.load(Ordering::Relaxed) == 1 {
            return false;
        }
        let Ok(rtxn) = self.env.read_txn() else {
            return true;
        };
        let Ok(stat) = self.db.stat(&rtxn) else {
            return true;
        };
        let empty = stat.entries == 0;
        self.empty_hint.store(u8::from(!empty), Ordering::Relaxed);
        empty
    }

    fn get_now(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    fn path_to_hash_bytes(path: &Path) -> Result<[u8; 32]> {
        // On Windows, resolve to the canonical form (short-name/case/symlink)
        // so the same file always hashes to one key regardless of how the
        // caller spelled it. Falls back to the raw path when the file no
        // longer exists (e.g. watcher delete events), so the op is never
        // dropped. No-op on other platforms.
        #[cfg(windows)]
        let canonical: Option<std::path::PathBuf> = crate::path_utils::canonicalize(path).ok();
        #[cfg(windows)]
        let path: &Path = canonical.as_deref().unwrap_or(path);

        Self::path_hash_raw(path)
    }

    /// Hash many paths in parallel on the bg pool. Scan hydration hashes
    /// thousands of absolute paths; serial blake3 before the single read txn
    /// was a visible slice of that phase on 200k trees.
    pub fn path_hashes_parallel(paths: &[&Path]) -> Vec<[u8; 32]> {
        if paths.is_empty() {
            return Vec::new();
        }
        crate::parallelism::run_on_bg_pool(|| {
            use rayon::prelude::*;
            paths
                .par_iter()
                .map(|p| Self::path_hash_raw(p).unwrap_or([0u8; 32]))
                .collect()
        })
    }

    /// Hash a path without canonicalize. For scan-built absolute paths that
    /// are already spelled consistently from `base_path` this avoids one
    /// Windows `canonicalize` syscall per file (families: io, cpu).
    pub fn path_hash_raw(path: &Path) -> Result<[u8; 32]> {
        let Some(key) = path.to_str() else {
            return Err(Error::InvalidPath(path.to_path_buf()));
        };

        Ok(*blake3::hash(key.as_bytes()).as_bytes())
    }

    /// Returns seconds since the most recent tracked access, or `None` if the
    /// file has never been tracked.
    pub fn seconds_since_last_access(&self, path: &Path) -> Result<Option<u64>> {
        let accesses = self.get_accesses(path)?;
        let last = accesses.and_then(|a| a.back().copied());
        Ok(last.map(|ts| self.get_now().saturating_sub(ts)))
    }

    /// Number of tracked access for file path
    pub fn access_count(&self, path: &Path) -> Result<usize> {
        Ok(self.get_accesses(path)?.map_or(0, |a| a.len()))
    }

    pub fn track_access(&self, path: &Path) -> Result<()> {
        let key_hash = Self::path_to_hash_bytes(path)?;
        let mut accesses = self.get_accesses(path)?.unwrap_or_default();

        let now = self.get_now();
        let cutoff_time = now.saturating_sub((MAX_HISTORY_DAYS * SECONDS_PER_DAY) as u64);

        // Drop stale timestamps from the front while also enforcing the
        // per-file cap. Reserves one slot for the `push_back` below.
        while let Some(&front_time) = accesses.front() {
            if front_time < cutoff_time || accesses.len() >= MAX_TIMESTAMPS_PER_FILE {
                accesses.pop_front();
            } else {
                break;
            }
        }

        accesses.push_back(now);
        tracing::debug!(?path, accesses = accesses.len(), "Tracking access");

        let mut wtxn = self
            .env
            .write_txn()
            .map_err(|source| Error::DbStartWriteTxn {
                db: Self::LABEL,
                source,
            })?;
        if let Err(e) = self.db.put(&mut wtxn, &key_hash, &accesses) {
            if is_map_full(&e) {
                self.health.mark_unhealthy("MDB_MAP_FULL on put");
                tracing::error!(
                    ?path,
                    "Frecency DB hit MDB_MAP_FULL; dropping write — db will be \
                     erased on next open via LmdbStore::erase_if_oversized"
                );
                return Ok(());
            }

            return Err(Error::DbWrite {
                db: Self::LABEL,
                source: e,
            });
        }

        wtxn.commit()
            .inspect_err(|e| {
                if is_map_full(e) {
                    self.health.mark_unhealthy("MDB_MAP_FULL on commit");
                    tracing::error!(
                        ?path,
                        "Frecency DB hit MDB_MAP_FULL on commit; dropping write"
                    );
                }
            })
            .map_err(|source| Error::DbCommit {
                db: Self::LABEL,
                source,
            })?;
        self.empty_hint.store(1, Ordering::Relaxed);
        Ok(())
    }

    /// Batch variant: one read txn for all lookups, then chunked write txns
    /// of `WRITE_TXN_CHUNK` puts. Replaces N×(read+write) with 1 read + N/5k
    /// writes for bulk imports / watcher replays.
    pub fn track_accesses(&self, paths: &[&Path]) -> Result<usize> {
        if paths.is_empty() {
            return Ok(0);
        }
        // Unhashable (non-UTF8) paths cannot round-trip through reads
        // either; skip them rather than colliding on the zero key.
        let hashes: Vec<[u8; 32]> = crate::parallelism::run_on_bg_pool(|| {
            use rayon::prelude::*;
            paths
                .par_iter()
                .filter_map(|p| Self::path_to_hash_bytes(p).ok())
                .collect()
        });
        let now = self.get_now();
        let cutoff_time = now.saturating_sub((MAX_HISTORY_DAYS * SECONDS_PER_DAY) as u64);
        let rtxn = self
            .env
            .read_txn()
            .map_err(|source| Error::DbStartReadTxn {
                db: Self::LABEL,
                source,
            })?;
        let mut pending: Vec<([u8; 32], VecDeque<u64>)> = Vec::with_capacity(hashes.len());
        for hash in &hashes {
            let mut accesses: VecDeque<u64> = self
                .db
                .get(&rtxn, hash)
                .map_err(|source| Error::DbRead {
                    db: Self::LABEL,
                    source,
                })?
                .unwrap_or_default();
            while let Some(&front) = accesses.front() {
                if front < cutoff_time || accesses.len() >= MAX_TIMESTAMPS_PER_FILE {
                    accesses.pop_front();
                } else {
                    break;
                }
            }
            accesses.push_back(now);
            pending.push((*hash, accesses));
        }
        drop(rtxn);
        let mut done = 0usize;
        for chunk in pending.chunks(WRITE_TXN_CHUNK) {
            let mut wtxn = self
                .env
                .write_txn()
                .map_err(|source| Error::DbStartWriteTxn {
                    db: Self::LABEL,
                    source,
                })?;
            for (hash, accesses) in chunk {
                if let Err(e) = self.db.put(&mut wtxn, hash, accesses) {
                    if is_map_full(&e) {
                        self.health.mark_unhealthy("MDB_MAP_FULL on batch put");
                        return Ok(done);
                    }
                    return Err(Error::DbWrite {
                        db: Self::LABEL,
                        source: e,
                    });
                }
                done += 1;
            }
            if let Err(e) = wtxn.commit() {
                if is_map_full(&e) {
                    self.health.mark_unhealthy("MDB_MAP_FULL on batch commit");
                    return Ok(done);
                }
                return Err(Error::DbCommit {
                    db: Self::LABEL,
                    source: e,
                });
            }
        }
        if done > 0 {
            self.empty_hint.store(1, Ordering::Relaxed);
        }
        Ok(done)
    }

    pub fn copy_history(&self, from: &Path, to: &Path) -> Result<bool> {
        if from == to {
            return Ok(false);
        }

        let Some(source) = self.get_accesses(from)?.filter(|a| !a.is_empty()) else {
            return Ok(false);
        };

        let target_key = Self::path_to_hash_bytes(to)?;
        if self.get_accesses(to)?.as_ref() == Some(&source) {
            return Ok(false);
        }

        tracing::debug!(
            ?from,
            ?to,
            accesses = source.len(),
            "Copying frecency history"
        );

        let mut wtxn = self
            .env
            .write_txn()
            .map_err(|source| Error::DbStartWriteTxn {
                db: Self::LABEL,
                source,
            })?;
        if let Err(e) = self.db.put(&mut wtxn, &target_key, &source) {
            if is_map_full(&e) {
                self.health.mark_unhealthy("MDB_MAP_FULL");
                tracing::error!(?to, "Frecency DB is full on commit");
                return Ok(false);
            }
            return Err(Error::DbWrite {
                db: Self::LABEL,
                source: e,
            });
        }

        if let Err(e) = wtxn.commit() {
            if is_map_full(&e) {
                self.health.mark_unhealthy("MDB_MAP_FULL on commit");
                tracing::error!(?to, "Frecency DB is full on commit");
                return Ok(false);
            }
            return Err(Error::DbCommit {
                db: Self::LABEL,
                source: e,
            });
        }

        Ok(true)
    }

    pub fn get_access_score(&self, file_path: &Path, mode: FFFMode) -> i64 {
        let accesses = self
            .get_accesses(file_path)
            .ok()
            .flatten()
            .unwrap_or_default();
        self.score_accesses(&accesses, self.get_now(), mode)
    }

    /// Score a batch of precomputed path hashes with one LMDB read txn.
    ///
    /// Use [`Self::path_hash_raw`] for scan-built absolute paths so hydration
    /// does not canonicalize every file on Windows.
    pub fn get_access_scores_for_hashes(&self, hashes: &[[u8; 32]], mode: FFFMode) -> Vec<i64> {
        let mut out = vec![0i64; hashes.len()];
        if hashes.is_empty() {
            return out;
        }

        let Ok(rtxn) = self.env.read_txn() else {
            return out;
        };
        let now = self.get_now();
        for (slot, hash) in out.iter_mut().zip(hashes.iter()) {
            let Ok(accesses) = self.db.get(&rtxn, hash) else {
                continue;
            };
            let Some(accesses) = accesses else {
                continue;
            };
            *slot = self.score_accesses(&accesses, now, mode);
        }
        drop(rtxn);
        out
    }

    fn score_accesses(&self, accesses: &VecDeque<u64>, now: u64, mode: FFFMode) -> i64 {
        if accesses.is_empty() {
            return 0;
        }

        let decay_constant = if mode.is_ai() {
            AI_DECAY_CONSTANT
        } else {
            DECAY_CONSTANT
        };
        let max_history_days = if mode.is_ai() {
            AI_MAX_HISTORY_DAYS
        } else {
            MAX_HISTORY_DAYS
        };

        let mut total_frecency = 0.0;

        let cutoff_time = now.saturating_sub((max_history_days * SECONDS_PER_DAY) as u64);

        for &access_time in accesses.iter().rev() {
            if access_time < cutoff_time {
                break; // All remaining entries are older, stop processing
            }

            let days_ago = (now.saturating_sub(access_time) as f64) / SECONDS_PER_DAY;
            let decay_factor = (-decay_constant * days_ago).exp();
            total_frecency += decay_factor;
        }

        let normalized_frecency = if total_frecency <= 10.0 {
            total_frecency
        } else {
            10.0 + (total_frecency - 10.0).sqrt() // Diminishing: >10 accesses grow slowly
        };

        normalized_frecency.round() as i64
    }

    /// Calculating modification score but only if the file is modified in the current git dir
    pub fn get_modification_score(
        &self,
        modified_time: u64,
        git_status: Option<git2::Status>,
        mode: FFFMode,
    ) -> i64 {
        let is_modified_git_status = git_status.is_some_and(is_modified_status);
        if !is_modified_git_status {
            return 0;
        }

        let thresholds = if mode.is_ai() {
            &AI_MODIFICATION_THRESHOLDS
        } else {
            &MODIFICATION_THRESHOLDS
        };

        let now = self.get_now();
        let duration_since = now.saturating_sub(modified_time);

        for i in 0..thresholds.len() {
            let (current_points, current_threshold) = thresholds[i];

            if duration_since <= current_threshold {
                if i == 0 || duration_since == current_threshold {
                    return current_points;
                }

                let (prev_points, prev_threshold) = thresholds[i - 1];

                let time_range = current_threshold - prev_threshold;
                let time_offset = duration_since - prev_threshold;
                let points_diff = prev_points - current_points;

                let interpolated_score =
                    prev_points - (points_diff * time_offset as i64) / time_range as i64;

                return interpolated_score;
            }
        }

        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file_picker::FFFMode;

    // A path that doesn't exist on disk must still hash (canonicalize fails on
    // Windows → falls back to the raw string), so watcher delete events and
    // raced files never drop their frecency op.
    #[test]
    fn hashes_nonexistent_path_without_error() {
        let missing = Path::new("/this/path/definitely/does/not/exist/frecency_test_xyz");
        assert!(FrecencyTracker::path_to_hash_bytes(missing).is_ok());
    }

    #[test]
    fn batch_scores_match_single_lookup() {
        let dir = std::env::temp_dir().join(format!("fff-frecency-batch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let tracker = FrecencyTracker::open(&dir).unwrap();

        let a = Path::new("/tmp/fff_batch_a.rs");
        let b = Path::new("/tmp/fff_batch_b.rs");
        let now = tracker.get_now();
        tracker.track_access(a).unwrap();

        let hashes = [
            FrecencyTracker::path_hash_raw(a).unwrap(),
            FrecencyTracker::path_hash_raw(b).unwrap(),
        ];
        let batch = tracker.get_access_scores_for_hashes(&hashes, FFFMode::Neovim);
        let single_a = tracker.get_access_score(a, FFFMode::Neovim);
        let single_b = tracker.get_access_score(b, FFFMode::Neovim);
        assert_eq!(batch.len(), 2);
        // Single lookup canonicalizes on Windows; raw hash matches when the
        // path spelling is already canonical (these temp paths are).
        assert_eq!(batch[0], single_a);
        assert_eq!(batch[1], single_b);
        assert!(batch[0] > 0);
        assert_eq!(batch[1], 0);
        let _ = now;
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn batch_track_matches_single_writes() {
        let dir = std::env::temp_dir().join(format!("fff-frecency-batchw-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let tracker = FrecencyTracker::open(&dir).unwrap();
        assert!(tracker.is_empty());

        let a = Path::new("/tmp/fff_batchw_a.rs");
        let b = Path::new("/tmp/fff_batchw_b.rs");
        let n = tracker.track_accesses(&[a, b]).unwrap();
        assert_eq!(n, 2);
        assert!(!tracker.is_empty());
        // Cached hint avoids a second read txn.
        assert!(!tracker.is_empty());
        assert_eq!(tracker.access_count(a).unwrap(), 1);
        assert_eq!(tracker.access_count(b).unwrap(), 1);
        assert_eq!(
            FrecencyTracker::path_hashes_parallel(&[a, b]),
            vec![
                FrecencyTracker::path_hash_raw(a).unwrap(),
                FrecencyTracker::path_hash_raw(b).unwrap(),
            ]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn calculate_test_frecency_score(access_timestamps: &[u64], current_time: u64) -> i64 {
        let mut total_frecency = 0.0;

        for &access_time in access_timestamps {
            let days_ago = (current_time.saturating_sub(access_time) as f64) / SECONDS_PER_DAY;
            let decay_factor = (-DECAY_CONSTANT * days_ago).exp();
            total_frecency += decay_factor;
        }

        let normalized_frecency = if total_frecency <= 20.0 {
            total_frecency
        } else {
            20.0 + (total_frecency - 10.0).sqrt()
        };

        normalized_frecency.round() as i64
    }

    #[test]
    fn test_frecency_calculation() {
        let current_time = 1000000000; // Base timestamp

        let score = calculate_test_frecency_score(&[], current_time);
        assert_eq!(score, 0);

        let accesses = [current_time]; // Accessed right now
        let score = calculate_test_frecency_score(&accesses, current_time);
        assert_eq!(score, 1); // 1.0 decay factor = 1

        let ten_days_seconds = 10 * 86400; // 10 days in seconds
        let accesses = [current_time - ten_days_seconds];
        let score = calculate_test_frecency_score(&accesses, current_time);
        assert_eq!(score, 1); // ~0.5 decay factor rounds to 1

        let accesses = [
            current_time,          // Today
            current_time - 86400,  // 1 day ago
            current_time - 172800, // 2 days ago
        ];
        let score = calculate_test_frecency_score(&accesses, current_time);
        assert!(score > 2 && score < 4, "Score: {}", score); // About 3 accesses with decay

        let thirty_days = 30 * 86400;
        let accesses = [current_time - thirty_days]; // 30 days ago
        let score = calculate_test_frecency_score(&accesses, current_time);
        assert!(
            score < 2,
            "Old access should have minimal score, got: {}",
            score
        );

        let recent_frequent = [current_time, current_time - 86400, current_time - 172800];
        let old_single = [current_time - ten_days_seconds];

        let recent_score = calculate_test_frecency_score(&recent_frequent, current_time);
        let old_score = calculate_test_frecency_score(&old_single, current_time);

        assert!(
            recent_score > old_score,
            "Recent frequent access ({}) should score higher than old single access ({})",
            recent_score,
            old_score
        );
    }

    #[test]
    fn test_modification_score_interpolation() {
        let temp_dir = std::env::temp_dir().join("fff_test_interpolation");
        let _ = std::fs::remove_dir_all(&temp_dir);
        let tracker = FrecencyTracker::open(temp_dir.to_str().unwrap()).unwrap();

        let current_time = tracker.get_now();
        let git_status = Some(git2::Status::WT_MODIFIED);

        // At 5 minutes: should interpolate between 16 and 8 points
        let five_minutes_ago = current_time - (5 * 60);
        let score = tracker.get_modification_score(five_minutes_ago, git_status, FFFMode::Neovim);

        // Expected: 16 - (8 * 3 / 13) = 16 - 1 = 15 points
        // (time_offset = 5-2 = 3, time_range = 15-2 = 13, points_diff = 16-8 = 8)
        assert_eq!(score, 15, "5 minutes should interpolate to 15 points");

        let two_minutes_ago = current_time - (2 * 60);
        let score = tracker.get_modification_score(two_minutes_ago, git_status, FFFMode::Neovim);
        assert_eq!(score, 16, "2 minutes should be exactly 16 points");

        let fifteen_minutes_ago = current_time - (15 * 60);
        let score =
            tracker.get_modification_score(fifteen_minutes_ago, git_status, FFFMode::Neovim);
        assert_eq!(score, 8, "15 minutes should be exactly 8 points");

        // At 12 hours: should interpolate between 4 and 2 points
        let twelve_hours_ago = current_time - (12 * 60 * 60);
        let score = tracker.get_modification_score(twelve_hours_ago, git_status, FFFMode::Neovim);
        // Expected: 4 - (2 * 11 / 23) = 4 - 0 = 4 points (integer division)
        // (time_offset = 12-1 = 11 hours, time_range = 24-1 = 23 hours, points_diff = 4-2 = 2)
        assert_eq!(score, 4, "12 hours should interpolate to 4 points");

        // at 18 hours for more significant interpolation
        let eighteen_hours_ago = current_time - (18 * 60 * 60);
        let score = tracker.get_modification_score(eighteen_hours_ago, git_status, FFFMode::Neovim);
        // Expected: 4 - (2 * 17 / 23) = 4 - 1 = 3 points
        assert_eq!(score, 3, "18 hours should interpolate to 3 points");

        let score = tracker.get_modification_score(five_minutes_ago, None, FFFMode::Neovim);
        assert_eq!(score, 0, "No git status should return 0");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
