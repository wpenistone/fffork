use fff::file_picker::FilePicker;
use fff::git::format_git_status;
use fff::{
    DirItem, DirSearchResult, FileItem, GrepResult, Location, MixedItemRef, MixedSearchResult,
    Score, SearchResult,
};
use mlua::prelude::*;

const PATH_SCRATCH: usize = 8192;

struct PathBlob {
    bytes: std::sync::Arc<Vec<u8>>,
}

struct FileSnap {
    path_start: u32,
    path_len: u32,
    name_off: u32,
    size: u64,
    modified: u64,
    access_frecency_score: i16,
    modification_frecency_score: i16,
    total_frecency_score: i32,
    git_status: &'static str,
    git_recency_score: i16,
    is_binary: bool,
}

struct DirSnap {
    path_start: u32,
    path_len: u32,
    name_off: u32,
    max_access_frecency: i32,
}

enum MixedSnap {
    File(FileSnap),
    Dir(DirSnap),
}

impl PathBlob {
    fn push(&mut self, rel: &str) -> (u32, u32) {
        let bytes = std::sync::Arc::make_mut(&mut self.bytes);
        let start = bytes.len() as u32;
        bytes.extend_from_slice(rel.as_bytes());
        (start, rel.len() as u32)
    }

    fn get(&self, start: u32, len: u32) -> &str {
        let s = start as usize;
        let e = s + len as usize;
        std::str::from_utf8(&self.bytes[s..e]).unwrap_or("")
    }
}

fn path_blob(cap: usize) -> PathBlob {
    PathBlob {
        bytes: std::sync::Arc::new(Vec::with_capacity(cap)),
    }
}

fn empty_blob() -> PathBlob {
    PathBlob {
        bytes: std::sync::Arc::new(Vec::new()),
    }
}

fn snap_from_row(row: &fff::CachedFileRow) -> FileSnap {
    FileSnap {
        path_start: row.start,
        path_len: row.len,
        name_off: row.name_off,
        size: row.size,
        modified: row.modified,
        access_frecency_score: row.access_frecency_score,
        modification_frecency_score: row.modification_frecency_score,
        total_frecency_score: row.total_frecency_score,
        git_status: row.git_status,
        git_recency_score: row.git_recency_score,
        is_binary: row.is_binary,
    }
}

fn snap_meta(item: &FileItem, path_start: u32, path_len: u32) -> FileSnap {
    FileSnap {
        path_start,
        path_len,
        name_off: item.filename_offset_in_relative_path().min(path_len as usize) as u32,
        size: item.size(),
        modified: item.modified(),
        access_frecency_score: item.access_frecency_score(),
        modification_frecency_score: item.modification_frecency_score(),
        total_frecency_score: item.total_frecency_score(),
        git_status: format_git_status(item.git_status()),
        git_recency_score: item.git_recency_score(),
        is_binary: item.is_binary(),
    }
}

fn snap_file(blob: &mut PathBlob, scratch: &mut [u8], item: &FileItem, picker: &FilePicker) -> FileSnap {
    let len = item.relative_path_len();
    let (path_start, path_len) = if len <= scratch.len() {
        let rel = item.read_relative_path(picker, scratch);
        blob.push(rel)
    } else {
        blob.push(&item.relative_path(picker))
    };
    snap_meta(item, path_start, path_len)
}

fn snap_dir(blob: &mut PathBlob, scratch: &mut [u8], item: &DirItem, picker: &FilePicker) -> DirSnap {
    let cap = scratch.len();
    let rel = item.read_relative_path_into(picker, scratch);
    let (path_start, path_len) = if rel.len() < cap {
        blob.push(rel)
    } else {
        blob.push(&item.relative_path(picker))
    };
    DirSnap {
        path_start,
        path_len,
        name_off: (item.last_segment_offset() as usize).min(path_len as usize) as u32,
        max_access_frecency: item.max_access_frecency(),
    }
}

pub struct SearchResultLua {
    paths: PathBlob,
    items: Vec<FileSnap>,
    scores: fff::SharedScores,
    match_ranges: fff::SharedOffsets,
    total_matched: usize,
    total_files: usize,
    location: Option<Location>,
}

impl SearchResultLua {
    /// Path bytes are copied out so Lua runs after the picker read guard drops.
    pub fn new(mut inner: SearchResult<'_>, picker: &FilePicker) -> Self {
        let mut scratch = [0u8; PATH_SCRATCH];
        // A scored page already holds path bytes. Summing `items` would allocate a buffer and drop it.
        let (paths, items) = if let Some(snapshot) = inner.paths.take() {
            (
                PathBlob {
                    bytes: snapshot.bytes,
                },
                snapshot.rows.iter().map(snap_from_row).collect(),
            )
        } else {
            let mut paths =
                path_blob(inner.items.iter().map(|item| item.relative_path_len()).sum());
            let items = inner
                .items
                .iter()
                .map(|item| snap_file(&mut paths, &mut scratch, item, picker))
                .collect();
            (paths, items)
        };
        Self {
            paths,
            items,
            scores: std::sync::Arc::clone(&inner.scores),
            match_ranges: std::sync::Arc::clone(&inner.match_byte_offsets),
            total_matched: inner.total_matched,
            total_files: inner.total_files,
            location: inner.location,
        }
    }

    pub fn empty(total_files: usize, location: Option<Location>) -> Self {
        Self {
            paths: empty_blob(),
            items: Vec::new(),
            scores: fff::SharedScores::new(Vec::new()),
            match_ranges: fff::SharedOffsets::new(Vec::new()),
            total_matched: 0,
            total_files,
            location,
        }
    }
}

pub struct GrepResultLua {
    paths: PathBlob,
    matches: Vec<fff::GrepMatch>,
    files: Vec<FileSnap>,
    total_files_searched: usize,
    total_files: usize,
    filtered_file_count: usize,
    next_file_offset: usize,
    regex_fallback_error: Option<String>,
}

impl GrepResultLua {
    /// Path bytes are copied out so Lua runs after the picker read guard drops.
    pub fn new(inner: GrepResult<'_>, picker: &FilePicker) -> Self {
        let mut paths = path_blob(inner.files.iter().map(|item| item.relative_path_len()).sum());
        let mut scratch = [0u8; PATH_SCRATCH];
        let files = inner
            .files
            .iter()
            .map(|item| snap_file(&mut paths, &mut scratch, item, picker))
            .collect();
        Self {
            paths,
            files,
            matches: inner.matches,
            total_files_searched: inner.total_files_searched,
            total_files: inner.total_files,
            filtered_file_count: inner.filtered_file_count,
            next_file_offset: inner.next_file_offset,
            regex_fallback_error: inner.regex_fallback_error,
        }
    }
}

pub struct DirSearchResultLua {
    paths: PathBlob,
    items: Vec<DirSnap>,
    scores: Vec<Score>,
    total_matched: usize,
    total_dirs: usize,
}

impl DirSearchResultLua {
    /// Path bytes are copied out so Lua runs after the picker read guard drops.
    pub fn new(inner: DirSearchResult<'_>, picker: &FilePicker) -> Self {
        let mut paths = empty_blob();
        let mut scratch = [0u8; PATH_SCRATCH];
        let items = inner
            .items
            .iter()
            .map(|item| snap_dir(&mut paths, &mut scratch, item, picker))
            .collect();
        Self {
            paths,
            items,
            scores: inner.scores,
            total_matched: inner.total_matched,
            total_dirs: inner.total_dirs,
        }
    }
}

pub struct MixedSearchResultLua {
    paths: PathBlob,
    items: Vec<MixedSnap>,
    scores: Vec<Score>,
    total_matched: usize,
    total_files: usize,
    total_dirs: usize,
    location: Option<Location>,
}

impl MixedSearchResultLua {
    /// Path bytes are copied out so Lua runs after the picker read guard drops.
    pub fn new(inner: MixedSearchResult<'_>, picker: &FilePicker) -> Self {
        let mut paths = empty_blob();
        let mut scratch = [0u8; PATH_SCRATCH];
        let items = inner
            .items
            .into_iter()
            .map(|item| match item {
                MixedItemRef::File(file) => MixedSnap::File(snap_file(&mut paths, &mut scratch, file, picker)),
                MixedItemRef::Dir(dir) => MixedSnap::Dir(snap_dir(&mut paths, &mut scratch, dir, picker)),
            })
            .collect();
        Self {
            paths,
            items,
            scores: inner.scores,
            total_matched: inner.total_matched,
            total_files: inner.total_files,
            total_dirs: inner.total_dirs,
            location: inner.location,
        }
    }
}

struct LuaPosition((i32, i32));

impl IntoLua for LuaPosition {
    fn into_lua(self, lua: &Lua) -> LuaResult<LuaValue> {
        let table = lua.create_table_with_capacity(0, 2)?;
        table.set("line", self.0.0)?;
        table.set("col", self.0.1)?;
        Ok(LuaValue::Table(table))
    }
}

// Field names and the two repeated type tags, built once per result.
struct Keys {
    typ: LuaString,
    file_ty: LuaString,
    dir_ty: LuaString,
    empty: LuaString,
    relative_path: LuaString,
    name: LuaString,
    directory: LuaString,
    size: LuaString,
    modified: LuaString,
    access_frecency_score: LuaString,
    modification_frecency_score: LuaString,
    total_frecency_score: LuaString,
    git_status: LuaString,
    git_recency_score: LuaString,
    is_binary: LuaString,
    match_ranges: LuaString,
    max_access_frecency: LuaString,
    total: LuaString,
    base_score: LuaString,
    filename_bonus: LuaString,
    special_filename_bonus: LuaString,
    frecency_boost: LuaString,
    git_status_boost: LuaString,
    git_recency_boost: LuaString,
    distance_penalty: LuaString,
    current_file_penalty: LuaString,
    combo_match_boost: LuaString,
    path_alignment_bonus: LuaString,
    match_type: LuaString,
    exact_match: LuaString,
    line_number: LuaString,
    col: LuaString,
    byte_offset: LuaString,
    is_binary_content: LuaString,
    line_content: LuaString,
    is_definition: LuaString,
    fuzzy_score: LuaString,
    items: LuaString,
    scores: LuaString,
    total_matched: LuaString,
    total_files: LuaString,
    total_dirs: LuaString,
    location: LuaString,
    total_files_searched: LuaString,
    filtered_file_count: LuaString,
    next_file_offset: LuaString,
    regex_fallback_error: LuaString,
}

impl Keys {
    fn load(lua: &Lua) -> LuaResult<Self> {
        let s = |text: &str| lua.create_string(text);
        Ok(Self {
            typ: s("type")?,
            file_ty: s("file")?,
            dir_ty: s("directory")?,
            empty: s("")?,
            relative_path: s("relative_path")?,
            name: s("name")?,
            directory: s("directory")?,
            size: s("size")?,
            modified: s("modified")?,
            access_frecency_score: s("access_frecency_score")?,
            modification_frecency_score: s("modification_frecency_score")?,
            total_frecency_score: s("total_frecency_score")?,
            git_status: s("git_status")?,
            git_recency_score: s("git_recency_score")?,
            is_binary: s("is_binary")?,
            match_ranges: s("match_ranges")?,
            max_access_frecency: s("max_access_frecency")?,
            total: s("total")?,
            base_score: s("base_score")?,
            filename_bonus: s("filename_bonus")?,
            special_filename_bonus: s("special_filename_bonus")?,
            frecency_boost: s("frecency_boost")?,
            git_status_boost: s("git_status_boost")?,
            git_recency_boost: s("git_recency_boost")?,
            distance_penalty: s("distance_penalty")?,
            current_file_penalty: s("current_file_penalty")?,
            combo_match_boost: s("combo_match_boost")?,
            path_alignment_bonus: s("path_alignment_bonus")?,
            match_type: s("match_type")?,
            exact_match: s("exact_match")?,
            line_number: s("line_number")?,
            col: s("col")?,
            byte_offset: s("byte_offset")?,
            is_binary_content: s("is_binary_content")?,
            line_content: s("line_content")?,
            is_definition: s("is_definition")?,
            fuzzy_score: s("fuzzy_score")?,
            items: s("items")?,
            scores: s("scores")?,
            total_matched: s("total_matched")?,
            total_files: s("total_files")?,
            total_dirs: s("total_dirs")?,
            location: s("location")?,
            total_files_searched: s("total_files_searched")?,
            filtered_file_count: s("filtered_file_count")?,
            next_file_offset: s("next_file_offset")?,
            regex_fallback_error: s("regex_fallback_error")?,
        })
    }
}

// Field names live on the Lua state. Later results reuse those strings.
fn cached_keys(lua: &Lua) -> LuaResult<mlua::AppDataRef<'_, Keys>> {
    if let Some(keys) = lua.app_data_ref::<Keys>() {
        return Ok(keys);
    }
    lua.set_app_data(Keys::load(lua)?);
    Ok(lua
        .app_data_ref::<Keys>()
        .expect("field names stay on this Lua state"))
}

// One static Lua string per distinct text inside a single result.
struct Intern {
    text: Option<&'static str>,
    lua: Option<LuaString>,
}

impl Intern {
    fn new() -> Self {
        Self {
            text: None,
            lua: None,
        }
    }

    fn set(&mut self, lua: &Lua, table: &LuaTable, key: &LuaString, text: &'static str) -> LuaResult<()> {
        if self.text != Some(text) {
            self.lua = Some(lua.create_string(text)?);
            self.text = Some(text);
        }
        table.raw_set(key, self.lua.as_ref().unwrap())
    }
}

enum Piece {
    Empty,
    Same,
    Owned(LuaString),
}

// Last directory and filename Lua strings. Reused when the next row's bytes match.
struct DirCache {
    start: u32,
    len: u32,
    lua: Option<LuaString>,
    name_start: u32,
    name_len: u32,
    name_lua: Option<LuaString>,
}

impl DirCache {
    fn new() -> Self {
        Self {
            start: 0,
            len: 0,
            lua: None,
            name_start: 0,
            name_len: 0,
            name_lua: None,
        }
    }

    fn reuse_slot<'a>(blob: &[u8], text: &str, start: u32, len: u32, lua: &'a Option<LuaString>) -> Option<&'a LuaString> {
        let lua = lua.as_ref()?;
        let start = start as usize;
        let end = start + len as usize;
        if end <= blob.len() && &blob[start..end] == text.as_bytes() {
            Some(lua)
        } else {
            None
        }
    }

    fn reuse_dir(&self, blob: &[u8], dir: &str) -> Option<&LuaString> {
        Self::reuse_slot(blob, dir, self.start, self.len, &self.lua)
    }

    fn reuse_name(&self, blob: &[u8], name: &str) -> Option<&LuaString> {
        Self::reuse_slot(blob, name, self.name_start, self.name_len, &self.name_lua)
    }

    fn store(blob: &[u8], text: &str, lua: LuaString, start: &mut u32, len: &mut u32, slot: &mut Option<LuaString>) {
        let base = blob.as_ptr() as usize;
        let ptr = text.as_ptr() as usize;
        if ptr >= base && ptr + text.len() <= base + blob.len() {
            *start = (ptr - base) as u32;
            *len = text.len() as u32;
            *slot = Some(lua);
        }
    }
}

struct SharedPath {
    index: usize,
    path: LuaString,
    name: Piece,
    dir: Piece,
}

fn put_piece(table: &LuaTable, key: &LuaString, piece: &Piece, path: &LuaString, empty: &LuaString) -> LuaResult<()> {
    match piece {
        Piece::Owned(text) => table.raw_set(key, text),
        Piece::Same => table.raw_set(key, path),
        Piece::Empty => table.raw_set(key, empty),
    }
}

// `directory` is the parent path, "" at the root. Same text as fnamemodify(':h')
// after '.' is dropped. Name and directory are slices of `rel`.
fn set_path_fields(
    lua: &Lua,
    keys: &Keys,
    table: &LuaTable,
    blob: &[u8],
    rel: &str,
    name: &str,
    dir: &str,
    dir_cache: &mut DirCache,
) -> LuaResult<(LuaString, Piece, Piece)> {
    let path = lua.create_string(rel)?;
    table.raw_set(&keys.relative_path, &path)?;
    let name_piece = if name.is_empty() {
        table.raw_set(&keys.name, &keys.empty)?;
        Piece::Empty
    } else if name == rel {
        table.raw_set(&keys.name, &path)?;
        Piece::Same
    } else if let Some(cached) = dir_cache.reuse_name(blob, name) {
        let owned = cached.clone();
        table.raw_set(&keys.name, &owned)?;
        Piece::Owned(owned)
    } else {
        let owned = lua.create_string(name)?;
        DirCache::store(
            blob,
            name,
            owned.clone(),
            &mut dir_cache.name_start,
            &mut dir_cache.name_len,
            &mut dir_cache.name_lua,
        );
        table.raw_set(&keys.name, &owned)?;
        Piece::Owned(owned)
    };
    let dir_piece = if dir.is_empty() {
        table.raw_set(&keys.directory, &keys.empty)?;
        Piece::Empty
    } else if dir == rel {
        table.raw_set(&keys.directory, &path)?;
        Piece::Same
    } else if let Some(cached) = dir_cache.reuse_dir(blob, dir) {
        let owned = cached.clone();
        table.raw_set(&keys.directory, &owned)?;
        Piece::Owned(owned)
    } else {
        let owned = lua.create_string(dir)?;
        DirCache::store(
            blob,
            dir,
            owned.clone(),
            &mut dir_cache.start,
            &mut dir_cache.len,
            &mut dir_cache.lua,
        );
        table.raw_set(&keys.directory, &owned)?;
        Piece::Owned(owned)
    };
    Ok((path, name_piece, dir_piece))
}

fn name_text(rel: &str, name_off: u32) -> &str {
    let off = (name_off as usize).min(rel.len());
    &rel[off..]
}

fn fill_file_row(
    lua: &Lua,
    keys: &Keys,
    paths: &PathBlob,
    item: &FileSnap,
    git: &mut Intern,
    dir_cache: &mut DirCache,
) -> LuaResult<LuaTable> {
    let table = lua.create_table_with_capacity(0, 16)?;
    table.raw_set(&keys.typ, &keys.file_ty)?;
    let rel = paths.get(item.path_start, item.path_len);
    let name = name_text(rel, item.name_off);
    let dir = parent_dir_text(rel);
    set_path_fields(
        lua,
        keys,
        &table,
        paths.bytes.as_slice(),
        rel,
        name,
        dir,
        dir_cache,
    )?;
    table.raw_set(&keys.size, item.size)?;
    table.raw_set(&keys.modified, item.modified)?;
    table.raw_set(&keys.access_frecency_score, item.access_frecency_score)?;
    table.raw_set(
        &keys.modification_frecency_score,
        item.modification_frecency_score,
    )?;
    table.raw_set(&keys.total_frecency_score, item.total_frecency_score)?;
    git.set(lua, &table, &keys.git_status, item.git_status)?;
    table.raw_set(&keys.git_recency_score, item.git_recency_score)?;
    table.raw_set(&keys.is_binary, item.is_binary)?;
    Ok(table)
}

fn fill_dir_row(
    lua: &Lua,
    keys: &Keys,
    paths: &PathBlob,
    item: &DirSnap,
    dir_cache: &mut DirCache,
) -> LuaResult<LuaTable> {
    let table = lua.create_table_with_capacity(0, 8)?;
    table.raw_set(&keys.typ, &keys.dir_ty)?;
    let rel = paths.get(item.path_start, item.path_len);
    let name = name_text(rel, item.name_off).trim_end_matches(['/', '\\']);
    let dir = parent_dir_text(rel);
    set_path_fields(
        lua,
        keys,
        &table,
        paths.bytes.as_slice(),
        rel,
        name,
        dir,
        dir_cache,
    )?;
    table.raw_set(&keys.max_access_frecency, item.max_access_frecency)?;
    Ok(table)
}

fn fill_score(lua: &Lua, keys: &Keys, score: &Score, kinds: &mut Intern) -> LuaResult<LuaTable> {
    let table = lua.create_table_with_capacity(0, 16)?;
    table.raw_set(&keys.total, score.total)?;
    table.raw_set(&keys.base_score, score.base_score)?;
    table.raw_set(&keys.filename_bonus, score.filename_bonus)?;
    table.raw_set(&keys.special_filename_bonus, score.special_filename_bonus)?;
    table.raw_set(&keys.frecency_boost, score.frecency_boost)?;
    table.raw_set(&keys.git_status_boost, score.git_status_boost)?;
    table.raw_set(&keys.git_recency_boost, score.git_recency_boost)?;
    table.raw_set(&keys.distance_penalty, score.distance_penalty)?;
    table.raw_set(&keys.current_file_penalty, score.current_file_penalty)?;
    table.raw_set(&keys.combo_match_boost, score.combo_match_boost)?;
    table.raw_set(&keys.path_alignment_bonus, score.path_alignment_bonus)?;
    kinds.set(lua, &table, &keys.match_type, score.match_type)?;
    table.raw_set(&keys.exact_match, score.exact_match)?;
    Ok(table)
}

fn set_match_ranges(lua: &Lua, keys: &Keys, item: &LuaTable, ranges: &[(u32, u32)]) -> LuaResult<()> {
    let ranges_table = lua.create_table_with_capacity(ranges.len(), 0)?;
    for (i, &(start, end)) in ranges.iter().enumerate() {
        let range = lua.create_table_with_capacity(2, 0)?;
        range.raw_seti(1, start)?;
        range.raw_seti(2, end)?;
        ranges_table.raw_seti(i + 1, range)?;
    }
    item.raw_set(&keys.match_ranges, ranges_table)
}

fn location_into_lua(location: &Location, lua: &Lua) -> LuaResult<LuaValue> {
    let table = lua.create_table_with_capacity(0, 4)?;
    match location {
        Location::Line(line) => {
            table.set("line", *line)?;
        }
        Location::Position { line, col } => {
            table.set("line", *line)?;
            table.set("col", *col)?;
        }
        Location::Range { start, end } => {
            table.set("start", LuaPosition(*start))?;
            table.set("end", LuaPosition(*end))?;
        }
    }
    Ok(LuaValue::Table(table))
}

fn push_scores(lua: &Lua, keys: &Keys, scores: &[Score], into: &LuaTable) -> LuaResult<()> {
    let mut kinds = Intern::new();
    for (i, score) in scores.iter().enumerate() {
        into.raw_seti(i + 1, fill_score(lua, keys, score, &mut kinds)?)?;
    }
    Ok(())
}

impl IntoLua for SearchResultLua {
    fn into_lua(self, lua: &Lua) -> LuaResult<LuaValue> {
        let keys = cached_keys(lua)?;
        let mut git = Intern::new();
        let mut dir_cache = DirCache::new();
        let table = lua.create_table_with_capacity(0, 6)?;
        let items_table = lua.create_table_with_capacity(self.items.len(), 0)?;
        for (i, item) in self.items.iter().enumerate() {
            let lua_item = fill_file_row(lua, &keys, &self.paths, item, &mut git, &mut dir_cache)?;
            if let Some(ranges) = self.match_ranges.get(i) {
                set_match_ranges(lua, &keys, &lua_item, ranges.as_slice())?;
            }
            items_table.raw_seti(i + 1, lua_item)?;
        }
        table.raw_set(&keys.items, items_table)?;

        let scores_table = lua.create_table_with_capacity(self.scores.len(), 0)?;
        push_scores(lua, &keys, &self.scores, &scores_table)?;
        table.raw_set(&keys.scores, scores_table)?;
        table.raw_set(&keys.total_matched, self.total_matched)?;
        table.raw_set(&keys.total_files, self.total_files)?;
        if let Some(location) = &self.location {
            table.raw_set(&keys.location, location_into_lua(location, lua)?)?;
        }
        Ok(LuaValue::Table(table))
    }
}

impl IntoLua for DirSearchResultLua {
    fn into_lua(self, lua: &Lua) -> LuaResult<LuaValue> {
        let keys = cached_keys(lua)?;
        let mut dir_cache = DirCache::new();
        let table = lua.create_table_with_capacity(0, 4)?;
        let items_table = lua.create_table_with_capacity(self.items.len(), 0)?;
        for (i, item) in self.items.iter().enumerate() {
            items_table.raw_seti(
                i + 1,
                fill_dir_row(lua, &keys, &self.paths, item, &mut dir_cache)?,
            )?;
        }
        table.raw_set(&keys.items, items_table)?;
        let scores_table = lua.create_table_with_capacity(self.scores.len(), 0)?;
        push_scores(lua, &keys, &self.scores, &scores_table)?;
        table.raw_set(&keys.scores, scores_table)?;
        table.raw_set(&keys.total_matched, self.total_matched)?;
        table.raw_set(&keys.total_dirs, self.total_dirs)?;
        Ok(LuaValue::Table(table))
    }
}

impl IntoLua for MixedSearchResultLua {
    fn into_lua(self, lua: &Lua) -> LuaResult<LuaValue> {
        let keys = cached_keys(lua)?;
        let mut git = Intern::new();
        let mut dir_cache = DirCache::new();
        let table = lua.create_table_with_capacity(0, 8)?;
        let items_table = lua.create_table_with_capacity(self.items.len(), 0)?;
        for (i, item) in self.items.iter().enumerate() {
            let lua_item = match item {
                MixedSnap::File(file) => {
                    fill_file_row(lua, &keys, &self.paths, file, &mut git, &mut dir_cache)?
                }
                MixedSnap::Dir(dir) => fill_dir_row(lua, &keys, &self.paths, dir, &mut dir_cache)?,
            };
            items_table.raw_seti(i + 1, lua_item)?;
        }
        table.raw_set(&keys.items, items_table)?;
        let scores_table = lua.create_table_with_capacity(self.scores.len(), 0)?;
        push_scores(lua, &keys, &self.scores, &scores_table)?;
        table.raw_set(&keys.scores, scores_table)?;
        table.raw_set(&keys.total_matched, self.total_matched)?;
        table.raw_set(&keys.total_files, self.total_files)?;
        table.raw_set(&keys.total_dirs, self.total_dirs)?;
        if let Some(location) = &self.location {
            table.raw_set(&keys.location, location_into_lua(location, lua)?)?;
        }
        Ok(LuaValue::Table(table))
    }
}

impl IntoLua for GrepResultLua {
    fn into_lua(self, lua: &Lua) -> LuaResult<LuaValue> {
        let keys = cached_keys(lua)?;
        let mut git = Intern::new();
        let mut dir_cache = DirCache::new();
        let mut shared: Option<SharedPath> = None;
        let table = lua.create_table_with_capacity(0, 8)?;
        let items_table = lua.create_table_with_capacity(self.matches.len(), 0)?;
        for (i, m) in self.matches.iter().enumerate() {
            let item = lua.create_table_with_capacity(0, 20)?;
            let file = &self.files[m.file_index];
            if shared.as_ref().is_some_and(|s| s.index == m.file_index) {
                let shared = shared.as_ref().unwrap();
                item.raw_set(&keys.relative_path, &shared.path)?;
                put_piece(&item, &keys.name, &shared.name, &shared.path, &keys.empty)?;
                put_piece(&item, &keys.directory, &shared.dir, &shared.path, &keys.empty)?;
            } else {
                let rel = self.paths.get(file.path_start, file.path_len);
                let name = name_text(rel, file.name_off);
                let dir = parent_dir_text(rel);
                let (path, name, dir) = set_path_fields(
                    lua,
                    &keys,
                    &item,
                    self.paths.bytes.as_slice(),
                    rel,
                    name,
                    dir,
                    &mut dir_cache,
                )?;
                shared = Some(SharedPath {
                    index: m.file_index,
                    path,
                    name,
                    dir,
                });
            }
            item.raw_set(&keys.is_binary, file.is_binary)?;
            git.set(lua, &item, &keys.git_status, file.git_status)?;
            item.raw_set(&keys.size, file.size)?;
            item.raw_set(&keys.modified, file.modified)?;
            item.raw_set(&keys.total_frecency_score, file.total_frecency_score)?;
            item.raw_set(&keys.access_frecency_score, file.access_frecency_score)?;
            item.raw_set(
                &keys.modification_frecency_score,
                file.modification_frecency_score,
            )?;
            item.raw_set(&keys.line_number, m.line_number)?;
            item.raw_set(&keys.col, m.col)?;
            item.raw_set(&keys.byte_offset, m.byte_offset)?;
            let is_binary_content = m.line_content.as_bytes().contains(&0u8);
            item.raw_set(&keys.is_binary_content, is_binary_content)?;
            item.raw_set(&keys.line_content, m.line_content.as_str())?;
            item.raw_set(&keys.is_definition, !is_binary_content && m.is_definition)?;
            set_match_ranges(lua, &keys, &item, m.match_byte_offsets.as_slice())?;
            if let Some(score) = m.fuzzy_score {
                item.raw_set(&keys.fuzzy_score, score)?;
            }
            items_table.raw_seti(i + 1, item)?;
        }
        table.raw_set(&keys.items, items_table)?;
        table.raw_set(&keys.total_matched, self.matches.len())?;
        table.raw_set(&keys.total_files_searched, self.total_files_searched)?;
        table.raw_set(&keys.total_files, self.total_files)?;
        table.raw_set(&keys.filtered_file_count, self.filtered_file_count)?;
        table.raw_set(&keys.next_file_offset, self.next_file_offset)?;
        if let Some(ref err) = self.regex_fallback_error {
            table.raw_set(&keys.regex_fallback_error, err.as_str())?;
        }
        Ok(LuaValue::Table(table))
    }
}

// Parent of `path`, "" when there is no parent. Trailing separators are ignored.
// A root slash or `C:\` is kept so absolute heads stay absolute.
pub(crate) fn parent_dir_text(path: &str) -> &str {
    let path = path.trim_end_matches(['/', '\\']);
    let bytes = path.as_bytes();
    let Some(slash) = bytes.iter().rposition(|b| *b == b'/' || *b == b'\\') else {
        return "";
    };
    if slash == 0 || (slash == 2 && bytes.get(1) == Some(&b':')) {
        return &path[..=slash];
    }
    &path[..slash]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn result_field_names_stay_on_the_lua_state() {
        let lua = Lua::new();
        let first = SearchResultLua::empty(4, None).into_lua(&lua).unwrap();
        let second = SearchResultLua::empty(9, None).into_lua(&lua).unwrap();
        let a = first.as_table().unwrap();
        let b = second.as_table().unwrap();
        assert_eq!(a.get::<usize>("total_files").unwrap(), 4);
        assert_eq!(b.get::<usize>("total_files").unwrap(), 9);
        assert!(lua.app_data_ref::<Keys>().is_some());
    }
}
