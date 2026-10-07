//! Network / remote environment identity and loopback fixture probes.
//!
//! FFF local file-picker runs are `native_local`. Loopback profiles run a real
//! TCP probe (with optional userspace delay/jitter/loss shaping) and record
//! delivered RTT evidence. External shared endpoints stay informational.

use crate::identity;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

pub const PROTOCOL: &str = "network-profile/v1";
const PROBE_ROUNDS: usize = 32;
const PROBE_PAYLOAD: &[u8] = b"fff-bench-net-probe-v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkProfile {
    pub id: String,
    /// native_local | loopback_fixture | same_host_processes | controlled_lan | external_shared
    pub environment_class: String,
    /// client | server | end_to_end | driver_only | n/a
    pub subject_boundary: String,
    pub topology: String,
    pub transport: String,
    /// confirm | informational
    pub default_grade: String,
    #[serde(default)]
    pub shaping: Option<ShapingSpec>,
    #[serde(default)]
    pub endpoint: Option<EndpointId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapingSpec {
    pub bandwidth_mbps: Option<f64>,
    pub delay_ms: Option<f64>,
    pub jitter_ms: Option<f64>,
    pub loss: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointId {
    pub kind: String,
    pub address: String,
    pub protocol_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkEvidence {
    pub profile_id: String,
    pub protocol: String,
    pub environment_class: String,
    pub subject_boundary: String,
    pub decision_grade: String,
    pub clock_model: String,
    pub provider: NetworkProvider,
    pub valid: bool,
    #[serde(default)]
    pub reason_codes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shaping_requested: Option<ShapingSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shaping_delivered: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkProvider {
    pub id: String,
    pub version: String,
    pub digest: String,
}

pub fn native_local_v1() -> NetworkProfile {
    NetworkProfile {
        id: "native_local_v1".into(),
        environment_class: "native_local".into(),
        subject_boundary: "n/a".into(),
        topology: "same_process".into(),
        transport: "none".into(),
        default_grade: "confirm".into(),
        shaping: None,
        endpoint: None,
    }
}

pub fn loopback_fixture_v1() -> NetworkProfile {
    NetworkProfile {
        id: "loopback_fixture_v1".into(),
        environment_class: "same_process/loopback_fixture".into(),
        subject_boundary: "end_to_end".into(),
        topology: "loopback".into(),
        transport: "tcp".into(),
        default_grade: "confirm".into(),
        shaping: Some(ShapingSpec {
            bandwidth_mbps: None,
            delay_ms: Some(0.0),
            jitter_ms: Some(0.0),
            loss: Some(0.0),
        }),
        endpoint: Some(EndpointId {
            kind: "loopback".into(),
            address: "127.0.0.1".into(),
            protocol_version: "fixture/v1".into(),
        }),
    }
}

/// Loopback with applied userspace delay (confirms shaping delivery path).
pub fn loopback_delay_20ms_v1() -> NetworkProfile {
    NetworkProfile {
        id: "loopback_delay_20ms_v1".into(),
        environment_class: "same_process/loopback_fixture".into(),
        subject_boundary: "end_to_end".into(),
        topology: "loopback".into(),
        transport: "tcp".into(),
        default_grade: "confirm".into(),
        shaping: Some(ShapingSpec {
            bandwidth_mbps: None,
            delay_ms: Some(20.0),
            jitter_ms: Some(0.0),
            loss: Some(0.0),
        }),
        endpoint: Some(EndpointId {
            kind: "loopback".into(),
            address: "127.0.0.1".into(),
            protocol_version: "fixture/v1".into(),
        }),
    }
}

/// External shared service — informational only (cannot establish small code wins).
pub fn external_shared_v1() -> NetworkProfile {
    NetworkProfile {
        id: "external_shared_v1".into(),
        environment_class: "external_shared".into(),
        subject_boundary: "client".into(),
        topology: "uncontrolled_internet".into(),
        transport: "https".into(),
        default_grade: "informational".into(),
        shaping: None,
        endpoint: Some(EndpointId {
            kind: "external".into(),
            address: "unspecified".into(),
            protocol_version: "unknown".into(),
        }),
    }
}

pub fn resolve_profile(id: &str) -> Option<NetworkProfile> {
    match id {
        "" | "none" | "local" | "native" | "native_local" | "native_local_v1" => {
            Some(native_local_v1())
        }
        "loopback" | "loopback_fixture" | "loopback_fixture_v1" => Some(loopback_fixture_v1()),
        "loopback_delay" | "loopback_delay_20ms" | "loopback_delay_20ms_v1" => {
            Some(loopback_delay_20ms_v1())
        }
        "external" | "external_shared" | "external_shared_v1" => Some(external_shared_v1()),
        _ => None,
    }
}

pub fn list_profiles() -> Vec<&'static str> {
    vec![
        "native_local_v1",
        "loopback_fixture_v1",
        "loopback_delay_20ms_v1",
        "external_shared_v1",
    ]
}

pub fn activate(profile: &NetworkProfile) -> NetworkEvidence {
    let dig = identity::sha256_hex(format!("{PROTOCOL}|{}|1.1.0", profile.id).as_bytes());
    let provider = NetworkProvider {
        id: "builtin.network-loopback".into(),
        version: "1.1.0".into(),
        digest: format!("sha256:{dig}"),
    };

    if profile.environment_class == "native_local" {
        return NetworkEvidence {
            profile_id: profile.id.clone(),
            protocol: PROTOCOL.into(),
            environment_class: profile.environment_class.clone(),
            subject_boundary: profile.subject_boundary.clone(),
            decision_grade: profile.default_grade.clone(),
            clock_model: "single_host_monotonic".into(),
            provider: NetworkProvider {
                id: "builtin.network-declare".into(),
                version: "1.1.0".into(),
                digest: format!("sha256:{dig}"),
            },
            valid: true,
            reason_codes: vec![],
            shaping_requested: None,
            shaping_delivered: Some(serde_json::json!({
                "mode": "none",
                "transport": "none",
            })),
        };
    }

    if profile.environment_class == "external_shared" {
        return NetworkEvidence {
            profile_id: profile.id.clone(),
            protocol: PROTOCOL.into(),
            environment_class: profile.environment_class.clone(),
            subject_boundary: profile.subject_boundary.clone(),
            decision_grade: "informational".into(),
            clock_model: "uncontrolled_remote".into(),
            provider: NetworkProvider {
                id: "builtin.network-external".into(),
                version: "1.1.0".into(),
                digest: format!("sha256:{dig}"),
            },
            valid: true,
            reason_codes: vec!["external_shared_informational_only".into()],
            shaping_requested: profile.shaping.clone(),
            shaping_delivered: None,
        };
    }

    // Loopback / controlled fixture: real TCP probe + userspace shaping on the echo path.
    match run_loopback_probe(profile) {
        Ok(delivered) => {
            let mut reasons = Vec::new();
            let mut valid = true;
            let mut grade = profile.default_grade.clone();

            if let Some(ref sh) = profile.shaping {
                if let Some(want_delay) = sh.delay_ms {
                    if want_delay > 0.5 {
                        let p50 = delivered
                            .get("rtt_ms")
                            .and_then(|r| r.get("p50"))
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0);
                        // Delivered RTT must reflect applied delay (allow 50% slack + 5ms floor).
                        let min_ok = want_delay * 0.5;
                        let max_ok = want_delay * 2.5 + 15.0;
                        if p50 < min_ok || p50 > max_ok {
                            valid = false;
                            reasons.push(format!(
                                "shaping_delay_mismatch:want={want_delay:.1},p50={p50:.2}"
                            ));
                        }
                    }
                }
                if let Some(want_loss) = sh.loss {
                    if want_loss > 0.0 {
                        let obs = delivered
                            .get("loss_observed")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0);
                        if (obs - want_loss).abs() > 0.25 + want_loss * 0.5 {
                            valid = false;
                            reasons.push(format!(
                                "shaping_loss_mismatch:want={want_loss:.2},got={obs:.2}"
                            ));
                        }
                    }
                }
            }

            let probes_ok = delivered
                .get("probes_ok")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            if probes_ok == 0 {
                valid = false;
                reasons.push("loopback_probe_no_success".into());
            }

            if !valid {
                grade = "informational".into();
            }

            NetworkEvidence {
                profile_id: profile.id.clone(),
                protocol: PROTOCOL.into(),
                environment_class: profile.environment_class.clone(),
                subject_boundary: profile.subject_boundary.clone(),
                decision_grade: grade,
                clock_model: "single_host_monotonic".into(),
                provider,
                valid,
                reason_codes: reasons,
                shaping_requested: profile.shaping.clone(),
                shaping_delivered: Some(delivered),
            }
        }
        Err(e) => NetworkEvidence {
            profile_id: profile.id.clone(),
            protocol: PROTOCOL.into(),
            environment_class: profile.environment_class.clone(),
            subject_boundary: profile.subject_boundary.clone(),
            decision_grade: "informational".into(),
            clock_model: "single_host_monotonic".into(),
            provider,
            valid: false,
            reason_codes: vec![format!("loopback_probe_failed:{e}")],
            shaping_requested: profile.shaping.clone(),
            shaping_delivered: None,
        },
    }
}

struct ShapeParams {
    delay_ms: f64,
    jitter_ms: f64,
    loss: f64,
    /// Minimum microseconds between bytes for crude bandwidth cap (0 = unlimited).
    bytes_per_us: f64,
}

impl ShapeParams {
    fn from_spec(sh: Option<&ShapingSpec>) -> Self {
        let delay_ms = sh.and_then(|s| s.delay_ms).unwrap_or(0.0).max(0.0);
        let jitter_ms = sh.and_then(|s| s.jitter_ms).unwrap_or(0.0).max(0.0);
        let loss = sh.and_then(|s| s.loss).unwrap_or(0.0).clamp(0.0, 1.0);
        let bytes_per_us = sh
            .and_then(|s| s.bandwidth_mbps)
            .filter(|b| *b > 0.0)
            .map(|mbps| (mbps * 1_000_000.0 / 8.0) / 1_000_000.0) // bytes per µs
            .unwrap_or(0.0);
        Self {
            delay_ms,
            jitter_ms,
            loss,
            bytes_per_us,
        }
    }

    fn sleep_before_reply(&self, rng: &mut u64) {
        let mut ms = self.delay_ms;
        if self.jitter_ms > 0.0 {
            // xorshift fraction in [0,1)
            *rng ^= *rng << 13;
            *rng ^= *rng >> 7;
            *rng ^= *rng << 17;
            let f = (*rng as f64) / (u64::MAX as f64);
            ms += (f * 2.0 - 1.0) * self.jitter_ms;
        }
        if ms > 0.0 {
            thread::sleep(Duration::from_secs_f64(ms / 1000.0));
        }
    }

    fn drop_packet(&self, rng: &mut u64) -> bool {
        if self.loss <= 0.0 {
            return false;
        }
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        let f = (*rng as f64) / (u64::MAX as f64);
        f < self.loss
    }
}

fn run_loopback_probe(profile: &NetworkProfile) -> Result<serde_json::Value, String> {
    let shape = ShapeParams::from_spec(profile.shaping.as_ref());
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| format!("bind: {e}"))?;
    listener
        .set_nonblocking(false)
        .map_err(|e| format!("set_blocking: {e}"))?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    let stop = Arc::new(AtomicBool::new(false));
    let stop_c = Arc::clone(&stop);
    let shape_srv = ShapeParams {
        delay_ms: shape.delay_ms,
        jitter_ms: shape.jitter_ms,
        loss: shape.loss,
        bytes_per_us: shape.bytes_per_us,
    };

    let server = thread::Builder::new()
        .name("fff-bench-net-echo".into())
        .spawn(move || {
            let mut rng = 0xDEAD_BEEF_CAFE_u64;
            // Accept a single connection (reuse for all rounds).
            if let Ok((mut stream, _)) = listener.accept() {
                let _ = stream.set_nodelay(true);
                let mut buf = [0u8; 256];
                while !stop_c.load(Ordering::Relaxed) {
                    match stream.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => {
                            if shape_srv.drop_packet(&mut rng) {
                                // Simulated loss: do not reply.
                                continue;
                            }
                            shape_srv.sleep_before_reply(&mut rng);
                            if shape_srv.bytes_per_us > 0.0 {
                                let us = (n as f64) / shape_srv.bytes_per_us;
                                if us > 1.0 {
                                    thread::sleep(Duration::from_micros(us as u64));
                                }
                            }
                            if stream.write_all(&buf[..n]).is_err() {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
            }
        })
        .map_err(|e| format!("spawn echo: {e}"))?;

    // Connect client and measure RTT samples.
    thread::sleep(Duration::from_millis(5));
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(2))
        .map_err(|e| format!("connect: {e}"))?;
    stream
        .set_nodelay(true)
        .map_err(|e| format!("nodelay: {e}"))?;
    stream
        .set_read_timeout(Some(Duration::from_millis(
            ((shape.delay_ms + shape.jitter_ms) as u64)
                .saturating_mul(4)
                .max(200)
                + 500,
        )))
        .ok();
    stream.set_write_timeout(Some(Duration::from_secs(2))).ok();

    let mut rtts_ms = Vec::with_capacity(PROBE_ROUNDS);
    let mut failed = 0u64;
    let mut ok = 0u64;
    let mut reply = vec![0u8; PROBE_PAYLOAD.len()];

    for _ in 0..PROBE_ROUNDS {
        let t0 = Instant::now();
        if stream.write_all(PROBE_PAYLOAD).is_err() {
            failed += 1;
            continue;
        }
        match stream.read_exact(&mut reply) {
            Ok(()) if reply == PROBE_PAYLOAD => {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                rtts_ms.push(ms);
                ok += 1;
            }
            _ => {
                failed += 1;
            }
        }
    }

    stop.store(true, Ordering::SeqCst);
    // Unblock server accept/read if still waiting.
    let _ = TcpStream::connect_timeout(&addr, Duration::from_millis(50));
    drop(stream);
    let _ = server.join();

    rtts_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = rtts_ms.len();
    let p50 = if n == 0 { 0.0 } else { rtts_ms[n / 2] };
    let p95 = if n == 0 {
        0.0
    } else {
        rtts_ms[(((n as f64) * 0.95) as usize).min(n - 1)]
    };
    let mean = if n == 0 {
        0.0
    } else {
        rtts_ms.iter().sum::<f64>() / n as f64
    };
    let min = rtts_ms.first().copied().unwrap_or(0.0);
    let max = rtts_ms.last().copied().unwrap_or(0.0);
    let total = ok + failed;
    let loss_obs = if total == 0 {
        1.0
    } else {
        failed as f64 / total as f64
    };

    Ok(serde_json::json!({
        "mode": "userspace_tcp_echo",
        "endpoint": addr.to_string(),
        "transport": "tcp",
        "probes_attempted": PROBE_ROUNDS,
        "probes_ok": ok,
        "probes_failed": failed,
        "loss_observed": loss_obs,
        "rtt_ms": {
            "min": min,
            "p50": p50,
            "p95": p95,
            "mean": mean,
            "max": max,
            "n": n,
        },
        "shaping_applied": {
            "delay_ms": shape.delay_ms,
            "jitter_ms": shape.jitter_ms,
            "loss": shape.loss,
            "bandwidth_mbps": profile.shaping.as_ref().and_then(|s| s.bandwidth_mbps),
        },
        "provider": "builtin.userspace_shaper",
    }))
}

pub fn validity_gate(ev: &NetworkEvidence) -> crate::gates::GateEntry {
    let required = ev.decision_grade == "confirm";
    crate::gates::GateEntry {
        gate_id: "treatment.network_identity".into(),
        status: if ev.valid || !required {
            "pass"
        } else {
            "fail"
        }
        .into(),
        required,
        reason_code: if ev.valid {
            None
        } else {
            ev.reason_codes.first().cloned()
        },
        evidence: Some(serde_json::json!({
            "profile": ev.profile_id,
            "environment_class": ev.environment_class,
            "decision_grade": ev.decision_grade,
            "valid": ev.valid,
            "reason_codes": ev.reason_codes,
            "shaping_delivered": ev.shaping_delivered,
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_local_is_confirm_valid() {
        let ev = activate(&native_local_v1());
        assert!(ev.valid);
        assert_eq!(ev.decision_grade, "confirm");
        assert!(ev.shaping_delivered.is_some());
    }

    #[test]
    fn external_is_informational() {
        let ev = activate(&external_shared_v1());
        assert_eq!(ev.decision_grade, "informational");
        assert!(ev.reason_codes.iter().any(|r| r.contains("informational")));
    }

    #[test]
    fn resolve_known() {
        assert!(resolve_profile("native_local_v1").is_some());
        assert!(resolve_profile("loopback_fixture_v1").is_some());
        assert!(resolve_profile("loopback_delay_20ms_v1").is_some());
        assert!(resolve_profile("nope").is_none());
    }

    #[test]
    fn loopback_probe_measures_rtt() {
        let ev = activate(&loopback_fixture_v1());
        assert!(ev.valid, "{:?}", ev.reason_codes);
        let del = ev.shaping_delivered.expect("delivered");
        let ok = del.get("probes_ok").and_then(|v| v.as_u64()).unwrap_or(0);
        assert!(ok > 0, "{del}");
        let p50 = del
            .pointer("/rtt_ms/p50")
            .and_then(|v| v.as_f64())
            .unwrap_or(-1.0);
        assert!(p50 >= 0.0, "p50={p50}");
    }

    #[test]
    fn loopback_delay_delivers_shaped_rtt() {
        let ev = activate(&loopback_delay_20ms_v1());
        assert!(ev.valid, "{:?}", ev.reason_codes);
        let p50 = ev
            .shaping_delivered
            .as_ref()
            .and_then(|d| d.pointer("/rtt_ms/p50"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        assert!(p50 >= 10.0, "expected ~20ms delay, p50={p50}");
    }
}
