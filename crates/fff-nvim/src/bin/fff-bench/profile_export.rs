//! Optional T5 profile artifacts (flamegraph via external tools).

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProfileArtifacts {
    pub flamegraph_path: Option<String>,
    pub backend: Option<String>,
    pub note: Option<String>,
}

/// Produce a flamegraph for a short command when a profiler backend is available.
/// Returns measured paths when tools succeed; records an honest unavailable note otherwise.
pub fn try_flamegraph(
    log_dir: &Path,
    enabled: bool,
    backend: &str,
    sample_cmd: &[&str],
) -> ProfileArtifacts {
    if !enabled {
        return ProfileArtifacts {
            note: Some("flamegraph disabled".into()),
            ..Default::default()
        };
    }
    let _ = std::fs::create_dir_all(log_dir);
    let out_svg = log_dir.join("flamegraph.svg");
    let backend = if backend == "auto" {
        detect_backend()
    } else {
        backend.to_string()
    };

    match backend.as_str() {
        "samply" => run_samply(log_dir, sample_cmd),
        "perf" => run_perf(log_dir, &out_svg, sample_cmd),
        "cargo-flamegraph" | "flamegraph" => run_cargo_flamegraph(log_dir, &out_svg, sample_cmd),
        "none" | "" => ProfileArtifacts {
            note: Some("no profile backend on PATH".into()),
            ..Default::default()
        },
        other => ProfileArtifacts {
            backend: Some(other.into()),
            note: Some(format!("unknown backend {other}")),
            ..Default::default()
        },
    }
}

fn run_samply(log_dir: &Path, sample_cmd: &[&str]) -> ProfileArtifacts {
    let prof = log_dir.join("samply.json");
    let mut args: Vec<String> = vec![
        "record".into(),
        "-o".into(),
        prof.display().to_string(),
        "--".into(),
    ];
    args.extend(sample_cmd.iter().map(|s| s.to_string()));
    match Command::new("samply").args(&args).status() {
        Ok(st) if st.success() && prof.exists() => ProfileArtifacts {
            flamegraph_path: Some(prof.display().to_string()),
            backend: Some("samply".into()),
            note: Some("samply profile written; open with samply load".into()),
        },
        Ok(_) => ProfileArtifacts {
            backend: Some("samply".into()),
            note: Some("samply exited non-zero or missing output".into()),
            ..Default::default()
        },
        Err(e) => ProfileArtifacts {
            backend: Some("samply".into()),
            note: Some(format!("samply unavailable: {e}")),
            ..Default::default()
        },
    }
}

fn run_perf(log_dir: &Path, out_svg: &Path, sample_cmd: &[&str]) -> ProfileArtifacts {
    let data = log_dir.join("perf.data");
    let mut record_args: Vec<String> = vec![
        "record".into(),
        "-g".into(),
        "-o".into(),
        data.display().to_string(),
        "--".into(),
    ];
    record_args.extend(sample_cmd.iter().map(|s| s.to_string()));
    match Command::new("perf").args(&record_args).status() {
        Ok(st) if st.success() && data.exists() => {
            // Fold stacks then render SVG when inferno-flamegraph / flamegraph.pl is present.
            let folded = log_dir.join("perf.folded");
            let script_ok = Command::new("perf")
                .args(["script", "-i", &data.display().to_string()])
                .stdout(std::process::Stdio::piped())
                .spawn()
                .and_then(|mut child| {
                    let stdout = child.stdout.take();
                    let status = child.wait()?;
                    if !status.success() {
                        return Err(std::io::Error::other("perf script failed"));
                    }
                    let stackcollapse = Command::new("inferno-collapse-perf")
                        .stdin(stdout.unwrap())
                        .stdout(std::process::Stdio::piped())
                        .spawn()?;
                    let out = stackcollapse.wait_with_output()?;
                    if out.status.success() {
                        std::fs::write(&folded, out.stdout)?;
                    }
                    Ok(())
                })
                .is_ok();

            if script_ok && folded.exists() {
                let rendered = Command::new("inferno-flamegraph")
                    .arg(&folded)
                    .stdout(std::process::Stdio::piped())
                    .output();
                if let Ok(out) = rendered {
                    if out.status.success() {
                        let _ = std::fs::write(out_svg, out.stdout);
                    }
                }
            }

            ProfileArtifacts {
                flamegraph_path: if out_svg.exists() {
                    Some(out_svg.display().to_string())
                } else if data.exists() {
                    Some(data.display().to_string())
                } else {
                    None
                },
                backend: Some("perf".into()),
                note: Some(if out_svg.exists() {
                    "perf record + inferno flamegraph written".into()
                } else {
                    "perf.data written; install inferno-collapse-perf and inferno-flamegraph for SVG"
                        .into()
                }),
            }
        }
        Ok(_) => ProfileArtifacts {
            backend: Some("perf".into()),
            note: Some("perf record exited non-zero".into()),
            ..Default::default()
        },
        Err(e) => ProfileArtifacts {
            backend: Some("perf".into()),
            note: Some(format!("perf unavailable: {e}")),
            ..Default::default()
        },
    }
}

fn run_cargo_flamegraph(log_dir: &Path, out_svg: &Path, sample_cmd: &[&str]) -> ProfileArtifacts {
    // cargo flamegraph --output <svg> -- <cmd>
    let mut args: Vec<String> = vec![
        "flamegraph".into(),
        "--output".into(),
        out_svg.display().to_string(),
        "--".into(),
    ];
    args.extend(sample_cmd.iter().map(|s| s.to_string()));
    match Command::new("cargo")
        .args(&args)
        .current_dir(log_dir)
        .status()
    {
        Ok(st) if st.success() && out_svg.exists() => ProfileArtifacts {
            flamegraph_path: Some(out_svg.display().to_string()),
            backend: Some("cargo-flamegraph".into()),
            note: Some("cargo flamegraph svg written".into()),
        },
        Ok(_) => ProfileArtifacts {
            backend: Some("cargo-flamegraph".into()),
            note: Some("cargo flamegraph exited non-zero or missing svg".into()),
            ..Default::default()
        },
        Err(e) => ProfileArtifacts {
            backend: Some("cargo-flamegraph".into()),
            note: Some(format!("cargo flamegraph unavailable: {e}")),
            ..Default::default()
        },
    }
}

fn detect_backend() -> String {
    if command_exists("samply") {
        "samply".into()
    } else if command_exists("perf") {
        "perf".into()
    } else if command_exists("cargo") && cargo_has_flamegraph() {
        "cargo-flamegraph".into()
    } else {
        "none".into()
    }
}

fn cargo_has_flamegraph() -> bool {
    Command::new("cargo")
        .args(["flamegraph", "--help"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn command_exists(name: &str) -> bool {
    Command::new(name)
        .arg("--help")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success() || s.code().is_some())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_flamegraph() {
        let a = try_flamegraph(Path::new("."), false, "auto", &["echo"]);
        assert!(a.note.unwrap().contains("disabled"));
    }

    #[test]
    fn unknown_backend_is_honest() {
        let a = try_flamegraph(Path::new("."), true, "not-a-real-backend", &["echo"]);
        assert!(a.note.unwrap().contains("unknown"));
    }
}
