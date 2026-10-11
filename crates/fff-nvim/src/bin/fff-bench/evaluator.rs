//! Evaluation-bundle digests (COMPARABILITY / CONTRACT).

use crate::identity;
use crate::scenarios;
use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: &str = "1.0";
pub const ADAPTER_PROTOCOL: &str = "fff/public-api/v1";
pub const GATE_PACK: &str = "fff-gates/v1";
pub const SCENARIO_PACK: &str = "fff-scenarios/v1";
pub const DECISION_POLICY: &str = "fff-policy/v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluatorIdentity {
    pub schema_version: String,
    pub harness_name: String,
    pub harness_version: String,
    pub harness_digest: String,
    pub adapter_id: String,
    pub adapter_protocol: String,
    pub adapter_digest: String,
    pub gate_pack: String,
    pub gate_pack_digest: String,
    pub scenario_pack: String,
    pub scenario_pack_digest: String,
    pub decision_policy: String,
    pub decision_policy_digest: String,
    /// Combined content-address of the evaluator bundle.
    pub evaluator_digest: String,
}

pub fn current(adapter_id: &str) -> EvaluatorIdentity {
    let harness = identity::harness_meta();
    let harness_digest = identity::sha256_hex(
        format!(
            "fff-bench|{}|{}|{}",
            harness.version,
            harness.git_commit.as_deref().unwrap_or("-"),
            SCHEMA_VERSION
        )
        .as_bytes(),
    );
    let adapter_digest =
        identity::sha256_hex(format!("{adapter_id}|{ADAPTER_PROTOCOL}").as_bytes());
    let gate_pack_digest = identity::sha256_hex(GATE_PACK.as_bytes());
    // Scenario catalog ids form the pack digest surface.
    let catalog: Vec<_> = scenarios::list_scenarios()
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let scenario_pack_digest = identity::sha256_hex(catalog.join("\n").as_bytes());
    let decision_policy_digest = identity::sha256_hex(DECISION_POLICY.as_bytes());

    let mut parts = [
        harness_digest.as_str(),
        adapter_digest.as_str(),
        gate_pack_digest.as_str(),
        scenario_pack_digest.as_str(),
        decision_policy_digest.as_str(),
        SCHEMA_VERSION,
    ];
    parts.sort();
    let evaluator_digest = identity::sha256_hex(parts.join("|").as_bytes());

    EvaluatorIdentity {
        schema_version: SCHEMA_VERSION.into(),
        harness_name: harness.name,
        harness_version: harness.version,
        harness_digest: format!("sha256:{harness_digest}"),
        adapter_id: adapter_id.into(),
        adapter_protocol: ADAPTER_PROTOCOL.into(),
        adapter_digest: format!("sha256:{adapter_digest}"),
        gate_pack: GATE_PACK.into(),
        gate_pack_digest: format!("sha256:{gate_pack_digest}"),
        scenario_pack: SCENARIO_PACK.into(),
        scenario_pack_digest: format!("sha256:{scenario_pack_digest}"),
        decision_policy: DECISION_POLICY.into(),
        decision_policy_digest: format!("sha256:{decision_policy_digest}"),
        evaluator_digest: format!("sha256:{evaluator_digest}"),
    }
}

pub fn plan_digest(suite: &str, repo_ids: &[String], cell_keys: &[String]) -> String {
    let mut keys = cell_keys.to_vec();
    keys.sort();
    let body = format!(
        "suite={suite}\nrepos={}\ncells=\n{}",
        repo_ids.join(","),
        keys.join("\n")
    );
    format!("sha256:{}", identity::sha256_hex(body.as_bytes()))
}

/// Exact match required for decision-grade comparison.
pub fn digests_match(a: &EvaluatorIdentity, b: &EvaluatorIdentity) -> bool {
    a.evaluator_digest == b.evaluator_digest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_stable() {
        let a = current("fff");
        let b = current("fff");
        assert_eq!(a.evaluator_digest, b.evaluator_digest);
        assert!(a.evaluator_digest.starts_with("sha256:"));
    }
}
