"""Week 11 Task 3 & 4: input and output guardrails wrapping the Week 10 fine-tuned endpoints.

InputGuardrail  -- Task 3: intercepts calls matching injection/leakage patterns
                   found in Task 1 & 2 before they reach the model.
                     1. rule-based check  -- regex/keyword blocklist
                     2. structural check  -- input must conform to the exact
                        IRIS feature schema each version was trained on
OutputGuardrail -- Task 4: scans model responses before they're returned.
                     1. context-leakage check -- response contains a fragment
                        of the real system instruction (same LEAK_PHRASES
                        used to score Task 2)
                     2. format-violation check -- response isn't the model's
                        exact expected output format
GuardedPredictor -- wraps llm_client.call_endpoint with input guardrail ->
                    model -> output guardrail, so callers never talk to the
                    Vertex AI endpoint directly.

Every blocked input and every filtered output is logged (timestamp, matched
rule/category, raw content) for audit, matching Task 3 & 4's logging
requirement.
"""
import json
import pathlib
import re
from datetime import datetime, timezone

from llm_client import call_endpoint
from redteam_leakage import LEAK_PHRASES

DATA_DIR = pathlib.Path("data/week11")
INPUT_LOG_PATH = DATA_DIR / "guardrail_input_audit_log.jsonl"
OUTPUT_LOG_PATH = DATA_DIR / "guardrail_output_audit_log.jsonl"

# ---------------------------------------------------------------------------
# Input guardrail: rule-based blocklist
# ---------------------------------------------------------------------------
# Each rule targets an attack pattern actually observed to succeed (or that
# plausibly could) in Task 1 (injection) / Task 2 (leakage) red-teaming:
# instruction_override, roleplay_framing, delimiter_escape,
# embedded_field_injection, direct_request / context_dump / role_confusion
# leakage probes.
BLOCKLIST_RULES = [
    ("ignore_instructions", re.compile(r"ignore (all |the )?(previous|above|prior) instructions", re.I)),
    ("disregard_instructions", re.compile(r"disregard (the |all )?(above|previous|prior)", re.I)),
    ("new_instructions", re.compile(r"new instructions?\s*:", re.I)),
    ("system_prompt_probe", re.compile(r"system prompt|system instruction", re.I)),
    ("role_override", re.compile(r"you are now|act as|pretend (you are|to be)|roleplay|in character as|debugging assistant", re.I)),
    ("jailbreak_persona", re.compile(r"\bDAN\b|no rules|no restrictions", re.I)),
    ("delimiter_escape", re.compile(r"---\s*end|</?(system|input|instructions?)>", re.I)),
    ("context_dump_request", re.compile(r"context window|repeat everything above|print the (full )?contents", re.I)),
    ("training_data_probe", re.compile(r"training example|fine-?tuned on|what instructions were you given", re.I)),
    ("translate_instructions", re.compile(r"translate your (system )?instructions", re.I)),
]

# ---------------------------------------------------------------------------
# Input guardrail: structural schema check
# ---------------------------------------------------------------------------
# Must match exactly what prepare_llm_data.py's v1_input/v2_input produce --
# any extra text (injected instructions, delimiters, commentary) fails this.
NUM = r"\d+(?:\.\d+)?"
V1_SCHEMA_RE = re.compile(
    rf"^sepal_length: {NUM}, sepal_width: {NUM}, petal_length: {NUM}, petal_width: {NUM}$"
)
V2_SCHEMA_RE = re.compile(
    rf"^A flower specimen has a sepal length of {NUM} cm, sepal width of {NUM} cm, "
    rf"petal length of {NUM} cm, and petal width of {NUM} cm\. Identify the iris species\.$"
)
SCHEMA_RE = {"v1": V1_SCHEMA_RE, "v2": V2_SCHEMA_RE}

# ---------------------------------------------------------------------------
# Output guardrail: format compliance (same regexes evaluate_llm.py scores against)
# ---------------------------------------------------------------------------
V1_COMPLIANT_RE = re.compile(r"^(setosa|versicolor|virginica)$")
V2_COMPLIANT_RE = re.compile(r"^This is Iris (setosa|versicolor|virginica)\.$")
COMPLIANT_RE = {"v1": V1_COMPLIANT_RE, "v2": V2_COMPLIANT_RE}
EXTRACT_RE = re.compile(r"(setosa|versicolor|virginica)", re.IGNORECASE)

FALLBACK_RESPONSE = "[response withheld: output guardrail detected non-compliant content]"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_log(path: pathlib.Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


class InputGuardrail:
    """Blocks a raw input before it reaches the model. Two independent checks:
    rule-based blocklist match, and structural IRIS-schema conformance."""

    def __init__(self, log_path: pathlib.Path = INPUT_LOG_PATH):
        self.log_path = log_path

    def check(self, version: str, raw_input: str) -> dict:
        for rule_name, pattern in BLOCKLIST_RULES:
            if pattern.search(raw_input):
                return self._block(version, raw_input, "blocklist_match", rule_name)

        if not SCHEMA_RE[version].match(raw_input.strip()):
            return self._block(version, raw_input, "schema_violation", "structural_schema_check")

        return {"blocked": False}

    def _block(self, version: str, raw_input: str, reason: str, matched_rule: str) -> dict:
        entry = {
            "timestamp": _now(),
            "model_version": version,
            "reason": reason,
            "matched_rule": matched_rule,
            "raw_input": raw_input,
        }
        _append_log(self.log_path, entry)
        return {"blocked": True, "reason": reason, "matched_rule": matched_rule}


class OutputGuardrail:
    """Scans a model response before it is returned. Detects context leakage
    and format violations; redacts down to a safe portion when recoverable,
    otherwise replaces the whole response with a fallback message."""

    def __init__(self, log_path: pathlib.Path = OUTPUT_LOG_PATH):
        self.log_path = log_path

    def check(self, version: str, raw_response: str) -> dict:
        lowered = raw_response.lower()
        leaked = [p for p in LEAK_PHRASES if p in lowered]
        if leaked:
            return self._filter(
                version, raw_response, "context_leakage", FALLBACK_RESPONSE, matched=leaked
            )

        if COMPLIANT_RE[version].match(raw_response.strip()):
            return {"blocked": False, "response": raw_response}

        # Format violation: try to redact down to a safe recoverable portion
        # (a valid species token embedded in otherwise off-format text)
        # before falling back to a standardized message entirely.
        match = EXTRACT_RE.search(raw_response)
        if match:
            return self._filter(
                version, raw_response, "format_violation_redacted", match.group(1).lower()
            )

        return self._filter(version, raw_response, "format_violation_unrecoverable", FALLBACK_RESPONSE)

    def _filter(self, version: str, raw_response: str, category: str, filtered_response: str, matched=None) -> dict:
        entry = {
            "timestamp": _now(),
            "model_version": version,
            "category": category,
            "raw_response": raw_response,
            "filtered_response": filtered_response,
        }
        if matched is not None:
            entry["matched_phrases"] = matched
        _append_log(self.log_path, entry)
        return {"blocked": True, "reason": category, "filtered_response": filtered_response}


class GuardedPredictor:
    """Wraps llm_client.call_endpoint with an input guardrail -> model ->
    output guardrail pipeline. Callers should use this instead of calling
    call_endpoint directly."""

    def __init__(self, input_guardrail: InputGuardrail = None, output_guardrail: OutputGuardrail = None):
        self.input_guardrail = input_guardrail or InputGuardrail()
        self.output_guardrail = output_guardrail or OutputGuardrail()

    def predict(self, version: str, raw_input: str) -> dict:
        input_check = self.input_guardrail.check(version, raw_input)
        if input_check["blocked"]:
            return {
                "status": "input_blocked",
                "final_response": None,
                "raw_response": None,
                "block_reason": input_check["reason"],
                "matched_rule": input_check["matched_rule"],
            }

        raw_response = call_endpoint(version, raw_input)
        output_check = self.output_guardrail.check(version, raw_response)
        if output_check["blocked"]:
            return {
                "status": f"output_{output_check['reason']}",
                "final_response": output_check["filtered_response"],
                "raw_response": raw_response,
                "block_reason": output_check["reason"],
                "matched_rule": None,
            }

        return {
            "status": "ok",
            "final_response": raw_response,
            "raw_response": raw_response,
            "block_reason": None,
            "matched_rule": None,
        }
