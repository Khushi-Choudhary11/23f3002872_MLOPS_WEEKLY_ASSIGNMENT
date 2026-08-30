"""Week 11: thin client for the Week 10 fine-tuned Gemini endpoints.

This is "the existing Vertex AI prediction call" that guardrails.py wraps,
and the raw/unguarded call used directly by the Task 1 & 2 red-team scripts
(red-teaming has to hit the model with no guardrail in front of it, or there
would be nothing to red-team).
"""
import json
import pathlib
import time

import vertexai
from vertexai.generative_models import GenerativeModel

PROJECT = "mlops-project-500020"
LOCATION = "us-central1"
JOBS_PATH = pathlib.Path("data/week10/tuning_jobs.json")

# Same system instructions evaluate_llm.py used at inference time in Week 10.
SYSTEM_INSTRUCTIONS = {
    "v1": "Respond with only the species name: setosa, versicolor, or virginica. No other words.",
    "v2": "Respond with exactly one sentence in the form: This is Iris <species>.",
}

GENERATION_CONFIG = {
    "temperature": 0.0,
    # Week 10 used 64 -- adversarial prompts can provoke much longer,
    # non-compliant responses (e.g. a recited system prompt) that we need to
    # capture in full to red-team, so the budget is raised here.
    "max_output_tokens": 256,
    "thinking_config": {"thinking_budget": 0},
}

_models: dict[str, GenerativeModel] = {}
_initialized = False


def _load_model(version: str) -> GenerativeModel:
    global _initialized
    if version not in _models:
        if not _initialized:
            vertexai.init(project=PROJECT, location=LOCATION)
            _initialized = True
        jobs = json.loads(JOBS_PATH.read_text())
        endpoint = jobs[version]["tuned_model_endpoint_name"]
        _models[version] = GenerativeModel(endpoint, system_instruction=[SYSTEM_INSTRUCTIONS[version]])
    return _models[version]


def call_endpoint(version: str, prompt: str) -> str:
    """Direct, unguarded call to the tuned v1/v2 endpoint. Retries transient errors."""
    model = _load_model(version)
    for attempt in range(3):
        try:
            response = model.generate_content(prompt, generation_config=GENERATION_CONFIG)
            if not response.candidates or not response.candidates[0].content.parts:
                return "__EMPTY__"
            return response.text.strip()
        except Exception as exc:  # noqa: BLE001 -- malformed/blocked/empty responses count as non-compliant
            if attempt == 2:
                return f"__ERROR__: {exc}"
            time.sleep(2)
    return "__ERROR__"
