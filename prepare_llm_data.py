"""Week 10 Task 1 & 2: convert IRIS into two JSONL fine-tuning representations.

v1 (raw)         -> input_text: "sepal_length: 5.1, sepal_width: 3.5, ..."   output_text: "setosa"
v2 (description)  -> input_text: "A flower specimen has a sepal length of ... Identify the iris species."
                      output_text: "This is Iris setosa."

Also emits a Gemini-chat-format ("contents": [...]) copy of each *train* split,
since Vertex AI's current supervised-tuning API for Gemini models consumes the
`contents`/`role`/`parts` schema rather than the legacy input_text/output_text
schema from the older PaLM tuning API. The input_text/output_text files remain
the primary deliverable (matches the assignment spec) and are what get scored
for accuracy at evaluation time; the *_gemini.jsonl files are a mechanical
conversion used only to submit the actual tuning jobs.
"""
import json
import pathlib

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_CSV = pathlib.Path("data/raw/iris.csv")
OUT_DIR = pathlib.Path("data/week10")
RANDOM_STATE = 42

SYSTEM_INSTRUCTION = (
    "You are an IRIS flower species classifier. Respond with only the "
    "requested output — no extra words, no punctuation beyond what is shown "
    "in the expected format, no explanation."
)


def v1_input(row: pd.Series) -> str:
    return (
        f"sepal_length: {row.sepal_length}, sepal_width: {row.sepal_width}, "
        f"petal_length: {row.petal_length}, petal_width: {row.petal_width}"
    )


def v1_output(row: pd.Series) -> str:
    return row.species


def v2_input(row: pd.Series) -> str:
    return (
        f"A flower specimen has a sepal length of {row.sepal_length} cm, "
        f"sepal width of {row.sepal_width} cm, petal length of "
        f"{row.petal_length} cm, and petal width of {row.petal_width} cm. "
        "Identify the iris species."
    )


def v2_output(row: pd.Series) -> str:
    return f"This is Iris {row.species}."


def to_jsonl(df: pd.DataFrame, input_fn, output_fn, path: pathlib.Path) -> None:
    with path.open("w") as f:
        for _, row in df.iterrows():
            record = {"input_text": input_fn(row), "output_text": output_fn(row)}
            f.write(json.dumps(record) + "\n")


def to_gemini_jsonl(df: pd.DataFrame, input_fn, output_fn, path: pathlib.Path) -> None:
    """Convert to the {"contents": [...]} chat schema Vertex AI's Gemini SFT expects."""
    with path.open("w") as f:
        for _, row in df.iterrows():
            record = {
                "systemInstruction": {"role": "system", "parts": [{"text": SYSTEM_INSTRUCTION}]},
                "contents": [
                    {"role": "user", "parts": [{"text": input_fn(row)}]},
                    {"role": "model", "parts": [{"text": output_fn(row)}]},
                ],
            }
            f.write(json.dumps(record) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_CSV)

    # 60/20/20 stratified split -> 90 train / 30 validation / 30 test (30/10/10 per class)
    train_df, rest_df = train_test_split(
        df, test_size=0.4, random_state=RANDOM_STATE, stratify=df.species
    )
    val_df, test_df = train_test_split(
        rest_df, test_size=0.5, random_state=RANDOM_STATE, stratify=rest_df.species
    )

    splits = {"train": train_df, "validation": val_df, "test": test_df}
    for version, (input_fn, output_fn) in {
        "v1": (v1_input, v1_output),
        "v2": (v2_input, v2_output),
    }.items():
        for split_name, split_df in splits.items():
            path = OUT_DIR / f"{version}_{split_name}.jsonl"
            to_jsonl(split_df, input_fn, output_fn, path)
            print(f"wrote {path} ({len(split_df)} rows)")

        # Gemini-format copies of train + validation, used to submit the tuning job
        for split_name in ("train", "validation"):
            gpath = OUT_DIR / f"{version}_{split_name}_gemini.jsonl"
            to_gemini_jsonl(splits[split_name], input_fn, output_fn, gpath)
            print(f"wrote {gpath} ({len(splits[split_name])} rows)")

    for split_name, split_df in splits.items():
        print(f"{split_name}: {len(split_df)} rows, class counts = {split_df.species.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
