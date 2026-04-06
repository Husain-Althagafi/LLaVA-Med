import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from PIL import Image
import torch

from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration,
)

# =========================
# Config
# =========================

MANIFEST_PATH = "data/vlmdata/manifest.jsonl"
OUTPUT_CSV = "results_all_models.csv"
MAX_NEW_TOKENS = 120

PROMPTS = [
    "Describe this ophthalmic image briefly.",
    "What abnormalities are visible? If none are visible, say so clearly.",
    "Give the most likely diagnosis and a one-sentence justification based only on visible evidence.",
]

# Add or remove models here
MODEL_SPECS = {
    "llava_med": {
        "family": "llava",
        "model_id": "chaoyinshe/llava-med-v1.5-mistral-7b-hf",
    },
    "base_llava": {
        "family": "llava",
        "model_id": "llava-hf/llava-1.5-7b-hf",
    },
    "blip2": {
        "family": "blip2",
        "model_id": "Salesforce/blip2-opt-2.7b",
    },
}


# =========================
# Data loading
# =========================

def load_manifest(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


# =========================
# Model wrappers
# =========================

class BaseVLMWrapper:
    def __init__(self, model_name: str, model_id: str):
        self.model_name = model_name
        self.model_id = model_id
        self.processor = None
        self.model = None

    def load(self) -> None:
        raise NotImplementedError

    def build_prompt(self, prompt_text: str) -> str:
        raise NotImplementedError

    def generate(self, image: Image.Image, prompt_text: str, max_new_tokens: int = 120) -> str:
        raise NotImplementedError


class LlavaWrapper(BaseVLMWrapper):
    def load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=False)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True,
        )

    def build_prompt(self, prompt_text: str) -> str:
        return f"USER: <image>\n{prompt_text} ASSISTANT:"

    def generate(self, image: Image.Image, prompt_text: str, max_new_tokens: int = 120) -> str:
        prompt = self.build_prompt(prompt_text)
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


class Blip2Wrapper(BaseVLMWrapper):
    def load(self) -> None:
        self.processor = Blip2Processor.from_pretrained(self.model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True,
        )

    def build_prompt(self, prompt_text: str) -> str:
        return f"Question: {prompt_text} Answer:"

    def generate(self, image: Image.Image, prompt_text: str, max_new_tokens: int = 120) -> str:
        prompt = self.build_prompt(prompt_text)
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def build_wrapper(model_name: str, spec: Dict[str, str]) -> BaseVLMWrapper:
    family = spec["family"]
    model_id = spec["model_id"]

    if family == "llava":
        return LlavaWrapper(model_name=model_name, model_id=model_id)
    if family == "blip2":
        return Blip2Wrapper(model_name=model_name, model_id=model_id)

    raise ValueError(f"Unsupported model family: {family}")


# =========================
# Scoring helpers
# =========================

LABEL_KEYWORDS = {
    "proliferative_retinopathy": [
        "proliferative diabetic retinopathy",
        "proliferative retinopathy",
        "pdr",
    ],
    "moderate_retinopathy": [
        "moderate diabetic retinopathy",
        "moderate retinopathy",
    ],
    "mild_retinopathy": [
        "mild diabetic retinopathy",
        "mild retinopathy",
    ],
    "no_diabetic_retinopathy": [
        "no diabetic retinopathy",
        "no evidence of diabetic retinopathy",
        "normal retina",
        "no abnormality",
        "no abnormalities",
    ],
}


def extract_pred_label(text: str) -> str:
    if not isinstance(text, str):
        return "unclear"

    t = text.lower()
    for label, phrases in LABEL_KEYWORDS.items():
        for phrase in phrases:
            if phrase in t:
                return label
    return "unclear"


def is_abnormal(label: str) -> bool:
    return label != "no_diabetic_retinopathy"


# =========================
# Experiment runner
# =========================

def run_model_on_manifest(
    wrapper: BaseVLMWrapper,
    manifest_rows: List[Dict[str, Any]],
    prompts: List[str],
    max_new_tokens: int = 120,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for sample in manifest_rows:
        image_path = sample["image"]
        image = load_image(image_path)

        for prompt_text in prompts:
            try:
                response = wrapper.generate(
                    image=image,
                    prompt_text=prompt_text,
                    max_new_tokens=max_new_tokens,
                )
                error = ""
            except Exception as e:
                response = ""
                error = f"{type(e).__name__}: {e}"

            row = {
                "image": image_path,
                "label_code": sample.get("label_code"),
                "label": sample.get("label"),
                "model": wrapper.model_name,
                "model_id": wrapper.model_id,
                "prompt": prompt_text,
                "response": response,
                "error": error,
            }
            results.append(row)

    return results


def add_scoring_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["pred_label"] = df["response"].apply(extract_pred_label)
    df["correct_exact"] = (df["pred_label"] == df["label"]).astype(int)

    df["gt_abnormal"] = df["label"].apply(is_abnormal)
    df["pred_abnormal"] = df["pred_label"].apply(
        lambda x: None if x == "unclear" else is_abnormal(x)
    )

    def coarse_correct(row: pd.Series) -> int:
        if row["pred_abnormal"] is None:
            return 0
        return int(row["gt_abnormal"] == row["pred_abnormal"])

    df["correct_normal_abnormal"] = df.apply(coarse_correct, axis=1)

    # manual columns to fill later
    for col in ["grounding_score", "hallucination", "abstention", "notes"]:
        if col not in df.columns:
            df[col] = ""

    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    def mean_grounding(series: pd.Series) -> float:
        return pd.to_numeric(series, errors="coerce").mean()

    def halluc_rate(series: pd.Series) -> float:
        return (series == "yes").mean()

    out = (
        df.groupby("model")
        .agg(
            exact_accuracy=("correct_exact", "mean"),
            coarse_accuracy=("correct_normal_abnormal", "mean"),
            avg_grounding=("grounding_score", mean_grounding),
            hallucination_rate=("hallucination", halluc_rate),
        )
        .reset_index()
    )
    return out


# =========================
# Main
# =========================

def main() -> None:
    manifest_rows = load_manifest(MANIFEST_PATH)
    all_rows: List[Dict[str, Any]] = []

    for model_name, spec in MODEL_SPECS.items():
        print(f"\nLoading {model_name} :: {spec['model_id']}")
        wrapper = build_wrapper(model_name, spec)
        wrapper.load()

        print(f"Running {model_name} on {len(manifest_rows)} images...")
        model_rows = run_model_on_manifest(
            wrapper=wrapper,
            manifest_rows=manifest_rows,
            prompts=PROMPTS,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        all_rows.extend(model_rows)

        # free memory before next model
        del wrapper.model
        del wrapper.processor
        torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    df = add_scoring_columns(df)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved results to {OUTPUT_CSV}")
    print(df.head())


if __name__ == "__main__":
    main()