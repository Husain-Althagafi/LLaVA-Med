import argparse
import json
import os
from typing import List, Dict, Any

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration


OUTPUT_PATH = "outputs"

PROMPTS = [
    "Describe this ophthalmic image briefly.",
    "What abnormalities are visible? If none are visible, say so clearly.",
    "Give the most likely diagnosis and a one-sentence justification based only on visible evidence.",
]


# -----------------------------
# Utilities
# -----------------------------

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_manifest(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_to_csv(results: List[Dict[str, Any]], file_name: str) -> None:
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_PATH, f"{file_name}.csv"), index=False)


# -----------------------------
# Wrapper base class
# -----------------------------

class ModelWrapper:
    def generate(self, image: Image.Image, text: str) -> str:
        raise NotImplementedError


# -----------------------------
# LLaVA-Med wrapper
# Keeps your original loading/generation logic
# -----------------------------

class LlavaMedWrapper(ModelWrapper):
    def __init__(self, tokenizer, model, image_processor, context_len):
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len

    def generate(self, image: Image.Image, text: str) -> str:
        from llava.constants import (
            IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
        )
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import (
            tokenizer_image_token,
            KeywordsStoppingCriteria,
            process_images,
        )

        image_tensor = process_images([image], self.image_processor, self.model.config)[0]

        prompt = text.replace(DEFAULT_IMAGE_TOKEN, "").strip()
        if self.model.config.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        conv = conv_templates["mistral_instruct"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            full_prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        ).unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=0.2,
                top_p=None,
                num_beams=1,
                max_new_tokens=1024,
                use_cache=True,
                # stopping_criteria=[stopping_criteria],  # optional
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs


# -----------------------------
# Base LLaVA wrapper
# -----------------------------

class LlavaBaseHFWrapper(ModelWrapper):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to("cuda")

        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)

    def generate(self, image: Image.Image, text: str) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image"},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(0, torch.float16)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False
            )

        return self.processor.decode(output[0][2:], skip_special_tokens=True)


# -----------------------------
# BLIP-2 wrapper
# -----------------------------

class Blip2Wrapper(ModelWrapper):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.processor = Blip2Processor.from_pretrained(model_id, use_fast=False)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to("cuda")

    def generate(self, image: Image.Image, text: str) -> str:
        prompt = f"Question: {text} Answer:"

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to("cuda", torch.float16)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False
            )

        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


# -----------------------------
# Common eval loop
# -----------------------------

def run_eval(data: List[Dict[str, Any]], wrapper: ModelWrapper) -> List[Dict[str, Any]]:
    results = []

    for sample in tqdm(data):
        img = load_image(sample["image"])

        for prompt_text in PROMPTS:
            try:
                response = wrapper.generate(img, prompt_text)
                error = ""
            except Exception as e:
                response = ""
                error = f"{type(e).__name__}: {e}"

            results.append({
                "image": sample["image"],
                "label_code": sample.get("label_code"),
                "label": sample.get("label"),
                "model_prompt": prompt_text,
                "response": response,
                "error": error,
            })

    return results


# -----------------------------
# Args
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", type=str, default="med", choices=["med", "base", "blip2"])

    # llava-med args
    parser.add_argument("--model_path", type=str, default="/work/vlmwork/LLaVA-Med/llavamodel")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="llava-med-v1.5-mistral-7b")

    # hf model ids for other wrappers
    parser.add_argument("--base_model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--blip2_model_id", type=str, default="Salesforce/blip2-opt-2.7b")

    parser.add_argument("--manifest_path", type=str, default="data/vlmdata/manifest.jsonl")
    parser.add_argument("--csv_name", type=str, required=True)

    return parser.parse_args()


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()

    print("Loading manifest...")
    data = load_manifest(args.manifest_path)

    print(f"Loading model type: {args.model_type}")

    if args.model_type == "med":
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=args.model_path,
            model_base=args.model_base,
            model_name=args.model_name,
        )
        wrapper = LlavaMedWrapper(
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            context_len=context_len,
        )

    elif args.model_type == "base":
        wrapper = LlavaBaseHFWrapper(args.base_model_id)

    elif args.model_type == "blip2":
        wrapper = Blip2Wrapper(args.blip2_model_id)

    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    print("Running eval...")
    results = run_eval(data, wrapper)

    print("Saving to csv...")
    save_to_csv(results, args.csv_name)

    print("Done.")


if __name__ == "__main__":
    main()