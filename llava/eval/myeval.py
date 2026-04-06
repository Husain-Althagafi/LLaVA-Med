from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
from llava.model.builder import load_pretrained_model

from llava.eval.wrappers.llava_base import LLaVABaseWrapper

import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
import math
from transformers import set_seed, logging
import pandas as pd

OUTPUT_PATH = 'outputs'

PROMPTS = [
    "Describe this ophthalmic image briefly.",
    "What abnormalities are visible? If none are visible, say so clearly.",
    "Give the most likely diagnosis and a one-sentence justification based only on visible evidence."
]


from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch


class LLaVABaseWrapper:
    def __init__(self, model_id):
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to('cuda')

        self.processor = AutoProcessor.from_pretrained(model_id)

    
    def generate(self, image, text):
        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image"},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)

                
        output = self.model.generate(**inputs, max_new_tokens=400, do_sample=False)
        return self.processor.decode(output[0][2:], skip_special_tokens=True)



   


def load_image(path):
    return Image.open(path)


def load_manifest(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    return data


def save_to_csv(results, file_name):    
    df = pd.DataFrame(results)
    df.to_csv(f"outputs/{file_name}.csv", index=False)


def run_eval(data, tokenizer, model, image_processor, context_len):
    results = []

    for sample in data:
        img = load_image(sample['image'])
        image_tensor = process_images([img], image_processor, model.config)[0]

        for prompt in PROMPTS:
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, '').strip()
            if model.config.mm_use_im_start_end:
                prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
            else:
                prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
            
            conv = conv_templates['mistral_instruct'].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()    

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)


            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True,
                    temperature=0.2,
                    top_p=None,
                    num_beams=1,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            results.append({
                "image": sample["image"],
                "label": sample["label"],
                "prompt": prompt,
                "response": outputs
            })
        
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='med')
    parser.add_argument('--model_path', type=str, default='/work/vlmwork/LLaVA-Med/llavamodel')
    parser.add_argument('--model_base', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='llava-med-v1.5-mistral-7b')
    parser.add_argument('--manifest_path', type=str, default='data/vlmdata/manifest.jsonl')
    parser.add_argument('--image_folder', type=str, default='data/vlmdata')
    parser.add_argument('--csv_name', type=str, required=True)
    return parser.parse_args()
    


def main():
    args = parse_args()

    print(f'Loading model...')
    tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=args.model_path,
            model_base=args.model_base,
            model_name=args.model_name
    ) if 

    print(f'Loading manifest...')
    data = load_manifest(args.manifest_path)

    print(f'Running eval...')
    results = run_eval(data, tokenizer, model, image_processor, context_len)

    print(f'Saving to csv...')
    save_to_csv(results, args.csv_name)

    

if __name__ == '__main__':
    main()