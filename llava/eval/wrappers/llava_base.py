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



   
