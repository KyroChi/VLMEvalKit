import torch

from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

model_path = 'llava-hf/llava-v1.6-vicuna-7b-hf'

processor = LlavaNextProcessor.from_pretrained(model_path)

model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to("cuda:0")

# prepare image and text prompt, using the appropriate prompt template
image_path = '/home/kyle/luka/qtvit/assets/486A3683.jpg'
image = Image.open(image_path)

text = "How many flowers are in the bridesmaid's bouquet?"

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": text},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(image, prompt, return_tensors="pt").to("cuda:0")

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=100)

print(processor.decode(output[0], skip_special_tokens=True))