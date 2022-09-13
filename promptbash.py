import sys
import torch
import argparse

from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from math import floor
from random import choice, sample, random
from pathlib import Path

parser = argparse.ArgumentParser(description='Generate SD images based on a prompt matrix')

parser.add_argument('--num-prompts', type=int, default=10, help='Number of prompts to generate')
parser.add_argument('--images-per-prompt', type=int, default=10, help='Number of images to generate for each prompt')
parser.add_argument('--model-dir', type=str, required=True, help='Path to Stable Diffusion model directory')
parser.add_argument('--prompt-dir', type=str, required=True, help='Path to directory containing prompt text files')
parser.add_argument('--steps', type=int, default=50, help='Number of steps per image')
parser.add_argument('--width', type=int, default=512, help='Image width')
parser.add_argument('--height', type=int, default=512, help='Image height')

args = parser.parse_args()

def read_prompt_file(name):
    print(f'Reading {name}.txt')
    with open(f'{args.prompt_dir}/{name}.txt', 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]

        print(f'Loaded {len(lines)} items')
        return lines

def pick_one(arr):
    return choice(arr).strip().replace('\n', '')

def pick_some(arr, min = 2, max = 5):
    count = min + floor(random() * (max - min))
    return ', '.join(sample(arr, count)).strip().replace('\n', '')

subjects = read_prompt_file('subjects')
settings = read_prompt_file('settings')
modifiers = read_prompt_file('modifiers')
styles = read_prompt_file('styles')

total_images = args.num_prompts * args.images_per_prompt
print(f'Will generate {total_images} images total')

print('Ensuring output directory exists...')
Path('./output/').mkdir(exist_ok=True)

print(f'Loading Stable Diffusion with model at {args.model_dir}')
device = torch.device('cuda')
lms = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear"
)
pipe = StableDiffusionPipeline.from_pretrained(args.model_dir, scheduler=lms, revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to(device)

for prompt_index in range(args.num_prompts):
    prompt = ', '.join([
        pick_one(subjects),
        pick_one(settings),
        pick_some(modifiers),
        pick_one(styles)
    ])

    print(f'Running prompt {prompt}')
    for run in range(args.images_per_prompt):
        with autocast("cuda"):
            seed = floor(random() * 1000000000)
            generator = torch.Generator("cuda").manual_seed(seed)
            image = pipe(prompt, num_inference_steps=args.steps, generator=generator, height=args.height, width=args.width)["sample"][0]
            prompt_slug = prompt.replace(', ', '_')
            image_index = (prompt_index * args.images_per_prompt) + run
            image.save(f'./output/{image_index}-{seed}-{prompt_slug}.png', 'PNG')