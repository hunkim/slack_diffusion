import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

def diffusion(prompt):
    with autocast("cuda"):
        image = pipe(prompt, guidance_scale=7.5)["sample"][0]
        return image  
    

if __name__ == "__main__":
    prompt = "diffusion slack app icon for upstage ai company."
    image = diffusion(prompt)
    image.save("astronaut_rides_horse.png")

