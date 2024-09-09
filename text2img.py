"""
Goal here is to take a .imagine file as an input and feed the whole file to an image-generation model, like Flux or Stable Diffusion
"""


# Literally cannot run on my GPU atm (RTX 3070)
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "The image presents a clear view of the moon, captured in a state of partial illumination against the backdrop of a dark sky. The moon is positioned centrally in the frame, occupying a significant portion of the visual space. Its surface is detailed, showcasing a variety of craters and maria, which are the dark, flat areas on the lunar surface. The moon's surface appears textured, with a range of shades from white to gray, indicating the presence of different materials and elevations. The surrounding sky is uniformly dark, providing a stark contrast that accentuates the moon's luminosity. There are no other celestial bodies or objects visible in the image, and no additional context or background story is provided. The image is a straightforward representation of the moon in its natural environment, captured in a moment of celestial beauty"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cuda").manual_seed(0)
).images[0]
image.save("flux-dev.png")