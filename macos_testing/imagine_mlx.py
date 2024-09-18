from diffusionkit.mlx import DiffusionPipeline
pipeline = DiffusionPipeline(
  shift=3.0,
  use_t5=False,
  model_version="argmaxinc/mlx-stable-diffusion-3-medium",
  low_memory_mode=True,
  a16=True,
  w16=True,
)

HEIGHT = 1024
WIDTH = 1024
NUM_STEPS = 50  #  4 for FLUX.1-schnell, 50 for SD3 and FLUX.1-dev
CFG_WEIGHT = 7. # for FLUX.1-schnell, 5. for SD3

image, _ = pipeline.generate_image(
  "A picture of a rubix cube",
  cfg_weight=CFG_WEIGHT,
  num_steps=NUM_STEPS,
  latent_size=(HEIGHT // 8, WIDTH // 8),
)

image.save("test.png")