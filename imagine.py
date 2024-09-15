from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True)
pipe.to("cuda")

pipe.enable_xformers_memory_efficient_attention()

prompt = """
The image depicts a night sky with a prominent celestial body, the moon, in the upper central portion. The moon is captured in a phase where it is partially illuminated, showing a crescent shape with a significant portion of its surface visible. The surface of the moon is detailed, with various craters and maria (lunar plains) discernible, indicating its rugged and uneven terrain. The moon is set against a stark black background, which suggests that the photograph was taken during the night when the sky is dark.

Spatially, the moon is centrally located in the image, occupying a significant portion of the frame. There are no other objects or celestial bodies visible in the image, placing the entire focus on the moon. The moon's orientation is such that it appears to be slightly tilted, with the left side of the crescent being more illuminated than the right side.

The colors in the image are monochromatic, with the moon's surface appearing in shades of gray against the black backdrop of the night sky. The texture of the moon's surface is detailed and varied, with different shades indicating the presence of craters and other lunar features.

There are no discernible sizes or dimensions provided in the image, as the moon's actual size is not visible. However, it is clear that the moon is the largest object in the frame, and its crescent shape suggests it is not at full or new moon, but rather in a phase where only a portion of the moon is visible from Earth.

The shapes in the image are primarily the crescent shape of the moon, with its surface showing circular and oval forms due to the craters and maria. The background is a uniform black, providing a high contrast to the moon's grayish tones.

There is no text or numerical data present in the image, and the context is limited to the natural phenomenon of the moon in the night sky. The lighting conditions are such that the moon is the primary source of light in the image, illuminating its surface and casting subtle shadows within the craters. The shadows add depth to the moon's surface, highlighting the texture and topography.

In summary, the image is a clear and detailed photograph of a partially illuminated moon in the night sky, with a focus on the moon's surface features and the stark contrast between the moon and the dark background. -- The RGB colors you must use in the image are the following: [(0, 0, 0), (1, 1, 1), (2, 2, 2), (0, 2, 0), (0, 2, 1), (0, 1, 0), (1, 1, 0), (3, 3, 3), (2, 2, 0), (3, 3, 1), (1, 1, 3), (2, 0, 1), (2, 3, 0), (3, 2, 0), (1, 3, 0), (4, 4, 2), (5, 5, 3), (6, 6, 4), (4, 4, 4), (7, 7, 5), (3, 5, 1), (4, 7, 3), (6, 8, 4), (8, 8, 6), (7, 9, 5), (4, 4, 0), (6, 6, 0), (7, 6, 3), (9, 9, 7), (2, 2, 4), (3, 3, 5), (5, 5, 5), (10, 10, 8), (6, 6, 6), (7, 7, 7), (8, 8, 8), (8, 9, 4), (9, 11, 8), (9, 9, 9), (10, 10, 10), (11, 8, 8), (8, 10, 7), (10, 13, 10), (11, 11, 9), (11, 11, 11), (11, 14, 10), (12, 12, 10), (13, 14, 11), (14, 15, 12), (11, 12, 14), (16, 16, 13), (22, 24, 21), (0, 2, 3), (34, 35, 33), (43, 44, 42), (55, 57, 56), (66, 68, 67), (72, 74, 73), (81, 82, 81), (96, 97, 96), (110, 111, 110), (127, 128, 128), (138, 139, 139), (144, 144, 144), (147, 149, 148), (152, 154, 153), (158, 159, 158), (164, 165, 164), (169, 170, 169), (162, 163, 161), (134, 135, 135), (117, 119, 119), (5, 5, 7), (167, 167, 166), (173, 174, 172), (178, 179, 177), (184, 185, 183), (189, 190, 188), (194, 195, 193), (199, 199, 197), (197, 197, 194), (200, 202, 199), (12, 12, 6), (203, 204, 201), (131, 132, 131), (207, 208, 205), (122, 123, 123), (216, 216, 213), (3, 4, 0), (9, 8, 2), (10, 10, 1), (3, 3, 0), (6, 4, 1), (7, 7, 0), (10, 6, 1), (3, 1, 1), (5, 5, 0), (4, 2, 3), (6, 5, 3), (0, 0, 2)]
"""

images = pipe(prompt=prompt).images[0]
images.save("test.png")
