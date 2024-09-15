from torch import autocast, compile
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    use_auth_token=True,
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()

prompt = """
The image features a Rubik's Cube, a well-known puzzle toy, prominently displayed against a gradient blue background. The cube is composed of smaller cubes, each with a different color: yellow, green, red, orange, white, and blue. The colors are arranged in a standard Rubik's Cube pattern, with each face of the cube having a uniform color scheme. The cube is positioned on a reflective surface, which creates a subtle reflection of the cube's bottom face.

Spatially, the Rubik's Cube is the central object in the image, with its top face slightly out of focus, suggesting a shallow depth of field. The cube is tilted at an angle, with the bottom face closer to the viewer than the top face. The lighting in the image is soft and diffused, casting gentle shadows to the right of the cube, indicating a light source to the left of the frame.

The colors of the Rubik's Cube are vivid and distinct, with each color block clearly visible. The yellow faces are on the top and the left side, the green faces are on the top and the right side, the red faces are on the front and the left side, the orange faces are on the front and the right side, the white faces are on the back and the left side, and the blue faces are on the back and the right side. The edges and corners of the cube are sharp, and the individual squares are well-defined, indicating that the cube is intact and not scrambled.

There is no text or numerical data present in the image. The background is a smooth gradient of blue, which provides a contrasting backdrop that highlights the colors of the Rubik's Cube. The overall composition of the image focuses on the Rubik's Cube, with the background and lighting serving to emphasize its details and colors. -- The RGB colors you must use in the image are the following: [(9, 32, 47), (14, 33, 45), (17, 41, 54), (12, 42, 59), (19, 46, 59), (20, 49, 62), (21, 53, 73), (23, 57, 77), (25, 61, 79), (30, 66, 86), (32, 70, 90), (33, 73, 93), (33, 76, 98), (38, 77, 100), (34, 80, 103), (38, 82, 106), (32, 74, 98), (21, 66, 93), (10, 29, 38), (8, 26, 35), (11, 24, 22), (9, 18, 19), (38, 73, 92), (39, 86, 109), (16, 38, 44), (41, 83, 108), (40, 88, 111), (41, 88, 113), (31, 62, 80), (41, 90, 117), (46, 90, 115), (39, 93, 115), (46, 93, 120), (49, 97, 124), (46, 98, 129), (48, 101, 130), (51, 94, 119), (49, 102, 133), (53, 106, 136), (50, 107, 137), (53, 110, 141), (60, 106, 132), (59, 110, 140), (59, 114, 146), (54, 114, 146), (40, 81, 98), (59, 117, 149), (47, 84, 105), (59, 99, 123), (61, 122, 154), (69, 116, 144), (65, 123, 155), (66, 131, 168), (76, 135, 169), (86, 133, 161), (48, 92, 112), (67, 130, 150), (82, 139, 174), (31, 52, 59), (91, 140, 173), (73, 117, 144), (98, 149, 183), (102, 147, 176), (83, 146, 174), (106, 156, 188), (57, 86, 105), (116, 155, 182), (120, 166, 196), (123, 162, 185), (65, 106, 131), (131, 170, 195), (137, 175, 199), (47, 60, 69), (152, 186, 209), (158, 190, 211), (166, 195, 213), (39, 51, 58), (149, 177, 190), (136, 114, 90), (36, 97, 112), (44, 161, 142), (29, 105, 84), (118, 37, 32), (60, 48, 49), (9, 66, 33), (5, 40, 32), (31, 24, 19), (1, 8, 8), (1, 1, 6), (1, 1, 1), (9, 6, 6), (3, 185, 48), (1, 248, 1), (251, 253, 1), (208, 212, 151), (7, 98, 189), (179, 197, 199), (244, 248, 204), (19, 108, 190), (245, 17, 20)]
"""
with autocast("cuda"):
    image = pipe(prompt).images[0]

image.save("astronaut_rides-_horse.png")