"""
The goal is to take as an input an image and output a prompt
"""
import ollama
from PIL import Image

image_path = "./test_images/stars.jpg"
# Lossy is highly recommended
lossless = False

print("GENERATING PROMPT....")
response = ollama.chat(
	# 34b model is also an option but is way more demanding regarding processing power
	model="llava:13b",
	messages=[
		{
			'role': 'user',
			'content': 'Provide a highly detailed description of the image, starting with a general overview of the scene, including all main elements and their precise and exact spatial relationships. Describe the foreground and background in full detail, specifying the exact and precise positions, sizes, and orientations of objects, subjects, or entities. Include a comprehensive breakdown of the color palette, noting specific hues, shades, and gradients, as well as any lighting effects such as shadows, highlights, and reflections, with their sources. Explain the textures and patterns on all surfaces, describing whether they are smooth, rough, shiny, matte, or otherwise distinct, and identify the materials and their physical appearance. Detail the environment or setting, including natural or man-made elements like trees, buildings, skies, landscapes, and any relevant weather conditions. If there are people or animals present, provide a full description of their appearance, including facial expressions, clothing, poses, and notable features like hair, fur, or skin tone. Comment on the artistic style or technique used, such as realism, impressionism, or abstract qualities, and mention any unique artistic effects or stylizations. Describe any text, symbols, or intricate details within the image, and how these contribute to the overall composition and meaning. Finally, discuss the overall composition and balance of the image, explaining how the elements are arranged, including composition, balance, symmetry, and focal points, and how these elements work together to create a cohesive image that can be faithfully reproduced. Do not use metaphores, use only facts, and be as detailed as possible, as precise and mathematical as possible. Remember that an AI model must be able to generate the image from your description, so be as clear and detailed as possible. Think very hard before qualifying an object, as it may not be what it looks like, as such it is simpler for you to describe objects using shapes and colors, and thus be more precise.',
			'images': [image_path]
		}
	]
)

description = response['message']['content']

print("RETRIEVING PALETTE...")
colors = []
img = Image.open(image_path)
if not lossless:
	img = img.quantize(colors=100).convert("RGB")

for x in range(img.width):
	for y in range(img.height):
		color = img.getpixel((x, y))
		if color not in colors:
			colors.append(img.getpixel((x, y)))

print(colors)

print("WRITING TO FILE...")
with open("test.imagine", "w+") as f:
	f.write(description+" -- The RGB colors you must use in the image are the following: "+str(colors))
print("OPERATION COMPLETED SUCCESSFULLY")


