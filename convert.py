"""
The goal is to take as an input an image and output a prompt
"""


from PIL import Image
import ollama
try:
	ollama.chat()
except:
	print("INITIALIZING OLLAMA...")
	import subprocess, time
	ollama_proc = subprocess.Popen(["ollama","run","llava:13b"])
	time.sleep(1)
	ollama_proc.kill()



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
			'content': 'provide a detailed description for this photo without interpretations. Skip using word "image". No interpretations.',
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


