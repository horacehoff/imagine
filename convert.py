"""
The goal is to take as an input an image and output a prompt
"""
from PIL import Image
import ollama


image_path = "./test_images/rubixcube.jpg"
# Lossy is highly recommended
lossless = False
# 34b model is also an option but is way more demanding regarding processing power
model = "llava:13b"


try:
    ollama.chat(model=model)
except:
    print("INITIALIZING OLLAMA...")
    import subprocess, time

    ollama_proc = subprocess.Popen(["ollama", "run", model])
    time.sleep(1)
    ollama_proc.kill()


print("GENERATING PROMPT...")
response = ollama.chat(
    model=model,
    messages=[
        {
            'role': 'user',
            'content': 'Provide a very detailed and precise analysis of the image without assumptions or interpretations. Do not qualify objects, people, or actions. If needed, be as precise as to describe individual pixels and their position. Be exhaustive and detailed. Please provide all the required details for a perfect reproduction, describe the image as if I needed to recreate it from scratch using only your description.',
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
    f.write(description + " -- The RGB colors you must use in the image are the following: " + str(colors))
print("OPERATION COMPLETED SUCCESSFULLY")
