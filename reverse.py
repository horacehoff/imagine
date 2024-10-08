from llama_cpp import Llama
from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler
import base64
from PIL import Image
from time import time
start = time()


image_path = "./test_images/moon.jpg"
# Lossy is highly recommended
lossless = False

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"


chat_handler = MiniCPMv26ChatHandler(clip_model_path="model/minicpm-v2_6/mmproj-model-f16.gguf")
llm = Llama(
    model_path="model/minicpm-v2_6/ggml-model-Q8_0.gguf",
    chat_handler=chat_handler,
    n_ctx=4906,  # n_ctx should be increased to accommodate the image embedding
    verbose=True,
    n_threads=16,
)


#114 seconds with threads not specified
#126 seconds with 8 threads
#83 seconds with 16 threads
#144 seconds with 24 threads

data_uri = image_to_base64_data_uri(image_path)

print("GENERATING PROMPT...")


description = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are MiniCPMv-2.6, an image description model. Your task is to provide a precise and factual description of the given image. Describe the image by specifying all observable elements including: Objects and their spatial relationships, Colors and textures, Sizes and shapes, Positions and orientations, Any textual or numerical data present, Avoid assumptions, interpretations, or metaphors. Your description should be a direct and unambiguous account of what is visually present in the image."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": """
Provide a comprehensive and factual description of the image, focusing on all visible elements. Use precise numerical values to describe the size and position of objects. All measurements should be expressed as percentages of the image's width and height for both size and position.

Identify distinct objects and entities, detailing their appearance and notable characteristics. Describe their exact positions within the image using percentages relative to the image's dimensions (e.g., "an object is located 20% from the left edge and 30% from the top edge"). Similarly, indicate their sizes in terms of width and height as percentages of the total image size (e.g., "the object occupies 15% of the image's width and 10% of its height").

For orientation and rotation, specify the angle of each object relative to the image's axes or other elements.

Describe the color of each object, noting shading, gradients, and adjacent or overlapping colors. Ensure colors are clearly described with respect to their interaction and relationship within the image.

Compare the sizes of objects relative to each other, using the same percentage-based system. Outline the shapes of the objects, detailing geometric or irregular forms with clear reference to their curves, angles, or edges.

If text or numbers are visible, transcribe them and describe their placement using precise percentages for both position and size. Include font type, size, and color information.

Describe the background and context of the image, noting any textures, colors, or contextual clues, with positional details given where applicable. Pay attention to the lighting conditions and explain how they affect the visibility and appearance of objects, along with shadow placement and extent (also using percentages).

Include any additional relevant details to ensure a complete understanding of the image. If the image was generated by AI, specify the art style. Maintain an objective and precise tone, avoiding speculative or subjective interpretations throughout the description.
"""},
                {"type": "image_url", "image_url": {
                    "url": data_uri}}
            ]
        }
    ],
    temperature=0.1,
)["choices"][0]["message"]["content"]
print(description)



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
    f.write(str(img.width)+"#"+str(img.height)+"#"+description + " -- The RGB colors you must use in the image are the following: " + str(colors))
print("OPERATION COMPLETED SUCCESSFULLY")


end = time()
print(f"EXECUTED IN {end-start} SECONDS")
