from llama_cpp import Llama
from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler
import base64
from PIL import Image


from time import time
start = time()




image_path = "./test_images/rubixcube.jpg"
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
    # n_threads=16,

)

#83 seconds with 16 cores

data_uri = image_to_base64_data_uri(image_path)

print("GENERATING PROMPT...")


description = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are MiniCPMv-2.6, an image description model. Your task is to provide a precise and factual description of the given image. Describe the image by specifying all observable elements including: Objects and their spatial relationships, Colors and textures, Sizes and shapes, Positions and orientations, Any textual or numerical data present, Avoid assumptions, interpretations, or metaphors. Your description should be a direct and unambiguous account of what is visually present in the image."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": """
Please provide a comprehensive and factual description of the image. Your description should include the following detailed elements:

1. **Objects and Elements**:
   - Identify and list all distinct objects, elements, and entities present in the image.
   - Describe each object’s specific appearance and any notable characteristics.

2. **Spatial Arrangement**:
   - Detail the relative positioning of each object within the image.
   - Specify distances or spatial relationships between objects if discernible.

3. **Orientation and Rotation**:
   - Indicate the orientation or angle at which each object is positioned.
   - Note any rotation or tilt of objects with respect to the image’s axes or other objects.

4. **Colors**:
   - Describe the color of each object, including any gradients, shading, or patterns.
   - If multiple colors are present, specify which colors are adjacent or overlapping.

5. **Sizes and Dimensions**:
   - Provide the approximate size or dimensions of each object, using relative measurements where exact dimensions are not possible.
   - Compare sizes of objects to each other when relevant.

6. **Shapes**:
   - Detail the shape of each object, including any geometric or irregular forms.
   - Note any distinguishing features such as curves, angles, or edges.

7. **Text and Numbers**:
   - Identify and transcribe any text, labels, or numbers visible in the image.
   - Specify the font type, size, color, and placement of any text or numbers.

8. **Background and Context**:
   - Describe the background or setting of the image, including any textures or colors.
   - Note if there are any contextual clues that provide additional information about the scene.

9. **Lighting and Shadows**:
   - Mention the lighting conditions and how they affect the visibility and appearance of objects.
   - Describe any shadows cast by objects and their relative positions.

10. **Additional Details**:
    - Include any other pertinent details that contribute to a complete understanding of the image.
    - Be as specific and detailed as possible, avoiding any subjective interpretation or assumptions.

Ensure that your description is precise, objective, and avoids any speculative or interpretive language.
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
    f.write(description + " -- The RGB colors you must use in the image are the following: " + str(colors))
print("OPERATION COMPLETED SUCCESSFULLY")


end = time()
print(f"EXECUTED IN {end-start} SECONDS")
