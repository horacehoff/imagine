from llama_cpp import Llama
from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler
import base64
from PIL import Image



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
    n_ctx=2048,  # n_ctx should be increased to accommodate the image embedding
    verbose=True
)

file_path = 'test_images/rubixcube.jpg'
data_uri = image_to_base64_data_uri(file_path)

print("GENERATING PROMPT...")
description = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are an assistant who perfectly describes images."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Do an exhaustive reverse prompt of this image. Basically, create a prompt of this image, allowing any AI model to reproduce this image exactly, but don't do instructions, do rather a description."},
                {"type": "image_url", "image_url": {
                    "url": data_uri}}
            ]
        }
    ]
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
