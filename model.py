from llama_cpp import Llama
from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler
import base64
from PIL import Image



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
    verbose=True
)

data_uri = image_to_base64_data_uri(image_path)

print("GENERATING PROMPT...")
description = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are an assistant who perfectly describes images."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Can you describe super faithfully to me in english please ? I want to reproduce it using only your description and as such could you describe everything, every detail please ?"},
                {"type": "image_url", "image_url": {
                    "url": data_uri}}
            ]
        }
    ],
    temperature=0.5,
    repeat_penalty=0.5
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
