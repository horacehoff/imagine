# Imagine
An experimental image format whose goal is to demonstrate the ability of an AI to do a reverse prompt from an image and then reconstruct the image faithfully using only the prompt of the previous AI.

### Models
> This project is very early in development and, as such, models are subject to change. Below are the instructions/required files to download the models and use them in the project.

#### MiniCPM-v2.6
- 'model/minicpm-v2_6/mmproj-model-f16.gguf'\
- 'model/minicpm-v2_6/ggml-model-Q8_0.gguf' (or any other quantization you prefer)\
[MiniCPM repository](https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/tree/main)

### Installation
You need to install [Pytorch](https://pytorch.org/get-started/locally/) on your system. Then: 
```sh
pip install -r requirements.txt
```

### License
[MIT License](LICENSE)
