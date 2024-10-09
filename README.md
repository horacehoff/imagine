# Imagine
An experimental image format whose goal is to demonstrate the ability of an AI to do a reverse prompt(calculate the prompt that would result in the given image) from an image and then reconstruct the image faithfully using only the prompt of the previous AI.

### Why, and how ?
This project is just, as of right now, a proof-of-concept. The main goal here is to save a lot of storage space by, at the end, making an image look slightly (or hugely, depending on the AI) different. They say an image is worth 1000 words, well if that's the case then we could be able to save huge amounts of storage space. Having said that, the relevance of this project in the future will, no matter what, depend on the evolution of our technology. For example, this project would simply become irrelevant if breakthroughs were to occur in the way we store data, allowing us to store more data than ever, cheaper than ever (for which I wish!). Moreover, this project will not be truly viable until advances in artifical intelligence allow us to run large models at a fraction of the computing power they require today.

### Models
> This project is very early in development and, as such, models are subject to change. Below are the instructions/required files to download the models and use them in the project.

#### MiniCPM-v2.6
- 'model/minicpm-v2_6/mmproj-model-f16.gguf'
- 'model/minicpm-v2_6/ggml-model-Q8_0.gguf' (or any other quantization you prefer)\
[MiniCPM repository](https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/tree/main)

### Installation
You need to install [Pytorch](https://pytorch.org/get-started/locally/) on your system. Then: 
```sh
pip install -r requirements.txt
```

### License
[MIT License](LICENSE)
