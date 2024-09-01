# Imagine
An experimental image format whose goal is to demonstrate the ability of an AI to do a reverse prompt from an image and then reconstruct the image faithfully using only the prompt of the previous AI.

### Models
> This project is very early and, as such, models are subject to change

The models currently used are LLaVA (13b) for computing the reverse prompt, and Flux/Stable Diffusion (haven't decided yet) for generating the output image

### Installation
You need to have [Ollama](https://ollama.com/download) installed on your computer.
```sh
pip install -r requirements.txt
```

### License
[MIT License](LICENSE)
