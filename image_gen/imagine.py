from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True)
pipe.to("cuda")

pipe.enable_xformers_memory_efficient_attention()

prompt = """
The image depicts a celestial body, specifically the moon, in a night sky. Here is a detailed description based on the elements present in the image:

1. **Objects and Elements**:
   - The primary object in the image is the moon, which is visible in a crescent phase. The moon appears to be in the upper central part of the image, occupying a significant portion of the frame.
   - The background is entirely black, indicating that the photo was taken during nighttime or in a dark environment.

2. **Spatial Arrangement**:
   - The moon is centrally positioned in the image, with ample dark space surrounding it. There are no other objects or celestial bodies visible in the immediate vicinity of the moon.

3. **Orientation and Rotation**:
   - The moon is oriented such that the left side of the image shows a darker portion, indicating the shadowed side, while the right side shows a lighter portion, indicating the illuminated side. The crescent shape suggests that the moon is in a waxing or waning phase.

4. **Colors**:
   - The moon is predominantly white with varying shades of gray, indicating different levels of illumination and shadow. The surface of the moon shows some texture, with craters and maria (dark plains) visible.

5. **Sizes and Dimensions**:
   - The moon appears to be of a size that is typical for a full moon, but since it is only partially visible, it is not possible to determine its exact diameter. The moon's size is relative to the black background, which provides no scale reference.

6. **Shapes**:
   - The shape of the moon is irregular, with a smooth curve along the crescent edge. The surface features such as craters and maria create a textured appearance.

7. **Text and Numbers**:
   - There is no visible text or numerical data present in the image.

8. **Background and Context**:
   - The background is a uniform black, suggesting that the photo was taken in a dark environment, likely at night. There are no contextual clues or additional elements that provide further information about the location or time of the photograph.

9. **Lighting and Shadows**:
   - The lighting in the image is focused on the moon, highlighting its surface features. The contrast between the illuminated and shadowed parts of the moon is stark, with the dark areas appearing almost black against the bright lunar surface.

10. **Additional Details**:
    - The image appears to be a photograph rather than an artistic rendering or a digital creation. The clarity and detail suggest that it was taken with a high-quality camera or telescope.

In summary, the image is a clear and detailed photograph of the moon in a crescent phase against a black night sky. The moon's texture and the contrast between its illuminated and shadowed areas are the main focal points of the image."""

images = pipe(prompt=prompt).images[0]
images.save("test.png")
