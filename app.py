# # from flask import Flask, request, jsonify
# # from flask_cors import CORS
# # import torch
# # import logging
# # import base64
# # import io
# # from PIL import Image
# # from torchvision import transforms as tfms
# # from transformers import CLIPTextModel, CLIPTokenizer
# # from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
# # from concurrent.futures import ThreadPoolExecutor

# # # Disable warnings
# # logging.disable(logging.WARNING)

# # # Set the device
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # Initialize Flask app
# # app = Flask(__name__)
# # CORS(app)

# # # Load models globally but lazily
# # tokenizer = None
# # text_encoder = None
# # vae = None
# # unet = None
# # scheduler = None

# # def load_models():
# #     """Load models if they are not already loaded."""
# #     global tokenizer, text_encoder, vae, unet, scheduler
# #     if tokenizer is None:
# #         tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", clean_up_tokenization_spaces=False)
# #     if text_encoder is None:
# #         text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(device)
# #     if vae is None:
# #         vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float16).to(device)
# #     if unet is None:
# #         unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float16).to(device)
# #     if scheduler is None:
# #         scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
# #         scheduler.set_timesteps(num_inference_steps=50)  # Specify the number of inference steps

# # def magic_prompt(color, rash_type, body_part):
# #     """Construct a detailed prompt using the extracted values."""
# #     return (
# #         f"Create a highly detailed and realistic image showing the {body_part} of a person with {color} skin. "
# #         f"The {body_part} should display a typical {rash_type} infection, characterized by a clearly visible rash."
# #     )

# # def load_image(p):
# #     """Load images from a defined path."""
# #     return Image.open(p).convert('RGB').resize((512, 512))

# # def pil_to_latents(image):
# #     """Convert PIL image to latent space."""
# #     init_image = tfms.ToTensor()(image).unsqueeze(0)
# #     init_image = init_image.to(device=device, dtype=torch.float16)
# #     return vae.encode(init_image).latent_dist.sample()

# # def latents_to_pil(latents):
# #     """Convert latent space to PIL image."""
# #     latents = latents.to(device=device, dtype=torch.float16)
# #     image = vae.decode(latents)
# #     image = (image / 2 + 0.5).clamp(0, 1)
# #     image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
# #     images = (image * 255).round().astype("uint8")
# #     pil_images = [Image.fromarray(image) for image in images]
# #     return pil_images[0]

# # def prompt_to_image(prompt, save_int=False):
# #     """Generate an image from a prompt."""
# #     load_models()  # Ensure models are loaded
# #     with torch.no_grad():
# #         # Tokenize and encode the text prompt
# #         inp = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
# #         text_emb = text_encoder(**inp).last_hidden_state

# #         # Convert text embedding to float16
# #         text_emb = text_emb.to(dtype=torch.float16)

# #         # Generate initial latent space
# #         latents = torch.randn((1, unet.config.in_channels, 64, 64), device=device, dtype=torch.float16)

# #         # Set scheduler timesteps
# #         scheduler.set_timesteps(num_inference_steps=50)  # Adjust the number of inference steps if needed

# #         # Diffusion process
# #         for t in scheduler.timesteps:
# #             latent_model_input = scheduler.scale_model_input(latents, t)
# #             noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_emb).sample
# #             latents = scheduler.step(noise_pred, t, latents).prev_sample

# #         # Convert latents to image
# #         image = latents_to_pil(latents)
# #         return image
# import re
# import torch, logging

# import torch
# from PIL import Image
# import io

# ## disable warnings
# logging.disable(logging.WARNING)

# ## Imaging  library
# from PIL import Image
# from torchvision import transforms as tfms

# ## Basic libraries
# import numpy as np
# from tqdm.auto import tqdm
# import matplotlib.pyplot as plt
# from IPython.display import display
# import shutil
# import os

# ## For video display
# from IPython.display import HTML
# from base64 import b64encode


# ## Import the CLIP artifacts
# from transformers import CLIPTextModel, CLIPTokenizer
# from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
# from IPython.display import display, clear_output
# import os

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def magic_prompt(color, rash_type, body_part):
#     # Construct the detailed prompt using the extracted values
#     return f"Create a highly detailed and realistic image showing the {body_part} of a person with {color} skin. The {body_part} should display a typical {rash_type} infection, characterized by a clearly visible rash."


# ## Helper functions
# def load_image(p):
#     '''
#     Function to load images from a defined path
#     '''
#     return Image.open(p).convert('RGB').resize((512,512))

# def pil_to_latents(image):
#     '''
#     Function to convert image to latents
#     '''
#     init_image = tfms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0
#     init_image = init_image.to(device="cuda", dtype=torch.float16)
#     init_latent_dist = vae.encode(init_image).latent_dist.sample() * 0.18215
#     return init_latent_dist

# def latents_to_pil(latents):
#     '''
#     Function to convert latents to images
#     '''
#     latents = (1 / 0.18215) * latents
#     with torch.no_grad():
#         image = vae.decode(latents).sample
#     image = (image / 2 + 0.5).clamp(0, 1)
#     image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
#     images = (image * 255).round().astype("uint8")
#     pil_images = [Image.fromarray(image) for image in images]
#     return pil_images

# def text_enc(prompts, maxlen=None):
#     '''
#     A function to take a texual promt and convert it into embeddings
#     '''
#     if maxlen is None: maxlen = tokenizer.model_max_length
#     inp = tokenizer(prompts, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
#     return text_encoder(inp.input_ids.to("cuda"))[0].half()

# def prompt_2_img(prompts, g=7.5, seed=100, steps=70, dim=512, save_int=True):
#     """
#     Diffusion process to convert prompt to image, modified to yield images for Streamlit.
#     """

#     bs = len(prompts)
#     text = text_enc(prompts)
#     uncond = text_enc([""] * bs, text.shape[1])
#     emb = torch.cat([uncond, text])

#     if seed:
#         torch.manual_seed(seed)

#     latents = torch.randn((bs, unet.in_channels, dim//8, dim//8))
#     scheduler.set_timesteps(steps)
#     latents = latents.to("cuda").half() * scheduler.init_noise_sigma

#     for i, ts in enumerate(scheduler.timesteps):
#         inp = scheduler.scale_model_input(torch.cat([latents] * 2), ts)
#         with torch.no_grad():
#             u, t = unet(inp, ts, encoder_hidden_states=emb).sample.chunk(2)
#         pred = u + g*(t-u)
#         latents = scheduler.step(pred, ts, latents).prev_sample

#         if save_int and i % (steps // 70) == 0:  # Yield 10 images throughout the process
#             image = latents_to_pil(latents)[0]
#             buf = io.BytesIO()
#             image.save(buf, format="JPEG")
#             byte_im = buf.getvalue()
#             yield byte_im  # Yield image in bytes format for Streamlit to display

#     final_image = latents_to_pil(latents)
#     final_buf = io.BytesIO()
#     final_image[0].save(final_buf, format="JPEG")
#     final_byte_im = final_buf.getvalue()
#     yield final_byte_im  # Yield the final image

# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
# text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to("cpu")

# ## Initiating the VAE
# vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float16).to("cpu")

# ## Initializing a scheduler and Setting number of sampling steps
# scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
# scheduler.set_timesteps(50)

# ## Initializing the U-Net model
# unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float16).to("cpu")

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from PIL import Image
# import base64
# import io

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# @app.route('/generate', methods=['POST'])
# def generate():
#     data = request.json
#     skin_rash_type = data.get('skinRashType')
#     skin_color = data.get('skinColor')
#     body_part = data.get('bodyPart')
#     detailed_prompt = magic_prompt(skin_color, skin_rash_type, body_part)
#     generator_new = prompt_2_img([detailed_prompt], save_int=True)
#     total_steps = 70
#     try:
#       for i, image_bytes in enumerate(generator_new):
#         print(f"Processing image {i}")
#         pg = min((step) / total_steps, 100)
#         if step%13 == 0:
#             clear_output(wait=True)
#             print(f"Progress: {pg:.2f}%")
#     except Exception as e:
#         print(f"An error occurred: {e}")
#     for step, image_bytes in enumerate(generator_new):
#             pg = min((step) / total_steps, 100)
#             if step%13 == 0:
#                 clear_output(wait=True)
#                 print(f"Progress: {pg:.2f}%")
#     # try:
#     #   image_bytes = next(generator_new)
#     # except StopIteration:
#     #   print("No more images.")
#     # except Exception as e:
#     #     print(f"An error occurred: {e}")
#     img_str = base64.b64encode(image_bytes).decode("utf-8")

#     return jsonify({"image": img_str})

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import logging
import base64
import io
from PIL import Image
from torchvision import transforms as tfms
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler

# Disable warnings
logging.disable(logging.WARNING)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load models globally
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", clean_up_tokenization_spaces=False)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(device)
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float16).to(device)
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float16).to(device)
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
scheduler.set_timesteps(num_inference_steps=50)  # Specify the number of inference steps

def magic_prompt(color, rash_type, body_part):
    """Construct a detailed prompt using the extracted values."""
    return (
        f"Create a highly detailed and realistic image showing the {body_part} of a person with {color} skin. "
        f"The {body_part} should display a typical {rash_type} infection, characterized by a clearly visible rash."
    )

def load_image(p):
    """Load images from a defined path."""
    return Image.open(p).convert('RGB').resize((512, 512))

def pil_to_latents(image):
    """Convert PIL image to latent space."""
    init_image = tfms.ToTensor()(image).unsqueeze(0)
    init_image = init_image.to(device=device, dtype=torch.float16)
    return vae.encode(init_image).latent_dist.sample()

def latents_to_pil(latents):
    """Convert latent space to PIL image."""
    latents = latents.to(device=device, dtype=torch.float16)
    image = vae.decode(latents)
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images[0]

def prompt_to_image(prompt):
    """Generate an image from a prompt."""
    with torch.no_grad():
        # Tokenize and encode the text prompt
        inp = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
        text_emb = text_encoder(**inp).last_hidden_state

        # Convert text embedding to float16
        text_emb = text_emb.to(dtype=torch.float16)

        # Generate initial latent space
        latents = torch.randn((1, unet.config.in_channels, 64, 64), device=device, dtype=torch.float16)

        # Set scheduler timesteps
        scheduler.set_timesteps(num_inference_steps=50)  # Adjust the number of inference steps if needed

        # Diffusion process
        for t in scheduler.timesteps:
            latent_model_input = scheduler.scale_model_input(latents, t)
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_emb).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Convert latents to image
        image = latents_to_pil(latents)
        return image

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    skin_rash_type = data.get('skinRashType')
    skin_color = data.get('skinColor')
    body_part = data.get('bodyPart')
    detailed_prompt = magic_prompt(skin_color, skin_rash_type, body_part)
    print(detailed_prompt)
    image = prompt_to_image(detailed_prompt)
    total_steps = 70
    try:
      for i, image_bytes in enumerate(image):
        print(f"Processing image {i}")
        pg = min((step) / total_steps, 100)
        if step%13 == 0:
            clear_output(wait=True)
            print(f"Progress: {pg:.2f}%")
    except Exception as e:
        print(f"An error occurred: {e}")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify({"image": img_str})

if __name__ == '__main__':
    app.run(debug=True)
