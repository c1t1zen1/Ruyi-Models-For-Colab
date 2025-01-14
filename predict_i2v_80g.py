import os

import torch
from PIL import Image
from diffusers import (EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
                       DPMSolverMultistepScheduler, PNDMScheduler, DDIMScheduler)
from omegaconf import OmegaConf
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download

from ruyi.data.bucket_sampler import ASPECT_RATIO_512, get_closest_ratio
from ruyi.models.autoencoder_magvit import AutoencoderKLMagvit
from ruyi.models.transformer3d import HunyuanTransformer3DModel
from ruyi.pipeline.pipeline_ruyi_inpaint import RuyiInpaintPipeline
from ruyi.utils.lora_utils import merge_lora, unmerge_lora
from ruyi.utils.utils import get_image_to_video_latent, save_videos_grid

# Input and output
start_image_path    = "assets/girl_01.jpg"
end_image_path      = "assets/girl_02.jpg" # Can be None for start-image-to-video
output_video_path   = "outputs/example_01.mp4"

# Video settings
video_length        = 120       # The max video length is 120 frames (24 frames per second)
base_resolution     = 512       # # The pixels in the generated video are approximately 512 x 512. Values in the range of [384, 896] typically produce good video quality.
video_size          = None      # Override base_resolution. Format: [height, width], e.g., [384, 672]
# Control settings
aspect_ratio        = "16:9"    # Do not change, currently "16:9" works better
motion              = "auto"    # Motion control, choose in ["1", "2", "3", "4", "auto"]
camera_direction    = "auto"    # Camera control, choose in ["static", "left", "right", "up", "down", "auto"]
# Sampler settings
steps               = 25
cfg                 = 7.0
scheduler_name      = "DDIM"    # Choose in ["Euler", "Euler A", "DPM++", "PNDM","DDIM"]

# GPU memory settings
low_gpu_memory_mode = False     # Low gpu memory mode
gpu_offload_steps   = 0         # Choose in [0, 10, 7, 5, 1], the latter number requires less GPU memory but longer time

# Random seed
seed                = 42        # The Answer to the Ultimate Question of Life, The Universe, and Everything

# Model settings
config_path         = "config/default.yaml"
model_name          = "Ruyi-Mini-7B"
model_type          = "Inpaint"
model_path          = f"models/{model_name}"    # (Down)load mode in this path
auto_download       = True                      # Automatically download the model if the pipeline creation fails
auto_update         = True                      # If auto_download is enabled, check for updates and update the model if necessary

# FP8 settings
fp8_quant_mode      = "none"    # Choose in ["none", "lite", "strong", "extreme"]. GPU memory decreases depending on the modes: bf16 default > fp8 lite > fp8 strong > fp8 extreme.
fp8_data_type       = "auto"    # Choose in ["auto", "fp8_e5m2", "fp8_e4m3fn"]. The "extreme" mode with "fp8_e5m2" is not recommended for achieving good quality.

# LoRA settings
lora_path           = None
lora_weight         = 1.0

# Other settings
weight_dtype = torch.bfloat16
device = torch.device("cuda")

# TeaCache settings
tea_cache_enabled                  = False
tea_cache_threshold                = 0.10   # A smaller threshold results in fewer cached steps. 0.10 caches 6 ~ 8 steps, 0.15 caches 10 ~ 12 steps normally.
tea_cache_skip_start_steps         = 3      # First n steps do not use TeaCache, should be >= 1
tea_cache_skip_end_steps           = 1      # Last n steps do not use TeaCache, should be >= 1
tea_cache_offload_cpu              = False  # Offload TeaCache tensors to cpu, which could save some GPU memory

# Enhance-A-Video settings
enhance_a_video_enabled            = False
enhance_a_video_weight             = 1.0    # Should be smaller than 10. For smaller video length and lower resolution, should be smaller than 5.
enhance_a_video_skip_start_steps   = 0      # First n steps do not use Enhance-A-Video, should be >= 0
enhance_a_video_skip_end_steps     = 0      # Last n steps do not use Enhance-A-Video, should be >= 0


def get_control_embeddings(pipeline, aspect_ratio, motion, camera_direction):
    # Default keys
    p_default_key = "p.default"
    n_default_key = "n.default"

    # Load embeddings
    if motion == "auto":
        motion = "0"
    p_key = f"p.{aspect_ratio.replace(':', 'x')}movie{motion}{camera_direction}"
    embeddings = pipeline.embeddings

    # Get embeddings
    positive_embeds = embeddings.get(f"{p_key}.emb1", embeddings[f"{p_default_key}.emb1"])
    positive_attention_mask = embeddings.get(f"{p_key}.mask1", embeddings[f"{p_default_key}.mask1"])
    positive_embeds_2 = embeddings.get(f"{p_key}.emb2", embeddings[f"{p_default_key}.emb2"])
    positive_attention_mask_2 = embeddings.get(f"{p_key}.mask2", embeddings[f"{p_default_key}.mask2"])

    negative_embeds = embeddings[f"{n_default_key}.emb1"]
    negative_attention_mask = embeddings[f"{n_default_key}.mask1"]
    negative_embeds_2 = embeddings[f"{n_default_key}.emb2"]
    negative_attention_mask_2 = embeddings[f"{n_default_key}.mask2"]

    return {
        "positive_embeds": positive_embeds,
        "positive_attention_mask": positive_attention_mask,
        "positive_embeds_2": positive_embeds_2,
        "positive_attention_mask_2": positive_attention_mask_2,

        "negative_embeds": negative_embeds,
        "negative_attention_mask": negative_attention_mask,
        "negative_embeds_2": negative_embeds_2,
        "negative_attention_mask_2": negative_attention_mask_2,
    }


def try_setup_pipeline(model_path, weight_dtype, config, fp8_quant_mode, fp8_data_type):
    try:
        # Get Vae
        vae = AutoencoderKLMagvit.from_pretrained(
            model_path, 
            subfolder="vae"
        ).to(weight_dtype)
        print("Vae loaded ...")

        # Get Transformer
        transformer_additional_kwargs = OmegaConf.to_container(config['transformer_additional_kwargs'])
        transformer = HunyuanTransformer3DModel.from_pretrained_2d(
            model_path, 
            subfolder="transformer",
            transformer_additional_kwargs=transformer_additional_kwargs
        ).to(weight_dtype)
        print("Transformer loaded ...")

        # Transformer to fp8
        if fp8_quant_mode != 'none':
            count_f8 =0
            fp8_type = torch.float8_e5m2 if fp8_data_type == 'fp8_e5m2' else torch.float8_e4m3fn
            if fp8_quant_mode != 'extreme':

                shape_size = 2816 ** 2
                if fp8_quant_mode == 'strong': shape_size -= 1

                for module in transformer.modules():
                    if module.__class__.__name__ in ["Linear"]:
                        x,y = module.weight.shape
                        if x * y > shape_size:
                            module.to(fp8_type)
                            count_f8 += 1
            else:
                for module in transformer.modules():
                    if len(list(module.modules())) == 1 and list(module.named_parameters()):
                        if module.__class__.__name__ not in ["Embedding", 'LayerNorm', 'Conv2d', 'NonDynamicallyQuantizableLinear']:
                            module.to(fp8_type)
                            count_f8 += 1

            print (f'FP8: {count_f8} layers converted to {fp8_type}')

        # Load Clip
        clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            model_path, subfolder="image_encoder"
        ).to(weight_dtype)
        clip_image_processor = CLIPImageProcessor.from_pretrained(
            model_path, subfolder="image_encoder"
        )

        # Load sampler and create pipeline
        Choosen_Scheduler = DDIMScheduler
        scheduler = Choosen_Scheduler.from_pretrained(
            model_path, 
            subfolder="scheduler"
        )
        pipeline = RuyiInpaintPipeline.from_pretrained(
            model_path,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=weight_dtype,
            clip_image_encoder=clip_image_encoder,
            clip_image_processor=clip_image_processor,
        )

        # Load embeddings
        embeddings = load_safetensors(os.path.join(model_path, "embeddings.safetensors"))
        pipeline.embeddings = embeddings
        print("Pipeline loaded ...")

        return pipeline
    except Exception as e:
        print("[Ruyi] Setup pipeline failed:", e)
        return None


# Load config
config = OmegaConf.load(config_path)

# Load images
start_img = [Image.open(start_image_path).convert("RGB")]
end_img   = [Image.open(end_image_path).convert("RGB")] if end_image_path is not None else None

# Check for update
repo_id = f"IamCreateAI/{model_name}"
if auto_download and auto_update:
    print(f"Checking for {model_name} updates ...")

    # Download the model
    snapshot_download(repo_id=repo_id, local_dir=model_path)

# Init model
pipeline = try_setup_pipeline(model_path, weight_dtype, config, fp8_quant_mode, fp8_data_type)
if pipeline is None and auto_download:
    print(f"Downloading {model_name} ...")

    # Download the model
    snapshot_download(repo_id=repo_id, local_dir=model_path)

    pipeline = try_setup_pipeline(model_path, weight_dtype, config, fp8_quant_mode, fp8_data_type)

if pipeline is None:
    message = (f"[Load Model Failed] "
               f"Please download Ruyi model from huggingface repo '{repo_id}', "
               f"And put it into '{model_path}'.")
    if not auto_download:
        message += "\nOr just set auto_download to 'True'."
    raise FileNotFoundError(message)

# Setup GPU memory mode
if low_gpu_memory_mode:
    pipeline.enable_sequential_cpu_offload()
else:
    pipeline.enable_model_cpu_offload()

# Prepare LoRA config
loras = {
    'models': [lora_path] if lora_path is not None else [],
    'weights': [lora_weight] if lora_path is not None else [],
}

# Count most suitable height and width
if video_size is None:
    aspect_ratio_sample_size = {key : [x / 512 * base_resolution for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
    original_width, original_height = start_img[0].size if type(start_img) is list else Image.open(start_img).size
    closest_size, closest_ratio = get_closest_ratio(original_height, original_width, ratios=aspect_ratio_sample_size)
    height, width = [int(x / 16) * 16 for x in closest_size]
else:
    height, width = video_size

# Set hidden states offload steps
pipeline.transformer.hidden_cache_size = gpu_offload_steps

# Load Sampler
if scheduler_name == "DPM++":
    noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder='scheduler')
elif scheduler_name == "Euler":
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder='scheduler')
elif scheduler_name == "Euler A":
    noise_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_path, subfolder='scheduler')
elif scheduler_name == "PNDM":
    noise_scheduler = PNDMScheduler.from_pretrained(model_path, subfolder='scheduler')
elif scheduler_name == "DDIM":
    noise_scheduler = DDIMScheduler.from_pretrained(model_path, subfolder='scheduler')
pipeline.scheduler = noise_scheduler

# Set random seed
generator= torch.Generator(device).manual_seed(seed)

# Load control embeddings
embeddings = get_control_embeddings(pipeline, aspect_ratio, motion, camera_direction)

# Initialize TeaCache
pipeline.transformer.tea_cache.initialize(tea_cache_enabled, tea_cache_threshold, tea_cache_skip_start_steps, tea_cache_skip_end_steps, steps, tea_cache_offload_cpu)

# Initialize Enhance-A-Video
pipeline.transformer.enhance_a_video.initialize(enhance_a_video_enabled, enhance_a_video_weight, enhance_a_video_skip_start_steps, enhance_a_video_skip_end_steps, steps)

# Generate video
with torch.no_grad(), torch.autocast(str(device), dtype = pipeline.transformer.dtype):
    video_length = int(video_length // pipeline.vae.mini_batch_encoder * pipeline.vae.mini_batch_encoder) if video_length != 1 else 1
    input_video, input_video_mask, clip_image = get_image_to_video_latent(start_img, end_img, video_length=video_length, sample_size=(height, width))

    for _lora_path, _lora_weight in zip(loras.get("models", []), loras.get("weights", [])):
        pipeline = merge_lora(pipeline, _lora_path, _lora_weight)
    
    sample = pipeline(
        prompt_embeds = embeddings["positive_embeds"],
        prompt_attention_mask = embeddings["positive_attention_mask"],
        prompt_embeds_2 = embeddings["positive_embeds_2"],
        prompt_attention_mask_2 = embeddings["positive_attention_mask_2"],

        negative_prompt_embeds = embeddings["negative_embeds"],
        negative_prompt_attention_mask = embeddings["negative_attention_mask"],
        negative_prompt_embeds_2 = embeddings["negative_embeds_2"],
        negative_prompt_attention_mask_2 = embeddings["negative_attention_mask_2"],

        video_length = video_length,
        height      = height,
        width       = width,
        generator   = generator,
        guidance_scale = cfg,
        num_inference_steps = steps,

        video        = input_video,
        mask_video   = input_video_mask,
        clip_image   = clip_image, 
    ).videos

    for _lora_path, _lora_weight in zip(loras.get("models", []), loras.get("weights", [])):
        pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight)

# Log information
if tea_cache_enabled:
    print("TeaCache cached steps:", pipeline.transformer.tea_cache.skip_count)

# Save the video
output_folder = os.path.dirname(output_video_path)
if output_folder != '':
    os.makedirs(output_folder, exist_ok=True)
save_videos_grid(sample, output_video_path, fps=24)
