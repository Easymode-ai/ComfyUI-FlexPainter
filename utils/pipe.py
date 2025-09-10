import torch
from einops import rearrange
from PIL import Image
from typing import Optional, Any, Tuple
from comfy.samplers import KSampler  # adjust import path if needed
from diffusers import StableDiffusionControlNetPipeline
from comfy.sampler_helpers import convert_cond

def aggregate_embed(prompt_embeds_i, prompt_embeds_t, 
                    redux_strength, add_prompt_embeds_i = None):
    
    if prompt_embeds_i is None:
        return prompt_embeds_t
    elif add_prompt_embeds_i is not None:
        prompt_embeds_i_style = prompt_embeds_i - add_prompt_embeds_i
        prompt_embeds = torch.cat([prompt_embeds_t, prompt_embeds_i_style * redux_strength], dim=1)
    else:
        prompt_embeds = torch.cat([prompt_embeds_t, prompt_embeds_i * redux_strength], dim=1)

    return prompt_embeds
def _get_redux_embeddings_from_callable(redux_callable, image_or_pil):
    """Call a HF-style redux pipeline and normalize the result shape expected by the rest of the code."""
    redux_output = redux_callable(image_or_pil)
    # preserve compatibility: original code used redux_output["prompt_embeds"]
    if isinstance(redux_output, dict) and "prompt_embeds" in redux_output:
        return redux_output
    # Accept other return formats conservatively
    return {"prompt_embeds": redux_output}

def _try_extract_embedder_from_model_tuple(model_tuple) -> Optional[Any]:
    """
    Try to find a clip/text embedder inside a ComfyUI model tuple.
    This uses heuristics: look for an object with encode / get_input_embeddings / tokenize / __call__.
    Returns the first candidate callable or None.
    """
    # If model_tuple is already a single object rather than a tuple/list, allow that too
    if not isinstance(model_tuple, (tuple, list)):
        model_tuple = (model_tuple,)

    for element in model_tuple:
        if element is None:
            continue
        # If element itself is callable and looks like an embedder (heuristic)
        if callable(element):
            # skip calling here — we only return the callable for later use
            return element
        # If element has obvious embedder methods
        for attr in ("encode", "encode_text", "get_input_embeddings", "_encode", "tokenize", "__call__"):
            if hasattr(element, attr):
                # return the element (callable or object we can call later)
                return element
    return None

def _call_embedder(embedder, image_or_pil):
    """
    Attempt to call an embedder-like object with an image (PIL) to produce redux-style output.
    We try several calling conventions (dict output with 'prompt_embeds' being ideal).
    """
    # 1) if embedder is a HF pipeline-like callable that expects PIL -> call directly
    try:
        out = embedder(image_or_pil)
        if isinstance(out, dict) and "prompt_embeds" in out:
            return out
        # Some embedder returns a tensor directly -> wrap it
        if torch.is_tensor(out) or isinstance(out, (list, tuple)):
            return {"prompt_embeds": out}
    except Exception:
        pass

    # 2) try common methods
    for method_name in ("encode", "encode_text", "get_input_embeddings", "tokenize", "_encode"):
        fn = getattr(embedder, method_name, None)
        if callable(fn):
            try:
                out = fn(image_or_pil)
                if isinstance(out, dict) and "prompt_embeds" in out:
                    return out
                if torch.is_tensor(out) or isinstance(out, (list, tuple)):
                    return {"prompt_embeds": out}
            except Exception:
                continue

    # 3) not callable in expected way
    return None

# Helper: find device from a diffusers/ComfyUI pipeline-like object
def _get_pipeline_device(pipe) -> torch.device:
    # many diffusers pipelines expose .device, otherwise check submodules
    dev = getattr(pipe, "device", None)
    if dev is not None:
        return dev
    # try unet / vae
    for attr in ("unet", "vae", "text_encoder", "encoder", "clip"):
        sub = getattr(pipe, attr, None)
        if sub is not None and hasattr(sub, "device"):
            return getattr(sub, "device")
    # fallback
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper: encode prompts using SDXL / diffusers style. Returns (prompt_embeds, pooled_prompt_embeds_or_none)
def _encode_prompts_sdxl_style(pipe, prompt: str, negative_prompt: Optional[str] = None, device: Optional[torch.device] = None) -> Tuple[Any, Optional[Any]]:
    """
    Encode both positive (prompt) and negative prompts for SDXL pipelines.
    Returns: (prompt_embeds, pooled_prompt_embeds_or_none)
    Both prompt_embeds and pooled_prompt_embeds are shaped for classifier-free guidance:
    [batch_size * 2, ...] with first half = positive, second half = negative
    """
    if device is None:
        device = next(pipe.parameters()).device if hasattr(pipe, "parameters") else "cuda"

    # --- 1) Try the built-in _encode_prompt or encode_prompt ---
    for fn_name in ("_encode_prompt", "encode_prompt"):
        fn = getattr(pipe, fn_name, None)
        if callable(fn):
            try:
                out = fn(prompt=prompt, device=device, negative_prompt=negative_prompt)
                # Standard HF SDXL output: tuple(prompt_embeds, pooled_prompt_embeds)
                if isinstance(out, tuple):
                    return out[0].to(device), out[1].to(device) if len(out) > 1 else None
                return out.to(device), None
            except TypeError:
                continue  # try next method

    # --- 2) Try prepare_prompts / prepare_text_inputs ---
    for fn_name in ("prepare_prompts", "prepare_text_inputs", "_prepare_prompts"):
        fn = getattr(pipe, fn_name, None)
        if callable(fn):
            try:
                out = fn(prompt=prompt, negative_prompt=negative_prompt)
                if isinstance(out, dict):
                    pe = out.get("prompt_embeds") or out.get("input_embeddings")
                    pooled = out.get("pooled_prompt_embeds")
                    return pe.to(device), pooled.to(device) if pooled is not None else None
                if isinstance(out, tuple):
                    return out[0].to(device), out[1].to(device) if len(out) > 1 else None
            except Exception:
                continue

    # --- 3) Fallback: use tokenizer + text encoder ---
    tok = getattr(pipe, "tokenizer", None) or getattr(pipe, "tokenizer_2", None)
    enc = getattr(pipe, "text_encoder", None) or getattr(pipe, "text_encoder_2", None)
    if tok is not None and enc is not None:
        # Positive prompt
        pos_tokens = tok(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        pos_out = enc(**pos_tokens)
        pos_embeds = getattr(pos_out, "last_hidden_state", pos_out)
        pos_pooled = getattr(pos_out, "pooler_output", None)

        # Negative prompt
        if negative_prompt is None:
            neg_embeds = pos_embeds.clone() * 0
            neg_pooled = pos_pooled.clone() * 0 if pos_pooled is not None else None
        else:
            neg_tokens = tok(negative_prompt, return_tensors="pt", padding=True, truncation=True).to(device)
            neg_out = enc(**neg_tokens)
            neg_embeds = getattr(neg_out, "last_hidden_state", neg_out)
            neg_pooled = getattr(neg_out, "pooler_output", None)

        # Concatenate for classifier-free guidance: [pos; neg]
        prompt_embeds = torch.cat([pos_embeds, neg_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([pos_pooled, neg_pooled], dim=0) if pos_pooled is not None else None
        return prompt_embeds, pooled_prompt_embeds

    # If nothing worked
    raise RuntimeError(
        "Could not encode prompt with the provided pipeline. "
        "Ensure the pipeline is SDXL/diffusers style and exposes _encode_prompt, encode_prompt, or tokenizer + text_encoder."
    )
# ---------- Updated mv_generation (SDXL-compatible) ----------
@torch.no_grad()
def mv_generation(pipe, redux_pipe, prompt, image_prompt, depths, sample_steps, cfg_scale, generator, 
                  redux_strength, true_cfg, renderer, use_style_control = False):
    
    ksampler = KSampler(pipe) 
    """
    Multi-view generation that uses SDXL-style prompt encoding.
    Note: image-based `redux_pipe(image_prompt)` embedding is skipped by default for SDXL compatibility.
    If you need image-conditioning embeddings, pass them explicitly as redux embeddings or implement
    an image-embedder and wire it in upstream.
    """
    device = _get_pipeline_device(pipe)

    grid_depths = rearrange(depths, "(rows cols) c h w -> c (rows h) (cols w)", rows=2, cols=2)
    grid_depths = grid_depths.unsqueeze(0)
    if grid_depths.shape[1] == 1:
        grid_depths = grid_depths.repeat(1, 3, 1, 1)

    # For SDXL path: do not call redux_pipe(image_prompt). Instead use text encoding from pipe.
    prompt_embeds_i = None  # image-based redux embeddings are not used in SDXL path by default

    # Encode main prompt and negative prompt via SDXL-compatible helper
    # If you want to pass a negative prompt string, you can call with that; for now we set None
    negative_prompt = None
    prompt_embeds_t, pooled_prompt_embeds_t = _encode_prompts_sdxl_style(pipe, prompt, negative_prompt, device=device)

    # style control: optional; if requested, we attempt to obtain prompt embeddings for image_prompt,
    # but only if user provided an image embedding function via redux_pipe callable.
    if use_style_control and image_prompt is not None:
        # Only attempt if redux_pipe is a callable that accepts PIL and returns embeddings
        if callable(redux_pipe):
            try:
                redux_output = redux_pipe(image_prompt)
                if isinstance(redux_output, dict) and "prompt_embeds" in redux_output:
                    add_prompt_embeds_i = redux_output["prompt_embeds"]
                    add_pooled_prompt_embeds_i = redux_output.get("pooled_prompt_embeds", None)
                else:
                    add_prompt_embeds_i = None
                    add_pooled_prompt_embeds_i = None
            except Exception:
                add_prompt_embeds_i = None
                add_pooled_prompt_embeds_i = None
        else:
            add_prompt_embeds_i = None
            add_pooled_prompt_embeds_i = None
    else:
        add_prompt_embeds_i = None
        add_pooled_prompt_embeds_i = None
        # original code set true_cfg = 1.0 when not using style control
        true_cfg = 1.0

    # aggregate embeddings (keep your original aggregator)
    prompt_embeds = aggregate_embed(prompt_embeds_i, prompt_embeds_t, redux_strength, add_prompt_embeds_i)
    pooled_prompt_embeds = pooled_prompt_embeds_t

    images = pipe(
        true_cfg=true_cfg,
        prompt=None,
        control_image=grid_depths,
        height=grid_depths.shape[3],
        width=grid_depths.shape[2],
        num_inference_steps=sample_steps,
        guidance_scale=cfg_scale,
        generator=generator,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt_embeds=add_prompt_embeds_i,
        negative_pooled_prompt_embeds=add_pooled_prompt_embeds_i,
        output_type="pt",
    )

    # many diffusers pipelines return an object with .images
    if hasattr(images, "images"):
        images_t = images.images[0]
    else:
        images_t = images

    images_t = rearrange(images_t, 'c (rows h) (cols w) -> (rows cols) c h w', rows=2, cols=2)
    return images_t
# Move all sub-modules of the pipeline to device
def move_pipe_to_device(pipe, device):
    for attr_name in dir(pipe):
        attr = getattr(pipe, attr_name)
        if isinstance(attr, torch.nn.Module):
            attr.to(device)
        elif isinstance(attr, list) or isinstance(attr, tuple):
            for m in attr:
                if isinstance(m, torch.nn.Module):
                    m.to(device)
# ---------- Updated mv_sync_cfg_generation (SDXL-compatible) ----------
@torch.no_grad()
def mv_sync_cfg_generation(
    pipe,                   # ComfyUI model (Flux, SDXL, Krea, etc.)
    controlnet_model, 
    prompt: str,
    negative_prompt: Optional[str],
    image_prompt: Optional[Image.Image],
    depths: torch.Tensor,
    sample_steps: int,
    cfg_scale: float,
    generator: torch.Generator,
    true_cfg: float,
    tex_height: int,
    tex_width: int,
    mixing_step: int,
    renderer: Any,
    blank_txt: Optional[torch.Tensor] = None,
    blank_vec: Optional[torch.Tensor] = None,
    weighter: Optional[Any] = None,
    use_style_control: bool = False
):
    """
    Multi-view synchronous generation using KSampler.
    Automatically wraps the given model in a KSampler.
    Supports positive/negative prompts and style control.
    """

    device = next(pipe.parameters()).device if hasattr(pipe, "parameters") else "cuda"

    # --- 1) Prepare 2x2 multi-view grid of depths ---
    grid_depths = rearrange(depths, "(rows cols) c h w -> c (rows h) (cols w)", rows=2, cols=2)
    grid_depths = grid_depths.unsqueeze(0)
    if grid_depths.shape[1] == 1:
        grid_depths = grid_depths.repeat(1, 3, 1, 1)

    # --- 2) Optional style control via image prompt ---
    add_image_prompt = None
    if use_style_control and negative_prompt:
        add_image_prompt = Image.new("RGB", (grid_depths.shape[-1], grid_depths.shape[-2]), (255, 255, 255))

    # --- 3) Attach ControlNet and depth ---
    pipe.controlnet = controlnet_model
    pipe.control_image = grid_depths

    # --- 4) Prepare blank latent for KSampler ---
    batch, channels, height, width = grid_depths.shape
    blank_latent = torch.randn(batch, channels, height // 8, width // 8, device=device)

    # --- 5) Convert positive/negative prompts using ComfyUI format ---
    positive = convert_cond([[prompt, 1.0]])
    negative = convert_cond([[negative_prompt, 1.0]]) if negative_prompt else None

    # --- 6) Configure KSampler ---
    ksampler = KSampler(model=pipe, steps=sample_steps, device=device)

    # --- 7) Sample ---
    images_out = ksampler.sample(
        positive=positive,
        negative=negative,
        noise=1.0,
        cfg=cfg_scale,
        latent_image=blank_latent,
        denoise=1.0,
        sampler_name="k_euler_ancestral",
        scheduler="karras",
        steps=sample_steps,
        generator=generator,
        mixing_step=mixing_step,
    )

    # --- 8) Extract tensor from KSampler output ---
    if isinstance(images_out, dict) and "images" in images_out:
        images = images_out["images"][0]
    else:
        images = images_out

    # --- 9) Rearrange back to 2x2 multi-view ---
    images = rearrange(images, "c (rows h) (cols w) -> (rows cols) c h w", rows=2, cols=2)

    return images

def mv_sync_cfg_intermediate(pipe, redux_pipe, prompt, image_prompt, depths, timesteps, use_custom_timestep, 
                             cfg_scale, generator, redux_strength, true_cfg, blank_txt=None, blank_vec=None):
    grid_depths = rearrange(depths, "(rows cols) c h w -> c (rows h) (cols w)", rows=2, cols=2)
    grid_depths = grid_depths.unsqueeze(0)
    if grid_depths.shape[1] == 1:
        grid_depths = grid_depths.repeat(1, 3, 1, 1)

    if image_prompt is not None:
        redux_output = redux_pipe(image_prompt)
        prompt_embeds_i = redux_output["prompt_embeds"][:, 512: :]
    else:
        prompt_embeds_i = None
  
    prompt_embeds_t = pipe._get_t5_prompt_embeds([prompt])
    pooled_prompt_embeds_t = pipe._get_clip_prompt_embeds([prompt])

    if true_cfg > 1.0:
        add_prompt_embeds_i = blank_txt
        add_pooled_prompt_embeds_i = blank_vec
    else:
        add_prompt_embeds_i = None
        add_pooled_prompt_embeds_i = None
  
    prompt_embeds = aggregate_embed(prompt_embeds_i, prompt_embeds_t, 
                                    redux_strength, None)
    pooled_prompt_embeds = pooled_prompt_embeds_t

    images, ts = pipe.intermediate(
        true_cfg=true_cfg,
        prompt=None,
        control_image=grid_depths,
        height=grid_depths.shape[3],
        width=grid_depths.shape[2],
        timesteps=timesteps,
        guidance_scale=cfg_scale,
        generator=generator,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt_embeds=add_prompt_embeds_i,
        negative_pooled_prompt_embeds=add_pooled_prompt_embeds_i,
        use_custom_timestep=use_custom_timestep
    )

    images = rearrange(images, 'b c (rows h) (cols w) -> b (rows cols) c h w', rows=2, cols=2)

    return images, ts

