from functools import reduce
import torch
import math

from ..utils import CATEGORY_SAMPLING, DIRECTIONS

import comfy
from comfy import model_management
from comfy.utils import wait_cooldown
from nodes import common_ksampler, MAX_RESOLUTION, VAEEncode, VAEDecode, EmptyLatentImage
from comfy_extras.nodes_mask import composite
from comfy.sd import VAE
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from .facerestore import FACEDETECTION_MODELS, JN_FaceRestoreWithModel
from .image import JN_ImageCenterArea

def apply_seamless(tensor, direction, border_percent):
    (batch_size, channels, height, width) = tensor.shape

    if direction in ["both", "horizontal"]:
        gap = min(round(width * border_percent), width // 4)
        tensor[:, :, :, -gap:] = tensor[:, :, :, gap:(gap * 2)]
        tensor[:, :, :, :gap] = tensor[:, :, :, -(gap * 2):-gap]

    if direction in ["both", "vertical"]:
        gap = min(round(height * border_percent), height // 4)
        tensor[:, :, -gap:, :] = tensor[:, :, gap:(gap * 2), :]
        tensor[:, :, :gap, :] = tensor[:, :, -(gap * 2):-gap, :]

    return tensor

class JN_VAE:
    def __init__(self, vae, tile_size, fallback_method, *args, **kwargs):
        self.vae = vae
        self.tile_size = max(64, tile_size)
        self.fallback_method = fallback_method

    def __getattr__(self, attr):
        return getattr(self.vae, attr)

    def decode(self, samples_in):
        self.first_stage_model = self.first_stage_model.to(self.device)
        pixel_samples = None

        try:
            pixel_samples = self.decode_(samples_in, device=self.device)

        except model_management.OOM_EXCEPTION as e:
            pixel_samples = None

            if self.fallback_method == "tile":
                tile_size = 64
                while tile_size >= 8:
                    overlap = tile_size // 4
                    print(f"Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding with tile size {tile_size} and overlap {overlap}.")
                    try:
                        pixel_samples = self.decode_tiled_(samples_in, tile_x=tile_size, tile_y=tile_size, overlap=overlap)
                        break
                    except model_management.OOM_EXCEPTION as e:
                        pass
                    tile_size -= 8

            if self.fallback_method == "cpu":
                print(f"Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding with CPU.")
                self.first_stage_model = self.first_stage_model.to(torch.device("cpu"))
                try:
                    pixel_samples = self.decode_(samples_in, device=torch.device("cpu"))
                except model_management.OOM_EXCEPTION as e:
                    pass

            if pixel_samples is None:
                raise e

        finally:
            self.first_stage_model = self.first_stage_model.to(self.offload_device)

        pixel_samples = pixel_samples.cpu().movedim(1,-1)

        return pixel_samples

    def decode_(self, samples_in, device):
        memory_used = 2562 * samples_in.shape[2] * samples_in.shape[3] * 64 * 1.7
        model_management.free_memory(memory_used, device)
        free_memory = model_management.get_free_memory(device)
        batch_number = int(free_memory / memory_used)
        batch_number = max(1, batch_number)

        pixel_samples = torch.empty((samples_in.shape[0], 3, round(samples_in.shape[2] * 8), round(samples_in.shape[3] * 8)), "cpu", **("device",))

        for x in range(0, samples_in.shape[0], batch_number):
            samples = samples_in[x:x + batch_number].to(self.vae_dtype).to(device)
            pixel_samples[x:x + batch_number] = torch.clamp((self.first_stage_model.decode(samples).cpu().float() + 1) / 2, 0, 1, **("min", "max"))

        return pixel_samples

    def encode(self, pixel_samples):
        self.first_stage_model = self.first_stage_model.to(self.device)
        pixel_samples = pixel_samples.movedim(-1,1)
        samples = None

        try:
            samples = self.encode_(pixel_samples, device=self.device)

        except model_management.OOM_EXCEPTION as e:
            samples = None

            if self.fallback_method == "tile":
                tile_size = 512
                while tile_size >= 64:
                    overlap = tile_size // 8
                    print(f"Warning: Ran out of memory when regular VAE encoding, retrying with tiled VAE encoding with tile size {tile_size} and overlap {overlap}.")
                    try:
                        samples = self.encode_tiled_(pixel_samples, tile_x=tile_size, tile_y=tile_size, overlap=overlap)
                        break
                    except model_management.OOM_EXCEPTION as e:
                        pass
                    tile_size -= 64

            if self.fallback_method == "cpu":
                print(f"Warning: Ran out of memory when regular VAE encoding, retrying with tiled VAE encoding with CPU.")
                self.first_stage_model = self.first_stage_model.to(torch.device("cpu"))
                try:
                    samples = self.encode_(pixel_samples, device=torch.device("cpu"))
                except model_management.OOM_EXCEPTION as e:
                    pass

            if samples is None:
                raise e

        finally:
            self.first_stage_model = self.first_stage_model.to(self.offload_device)

        return samples

    def encode_(self, pixel_samples, device):
        memory_used = 2078 * pixel_samples.shape[2] * pixel_samples.shape[3] * 1.7
        model_management.free_memory(memory_used, device)
        free_memory = model_management.get_free_memory(device)
        batch_number = int(free_memory / memory_used)
        batch_number = max(1, batch_number)

        samples = torch.empty((pixel_samples.shape[0], 4, round(pixel_samples.shape[2] // 8), round(pixel_samples.shape[3] // 8)), "cpu", **("device",))

        for x in range(0, pixel_samples.shape[0], batch_number):
            pixels_in = (2 * pixel_samples[x:x + batch_number] - 1).to(self.vae_dtype).to(device)
            samples[x:x + batch_number] = self.first_stage_model.encode(pixels_in).cpu().float()

        return samples

class JN_VAEPatch:
    CATEGORY = CATEGORY_SAMPLING
    RETURN_TYPES = ("VAE",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "fallback_method": (["tile", "cpu"],),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
            },
        }

    def run(self, vae, fallback_method, tile_size, **kwargs):
        vae_wrapper = JN_VAE(vae=vae, tile_size=tile_size, fallback_method=fallback_method)
        return (vae_wrapper,)

class JN_KSamplerAdvancedParams:
    CATEGORY = CATEGORY_SAMPLING
    RETURN_TYPES = ("PARAMS",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "add_noise": ("BOOLEAN", {"default": True}),
                "return_with_leftover_noise": ("BOOLEAN", {"default": False}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
            },
        }

    def run(self, add_noise=True, return_with_leftover_noise=False, start_at_step=0, end_at_step=10000, **kwargs):
        params = {
            "__type__": "JN_KSamplerAdvancedParams",
            "add_noise": add_noise,
            "return_with_leftover_noise": return_with_leftover_noise,
            "start_at_step": start_at_step,
            "end_at_step": end_at_step,
        }
        return (params,)

class JN_KSamplerSeamlessParams:
    CATEGORY = CATEGORY_SAMPLING
    RETURN_TYPES = ("PARAMS",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "direction": (DIRECTIONS,),
                "border_percent": ("FLOAT", {"default": 0.125, "min": 0, "max": 0.25, "step": 0.001}),
                "start_percent": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.001}),
            },
        }

    def run(self, direction="both", border_percent=0.125, start_percent=0, end_percent=1, **kwargs):
        params = {
            "__type__": "JN_KSamplerSeamlessParams",
            "direction": direction,
            "border_percent": border_percent,
            "start_percent": start_percent,
            "end_percent": end_percent,
        }
        return (params,)

class JN_KSamplerTileParams:
    CATEGORY = CATEGORY_SAMPLING
    RETURN_TYPES = ("PARAMS",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "overlay_percent": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.001}),
                "steps_chunk": ("INT", {"default": 0, "min": 0, "max": 10000}),
            },
        }

    def run(self, width=0, height=0, overlay_percent=0.5, steps_chunk=0, **kwargs):
        params = {
            "__type__": "JN_KSamplerTileParams",
            "width": width,
            "height": height,
            "overlay_percent": overlay_percent,
            "steps_chunk": steps_chunk,
        }
        return (params,)

class JN_KSamplerResizeInputParams:
    CATEGORY = CATEGORY_SAMPLING
    RETURN_TYPES = ("PARAMS",)
    FUNCTION = "run"
    RESIZE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos", "bislerp"]
    CROP_METHODS = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "method": (s.RESIZE_METHODS,),
                "crop": (s.CROP_METHODS,),
                "width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "scale_by": ("FLOAT", {"default": 0, "min": 0, "max": 8, "step": 0.01}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
            },
        }

    def run(self, method="nearest-exact", crop="disabled", width=0, height=0, scale_by=0, upscale_model=None, **kwargs):
        params = {
            "__type__": "JN_KSamplerResizeInputParams",
            "method": method,
            "crop": crop,
            "width": width,
            "height": height,
            "scale_by": scale_by,
            "upscale_model": upscale_model,
        }
        return (params,)

class JN_KSamplerResizeOutputParams:
    CATEGORY = CATEGORY_SAMPLING
    RETURN_TYPES = ("PARAMS",)
    FUNCTION = "run"
    RESIZE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos", "bislerp"]
    CROP_METHODS = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "method": (s.RESIZE_METHODS,),
                "crop": (s.CROP_METHODS,),
                "width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "scale_by": ("FLOAT", {"default": 0, "min": 0, "max": 8, "step": 0.01}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
            },
    }

    def run(self, method="nearest-exact", crop="disabled", width=0, height=0, scale_by=0, upscale_model=None, **kwargs):
        params = {
            "__type__": "JN_KSamplerResizeOutputParams",
            "method": method,
            "crop": crop,
            "width": width,
            "height": height,
            "scale_by": scale_by,
            "upscale_model": upscale_model,
        }
        return (params,)

class JN_KSamplerFaceRestoreParams:
    CATEGORY = CATEGORY_SAMPLING
    RETURN_TYPES = ("PARAMS",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "facerestore_model": ("FACERESTORE_MODEL",),
                "facedetection": (FACEDETECTION_MODELS,),
                "center_direction": (DIRECTIONS,),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05}),
            },
    }

    def run(self, facerestore_model=None, facedetection="retinaface_resnet50", center_direction="none", strength=0.5, **kwargs):
        params = {
            "__type__": "JN_KSamplerFaceRestoreParams",
            "facerestore_model": facerestore_model,
            "facedetection": facedetection,
            "center_direction": center_direction,
            "strength": strength,
        }
        return (params,)

class JN_KSampler:
    CATEGORY = CATEGORY_SAMPLING
    RETURN_TYPES = ("LATENT", "IMAGE", "IMAGE", "LATENT", "IMAGE")
    RETURN_NAMES = ("LATENT", "IMAGE", "FINAL_IMAGE", "INPUT_LATENT", "INPUT_IMAGE")
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8, "min": 0, "max": 100, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "denoise": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                "decode_image": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "latent_image": ("LATENT",),
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "params": ("*", {"multiple": True}),
            },
        }

    def sample(self, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise=1,
            decode_image=True,
            resize_input=False,
            latent_image=None, image=None,
            width=512, height=512, batch_size=1,
            params=[],
            **kwargs):
        params = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), params, [None])
        params = [param for param in params if param is not None]
        options = {param["__type__"]: param for param in params if "__type__" in param}

        options_order = []
        options_order_ = [param["__type__"] for param in params if "__type__" in param]
        for option_order in options_order_:
            if option_order not in options_order:
                options_order.append(option_order)

        advanced_params = options["JN_KSamplerAdvancedParams"] if "JN_KSamplerAdvancedParams" in options else None
        seamless_params = options["JN_KSamplerSeamlessParams"] if "JN_KSamplerSeamlessParams" in options else None
        tile_params = options["JN_KSamplerTileParams"] if "JN_KSamplerTileParams" in options else None
        resize_input_params = options["JN_KSamplerResizeInputParams"] if "JN_KSamplerResizeInputParams" in options else None
        resize_output_params = options["JN_KSamplerResizeOutputParams"] if "JN_KSamplerResizeOutputParams" in options else None
        face_restore_params = options["JN_KSamplerFaceRestoreParams"] if "JN_KSamplerFaceRestoreParams" in options else None

        if tile_params and tile_params["width"] == 0 and tile_params["height"] == 0:
            tile_params = None

        if seamless_params and seamless_params["direction"] == "none":
            seamless_params = None

        if resize_input_params and resize_input_params["width"] == 0 and resize_input_params["height"] == 0 and resize_input_params["scale_by"] == 0:
            resize_input_params["width"] = width
            resize_input_params["height"] = height

        if image is not None and resize_input_params is not None:
            image = self.resize(image.clone().movedim(-1, 1), **resize_input_params).movedim(1, -1)

        if latent_image is None:
            if image is not None:
                latent_image = VAEEncode().encode(vae=vae, pixels=image.clone())[0]
                resize_input_params = None
            else:
                latent_image = EmptyLatentImage().generate(width=width, height=height, batch_size=batch_size)[0]

        if resize_input_params is not None:
            latent_image["samples"] = self.resize(latent_image["samples"], samples_type="latent", **resize_input_params)

        if advanced_params is not None:
            disable_noise = not advanced_params["add_noise"]
            force_full_denoise = not advanced_params["return_with_leftover_noise"]
            start_step = advanced_params["start_at_step"]
            last_step = advanced_params["end_at_step"]
        else:
            disable_noise = False
            force_full_denoise = False
            start_step = None
            last_step = None

        common_ksampler_params = {
            "model": model,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "positive": positive,
            "negative": negative,
            "latent": latent_image,
            "denoise": denoise,
            "disable_noise": disable_noise,
            "start_step": start_step,
            "last_step": last_step,
            "force_full_denoise": force_full_denoise,
        }

        wait_cooldown(kind="execution")

        if tile_params is not None:
            output_latent = self.tiled_ksampler(**tile_params, common_ksampler_params=common_ksampler_params, seamless_params=seamless_params)
        else:
            if seamless_params is not None:
                common_ksampler_params["model"] = self.seamless_patch(model, **seamless_params)

            output_latent = common_ksampler(**common_ksampler_params)[0]

        output_image = None
        final_image = None

        if decode_image:
            wait_cooldown(kind="execution")
            final_image = output_image = VAEDecode().decode(vae=vae, samples=output_latent)[0]

            for option_order in options_order:
                if option_order == "JN_KSamplerResizeOutputParams" and resize_output_params is not None:
                    wait_cooldown(kind="execution")
                    final_image = self.resize(final_image.clone().movedim(-1, 1), **resize_output_params).movedim(1, -1)

                if option_order == "JN_KSamplerSeamlessParams" and seamless_params is not None:
                    wait_cooldown(kind="execution")
                    final_image = self.seamless_crop(final_image, **seamless_params)

                if option_order == "JN_KSamplerFaceRestoreParams" and face_restore_params is not None:
                    wait_cooldown(kind="execution")
                    final_image = self.facerestore(final_image, **face_restore_params)

        if resize_output_params is not None:
            output_latent["samples"] = self.resize(output_latent["samples"], samples_type="latent", **resize_output_params)

        return (output_latent, output_image, final_image, latent_image, image)

    def tiled_ksampler(self, width, height, overlay_percent, steps_chunk=0, common_ksampler_params=None, seamless_params=None, **kwargs):
        latent_image = common_ksampler_params["latent"]
        latent_width = latent_image["samples"].shape[3]
        latent_height = latent_image["samples"].shape[2]

        width = width // 8
        height = height // 8

        if width == 0:
            width = max(4, round(height * latent_width / latent_height))
        if height == 0:
            height = max(4, round(width * latent_height / latent_width))

        columns = math.ceil(latent_width / width)
        rows = math.ceil(latent_height / height)

        width = latent_width // columns
        height = latent_height // rows

        overlay_width = max(2, round(width * overlay_percent * 2))
        overlay_height = max(2, round(height * overlay_percent * 2))

        half_overlay_width = overlay_width // 2
        half_overlay_height = overlay_height // 2

        mask_overlay_width = overlay_width // 2
        mask_overlay_height = overlay_height // 2

        total_tiles = columns * rows

        tiles = [None for i in range(total_tiles)]

        y = 0

        for r in range(0, rows):
            x = 0

            for c in range(0, columns):
                i = c + (r * columns)

                tw = width
                th = height

                tx = x
                ty = y

                # left overlay
                if c > 0:
                    tx -= half_overlay_width
                    tw += half_overlay_width

                # top overlay
                if r > 0:
                    ty -= half_overlay_height
                    th += half_overlay_height

                # right overlay
                if c + 1 < columns:
                    tw += half_overlay_width

                # bottom overlay
                if r + 1 < rows:
                    th += half_overlay_height

                tw = min(tw, latent_width - tx)
                th = min(th, latent_height - ty)

                tiles[i] = {
                    "column": c,
                    "row": r,
                    "x": tx,
                    "y": ty,
                    "width": tw,
                    "height": th,
                }

                x += width

            y += height

        output_latent = latent_image.copy()
        output_latent["samples"] = latent_image["samples"].clone()

        steps = common_ksampler_params["steps"]

        if steps_chunk <= 0:
            steps_chunk = steps

        if seamless_params is not None:
            seamless_start_step = steps * seamless_params["start_percent"]
            seamless_end_step = steps * seamless_params["end_percent"]
        else:
            seamless_start_step = 0
            seamless_end_step = steps

        for step in range(0, steps, steps_chunk):
            wait_cooldown(kind="progress")

            for tile in tiles:
                c = tile["column"]
                r = tile["row"]
                x = tile["x"]
                y = tile["y"]
                w = tile["width"]
                h = tile["height"]

                print(f"Step: {step}-{step + steps_chunk}/{steps} Tile: {c+1}x{r+1} => {x*8}-{(x+w)*8}:{y*8}-{(y+h)*8} => {w*8}x{h*8}")

                if seamless_params is not None and step + 1 >= seamless_start_step and step + steps_chunk <= seamless_end_step:
                    output_latent["samples"] = apply_seamless(output_latent["samples"], seamless_params["direction"], seamless_params["border_percent"])

                latent = output_latent.copy()
                latent["samples"] = output_latent["samples"][:, :, y:y+h, x:x+w].clone()

                mask = torch.ones((latent["samples"].shape[0], latent["samples"].shape[2], latent["samples"].shape[3]))

                # left overlay
                if c > 0:
                    for i in range(mask_overlay_width):
                        feather_rate = (i + 1.0) / mask_overlay_width
                        mask[:, :, i] *= feather_rate

                # top overlay
                if r > 0:
                    for i in range(mask_overlay_height):
                        feather_rate = (i + 1.0) / mask_overlay_height
                        mask[:, i, :] *= feather_rate

                # # right overlay
                # if c + 1 < columns:
                #     for i in range(mask_overlay_width):
                #         feather_rate = (i + 1.0) / mask_overlay_width
                #         mask[:, :, -i] *= feather_rate
                # 
                # # bottom overlay
                # if r + 1 < rows:
                #     for i in range(mask_overlay_height):
                #         feather_rate = (i + 1.0) / mask_overlay_height
                #         mask[:, -i, :] *= feather_rate

                latent["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))

                tiled_ksampler_params = {
                    "latent": latent,
                    "start_step": step,
                    "last_step": step + steps_chunk,
                    "disable_noise": True if step > 0 else False,
                    "force_full_denoise": True if (step + steps_chunk) >= steps else False,
                }

                output_latent_tile = common_ksampler(**{**common_ksampler_params, **tiled_ksampler_params})[0]

                output_latent["samples"] = composite(destination=output_latent["samples"], source=output_latent_tile["samples"], x=x, y=y, mask=mask, multiplier=1, resize_source=False)

                if seamless_params is not None and step + 1 >= seamless_start_step and step + steps_chunk <= seamless_end_step:
                    output_latent["samples"] = apply_seamless(output_latent["samples"], seamless_params["direction"], seamless_params["border_percent"])

        return output_latent

    def resize(self, samples, width, height, method, crop, upscale_model=None, scale_by=0, samples_type="image", **kwargs):
        if samples_type == "latent":
            width = width // 8
            height = height // 8

        if scale_by > 0:
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)
        elif width == 0:
            width = max(1, round(samples.shape[3] * height / samples.shape[2]))
        elif height == 0:
            height = max(1, round(samples.shape[2] * width / samples.shape[3]))

        if width == samples.shape[3] and height == samples.shape[2]:
            return samples

        if upscale_model is not None and samples_type == "image":
            while width > samples.shape[3] or height > samples.shape[2]:
                samples = ImageUpscaleWithModel().upscale(upscale_model, samples.clone().movedim(1, -1))[0].movedim(-1, 1)

            if width == samples.shape[3] and height == samples.shape[2]:
                return samples

        return comfy.utils.common_upscale(samples, width, height, method, crop)

    def facerestore(self, image, facerestore_model, facedetection, center_direction, strength, **kwargs):
        outputs = JN_FaceRestoreWithModel().restore_face(image=image, facerestore_model=facerestore_model, facedetection=facedetection, strength=strength)

        restored_image = outputs[0]

        if center_direction != "none":
            faces_areas = outputs[3]
            restored_image = JN_ImageCenterArea().run(image=restored_image, direction=center_direction, areas=faces_areas)[0]

        return restored_image

    def seamless_crop(self, image, direction, border_percent, **kwargs):
        def crop(tensor, direction, border_percent):
            (batch_size, channels, height, width) = tensor.shape

            if direction in ["both", "horizontal"]:
                gap = min(round(width * border_percent), width // 4)
                tensor = tensor[:, :, :, gap:-gap]

            if direction in ["both", "vertical"]:
                gap = min(round(height * border_percent), height // 4)
                tensor = tensor[:, :, gap:-gap, :]

            return tensor

        return crop(image.clone().movedim(-1, 1), direction, border_percent).movedim(1, -1)

    def seamless_patch(self, model, direction, border_percent, start_percent, end_percent, **kwargs):
        sigma_start = model.model.model_sampling.percent_to_sigma(start_percent)
        sigma_end = model.model.model_sampling.percent_to_sigma(end_percent)
        model_options = model.model_options

        def unet_wrapper_function(apply_model, options):
            input_x = options["input"]
            timestep_ = options["timestep"]
            c = options["c"]

            sigma = timestep_[0].item()

            if sigma <= sigma_start and sigma >= sigma_end:
                input_x = apply_seamless(input_x, direction, border_percent)

            if "model_function_wrapper" in model_options:
                output = model_options["model_function_wrapper"](apply_model, options)
            else:
                output = apply_model(input_x, timestep_, **c)

            if sigma <= sigma_start and sigma >= sigma_end:
                output = apply_seamless(output, direction, border_percent)

            return output

        m = model.clone()
        m.set_model_unet_function_wrapper(unet_wrapper_function)
        return m

    @staticmethod
    def vae_encode_crop_pixels(pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels

NODE_CLASS_MAPPINGS = {
    "JN_KSampler": JN_KSampler,
    "JN_KSamplerAdvancedParams": JN_KSamplerAdvancedParams,
    "JN_KSamplerResizeInputParams": JN_KSamplerResizeInputParams,
    "JN_KSamplerResizeOutputParams": JN_KSamplerResizeOutputParams,
    "JN_KSamplerSeamlessParams": JN_KSamplerSeamlessParams,
    "JN_KSamplerTileParams": JN_KSamplerTileParams,
    "JN_KSamplerFaceRestoreParams": JN_KSamplerFaceRestoreParams,
    "JN_VAEPatch": JN_VAEPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JN_KSampler": "KSampler",
    "JN_KSamplerAdvancedParams": "KSampler Advanced Params",
    "JN_KSamplerResizeInputParams": "KSampler Resize Input Params",
    "JN_KSamplerResizeOutputParams": "KSampler Resize Output Params",
    "JN_KSamplerSeamlessParams": "KSampler Seamless Params",
    "JN_KSamplerTileParams": "KSampler Tile Params",
    "JN_KSamplerFaceRestoreParams": "KSampler Face Restore Params",
    "JN_VAEPatch": "VAE Patch",
}
