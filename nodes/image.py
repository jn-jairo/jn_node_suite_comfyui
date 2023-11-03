from functools import reduce
import os
import re
import glob
import torch
import numpy as np
from PIL import Image, ImageOps
import rembg
import hashlib
from transformers import BlipProcessor, BlipForConditionalGeneration

import folder_paths
import comfy
from comfy_extras.nodes_mask import composite

from ..utils import CATEGORY_IMAGE

def batch_image(image1, image2):
    if image1.shape[1:] != image2.shape[1:]:
        image2 = comfy.utils.common_upscale(image2.movedim(-1,1), image1.shape[2], image1.shape[1], "bilinear", "center").movedim(1,-1)
    s = torch.cat((image1, image2), dim=0)
    return s

def batch_mask(mask1, mask2):
    image1 = mask1.reshape((-1, 1, mask1.shape[-2], mask1.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
    image2 = mask2.reshape((-1, 1, mask2.shape[-2], mask2.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
    s = batch_image(image1, image2)[:, :, :, 0]
    s = s.reshape((s.shape[0], s.shape[1], s.shape[2]))
    return s

def normalize_area(area, width=0xffffffffffffffff, height=0xffffffffffffffff, pad=0):
    if not isinstance(area, dict):
        if isinstance(area, list):
            x1 = round(area[0])
            y1 = round(area[1])
            x2 = round(area[2])
            y2 = round(area[3])
        else:
            x1 = 0
            y1 = 0
            x2 = 0
            y2 = 0

        area = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }

    multiple_of = 8

    area = dict(area)
    area["x1"] = int(max(round((area["x1"] - pad) / multiple_of) * multiple_of, 0))
    area["y1"] = int(max(round((area["y1"] - pad) / multiple_of) * multiple_of, 0))
    area["x2"] = int(min(round((area["x2"] + pad) / multiple_of) * multiple_of, width))
    area["y2"] = int(min(round((area["y2"] + pad) / multiple_of) * multiple_of, height))

    return area

def get_crop_region(mask, pad=0):
    """finds a rectangular region that contains all masked areas in an image. Returns (x1, y1, x2, y2) coordinates of the rectangle.
    For example, if a user has painted the top-right part of a 512x512 image", the result may be (256, 0, 512, 256)"""

    _, h, w = mask.shape

    crop_left = 0
    for i in range(w):
        if not (mask[:, :, i] == 0).all():
            break
        crop_left += 1

    crop_right = 0
    for i in reversed(range(w)):
        if not (mask[:, :, i] == 0).all():
            break
        crop_right += 1

    crop_top = 0
    for i in range(h):
        if not (mask[:, i] == 0).all():
            break
        crop_top += 1

    crop_bottom = 0
    for i in reversed(range(h)):
        if not (mask[:, i] == 0).all():
            break
        crop_bottom += 1

    multiple_of = 8

    return {
        "x1": int(max(round((crop_left-pad) / multiple_of) * multiple_of, 0)),
        "y1": int(max(round((crop_top-pad) / multiple_of) * multiple_of, 0)),
        "x2": int(min(round((w - crop_right + pad) / multiple_of) * multiple_of, w)),
        "y2": int(min(round((h - crop_bottom + pad) / multiple_of) * multiple_of, h)),
    }

class JN_AreaXY:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("AREA",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "x1": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "y1": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "x2": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "y2": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    def run(self, x1, y1, x2, y2):
        area = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }
        return (area,)

class JN_AreaWidthHeight:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("AREA",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "x": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "y": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "height": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    def run(self, x, y, width, height):
        area = {
            "x1": x,
            "y1": y,
            "x2": x + width,
            "y2": y + height,
        }
        return (area,)

class JN_AreaInfo:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("x1", "y1", "x2", "y2", "width", "height")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "area": ("AREA",),
            },
        }

    def run(self, area):
        if isinstance(area, dict):
            x1 = round(area["x1"])
            y1 = round(area["y1"])
            x2 = round(area["x2"])
            y2 = round(area["y2"])
        elif isinstance(area, list):
            x1 = round(area[0])
            y1 = round(area[1])
            x2 = round(area[2])
            y2 = round(area[3])
        else:
            x1 = 0
            y1 = 0
            x2 = 0
            y2 = 0
            
        width =  x2 - x1
        height = y2 - y1

        return (x1, y1, x2, y2, width, height)

class JN_ImageInfo:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "ARRAY")
    RETURN_NAMES = ("width", "height", "channels", "batches", "shape")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    def run(self, image):
        (batches, height, width, channels) = image.shape
        shape = list(image.shape)
        return (width, height, channels, batches, shape)

class JN_MaskInfo:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("INT", "INT", "INT", "ARRAY")
    RETURN_NAMES = ("width", "height", "batches", "shape")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }

    def run(self, mask):
        mask = mask.clone().reshape((-1, mask.shape[-2], mask.shape[-1]))
        (batches, height, width) = mask.shape
        shape = list(mask.shape)
        return (width, height, batches, shape)

class JN_ImageAddMask:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    def run(self, image, mask):
        image = image.clone()
        mask = mask.clone().reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1)
        image_masked = torch.cat((image, mask), dim=3)
        return (image_masked,)

class JN_ImageCrop:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("IMAGE", "MASK", "AREA")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "pad": ("INT", {"default": 100, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "area": ("AREA",),
                "mask": ("MASK",),
            },
        }

    def run(self, image, mask=None, area=None, pad=0):
        image = image.clone().movedim(-1,1)
        if mask is not None:
            mask = mask.clone().reshape((-1, mask.shape[-2], mask.shape[-1]))

        if area is not None:
            area = normalize_area(area, image.shape[3], image.shape[2], pad)
        else:
            area = get_crop_region(mask, pad)

        cropped_image = image[:, :, area["y1"]:area["y2"], area["x1"]:area["x2"]]

        if mask is not None:
            cropped_mask = mask[:, area["y1"]:area["y2"], area["x1"]:area["x2"]]
        else:
            cropped_mask = None

        cropped_image = cropped_image.movedim(1,-1)

        return (cropped_image, cropped_mask, area)

class JN_ImageUncrop:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "area": ("AREA",),
                "resize_source": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    def run(self, destination, source, area, resize_source, mask = None):
        destination = destination.clone().movedim(-1, 1)
        output = composite(destination, source.movedim(-1, 1), area["x1"], area["y1"], mask, 1, resize_source).movedim(1, -1)
        return (output,)

class JN_ImageBatch:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "images": ("*", {"multiple": True}),
            },
            "required": {
            },
        }

    def run(self, images=[]):
        images = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), images)
        images = [image for image in images if image is not None]

        if len(images) == 0:
            s = None
        else:
            if len(images) == 1:
                s = images[0]
            else:
                s = reduce(lambda a, b: batch_image(a, b), images)
        return (s,)

class JN_LoadImageDirectory:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("IMAGE", "MASK", "ARRAY", "ARRAY")
    RETURN_NAMES = ("IMAGE", "MASK", "IMAGE_ARRAY", "MASK_ARRAY")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        # dirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
        dirs = sorted(list(set([d.rstrip("/") for d in glob.iglob("**/**", root_dir=input_dir, recursive=True) if os.path.isdir(os.path.join(input_dir, d))])))
        return {
            "required": {
                "directory": (dirs,),
            },
            "optional": {
                "recursive": ("BOOLEAN", {"default": True}),
            },
        }

    def run(self, directory, recursive=True):
        directory_path = folder_paths.get_annotated_filepath(directory)
        # files = sorted([os.path.join(directory, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])
        files = sorted([os.path.join(directory, f) for f in glob.iglob("**/**", root_dir=directory_path, recursive=recursive) if os.path.isfile(os.path.join(directory_path, f))])

        images = None
        masks = None
        images_array = []
        masks_array = []

        for file in files:
            (image, mask) = self.load_image(image=file)
            image = image.reshape((-1, image.shape[-3], image.shape[-2], image.shape[-1]))
            mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))

            images_array.append(image)
            masks_array.append(mask)

            if images is None:
                images = image
                masks = mask
            else:
                images = batch_image(images, image)
                masks = batch_mask(masks, mask)

        return (images, masks, images_array, masks_array)

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, mask.unsqueeze(0))

    @classmethod
    def IS_CHANGED(s, directory, recursive=True):
        directory_path = folder_paths.get_annotated_filepath(directory)
        # files = sorted([os.path.join(directory, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])
        files = sorted([os.path.join(directory, f) for f in glob.iglob("**/**", root_dir=directory_path, recursive=recursive) if os.path.isfile(os.path.join(directory_path, f))])

        m = hashlib.sha256()

        for file in files:
            image_path = folder_paths.get_annotated_filepath(file)
            with open(image_path, 'rb') as f:
                m.update(f.read())

        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, directory, **kwargs):
        if not folder_paths.exists_annotated_filepath(directory):
            return "Invalid directory: {}".format(directory)

        return True

class JN_BlipLoader:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("BLIP_MODEL", "BLIP_PROCESSOR")
    FUNCTION = "run"

    MODEL_NAMES = {
        "base": "Salesforce/blip-image-captioning-base",
        "large": "Salesforce/blip-image-captioning-large",
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (list(s.MODEL_NAMES.keys()),),
                "precision": (["float32", "float16"],),
                "device": (["cpu", "gpu"],),
            },
        }

    def run(self, model, precision, device):
        model_name = self.MODEL_NAMES[model]
        kargs = {}

        if precision == "float16":
            kargs["torch_dtype"] = torch.float16

        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name, **kargs)

        if device == "gpu":
            device = comfy.model_management.get_torch_device()
            model = model.to(device)

        return (model, processor)

class JN_Blip:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("STRING", "ARRAY")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("BLIP_MODEL",),
                "processor": ("BLIP_PROCESSOR",),
                "images": ("IMAGE",),
            },
            "optional": {
                "conditioning": ("STRING", {"default": "", "dynamicPrompts": True}),
                "max_new_tokens": ("INT", {"default": 1000, "min": 0, "max": 0xffffffffffffffff}),
                "skip_special_tokens": ("BOOLEAN", {"default": True}),
            },
        }

    def run(self, model, processor, images, conditioning="", max_new_tokens=1000, skip_special_tokens=True):
        images = (images.clone().reshape((-1, images.shape[-3], images.shape[-2], images.shape[-1])) * 255).to(torch.uint8)

        kargs = {
            "images": images,
            "return_tensors": "pt",
        }

        if conditioning != "":
            kargs["text"] = [conditioning for i in range(0, images.shape[0])]

        inputs = processor(**kargs).to(model.device, model.dtype)

        out = model.generate(**inputs, max_new_tokens=max_new_tokens)

        texts = []

        for x in range(0, out.shape[0]):
            text = processor.decode(out[x], skip_special_tokens=skip_special_tokens)
            text = self.clear_text(text)
            texts.append(text)

        first_text = texts[0] if len(texts) > 0 else None

        return (first_text, texts)

    def clear_text(self, text):
        remove_texts = [
            "arafed",
        ]

        spaces = "\s*"

        for remove_text in remove_texts:
            text = re.sub(spaces + remove_text + spaces, "", text, flags=re.IGNORECASE)

        return text

class JN_RemoveBackground:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE_MASKED", "IMAGE", "MASK")
    FUNCTION = "run"

    MODELS = [
        "u2net",
        "u2netp",
        "u2net_human_seg",
        "u2net_cloth_seg",
        "silueta",
        "isnet-general-use",
        "isnet-anime",
        "sam",
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (s.MODELS,),
            },
            "optional": {
                "background_image": ("IMAGE",),
                "background_color": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFF, "step": 1, "display": "color"}),
            },
        }

    def run(self, image, model, background_image=None, background_color=0):
        session = rembg.new_session(model)

        removed = [
            torch.from_numpy(
                rembg.remove(
                    (image[x] * 255).to(torch.uint8).reshape((image.shape[1], image.shape[2], image.shape[3])).numpy(),
                    session=session,
                ).astype(np.float32) / 255.0
            ).reshape((-1, image.shape[1], image.shape[2], 4)) for x in range(0, image.shape[0])
        ]

        del session

        image_masked = torch.cat(removed, dim=0)
        image = image_masked[:, :, :, 0:3]
        mask = image_masked[:, :, :, 3]

        if background_image is not None:
            background = self.resize_background(background_image, image.shape[2], image.shape[1])
        else:
            background = self.create_background(image.shape[0], image.shape[2], image.shape[1], background_color)

        image = composite(image.movedim(-1, 1), background.movedim(-1, 1), 0, 0, 1.0 - mask, 1, False).movedim(1, -1)

        return (image_masked, image, mask)

    def resize_background(self, image, width, height, upscale_method="bilinear", crop="center"):
        samples = image.movedim(-1,1)
        s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
        s = s.movedim(1,-1)
        return s

    def create_background(self, batch_size, width, height, color):
        r = torch.full([batch_size, height, width, 1], ((color >> 16) & 0xFF) / 0xFF)
        g = torch.full([batch_size, height, width, 1], ((color >> 8) & 0xFF) / 0xFF)
        b = torch.full([batch_size, height, width, 1], ((color) & 0xFF) / 0xFF)
        return torch.cat((r, g, b), dim=-1)

NODE_CLASS_MAPPINGS = {
    "JN_ImageInfo": JN_ImageInfo,
    "JN_MaskInfo": JN_MaskInfo,
    "JN_ImageAddMask": JN_ImageAddMask,
    "JN_ImageCrop": JN_ImageCrop,
    "JN_ImageUncrop": JN_ImageUncrop,
    "JN_ImageBatch": JN_ImageBatch,
    "JN_LoadImageDirectory": JN_LoadImageDirectory,
    "JN_AreaXY": JN_AreaXY,
    "JN_AreaWidthHeight": JN_AreaWidthHeight,
    "JN_AreaInfo": JN_AreaInfo,
    "JN_BlipLoader": JN_BlipLoader,
    "JN_Blip": JN_Blip,
    "JN_RemoveBackground": JN_RemoveBackground,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JN_ImageInfo": "Image Info",
    "JN_MaskInfo": "Mask Info",
    "JN_ImageAddMask": "Image Add Mask",
    "JN_ImageCrop": "Image Crop",
    "JN_ImageUncrop": "Image Uncrop",
    "JN_ImageBatch": "Image Batch",
    "JN_LoadImageDirectory": "Load Image Directory",
    "JN_AreaXY": "Area X Y",
    "JN_AreaWidthHeight": "Area Width Height",
    "JN_AreaInfo": "Area Info",
    "JN_BlipLoader": "Blip Loader",
    "JN_Blip": "Blip",
    "JN_RemoveBackground": "Remove Background",
}
