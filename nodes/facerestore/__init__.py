import os
import folder_paths

if "facerestore_models" not in folder_paths.folder_names_and_paths:
    dir_facerestore_models = os.path.join(folder_paths.models_dir, "facerestore_models")
    os.makedirs(dir_facerestore_models, exist_ok=True)
    folder_paths.add_model_folder_path("facerestore_models", dir_facerestore_models)

if "facedetection" not in folder_paths.folder_names_and_paths:
    dir_facedetection = os.path.join(folder_paths.models_dir, "facedetection")
    os.makedirs(dir_facedetection, exist_ok=True)
    folder_paths.add_model_folder_path("facedetection", dir_facedetection)

import model_management
import torch
import comfy.utils
import numpy as np
import cv2
import math
from .facelib.utils.face_restoration_helper import FaceRestoreHelper
from .facelib.detection.retinaface import retinaface
from torchvision.transforms.functional import normalize
from comfy_extras.chainner_models import model_loading

from ...utils import CATEGORY_IMAGE

FACEDETECTION_MODELS = [
    "retinaface_resnet50",
    "retinaface_mobile0.25",
    "YOLOv5l",
    "YOLOv5n",
]

def detface2area(det_face):
    return {
        "x1": round(det_face[0]),
        "y1": round(det_face[1]),
        "x2": round(det_face[2]),
        "y2": round(det_face[3]),
    }

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. ' f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

class JN_FaceRestoreWithModel:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "ARRAY")
    RETURN_NAMES = ("IMAGE", "FACES_RESTORED_IMAGE", "FACES_IMAGE", "FACES_AREA_ARRAY")
    FUNCTION = "restore_face"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "facerestore_model": ("FACERESTORE_MODEL",),
                "image": ("IMAGE",),
                "facedetection": (FACEDETECTION_MODELS,),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05}),
            }
        }

    def __init__(self):
        self.face_helper = None

    def restore_face(self, facerestore_model, image, facedetection, strength=0.5):
        device = model_management.get_torch_device()
        facerestore_model.to(device)
        if self.face_helper is None:
            self.face_helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model=facedetection, save_ext='png', use_parse=True, device=device)

        image_np = 255. * image.clone().cpu().numpy()

        total_images = image_np.shape[0]
        out_images = np.ndarray(shape=image_np.shape)
        out_faces_areas = []
        out_faces_images_arr = []
        out_faces_images_restored_arr = []

        for i in range(total_images):
            cur_image_np = image_np[i,:, :, ::-1]

            original_resolution = cur_image_np.shape[0:2]

            scale = min(1, min(cur_image_np.shape[0], cur_image_np.shape[1]) / 512)

            if facerestore_model is None or self.face_helper is None:
                return image

            self.face_helper.clean_all()
            self.face_helper.read_image(cur_image_np)
            self.face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
            self.face_helper.align_warp_face()

            restored_face = None
            for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
                cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

                try:
                    with torch.no_grad():
                        #output = facerestore_model(cropped_face_t, w=strength, adain=True)[0]
                        output = facerestore_model(cropped_face_t, weight=strength)[0]
                        restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                    del output
                    torch.cuda.empty_cache()
                except Exception as error:
                    print(f'\tFailed inference for CodeFormer: {error}', file=sys.stderr)
                    restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

                restored_face = restored_face.astype('uint8')
                self.face_helper.add_restored_face(restored_face)

                cropped_face_t = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1)).astype('uint8')

                out_faces_images_arr.append(cv2.cvtColor(cropped_face_t, cv2.COLOR_BGR2RGB))
                out_faces_images_restored_arr.append(cv2.cvtColor(restored_face, cv2.COLOR_BGR2RGB))
                out_faces_areas.append(detface2area(list(self.face_helper.det_faces[idx] * scale)))

            self.face_helper.get_inverse_affine(None)

            restored_img = self.face_helper.paste_faces_to_input_image()
            restored_img = restored_img[:, :, ::-1]

            if original_resolution != restored_img.shape[0:2]:
                restored_img = cv2.resize(restored_img, (0, 0), fx=original_resolution[1]/restored_img.shape[1], fy=original_resolution[0]/restored_img.shape[0], interpolation=cv2.INTER_LINEAR)

            self.face_helper.clean_all()

            # restored_img = cv2.cvtColor(restored_face, cv2.COLOR_BGR2RGB)

            out_images[i] = restored_img

        if len(out_faces_images_arr) > 0:
            out_faces_images = np.ndarray(shape=(len(out_faces_images_arr), 512, 512, 3))
            for i, face in enumerate(out_faces_images_arr):
                out_faces_images[i] = face
        else:
            out_faces_images = np.ndarray(shape=(1, 512, 512, 3))

        if len(out_faces_images_restored_arr) > 0:
            out_faces_images_restored = np.ndarray(shape=(len(out_faces_images_restored_arr), 512, 512, 3))
            for i, face in enumerate(out_faces_images_restored_arr):
                out_faces_images_restored[i] = face
        else:
            out_faces_images_restored = np.ndarray(shape=(1, 512, 512, 3))

        out_faces_images = torch.from_numpy(np.array(out_faces_images).astype(np.float32) / 255.0)
        out_faces_images_restored = torch.from_numpy(np.array(out_faces_images_restored).astype(np.float32) / 255.0)

        restored_img_np = np.array(out_images).astype(np.float32) / 255.0
        restored_img_tensor = torch.from_numpy(restored_img_np)

        return (restored_img_tensor, out_faces_images_restored, out_faces_images, out_faces_areas)

class JN_CropFace:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("IMAGE", "ARRAY")
    RETURN_NAMES = ("IMAGE", "AREA_ARRAY")
    FUNCTION = "crop_face"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "facedetection": (FACEDETECTION_MODELS,),
            }
        }

    def __init__(self):
        self.face_helper = None

    def crop_face(self, image, facedetection):
        device = model_management.get_torch_device()
        if self.face_helper is None:
            self.face_helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model=facedetection, save_ext='png', use_parse=True, device=device)

        image_np = 255. * image.clone().cpu().numpy()

        total_images = image_np.shape[0]
        out_images = np.ndarray(shape=(total_images, 512, 512, 3))
        out_areas = []
        next_idx = 0

        for i in range(total_images):

            cur_image_np = image_np[i,:, :, ::-1]

            original_resolution = cur_image_np.shape[0:2]

            scale = min(1, min(cur_image_np.shape[0], cur_image_np.shape[1]) / 512)

            if self.face_helper is None:
                return image

            self.face_helper.clean_all()
            self.face_helper.read_image(cur_image_np)
            self.face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
            self.face_helper.align_warp_face()

            faces_found = len(self.face_helper.cropped_faces)
            if faces_found == 0:
                next_idx += 1 # output black image for no face
            if out_images.shape[0] < next_idx + faces_found:
                # print(out_images.shape)
                # print((next_idx + faces_found, 512, 512, 3))
                # print('aaaaa')
                out_images = np.resize(out_images, (next_idx + faces_found, 512, 512, 3))
                # print(out_images.shape)
            for j in range(faces_found):
                cropped_face_1 = self.face_helper.cropped_faces[j]
                cropped_face_2 = img2tensor(cropped_face_1 / 255., bgr2rgb=True, float32=True)
                normalize(cropped_face_2, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_3 = cropped_face_2.unsqueeze(0).to(device)
                cropped_face_4 = tensor2img(cropped_face_3, rgb2bgr=True, min_max=(-1, 1)).astype('uint8')
                cropped_face_5 = cv2.cvtColor(cropped_face_4, cv2.COLOR_BGR2RGB)
                out_images[next_idx] = cropped_face_5
                out_areas.append(detface2area(list(self.face_helper.det_faces[j] * scale)))
                next_idx += 1

        cropped_face_6 = np.array(out_images).astype(np.float32) / 255.0
        cropped_face_7 = torch.from_numpy(cropped_face_6)

        return (cropped_face_7, out_areas)

class JN_FaceRestoreModelLoader:
    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("FACERESTORE_MODEL",)
    FUNCTION = "load_model"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("facerestore_models"),),
            }
        }

    def load_model(self, model_name):
        model_path = folder_paths.get_full_path("facerestore_models", model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        out = model_loading.load_state_dict(sd).eval()
        return (out, )

NODE_CLASS_MAPPINGS = {
    "JN_FaceRestoreModelLoader": JN_FaceRestoreModelLoader,
    "JN_FaceRestoreWithModel": JN_FaceRestoreWithModel,
    "JN_CropFace": JN_CropFace,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JN_FaceRestoreModelLoader": "Face Restore Model Loader",
    "JN_FaceRestoreWithModel": "Face Restore With Model",
    "JN_CropFace": "Crop Face",
}
