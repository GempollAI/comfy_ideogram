import requests
from io import BytesIO
import torch
import numpy as np
import os
from PIL import Image, ImageSequence, ImageOps

API_KEY = os.environ.get("IDEOGRAM_KEY", None)

MIN_COLOR_WEIGHT_FLOAT = 0.05 # weight >= 0.05, required by Ideogram
ROUNDING_MULTIPLIER = 100 # 保留百分位
MIN_COLOR_WEIGHT_INT = int(MIN_COLOR_WEIGHT_FLOAT * ROUNDING_MULTIPLIER)

RESOLUTION_DEFAULT = "NotSelected"

RESOLUTIONS = [
    RESOLUTION_DEFAULT,
    "RESOLUTION_1024_640",
    "RESOLUTION_1024_768",
    "RESOLUTION_1024_832",
    "RESOLUTION_1024_896",
    "RESOLUTION_1024_960",
    "RESOLUTION_1024_1024",
    "RESOLUTION_1088_768",
    "RESOLUTION_1088_832",
    "RESOLUTION_1088_896",
    "RESOLUTION_1088_960",
    "RESOLUTION_1120_896",
    "RESOLUTION_1152_704",
    "RESOLUTION_1152_768",
    "RESOLUTION_1152_832",
    "RESOLUTION_1152_864",
    "RESOLUTION_1152_896",
    "RESOLUTION_1216_704",
    "RESOLUTION_1216_768",
    "RESOLUTION_1216_832",
    "RESOLUTION_1232_768",
    "RESOLUTION_1248_832",
    "RESOLUTION_1280_704",
    "RESOLUTION_1280_720",
    "RESOLUTION_1280_768",
    "RESOLUTION_1280_800",
    "RESOLUTION_1312_736",
    "RESOLUTION_1344_640",
    "RESOLUTION_1344_704",
    "RESOLUTION_1344_768",
    "RESOLUTION_1408_576",
    "RESOLUTION_1408_640",
    "RESOLUTION_1408_704",
    "RESOLUTION_1472_576",
    "RESOLUTION_1472_640",
    "RESOLUTION_1472_704",
    "RESOLUTION_1536_512",
    "RESOLUTION_1536_576",
    "RESOLUTION_1536_640",
    "RESOLUTION_512_1536",
    "RESOLUTION_576_1408",
    "RESOLUTION_576_1472",
    "RESOLUTION_576_1536",
    "RESOLUTION_640_1024",
    "RESOLUTION_640_1344",
    "RESOLUTION_640_1408",
    "RESOLUTION_640_1472",
    "RESOLUTION_640_1536",
    "RESOLUTION_704_1152",
    "RESOLUTION_704_1216",
    "RESOLUTION_704_1280",
    "RESOLUTION_704_1344",
    "RESOLUTION_704_1408",
    "RESOLUTION_704_1472",
    "RESOLUTION_720_1280",
    "RESOLUTION_736_1312",
    "RESOLUTION_768_1024",
    "RESOLUTION_768_1088",
    "RESOLUTION_768_1152",
    "RESOLUTION_768_1216",
    "RESOLUTION_768_1232",
    "RESOLUTION_768_1280",
    "RESOLUTION_768_1344",
    "RESOLUTION_832_960",
    "RESOLUTION_832_1024",
    "RESOLUTION_832_1088",
    "RESOLUTION_832_1152",
    "RESOLUTION_832_1216",
    "RESOLUTION_832_1248",
    "RESOLUTION_864_1152",
    "RESOLUTION_896_960",
    "RESOLUTION_896_1024",
    "RESOLUTION_896_1088",
    "RESOLUTION_896_1120",
    "RESOLUTION_896_1152",
    "RESOLUTION_960_832",
    "RESOLUTION_960_896",
    "RESOLUTION_960_1024",
    "RESOLUTION_960_1088",
]

RESOLUTION_MAPPING = {
    v.replace("RESOLUTION_", "").replace("_", "x"): v
    for v in RESOLUTIONS
}

ASPECT_RATIO_DEFAULT = "NotSelected"

ASPECT_RATIOS = [
    ASPECT_RATIO_DEFAULT,
    "ASPECT_10_16",
    "ASPECT_16_10",
    "ASPECT_9_16",
    "ASPECT_16_9",
    "ASPECT_3_2",
    "ASPECT_2_3",
    "ASPECT_4_3",
    "ASPECT_3_4",
    "ASPECT_4_5",
    "ASPECT_5_4",
    "ASPECT_1_1",
    "ASPECT_1_3",
    "ASPECT_3_1"
]

ASPECT_RATIO_MAPPING = {
    v.replace("ASPECT_", "").replace("_", ":"): v
    for v in ASPECT_RATIOS
}


def load_image(image_source):
    if image_source.startswith('http'):
        response = requests.get(image_source)
        img = Image.open(BytesIO(response.content))
        file_name = image_source.split('/')[-1]
    else:
        img = Image.open(image_source)
        file_name = os.path.basename(image_source)
    return img, file_name


def pil2tensor(img):
    output_images = []
    output_masks = []
    for i in ImageSequence.Iterator(img):
        i = ImageOps.exif_transpose(i)
        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        output_images.append(image)
        output_masks.append(mask.unsqueeze(0))

    if len(output_images) > 1:
        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)
    else:
        output_image = output_images[0]
        output_mask = output_masks[0]

    return output_image, output_mask


class IdeogramTxt2Img:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "(shadow), silver grey Cadillac xt5 SUV car on the edge of the mountain,sunlight,afternoon,shot from front right side,realistic film photo",
                    "multiline": True
                }),
                "aspect_ratio": (list(ASPECT_RATIO_MAPPING.keys()),),
                "resolution": (list(RESOLUTION_MAPPING.keys()),),
                "seed": ("INT", {"default": 1234}),
                "model": (["V_2", "V_2_TURBO", "V_1", "V_1_TURBO"],),
                "magic_prompt_option": (["AUTO", "ON", "OFF"],),
                "negative_prompt": ("STRING", {"default": ""}),
                "color_palette_hex1": ("STRING",),
                "color_palette_weight1": ("FLOAT", {"default": 10, "min": 0, "max": 10}),
                "color_palette_hex2": ("STRING",),
                "color_palette_weight2": ("FLOAT", {"default": 10, "min": 0, "max": 10}),
                "color_palette_hex3": ("STRING",),
                "color_palette_weight3": ("FLOAT", {"default": 10, "min": 0, "max": 10}),
                "color_palette_hex4": ("STRING",),
                "color_palette_weight4": ("FLOAT", {"default": 10, "min": 0, "max": 10}),
            },
            "optional": {
                "api_key": ("STRING",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "text2image"
    CATEGORY = "Ideogram/txt2img"

    def text2image(self,
                   prompt: str,
                   aspect_ratio: str,
                   resolution: str,
                   model: str,
                   magic_prompt_option: str,
                   negative_prompt: str,
                   seed: int,
                   color_palette_hex1: str, color_palette_weight1: float,
                   color_palette_hex2: str, color_palette_weight2: float,
                   color_palette_hex3: str, color_palette_weight3: float,
                   color_palette_hex4: str, color_palette_weight4: float,
                   api_key: str):
        if len(api_key) == 0 or api_key is None:
            if API_KEY is None:
                raise Exception("Must configure the API key in env_var `IDEOGRAM_KEY` or on the node.")
            api_key = API_KEY

        txt2img_generate_url = "https://api.ideogram.ai/generate"
        weights = [color_palette_weight1, color_palette_weight2, color_palette_weight3, color_palette_weight4]
        colors = [color_palette_hex1, color_palette_hex2, color_palette_hex3, color_palette_hex4]
        resolution = RESOLUTION_MAPPING[resolution]
        aspect_ratio = ASPECT_RATIO_MAPPING[aspect_ratio]

        if resolution == RESOLUTION_DEFAULT and aspect_ratio == ASPECT_RATIO_DEFAULT:
            raise Exception("Must select one of aspect ratio and resolution")
        if resolution != RESOLUTION_DEFAULT and aspect_ratio != ASPECT_RATIO_DEFAULT:
            raise Exception("Should not select both aspect ratio and resolution")

        # 移除权重为0的元素及其对应的颜色
        weights_colors = [(w, c) for w, c in zip(weights, colors) if w != 0.0]

        if not weights_colors:
            raise ValueError("All weights are zero, cannot proceed with image generation.")

        weights, colors = zip(*weights_colors)

        # 计算剩余元素的总和
        total_weight = sum(weights)

        # 对剩余的元素进行归一化处理
        normalized_weight = [i / total_weight for i in weights]

        # 去除小数点后两位的数字，并计算这些被去除值的总和
        truncated_weight_int = [int(i * ROUNDING_MULTIPLIER) for i in normalized_weight]
        total_weight_gap = ROUNDING_MULTIPLIER - sum(truncated_weight_int)
        # 找到剩余元素中的最大值
        min_index = truncated_weight_int.index(min(truncated_weight_int))
        # 将被去除值的总和补到最小的数字上
        truncated_weight_int[min_index] += total_weight_gap

        # 补充低于 MIN_COLOR_WEIGHT_INT 的权重
        for (i, val) in enumerate(truncated_weight_int):
            if val < MIN_COLOR_WEIGHT_INT:
                diff = MIN_COLOR_WEIGHT_INT - val
                truncated_weight_int[i] = MIN_COLOR_WEIGHT_INT
                max_index = truncated_weight_int.index(max(truncated_weight_int))
                truncated_weight_int[max_index] -= diff

        for val in truncated_weight_int:
            assert val >= MIN_COLOR_WEIGHT_INT, "Impossible"

        rectified_weights = [val / ROUNDING_MULTIPLIER for val in truncated_weight_int]
        color_palette = [
            {"color_hex": color, "color_weight": color_weight}
            for color, color_weight in zip(colors, rectified_weights)
        ]

        payload = {
            "image_request": {
                "prompt": prompt,
                "model": model,
                "magic_prompt_option": magic_prompt_option,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "color_palette": {
                    "members": color_palette
                },
            }
        }

        if resolution != RESOLUTION_DEFAULT:
            payload["image_request"]["resolution"] = resolution

        if aspect_ratio != ASPECT_RATIO_DEFAULT:
            payload["image_request"]["aspect_ratio"] = aspect_ratio

        headers = {
            "Api-Key": api_key,
            "Content-Type": "application/json"
        }

        response = requests.post(txt2img_generate_url, json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()["data"][0]
        is_image_safe = response_data["is_image_safe"]
        if not is_image_safe:
            raise Exception("Ideogram reports the generated image is not safe. Image not available.")
        img_url = response_data["url"]
        img, _name = load_image(img_url)
        img_out, mask_out = pil2tensor(img)
        return img_out, mask_out


NODE_CLASS_MAPPINGS = {
    "IdeogramTxt2Img": IdeogramTxt2Img
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IdeogramTxt2Img": "IdeogramTxt2Img"
}
