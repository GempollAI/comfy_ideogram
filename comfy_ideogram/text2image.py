import requests
from io import BytesIO
import torch
import numpy as np
import os
from PIL import Image, ImageSequence, ImageOps
from .utils import CONFIGS


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
    def __init__(self):
        self.configs = CONFIGS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "(shadow), silver grey Cadillac xt5 SUV car on the edge of the mountain,sunlight,afternoon,shot from front right side,realistic film photo",
                    "multiline": True}),
                "aspect_ratio": (["ASPECT_10_16",
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
                                  "ASPECT_3_1"],),
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
                "api": ("STRING",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "text2image"
    CATEGORY = "gempoll/ideo_t2i"

    def text2image(self, prompt, aspect_ratio, model, magic_prompt_option, negative_prompt, color_palette_hex1,
                   color_palette_weight1,
                   color_palette_hex2, color_palette_weight2, color_palette_hex3, color_palette_weight3,
                   color_palette_hex4, color_palette_weight4, api):
        if len(api) == 0 or api is None:
            api = self.configs.api_key
            if len(self.configs.api_key) == 0:
                raise Exception("Must configure the API key in configs.json or on the node.")

        url = "https://api.ideogram.ai/generate"
        weights = [color_palette_weight1, color_palette_weight2, color_palette_weight3, color_palette_weight4]
        colors = [color_palette_hex1, color_palette_hex2, color_palette_hex3, color_palette_hex4]

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
        truncated_weight_int = [int(i * 100) for i in normalized_weight]
        gap = 100 - sum(truncated_weight_int)
        # 找到剩余元素中的最大值
        min_index = truncated_weight_int.index(min(truncated_weight_int))
        # 将被去除值的总和补到最小的数字上
        truncated_weight_int[min_index] += gap

        # 补充低于0.05 * 100 的权重，因为 API 的要求，权重不能小于 0.05
        for (i, val) in enumerate(truncated_weight_int):
            if val < 5:
                diff = 5 - val
                truncated_weight_int[i] = 5
                max_index = truncated_weight_int.index(max(truncated_weight_int))
                truncated_weight_int[max_index] -= diff

        for val in truncated_weight_int:
            assert val >= 5, "Impossible"

        rectified_weights = [val / 100 for val in truncated_weight_int]
        color_palette = [
            {"color_hex": color, "color_weight": color_weight}
            for color, color_weight in zip(colors, rectified_weights)
        ]

        payload = {
            "image_request": {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "model": model,
                "magic_prompt_option": magic_prompt_option,
                "negative_prompt": negative_prompt,
                "color_palette": {
                    "members": color_palette},
            }
        }
        headers = {
            "Api-Key": api,
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)

        url = response.json()["data"][0]["url"]
        img, name = load_image(url)
        img_out, mask_out = pil2tensor(img)
        return img_out, mask_out


NODE_CLASS_MAPPINGS = {
    "IdeogramTxt2Img": IdeogramTxt2Img
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IdeogramTxt2Img": "IdeogramTxt2Img"
}
