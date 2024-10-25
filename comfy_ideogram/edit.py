from .utils import *

EDIT_URL = f"{API_ENDPOINT}/edit"


class IdeogramEdit:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "(shadow), silver grey Cadillac xt5 SUV car on the edge of the mountain,sunlight,afternoon,shot from front right side,realistic film photo",
                    "multiline": True
                }),
                "model": (IDEOGRAM_MODELS,),
                "magic_prompt_option": (MAGIC_PROMPT_OPTIONS,),
                "seed": ("INT", {"default": 1234}),
                "style_type": (STYLE_TYPES,),
            },
            "optional": {
                "api_key": ("STRING",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "edit"
    CATEGORY = "Ideogram/edit"

    def edit(self,
             image: torch.Tensor,
             mask: torch.Tensor,
             prompt: str,
             model: str,
             magic_prompt_option: str,
             seed: int,
             style_type: str,
             api_key: str):
        if len(api_key) == 0 or api_key is None:
            if API_KEY is None:
                raise Exception("Must configure the API key in env_var `IDEOGRAM_KEY` or on the node.")
            api_key = API_KEY

        # 检查输入图像的形状和数据类型
        if image.ndim != 4:
            raise ValueError("Input image tensor must be 4-dimensional (batch_size, channels, height, width)")
        else:
            image = image.squeeze(0)

        if mask.ndim != 4:
            raise ValueError("Input mask tensor must be 4-dimensional (batch_size, channels, height, width)")
        else:
            mask = mask.squeeze(0)

        # 确保输入图像的数据类型为float32
        if image.dtype != torch.float32:
            image = image.to(torch.float32)

        if mask.dtype != torch.float32:
            mask = mask.to(torch.float32)

        pil_image = Image.fromarray((image.mul(255).byte().cpu().numpy()).astype(np.uint8))
        image_bytes = BytesIO()
        pil_image.save(image_bytes, format='PNG')
        image_bytes.seek(0)  # 确保字节流的指针在开头位置

        pil_mask = Image.fromarray((mask.mul(255).byte().cpu().numpy()).astype(np.uint8))
        mask_bytes = BytesIO()
        pil_mask.save(mask_bytes, format='PNG')
        mask_bytes.seek(0)  # 确保字节流的指针在开头位置

        # 准备要发送的文件对象
        files = [
            ('image_file', ('image.png', image_bytes, 'application/octet-stream')),
            ('mask', ('mask.png', mask_bytes, 'application/octet-stream'))
        ]

        image_request = {
            "prompt": prompt,
            "model": model,
            "magic_prompt_option": magic_prompt_option,
            "seed": seed,
            "style_type": style_type
        }

        headers = {
            "Api-Key": api_key
        }

        response = requests.post(EDIT_URL, data=image_request, files=files, headers=headers)
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
    "IdeogramEdit": IdeogramEdit
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IdeogramEdit": "IdeogramEdit"
}
