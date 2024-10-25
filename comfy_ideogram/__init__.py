from .describe import IdeogramDescribe
from .edit import IdeogramEdit
from .helper_nodes import ColorSelect, SeedSelect
from .image2image import IdeogramImg2Img
from .text2image import IdeogramTxt2Img
from .upscale import IdeogramUpscale

NODE_CLASS_MAPPINGS = {
    "IdeogramTxt2Img": IdeogramTxt2Img,
    "ColorSelect": ColorSelect,
    "SeedSelect": SeedSelect,
    "IdeogramImg2Img": IdeogramImg2Img,
    "IdeogramDescribe": IdeogramDescribe,
    "IdeogramUpscale": IdeogramUpscale,
    "IdeogramEdit": IdeogramEdit
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IdeogramTxt2Img": "IdeogramTxt2Img",
    "ColorSelect": "ColorSelect",
    "SeedSelect": "SeedSelect",
    "IdeogramImg2Img": "IdeogramImg2Img",
    "IdeogramDescribe": "IdeogramDescribe",
    "IdeogramUpscale": "IdeogramUpscale",
    "IdeogramEdit": "IdeogramEdit"
}

all = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
