from .text2image import IdeogramTxt2Img
from .image2image import IdeogramImg2Img
from .describe import IdeogramDescribe
from .upscale import IdeogramUpscale
from .helper_nodes import ColorSelect,SeedSelect


NODE_CLASS_MAPPINGS = {
    "IdeogramTxt2Img": IdeogramTxt2Img,
    "ColorSelect": ColorSelect,
    "SeedSelect": SeedSelect,
    "IdeogramImg2Img": IdeogramImg2Img,
    "IdeogramDescribe": IdeogramDescribe,
    "IdeogramUpscale": IdeogramUpscale

}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IdeogramTxt2Img": "IdeogramTxt2Img",
    "ColorSelect": "ColorSelect",
    "SeedSelect": "SeedSelect",
    "IdeogramImg2Img": "IdeogramImg2Img",
    "IdeogramDescribe": "IdeogramDescribe",
    "IdeogramUpscale": "IdeogramUpscale"
}


all = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

