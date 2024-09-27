from .text2image import IdeogramTxt2Img
from .mode_select import ColorSelect,SeedSelect
from .image2image import IdeogramImg2Img


NODE_CLASS_MAPPINGS = {
    "IdeogramTxt2Img": IdeogramTxt2Img,
    "ColorSelect": ColorSelect,
    "SeedSelect": SeedSelect,
    "IdeogramImg2Img": IdeogramImg2Img

}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IdeogramTxt2Img": "IdeogramTxt2Img",
    "ColorSelect": "ColorSelect",
    "SeedSelect": "SeedSelect",
    "IdeogramImg2Img": "IdeogramImg2Img"
}


all = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

