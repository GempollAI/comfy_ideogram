from .text2image import IdeogramTxt2Img
from .helper_nodes import ColorSelect,SeedSelect


NODE_CLASS_MAPPINGS = {
    "IdeogramTxt2Img": IdeogramTxt2Img,
    "ColorSelect": ColorSelect,
    "SeedSelect": SeedSelect

}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IdeogramTxt2Img": "IdeogramTxt2Img",
    "ColorSelect": "ColorSelect",
    "SeedSelect": "SeedSelect"
}


all = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

