from .text2image import IdeogramTxt2Img
from .color_select import ColorSelect,Seed_select


NODE_CLASS_MAPPINGS = {
    "IdeogramTxt2Img": IdeogramTxt2Img,
    "ColorSelect": ColorSelect,
    "Seed_select": Seed_select

}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IdeogramTxt2Img": "IdeogramTxt2Img",
    "ColorSelect": "ColorSelect",
    "Seed_select": "Seed_select"
}


all = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

