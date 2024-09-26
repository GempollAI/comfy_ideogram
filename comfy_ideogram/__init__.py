from .text2image import IdeogramTxt2Img
from .color_select import Colors_select


NODE_CLASS_MAPPINGS = {
    "IdeogramTxt2Img": IdeogramTxt2Img,
    "color_select": Colors_select
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IdeogramTxt2Img": "IdeogramTxt2Img",
    "color_select": "color_select"
}


all = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

