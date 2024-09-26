from .text2image import ideogram_text2image
from .color_select import Colors_select


NODE_CLASS_MAPPINGS = {
    "ideogram_text2image": ideogram_text2image,
    "color_select": Colors_select

}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ideogram_text2image": "ideogram_text2image",
    "color_select": "color_select"

}

all = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
