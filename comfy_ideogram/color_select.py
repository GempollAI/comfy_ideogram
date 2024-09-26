
class Colors_select:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        mode_list = ['HEX']
        return {
            "required": {
                "color": ("COLOR", {"default": "#FFFFFF"},),
                "mode": (mode_list,),  # 输出模式
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value",)
    FUNCTION = 'picker'
    CATEGORY = 'select'

    def picker(self, color, mode):
        ret = color
        return (ret,)


NODE_CLASS_MAPPINGS = {
    "color_select": Colors_select
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "color_select": "color_select"
}