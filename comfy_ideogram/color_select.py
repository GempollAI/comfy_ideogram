
class ColorSelect:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "color": ("COLOR", {"default": "#FFFFFF"},),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value",)
    FUNCTION = 'picker'
    CATEGORY = 'select'

    def picker(self, color):
        ret = color
        return (ret,)


NODE_CLASS_MAPPINGS = {
    "ColorSelect": ColorSelect
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorSelect": "ColorSelect"
}