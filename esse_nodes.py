class EssePromptingPresets:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "category": ("STRING", {"default": "", "multiline": False}),
                "key": ("STRING", {"default": "", "multiline": False}),
                "value": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("key", "value")
    FUNCTION = "process"
    CATEGORY = "Esse"

    def process(self, category, key, value):
        return (key, value)

NODE_CLASS_MAPPINGS = {
    "EssePromptingPresets": EssePromptingPresets
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EssePromptingPresets": "Esse Prompting Presets"
}
