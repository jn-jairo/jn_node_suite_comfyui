from ..utils import CATEGORY_PRIMITIVE

class JN_PrimitiveBoolean:
    CATEGORY = CATEGORY_PRIMITIVE
    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("BOOLEAN", {"default": False}),
            },
        }

    def run(self, value):
        return (value,)

class JN_PrimitiveInt:
    CATEGORY = CATEGORY_PRIMITIVE
    RETURN_TYPES = ("INT",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    def run(self, value):
        return (value,)

class JN_PrimitiveFloat:
    CATEGORY = CATEGORY_PRIMITIVE
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 0.01}),
            },
        }

    def run(self, value):
        return (value,)

class JN_PrimitiveString:
    CATEGORY = CATEGORY_PRIMITIVE
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("STRING", {"default": "", "dynamicPrompts": False}),
            },
        }

    def run(self, value):
        return (value,)

class JN_PrimitiveStringMultiline:
    CATEGORY = CATEGORY_PRIMITIVE
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("STRING", {"default": "", "dynamicPrompts": False, "multiline": True}),
            },
        }

    def run(self, value):
        return (value,)

class JN_PrimitivePrompt:
    CATEGORY = CATEGORY_PRIMITIVE
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("STRING", {"default": "", "dynamicPrompts": True, "multiline": True}),
            },
        }

    def run(self, value):
        return (value,)

class JN_PrimitiveArrayInfo:
    CATEGORY = CATEGORY_PRIMITIVE
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("length",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("ARRAY",),
            },
        }

    def run(self, value):
        value_length = len(value)
        return (value_length,)

NODE_CLASS_MAPPINGS = {
    "JN_PrimitiveBoolean": JN_PrimitiveBoolean,
    "JN_PrimitiveInt": JN_PrimitiveInt,
    "JN_PrimitiveFloat": JN_PrimitiveFloat,
    "JN_PrimitiveString": JN_PrimitiveString,
    "JN_PrimitiveStringMultiline": JN_PrimitiveStringMultiline,
    "JN_PrimitivePrompt": JN_PrimitivePrompt,
    "JN_PrimitiveArrayInfo": JN_PrimitiveArrayInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JN_PrimitiveBoolean": "BOOLEAN",
    "JN_PrimitiveInt": "INT",
    "JN_PrimitiveFloat": "FLOAT",
    "JN_PrimitiveString": "STRING",
    "JN_PrimitiveStringMultiline": "STRING MULTILINE",
    "JN_PrimitivePrompt": "PROMPT",
    "JN_PrimitiveArrayInfo": "ARRAY INFO",
}
