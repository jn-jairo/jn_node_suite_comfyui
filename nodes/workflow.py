from ..utils import CATEGORY_WORKFLOW

class JN_Condition:
    CATEGORY = CATEGORY_WORKFLOW
    RETURN_TYPES = ("*",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "condition": ("BOOLEAN", {"default": True}),
                "if_true": ("*",),
                "if_false": ("*",),
            },
        }

    def run(self, condition=True, if_true=True, if_false=False):
        value = if_true if condition else if_false
        return (value,)

class JN_Route:
    CATEGORY = CATEGORY_WORKFLOW
    RETURN_TYPES = ("*",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "value": ("*",),
            },
        }

    def run(self, value):
        return (value,)

NODE_CLASS_MAPPINGS = {
    "JN_Condition": JN_Condition,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JN_Condition": "Condition",
}
