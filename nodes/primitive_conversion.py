import math
from functools import reduce

from ..utils import CATEGORY_PRIMITIVE_CONVERSION

def convert_constants(value):
    if value == "e":
        value = math.e
    elif value == "pi":
        value = math.pi
    elif value == "tau":
        value = math.tau

    return value

def to_boolean(value):
    if isinstance(value, str):
        value = value.lower() == "true"
    elif isinstance(value, (int, float, bool)):
        if math.isnan(value):
            value = False
        else:
            try:
                value = bool(value)
            except:
                value = None
    else:
        value = value is not None

    return value

def to_int_round(value, round_method="round"):
    if round_method == "floor":
        value = math.floor(value)
    elif round_method == "ceil":
        value = math.ceil(value)
    elif round_method == "round":
        value = round(value)
    else:
        value = int(value)
    return value

def to_int(value, round_method="truncate"):
    if value is None:
        return value

    value = convert_constants(value)

    if isinstance(value, float):
        value = to_int_round(value, round_method)
    elif not isinstance(value, int):
        try:
            value = int(value)
        except:
            value = None

    return value

def to_float_round(value, decimal_places, round_method="round"):
    factor = 10 ** decimal_places
    if round_method == "floor":
        value = math.floor(value * factor) / factor
    elif round_method == "ceil":
        value = math.ceil(value * factor) / factor
    elif round_method == "round":
        value = round(value, decimal_places)
    elif round_method == "truncate":
        value = float(int(value))
    return value

def to_float(value, decimal_places=0, round_method="disabled"):
    if value is None:
        return value

    value = convert_constants(value)

    try:
        value = to_float_round(float(value), decimal_places, round_method)
    except:
        value = None

    return value

def to_string(value):
    if value is None:
        return value

    try:
        value = str(value)
    except:
        value = None

    return value

class JN_PrimitiveToBoolean:
    CATEGORY = CATEGORY_PRIMITIVE_CONVERSION
    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("*",),
            },
        }

    def run(self, value):
        value = to_boolean(value)
        return (value,)

class JN_PrimitiveToInt:
    CATEGORY = CATEGORY_PRIMITIVE_CONVERSION
    RETURN_TYPES = ("INT",)
    FUNCTION = "run"

    ROUND_METHODS = ["truncate", "round", "floor", "ceil"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("*",),
                "round_method": (s.ROUND_METHODS, ),
            },
        }

    def run(self, value, round_method):
        value = to_int(value, round_method)
        return (value,)

class JN_PrimitiveToFloat:
    CATEGORY = CATEGORY_PRIMITIVE_CONVERSION
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "run"

    ROUND_METHODS = ["disabled", "truncate", "round", "floor", "ceil"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("*",),
                "decimal_places": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "round_method": (s.ROUND_METHODS, ),
            },
        }

    def run(self, value, decimal_places, round_method):
        value = to_float(value, decimal_places, round_method)
        return (value,)

class JN_PrimitiveToString:
    CATEGORY = CATEGORY_PRIMITIVE_CONVERSION
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("*",),
            },
        }

    def run(self, value):
        value = to_string(value)
        return (value,)

class JN_PrimitiveToArray:
    CATEGORY = CATEGORY_PRIMITIVE_CONVERSION
    RETURN_TYPES = ("ARRAY",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "only_active": ("BOOLEAN", {"default": True}),
                "merge_array_items": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "values": ("*", {"multiple": True}),
            },
        }

    def run(self, only_active, merge_array_items, values=[]):
        selected = None

        if merge_array_items:
            values = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), values)

        if only_active:
            values = [value for value in values if value is not None]

        return (values,)

class JN_PrimitiveStringToArray:
    CATEGORY = CATEGORY_PRIMITIVE_CONVERSION
    RETURN_TYPES = ("ARRAY",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("STRING", {"default": "", "dynamicPrompts": False, "multiline": True}),
            },
        }

    def run(self, value):
        value = value.split("\n")
        return (value,)

class JN_PrimitiveBatchToArray:
    CATEGORY = CATEGORY_PRIMITIVE_CONVERSION
    RETURN_TYPES = ("ARRAY",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("*",),
            },
        }

    def run(self, value):
        shape = list(value.shape)
        shape[0] = -1
        value = [value[x].clone().reshape(shape) for x in range(0, value.shape[0])]
        return (value,)

NODE_CLASS_MAPPINGS = {
    "JN_PrimitiveToBoolean": JN_PrimitiveToBoolean,
    "JN_PrimitiveToInt": JN_PrimitiveToInt,
    "JN_PrimitiveToFloat": JN_PrimitiveToFloat,
    "JN_PrimitiveToString": JN_PrimitiveToString,
    "JN_PrimitiveToArray": JN_PrimitiveToArray,
    "JN_PrimitiveStringToArray": JN_PrimitiveStringToArray,
    "JN_PrimitiveBatchToArray": JN_PrimitiveBatchToArray,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JN_PrimitiveToBoolean": "TO BOOLEAN",
    "JN_PrimitiveToInt": "TO INT",
    "JN_PrimitiveToFloat": "TO FLOAT",
    "JN_PrimitiveToString": "TO STRING",
    "JN_PrimitiveToArray": "TO ARRAY",
    "JN_PrimitiveStringToArray": "STRING TO ARRAY",
    "JN_PrimitiveBatchToArray": "BATCH TO ARRAY",
}
