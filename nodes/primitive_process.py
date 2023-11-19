import math
import statistics
from functools import reduce
import re

from ..utils import CATEGORY_PRIMITIVE_PROCESS
from .primitive_conversion import to_string, to_boolean

def math_operation(a, b, operation):
    value = None

    if operation == "a + b":
        value = a + b
    elif operation == "a - b":
        value = a - b
    elif operation == "a * b":
        value = a * b
    elif operation == "a / b":
        value = a / b
    elif operation == "a // b":
        value = a // b
    elif operation == "a % b":
        value = a % b
    elif operation == "a ** b":
        value = a ** b
    elif operation == "a ** (1/b)":
        value = a ** (1/b)
    elif operation == "sqrt(a)":
        value = math.sqrt(a)
    elif operation == "exp(a)":
        value = math.exp(a)
    elif operation == "log(a)":
        value = math.log(a)
    elif operation == "log(a, b)":
        value = math.log(a, b)

    return value

def math_operation_array(values, operation):
    value = None

    if operation == "min":
        value = min(values)
    elif operation == "max":
        value = max(values)
    elif operation == "mean":
        value = statistics.mean(values)
    elif operation == "median":
        value = statistics.median(values)

    return value

def boolean_operation(a, b, operation):
    a = to_boolean(a)
    b = to_boolean(b)
    value = None

    if operation == "a and b":
        value = a and b
    elif operation == "a or b":
        value = a or b
    elif operation == "a xor b":
        value = a ^ b
    elif operation == "not a":
        value = not a

    return value

def logic_operation(a, b, operation):
    value = None

    if operation == "a == b":
        value = a == b
    elif operation == "a != b":
        value = a != b
    elif operation == "a > b":
        value = a > b
    elif operation == "a >= b":
        value = a >= b
    elif operation == "a < b":
        value = a < b
    elif operation == "a <= b":
        value = a <= b

    return value

def slice_operation(values, a, b, operation):
    if operation == "[a:b]":
        values = values[a:b]
    elif operation == "[a:a+b]":
        values = values[a:a+b]
    elif operation == "[a-b:a]":
        values = values[a-b:a]
    elif operation == "[a-b:a+b]":
        values = values[a-b:a+b]
    elif operation == "[a:]":
        values = values[a:]
    elif operation == "[:a]":
        values = values[:a]

    return values

class JN_MathOperation:
    CATEGORY = CATEGORY_PRIMITIVE_PROCESS
    RETURN_TYPES = ("*",)
    FUNCTION = "run"

    OPERATIONS = ["a + b", "a - b", "a * b", "a / b", "a // b", "a % b", "a ** b", "a ** (1/b)", "sqrt(a)", "exp(a)", "log(a)", "log(a, b)"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a": ("*",),
                "operation": (s.OPERATIONS, ),
            },
            "optional": {
                "b": ("*",),
            },
        }

    def run(self, a, operation, b=0):
        value = math_operation(a, b, operation)
        return (value,)

class JN_MathOperationArray:
    CATEGORY = CATEGORY_PRIMITIVE_PROCESS
    RETURN_TYPES = ("*",)
    FUNCTION = "run"

    OPERATIONS = ["min", "max", "mean", "median"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "operation": (s.OPERATIONS, ),
            },
            "optional": {
                "values": ("*", {"multiple": True}),
            },
        }

    def run(self, operation, values=[]):
        values = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), values)
        values = [value for value in values if value is not None]
        if len(values) > 0:
            value = math_operation_array(values, operation)
        else:
            value = None
        return (value,)

class JN_BooleanOperation:
    CATEGORY = CATEGORY_PRIMITIVE_PROCESS
    RETURN_TYPES = ("*",)
    FUNCTION = "run"

    OPERATIONS = ["a and b", "a or b", "a xor b", "not a"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a": ("*",),
                "operation": (s.OPERATIONS, ),
            },
            "optional": {
                "b": ("*",),
            },
        }

    def run(self, a, operation, b=False):
        value = boolean_operation(a, b, operation)
        return (value,)

class JN_LogicOperation:
    CATEGORY = CATEGORY_PRIMITIVE_PROCESS
    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "run"

    OPERATIONS = ["a == b", "a != b", "a > b", "a >= b", "a < b", "a <= b"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a": ("*",),
                "operation": (s.OPERATIONS, ),
            },
            "optional": {
                "b": ("*",),
            },
        }

    def run(self, a, operation, b=False):
        value = logic_operation(a, b, operation)
        return (value,)

class JN_TextConcatenation:
    CATEGORY = CATEGORY_PRIMITIVE_PROCESS
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "glue": ("STRING", {"default": "", "dynamicPrompts": False}),
                "values": ("*", {"multiple": True}),
            },
        }

    def run(self, glue="", values=[]):
        if glue is None:
            glue = ""

        values = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), values)
        values = [to_string(value) for value in values]
        values = [value for value in values if value is not None]

        value = glue.join(values)

        return (value,)

class JN_TextReplace:
    CATEGORY = CATEGORY_PRIMITIVE_PROCESS
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("*",),
                "search": ("STRING", {"default": "", "dynamicPrompts": False}),
                "replace": ("STRING", {"default": "", "dynamicPrompts": False}),
                "regex": ("BOOLEAN", {"default": False}),
            },
        }

    def run(self, value, search, replace, regex=False):
        if regex:
            value = re.sub(search, replace, to_string(value))
        else:
            value = to_string(value).replace(search, replace)
        return (value,)

class JN_FirstActive:
    CATEGORY = CATEGORY_PRIMITIVE_PROCESS
    RETURN_TYPES = ("*",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "merge_array_items": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "values": ("*", {"multiple": True}),
            },
        }

    def run(self, merge_array_items, values=[]):
        first_active = None

        if merge_array_items:
            values = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), values)

        for value in values:
            if value is not None:
                first_active = value
                break

        return (first_active,)

class JN_SelectItem:
    CATEGORY = CATEGORY_PRIMITIVE_PROCESS
    RETURN_TYPES = ("*",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "only_active": ("BOOLEAN", {"default": True}),
                "merge_array_items": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "values": ("*", {"multiple": True}),
            },
        }

    def run(self, index, only_active, merge_array_items, values=[]):
        selected = None

        if merge_array_items:
            values = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), values)

        if only_active:
            values = [value for value in values if value is not None]

        if len(values) > 0:
            index = index % len(values)
            selected = values[index]

        return (selected,)

class JN_SliceOperation:
    CATEGORY = CATEGORY_PRIMITIVE_PROCESS
    RETURN_TYPES = ("*",)
    FUNCTION = "run"

    OPERATIONS = ["[a:b]", "[a:a+b]", "[a-b:a]", "[a-b:a+b]", "[a:]", "[:a]"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
                "b": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
                "operation": (s.OPERATIONS,),
                "only_active": ("BOOLEAN", {"default": True}),
                "merge_array_items": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "values": ("*", {"multiple": True}),
            },
        }

    def run(self, a, b, operation, only_active, merge_array_items, values=[]):
        if merge_array_items:
            values = reduce(lambda a, b: (a if isinstance(a, list) else [a]) + (b if isinstance(b, list) else [b]), values)

        if only_active:
            values = [value for value in values if value is not None]

        values = slice_operation(values, a, b, operation)

        return (values,)

NODE_CLASS_MAPPINGS = {
    "JN_MathOperation": JN_MathOperation,
    "JN_MathOperationArray": JN_MathOperationArray,
    "JN_BooleanOperation": JN_BooleanOperation,
    "JN_LogicOperation": JN_LogicOperation,
    "JN_SliceOperation": JN_SliceOperation,
    "JN_TextConcatenation": JN_TextConcatenation,
    "JN_TextReplace": JN_TextReplace,
    "JN_FirstActive": JN_FirstActive,
    "JN_SelectItem": JN_SelectItem,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JN_MathOperation": "Math Operation",
    "JN_MathOperationArray": "Math Operation Array",
    "JN_BooleanOperation": "Boolean Operation",
    "JN_LogicOperation": "Logic Operation",
    "JN_SliceOperation": "Slice Operation",
    "JN_TextConcatenation": "Text Concatenation",
    "JN_TextReplace": "Text Replace",
    "JN_FirstActive": "First Active",
    "JN_SelectItem": "Select Item",
}
