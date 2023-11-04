import time
import psutil
import GPUtil
import os, platform, subprocess, re
from datetime import datetime, timedelta
from comfy.utils import ProgressBar

from ..utils import CATEGORY_OTHER

def clear_line(n=1):
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for i in range(n):
        print(LINE_UP, end=LINE_CLEAR)

def func_sleep(seconds, pbar = None):
    while seconds > 0:
        print(f"Sleeping {seconds} seconds")
        time.sleep(1)
        seconds -= 1
        clear_line()
        if pbar is not None:
            pbar.update(1)

def get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command ="sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1).strip()
    return ""

def get_temperatures():
    temperatures = []

    if platform.system() == "Linux":
        cpu_max_temp = 0

        for k, v in psutil.sensors_temperatures(fahrenheit=False).items():
            for t in v:
                if t.current > cpu_max_temp:
                    cpu_max_temp = t.current

        temperatures.append({
            "label": get_processor_name(),
            "temperature": cpu_max_temp,
            "kind": "CPU",
        })

    for gpu in GPUtil.getGPUs():
        temperatures.append({
            "label": gpu.name,
            "temperature": gpu.temperature,
            "kind": "GPU",
        })

    return temperatures

def wait_cooldown(max_temperature=70, seconds=1, max_seconds=300):
    seconds = max(1, seconds)
    max_seconds = max(seconds, max_seconds)
    times = max_seconds // seconds

    hot = True

    while hot and times > 0:
        temperatures = [f"{t['kind']} {t['label']}: {t['temperature']}" for t in get_temperatures() if t["temperature"] > max_temperature]
        hot = len(temperatures) > 0

        if hot:
            print(f"Too hot! Max temperature: [ {max_temperature} ] Current temperature: [ " + " | ".join(temperatures) + " ]")
            pbar = ProgressBar(seconds)
            func_sleep(seconds, pbar)
            clear_line()
            times -= 1

class JN_CoolDown:
    CATEGORY = CATEGORY_OTHER
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("flow",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "flow": ("*",),
                "dependency": ("*", {"multiple": True}),
            },
            "required": {
                "max_temperature": ("INT", {"default": 70, "min": 0, "max": 0xffffffffffffffff}),
                "seconds": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
                "max_seconds": ("INT", {"default": 300, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    def run(self, max_temperature=70, seconds=1, max_seconds=300, flow=None, dependency=None):
        if max_temperature > 0:
            wait_cooldown(max_temperature, seconds, max_seconds)
        return (flow,)

class JN_CoolDownOutput:
    CATEGORY = CATEGORY_OTHER
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "dependency": ("*", {"multiple": True}),
            },
            "required": {
                "max_temperature": ("INT", {"default": 70, "min": 0, "max": 0xffffffffffffffff}),
                "seconds": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
                "max_seconds": ("INT", {"default": 300, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    def run(self, max_temperature=70, seconds=1, max_seconds=300, dependency=None):
        if max_temperature > 0:
            wait_cooldown(max_temperature, seconds, max_seconds)
        return {"ui": {}}

class JN_Sleep:
    CATEGORY = CATEGORY_OTHER
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("flow",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "flow": ("*",),
                "dependency": ("*", {"multiple": True}),
            },
            "required": {
                "seconds": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "every": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
                "count": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    def run(self, seconds, every=1, count=0, flow=None, dependency=None):
        every = max(1, every)
        if count % every == 0:
            pbar = ProgressBar(seconds)
            func_sleep(seconds, pbar)
        return (flow,)

class JN_SleepOutput:
    CATEGORY = CATEGORY_OTHER
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "dependency": ("*", {"multiple": True}),
            },
            "required": {
                "seconds": ("INT", {"default": 0, "min": 0, "max": 60 * 60}),
                "every": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
                "count": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    def run(self, seconds, every=1, count=0, dependency=None):
        every = max(1, every)
        if count % every == 0:
            pbar = ProgressBar(seconds)
            func_sleep(seconds, pbar)
        return {"ui": {}}

class JN_Dump:
    CATEGORY = CATEGORY_OTHER
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("flow",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "title": ("STRING", {"default": "", "dynamicPrompts": False}),
                "flow": ("*",),
                "dependency": ("*", {"multiple": True}),
            },
            "required": {
                "mode_repr": ("BOOLEAN", {"default": False}),
            },
        }

    def run(self, title="", flow=None, dependency=None, mode_repr=True):
        print("JN_Dump", title)

        print("flow:")
        print(repr(flow) if mode_repr else flow)

        print("dependency:")
        if isinstance(dependency, list):
            for dep in dependency:
                print(repr(dep) if mode_repr else dep)
        else:
            print(repr(dependency) if mode_repr else dependency)

        return (flow,)

class JN_DumpOutput:
    CATEGORY = CATEGORY_OTHER
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "title": ("STRING", {"default": "", "dynamicPrompts": False}),
                "dependency": ("*", {"multiple": True}),
            },
            "required": {
                "mode_repr": ("BOOLEAN", {"default": False}),
            },
        }

    def run(self, title="", dependency=None, mode_repr=True):
        print("JN_DumpOutput", title)

        print("dependency:")
        if isinstance(dependency, list):
            for dep in dependency:
                print(repr(dep) if mode_repr else dep)
        else:
            print(repr(dependency) if mode_repr else dependency)

        return {"ui": {}}

class JN_DatetimeNow:
    CATEGORY = CATEGORY_OTHER
    RETURN_TYPES = ("*", "DATETIME")
    RETURN_NAMES = ("flow", "DATETIME")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "title": ("STRING", {"default": "", "dynamicPrompts": False}),
                "flow": ("*",),
                "dependency": ("*", {"multiple": True}),
            },
            "required": {
            },
        }

    def run(self, title="", flow=None, dependency=None):
        now = datetime.now()
        print("JN_DatetimeNow", now, title)
        return (flow, now)

class JN_DatetimeInfo:
    CATEGORY = CATEGORY_OTHER
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("year", "month", "day", "hour", "minute", "second", "microsecond")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("DATETIME",),
            },
        }

    def run(self, value):
        year = value.year
        month = value.month
        day = value.day
        hour = value.hour
        minute = value.minute
        second = value.second
        microsecond = value.microsecond
        return (year, month, day, hour, minute, second, microsecond)

class JN_DatetimeFormat:
    CATEGORY = CATEGORY_OTHER
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("DATETIME",),
                "str_format": ("STRING", {"default": "", "dynamicPrompts": False}),
            },
        }

    def run(self, value, str_format):
        formatted = datetime.strftime(value, str_format)
        return (formatted,)

class JN_TimedeltaInfo:
    CATEGORY = CATEGORY_OTHER
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("days", "hours", "minutes", "seconds", "microseconds", "total_seconds")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("TIMEDELTA",),
            },
        }

    def run(self, value):
        # Extract days, seconds, and microseconds
        days = value.days
        seconds = value.seconds
        microseconds = value.microseconds

        # Calculate hours, minutes, and remaining seconds
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)

        total_seconds = value.total_seconds()

        return (days, hours, minutes, seconds, microseconds, total_seconds)

class JN_TimedeltaFormat:
    CATEGORY = CATEGORY_OTHER
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("TIMEDELTA",),
                "str_format": ("STRING", {"default": "", "dynamicPrompts": False}),
            },
        }

    def run(self, value, str_format):
        formatted = self._format_timedelta(value, str_format)
        return (formatted,)

    def _format_timedelta(self, td, format_str):
        # Extract days, seconds, and microseconds
        days, seconds = td.days, td.seconds
        microseconds = td.microseconds

        # Calculate hours, minutes, and remaining seconds
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)

        # Replace placeholders in the format string
        formatted_str = format_str.replace('%d', str(days))
        formatted_str = formatted_str.replace('%H', f'{hours:02}')
        formatted_str = formatted_str.replace('%M', f'{minutes:02}')
        formatted_str = formatted_str.replace('%S', f'{seconds:02}')
        formatted_str = formatted_str.replace('%f', f'{microseconds:06}')

        return formatted_str

class JN_StopIf:
    CATEGORY = CATEGORY_OTHER
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("flow",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "condition": ("BOOLEAN", {"default": False}),
                "title": ("STRING", {"default": "", "dynamicPrompts": False}),
                "flow": ("*",),
                "dependency": ("*", {"multiple": True}),
            },
            "required": {
            },
        }

    def run(self, condition=False, title="", flow=None, dependency=None):
        if condition:
            raise Exception(title)

        return (flow,)

class JN_StopIfOutput:
    CATEGORY = CATEGORY_OTHER
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "condition": ("BOOLEAN", {"default": False}),
                "title": ("STRING", {"default": "", "dynamicPrompts": False}),
                "dependency": ("*", {"multiple": True}),
            },
            "required": {
            },
        }

    def run(self, condition=False, title="", dependency=None):
        if condition:
            raise Exception(title)

        return {"ui": {}}

NODE_CLASS_MAPPINGS = {
    "JN_CoolDown": JN_CoolDown,
    "JN_CoolDownOutput": JN_CoolDownOutput,
    "JN_Sleep": JN_Sleep,
    "JN_SleepOutput": JN_SleepOutput,
    "JN_Dump": JN_Dump,
    "JN_DumpOutput": JN_DumpOutput,
    "JN_StopIf": JN_StopIf,
    "JN_StopIfOutput": JN_StopIfOutput,
    "JN_DatetimeNow": JN_DatetimeNow,
    "JN_DatetimeInfo": JN_DatetimeInfo,
    "JN_DatetimeFormat": JN_DatetimeFormat,
    "JN_TimedeltaInfo": JN_TimedeltaInfo,
    "JN_TimedeltaFormat": JN_TimedeltaFormat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JN_CoolDown": "Cool Down",
    "JN_CoolDownOutput": "Cool Down Output",
    "JN_Sleep": "Sleep",
    "JN_SleepOutput": "Sleep Output",
    "JN_Dump": "Dump",
    "JN_DumpOutput": "Dump Output",
    "JN_StopIf": "Stop If",
    "JN_StopIfOutput": "Stop If Output",
    "JN_DatetimeNow": "Datetime Now",
    "JN_DatetimeInfo": "Datetime Info",
    "JN_DatetimeFormat": "Datetime Format",
    "JN_TimedeltaInfo": "Timedelta Info",
    "JN_TimedeltaFormat": "Timedelta Format",
}
