import os
import importlib
import traceback
from .log import logger

MODULE_PATH = os.path.dirname(os.path.abspath(__file__))

try:
    from spandrel_extra_arches import EXTRA_REGISTRY
    from spandrel import MAIN_REGISTRY
    MAIN_REGISTRY.add(*EXTRA_REGISTRY)
    logger.info("Successfully imported spandrel_extra_arches: support for non commercial models.")
except:
    pass

def load_nodes():
    error_messages = []
    node_class_mappings = {}
    node_display_name_mappings = {}

    for filename in sorted(os.listdir(os.path.join(MODULE_PATH, "nodes"))):
        module_name, extension = os.path.splitext(filename)

        if extension not in ["", ".py"] or module_name.startswith("__"):
            continue

        try:
            module = importlib.import_module(
                f".nodes.{module_name}", package=__package__
            )

            if hasattr(module, "NODE_CLASS_MAPPINGS"):
                node_class_mappings.update(getattr(module, "NODE_CLASS_MAPPINGS"))

            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                node_display_name_mappings.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS"))

            logger.debug(f"Imported '{module_name}' nodes")

        except Exception:
            error_message = traceback.format_exc()
            error_messages.append(f"Failed to import module '{module_name}' because {error_message}")

    if len(error_messages) > 0:
        logger.warning(
            f"Some nodes failed to load:\n\n"
            + "\n".join(error_messages)
        )

    return node_class_mappings, node_display_name_mappings

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = load_nodes()
