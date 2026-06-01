from importlib import import_module

from sqlalchemy.orm import configure_mappers

MODEL_MODULES = (
    "modules.users.models",
    "modules.core.groups.models",
    "modules.core.standards.models",
    "modules.core.segments.models",
    "modules.tasks.models",
    "modules.cameras.models",
    "modules.yolo.training.models",
    "modules.yolo.inspection.models",
)


def import_models() -> None:
    for module_path in MODEL_MODULES:
        import_module(module_path)

    configure_mappers()
