"""Specialist lookup by task. Imports are lazy so one broken specialist cannot take down the API."""

from __future__ import annotations

import importlib

from app.schemas.common import TaskType
from app.specialists.base import Specialist

_IMPL = {
    TaskType.CAPTIONING: ("app.specialists.captioning.specialist", "CaptioningSpecialist"),
    TaskType.GROUNDING: ("app.specialists.grounding.specialist", "GroundingSpecialist"),
    TaskType.VQA: ("app.specialists.vqa.specialist", "VQASpecialist"),
    TaskType.CHANGE_ANALYSIS: ("app.specialists.change.specialist", "ChangeSpecialist"),
    TaskType.OPTICAL_SAR_FUSION: ("app.specialists.fusion.specialist", "FusionSpecialist"),
}
_CACHE: dict[TaskType, Specialist] = {}


def get_specialist(task: TaskType) -> Specialist | None:
    if task in _CACHE:
        return _CACHE[task]
    module_name, cls_name = _IMPL[task]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return None  # not implemented yet
        raise
    _CACHE[task] = getattr(module, cls_name)()
    return _CACHE[task]
