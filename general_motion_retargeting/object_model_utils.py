import os
import re
from typing import Iterable, Optional


DEFAULT_OBJECT_MODEL_PATH = "assets/objects/basketball.urdf"

OBJECT_MOTION_DEFAULTS = {
    "basketball": {
        "min_object_height": 0.13,
        "contact_links": [
            "left_hand_middle_0_link",
            "right_hand_middle_0_link",
        ],
    },
    "largebox": {
        "min_object_height": 0.5,
        "contact_links": [
            "left_hand_middle_0_link",
            "right_hand_middle_0_link",
        ],
    },
}


def _name_candidates(folder_name: str) -> Iterable[str]:
    base = folder_name.strip()
    if not base:
        return

    lowered = base.lower()
    variants = {
        base,
        lowered,
        lowered.replace("-", "_"),
        lowered.replace(" ", "_"),
    }

    # Remove trailing numeric suffixes such as "_001" or "-2".
    stripped = re.sub(r"([_-]\d+)$", "", lowered)
    if stripped:
        variants.add(stripped)

    for v in variants:
        if v:
            yield v


def infer_object_model_path_from_motion_path(
    motion_path: str,
    objects_root: str = "assets/objects",
    default_object_model_path: str = DEFAULT_OBJECT_MODEL_PATH,
) -> str:
    """
    Infer object URDF from the folder hierarchy of `motion_path`.

    Example:
      /.../intermimic/largebox/foo.pt -> assets/objects/largebox.urdf (if exists)

    If nothing matches, returns `default_object_model_path`.
    """
    norm_path = os.path.abspath(motion_path)
    curr = os.path.dirname(norm_path)
    root = os.path.abspath(os.path.sep)

    while curr and curr != root:
        folder = os.path.basename(curr)
        for name in _name_candidates(folder):
            candidate = os.path.join(objects_root, f"{name}.urdf")
            if os.path.isfile(candidate):
                return candidate
        curr = os.path.dirname(curr)

    return default_object_model_path


def resolve_object_model_path(
    motion_path: Optional[str],
    configured_object_model_path: Optional[str],
    auto_from_motion_path: bool = True,
    default_object_model_path: str = DEFAULT_OBJECT_MODEL_PATH,
) -> str:
    """
    Resolve final object model path:
      1. If auto mode and motion path is known, infer from folder names.
      2. Otherwise use configured path if provided.
      3. Fallback to basketball default.
    """
    if auto_from_motion_path and motion_path:
        inferred = infer_object_model_path_from_motion_path(
            motion_path=motion_path,
            default_object_model_path=default_object_model_path,
        )
        if inferred:
            return inferred

    if configured_object_model_path:
        return configured_object_model_path

    return default_object_model_path


def get_object_motion_defaults(object_model_path: str) -> dict:
    model_name = os.path.splitext(os.path.basename(object_model_path))[0].lower()
    cfg = OBJECT_MOTION_DEFAULTS.get(model_name, OBJECT_MOTION_DEFAULTS["basketball"])
    return {
        "min_object_height": float(cfg["min_object_height"]),
        "contact_links": list(cfg["contact_links"]),
    }
