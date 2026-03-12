import os
import re
from typing import Iterable, Optional


DEFAULT_OBJECT_MODEL_PATH = "assets/objects/basketball.urdf"

DEFAULT_ANCHOR_LINKS_SMPLX = ["left_index3", "right_index3"] # ["left_wrist", "right_wrist"], ["left_index3", "right_index3"]
DEFAULT_CONTACT_LINKS = ["left_hand_middle_0_link", "right_hand_middle_0_link"] # ["left_wrist_yaw_link", "right_wrist_yaw_link"], ["left_hand_middle_0_link", "right_hand_middle_0_link"]

OBJECT_MOTION_DEFAULTS = {
    "basketball": {
        "min_object_height": 0.16,
        "anchor_links_smplx": DEFAULT_ANCHOR_LINKS_SMPLX,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "clothesstand": {
        "min_object_height": 0.34,
        "anchor_links_smplx": DEFAULT_ANCHOR_LINKS_SMPLX,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "floorlamp": {
        "min_object_height": 0.34,
        "anchor_links_smplx": DEFAULT_ANCHOR_LINKS_SMPLX,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "largebox": {
        "min_object_height": 0.30,
        "anchor_links_smplx": DEFAULT_ANCHOR_LINKS_SMPLX,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "largetable": {
        "min_object_height": 0.43,
        "anchor_links_smplx": DEFAULT_ANCHOR_LINKS_SMPLX,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "monitor": {
        "min_object_height": 0.17,
        "anchor_links_smplx": DEFAULT_ANCHOR_LINKS_SMPLX,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "mop": {
        "min_object_height": 0.51,
        "anchor_links_smplx": DEFAULT_ANCHOR_LINKS_SMPLX,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "plasticbox": {
        "min_object_height": 0.27,
        "anchor_links_smplx": DEFAULT_ANCHOR_LINKS_SMPLX,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "smallbox": {
        "min_object_height": 0.17,
        "anchor_links_smplx": DEFAULT_ANCHOR_LINKS_SMPLX,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "smalltable": {
        "min_object_height": 0.222,
        "anchor_links_smplx": DEFAULT_ANCHOR_LINKS_SMPLX,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "suitcase": {
        "min_object_height": 0.32,
        "anchor_links_smplx": DEFAULT_ANCHOR_LINKS_SMPLX,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "trashcan": {
        "min_object_height": 0.205,
        "anchor_links_smplx": DEFAULT_ANCHOR_LINKS_SMPLX,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "tripod": {
        "min_object_height": 0.29,
        "anchor_links_smplx": DEFAULT_ANCHOR_LINKS_SMPLX,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "vacuum": {
        "min_object_height": 0.37,
        "anchor_links_smplx": DEFAULT_ANCHOR_LINKS_SMPLX,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "whitechair": {
        "min_object_height": 0.60,
        "anchor_links_smplx": DEFAULT_ANCHOR_LINKS_SMPLX,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "woodchair": {
        "min_object_height": 0.50,
        "anchor_links_smplx": DEFAULT_ANCHOR_LINKS_SMPLX,
        "contact_links": DEFAULT_CONTACT_LINKS,
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
    cfg = OBJECT_MOTION_DEFAULTS[model_name]
    return {
        "min_object_height": float(cfg["min_object_height"]),
        "anchor_links_smplx": list(cfg["anchor_links_smplx"]),
        "contact_links": list(cfg["contact_links"]),
    }
