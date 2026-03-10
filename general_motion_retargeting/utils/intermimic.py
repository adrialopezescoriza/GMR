import numpy as np
import torch
from smplx.joint_names import JOINT_NAMES

from general_motion_retargeting.rot_utils import (
    normalize_quat_xyzw_np,
)

INTERMIMIC_RAW_SLICE = {
    "root_pos": slice(0, 3),
    "root_rot_xyzw": slice(3, 7),
    "dof_pos": slice(9, 9 + 153),
    "body_pos": slice(162, 162 + 52 * 3),
    "obj_pos": slice(318, 321),
    "obj_rot_xyzw": slice(321, 325),
    "contact_obj": slice(330, 331),
    "contact_human": slice(331, 331 + 52),
    "body_rot_xyzw": slice(331 + 52, 331 + 52 + 52 * 4),
}

INTERMIMIC_OMOMO_BODY_NAMES = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Index1",
    "L_Index2",
    "L_Index3",
    "L_Middle1",
    "L_Middle2",
    "L_Middle3",
    "L_Pinky1",
    "L_Pinky2",
    "L_Pinky3",
    "L_Ring1",
    "L_Ring2",
    "L_Ring3",
    "L_Thumb1",
    "L_Thumb2",
    "L_Thumb3",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Index1",
    "R_Index2",
    "R_Index3",
    "R_Middle1",
    "R_Middle2",
    "R_Middle3",
    "R_Pinky1",
    "R_Pinky2",
    "R_Pinky3",
    "R_Ring1",
    "R_Ring2",
    "R_Ring3",
    "R_Thumb1",
    "R_Thumb2",
    "R_Thumb3",
]


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def load_intermimic_hoi_2d(input_path: str) -> np.ndarray:
    """
    Load a single InterMimic raw HOI motion as a 2D array [num_frames, 591].
    Supports either a direct tensor payload or a dict containing `hoi_data`.
    """
    payload = torch.load(input_path, map_location="cpu")
    hoi_data = payload.get("hoi_data") if isinstance(payload, dict) else payload
    if hoi_data is None:
        raise ValueError(f"Could not find InterMimic `hoi_data` in {input_path}.")

    hoi_np = _to_numpy(hoi_data).astype(np.float32)[1:]  # Skip the first frame which is often all zeros.

    if hoi_np.ndim == 3:
        lengths = payload.get("_motion_lengths") if isinstance(payload, dict) else None
        if lengths is not None:
            lengths_np = _to_numpy(lengths).astype(np.int64).reshape(-1)
            valid_len = int(lengths_np[0]) if lengths_np.size > 0 else int(hoi_np.shape[1])
        else:
            valid_len = int(hoi_np.shape[1])
        hoi_np = hoi_np[0, :valid_len]

    if hoi_np.ndim != 2:
        raise ValueError(
            f"InterMimic payload must be 2D [T, D] after loading. Got shape {hoi_np.shape}."
        )

    expected_dim = INTERMIMIC_RAW_SLICE["body_rot_xyzw"].stop
    if hoi_np.shape[1] != expected_dim:
        raise ValueError(
            f"InterMimic raw HOI must have {expected_dim} features, got {hoi_np.shape[1]} "
            f"from {input_path}."
        )

    return hoi_np


def parse_intermimic_hoi(hoi_2d: np.ndarray):
    num_frames = hoi_2d.shape[0]
    body_pos = hoi_2d[:, INTERMIMIC_RAW_SLICE["body_pos"]].reshape(num_frames, 52, 3).astype(np.float32)
    body_rot_xyzw = hoi_2d[:, INTERMIMIC_RAW_SLICE["body_rot_xyzw"]].reshape(num_frames, 52, 4).astype(np.float32)
    body_rot_xyzw = normalize_quat_xyzw_np(body_rot_xyzw).astype(np.float32)
    root_pos = hoi_2d[:, INTERMIMIC_RAW_SLICE["root_pos"]].astype(np.float32)
    root_rot_xyzw = hoi_2d[:, INTERMIMIC_RAW_SLICE["root_rot_xyzw"]].astype(np.float32)
    root_rot_xyzw = normalize_quat_xyzw_np(root_rot_xyzw).astype(np.float32)
    dof_pos = hoi_2d[:, INTERMIMIC_RAW_SLICE["dof_pos"]].astype(np.float32)
    obj_pos = hoi_2d[:, INTERMIMIC_RAW_SLICE["obj_pos"]].astype(np.float32)
    obj_rot_xyzw = hoi_2d[:, INTERMIMIC_RAW_SLICE["obj_rot_xyzw"]].astype(np.float32)
    obj_rot_xyzw = normalize_quat_xyzw_np(obj_rot_xyzw).astype(np.float32)
    contact_obj = np.round(hoi_2d[:, INTERMIMIC_RAW_SLICE["contact_obj"]]).astype(np.int32).reshape(-1, 1)
    contact_human = np.round(hoi_2d[:, INTERMIMIC_RAW_SLICE["contact_human"]]).astype(np.int32)

    return {
        "root_pos": root_pos,
        "root_rot_xyzw": root_rot_xyzw,
        "dof_pos": dof_pos,
        "body_pos": body_pos,
        "body_rot_xyzw": body_rot_xyzw,
        "obj_pos": obj_pos,
        "obj_rot_xyzw": obj_rot_xyzw,
        "contact_obj": contact_obj,
        "contact_human": contact_human,
    }


def smplx_body_to_omomo_body_name(smplx_name: str):
    if smplx_name == "pelvis":
        return "Pelvis"
    if smplx_name == "spine1":
        return "Torso"
    if smplx_name == "spine2":
        return "Spine"
    if smplx_name == "spine3":
        return "Chest"
    if smplx_name == "neck":
        return "Neck"
    if smplx_name == "head":
        return "Head"

    side, part = smplx_name.split("_", 1)
    side_prefix = "L_" if side == "left" else "R_" if side == "right" else None
    if side_prefix is None:
        return None

    if part == "foot":
        part_suffix = "Toe"
    elif part == "collar":
        part_suffix = "Thorax"
    else:
        part_suffix = part.capitalize()

    return f"{side_prefix}{part_suffix}"


def build_intermimic_pose_body(parsed) -> np.ndarray:
    # Build SMPL-X body pose (21*3 axis-angle) from InterMimic local dof_pos (51*3).
    # InterMimic dof_pos excludes pelvis and follows INTERMIMIC_OMOMO_BODY_NAMES[1:].
    smplx_pose_body = np.zeros((parsed["dof_pos"].shape[0], 21 * 3), dtype=np.float32)
    for i, smplx_joint in enumerate(JOINT_NAMES[1:22]):
        omomo_joint = smplx_body_to_omomo_body_name(smplx_joint)
        if omomo_joint is None:
            raise ValueError(f"Could not find corresponding InterMimic joint for SMPLX joint {smplx_joint}.")
        if omomo_joint not in INTERMIMIC_OMOMO_BODY_NAMES:
            raise ValueError(
                f"InterMimic joint {omomo_joint} corresponding to SMPLX joint {smplx_joint} not found in InterMimic data."
            )
        omomo_idx = INTERMIMIC_OMOMO_BODY_NAMES.index(omomo_joint)
        if omomo_idx == 0:
            raise ValueError("Pelvis is not part of SMPL-X pose_body and has no dof_pos entry.")
        dof_idx = omomo_idx - 1
        smplx_pose_body[:, i * 3 : (i + 1) * 3] = parsed["dof_pos"][:, dof_idx * 3 : (dof_idx + 1) * 3][:, [1, 2, 0]]
    return smplx_pose_body
