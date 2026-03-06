from __future__ import annotations

import os
import pathlib
import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from .dynamic_object_retargeting import optimize_object_traj_from_motion
from .kinematics_model import KinematicsModel
from .motion_retarget import GeneralMotionRetargeting
from .rot_utils import pad_or_truncate
from .utils.smpl import (
    convert_intermimic_to_smplx,
    convert_skillmimic_to_smplx,
    get_smplx_data_offline_fast,
    load_smplx_data,
)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _projection_keys_present(payload: Dict[str, Any]) -> bool:
    required = {
        "fps",
        "root_pos",
        "root_rot",
        "dof_pos",
        "world_body_pos",
        "world_body_orient",
        "local_body_pos",
        "local_body_orient",
        "link_body_list",
        "dof_names",
        "object_pos",
        "object_rot",
    }
    return isinstance(payload, dict) and required.issubset(set(payload.keys()))


def _convert_projection_dict_to_numpy(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in payload.items():
        out[key] = _to_numpy(value) if isinstance(value, torch.Tensor) else value
    return out


def _load_projection_motion_data(input_path: str) -> Optional[Dict[str, Any]]:
    ext = os.path.splitext(input_path)[1].lower()

    if ext in (".pkl", ".pickle"):
        try:
            with open(input_path, "rb") as f:
                payload = pickle.load(f)
            if _projection_keys_present(payload):
                return _convert_projection_dict_to_numpy(payload)
        except Exception:
            pass

    if ext in (".pt", ".pth"):
        try:
            payload = torch.load(input_path, map_location="cpu")
            if _projection_keys_present(payload):
                return _convert_projection_dict_to_numpy(payload)
        except Exception:
            pass

    try:
        payload = np.load(input_path, allow_pickle=True)
        if isinstance(payload, np.ndarray) and payload.shape == ():
            payload = payload.item()
        if _projection_keys_present(payload):
            return _convert_projection_dict_to_numpy(payload)
    except Exception:
        pass

    return None


def _extract_source_data(
    input_path: str,
    gender: str,
    tgt_fps: int,
    min_object_height: float,
) -> Tuple[Dict[str, Any], str]:
    smplx_folder = pathlib.Path(__file__).parent / ".." / "assets" / "body_models"

    if "intermimic" in input_path.lower():
        smplx_data = convert_intermimic_to_smplx(input_path, gender)
        source_name = "intermimic_omomo_gmr"
    elif "skillmimic" in input_path.lower():
        smplx_data = convert_skillmimic_to_smplx(input_path, gender)
        source_name = "skillmimic_gmr"
    else:
        raise ValueError(f"Unrecognized input path for GMR projection: {input_path}")

    body_model, smplx_output, actual_human_height, obj_data = load_smplx_data(
        smplx_data, smplx_folder
    )
    smplx_frames, aligned_fps, object_frames = get_smplx_data_offline_fast(
        smplx_data,
        body_model,
        smplx_output,
        tgt_fps=tgt_fps,
        object_data=obj_data,
    )

    if object_frames is None:
        raise ValueError("Input data has no object trajectory.")

    object_pos = np.asarray([frame[0] for frame in object_frames], dtype=np.float32)
    object_rot = np.asarray([frame[1] for frame in object_frames], dtype=np.float32)
    object_contact = np.asarray([frame[2] for frame in object_frames], dtype=np.float32).reshape(
        -1, 1
    )

    return (
        {
            "smplx_frames": smplx_frames,
            "fps": float(aligned_fps),
            "actual_human_height": actual_human_height,
            "object_pos": object_pos,
            "object_rot": object_rot,
            "object_contact": object_contact,
            "object_ground_contact": (object_pos[:, 2:3] <= min_object_height).astype(
                np.int8
            ),
        },
        source_name,
    )


def _build_motion_data_from_source_frames(
    *,
    smplx_frames: list,
    fps: float,
    object_pos: np.ndarray,
    object_rot_wxyz: np.ndarray,
    object_contact: np.ndarray,
    object_ground_contact: np.ndarray,
    robot: str,
    actual_human_height: Optional[float],
    device: Optional[str],
    contact_points_o_local: Dict[str, np.ndarray],
    object_speed_thresh: float,
) -> Dict[str, Any]:
    if not smplx_frames:
        raise ValueError("Empty motion sequence.")

    retarget = GeneralMotionRetargeting(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=robot,
    )
    required_joint_keys = (
        set(retarget.human_body_to_task1.keys())
        | set(retarget.human_body_to_task2.keys())
        | {retarget.human_root_name}
    )
    missing = sorted(k for k in required_joint_keys if k not in smplx_frames[0])
    if missing:
        raise ValueError(
            "Input motion is missing SMPL-X joints required by IK config: "
            + ", ".join(missing)
        )

    qpos = np.asarray([retarget.retarget(frame) for frame in smplx_frames], dtype=np.float32)
    root_pos = qpos[:, :3]
    root_rot = qpos[:, 3:7]
    dof_pos = qpos[:, 7:]

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    kinematics = KinematicsModel(retarget.xml_file, device=device)
    world_body_pos, world_body_orient = kinematics.forward_kinematics(
        root_pos=torch.from_numpy(root_pos).to(device=device, dtype=torch.float32),
        root_rot=torch.from_numpy(root_rot)[..., [1, 2, 3, 0]].to(
            device=device, dtype=torch.float32
        ),
        dof_pos=torch.from_numpy(dof_pos).to(device=device, dtype=torch.float32),
    )
    local_body_pos, local_body_orient = kinematics.forward_kinematics(
        root_pos=torch.zeros(root_pos.shape).to(device=device, dtype=torch.float32),
        root_rot=(
            torch.zeros(root_rot.shape).to(device=device, dtype=torch.float32)
            + torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float32)
        ),
        dof_pos=torch.from_numpy(dof_pos).to(device=device, dtype=torch.float32),
    )

    num_frames = root_pos.shape[0]
    object_pos = pad_or_truncate(np.asarray(object_pos, dtype=np.float32), num_frames)
    object_rot_wxyz = pad_or_truncate(
        np.asarray(object_rot_wxyz, dtype=np.float32), num_frames
    )
    object_contact = pad_or_truncate(
        np.asarray(object_contact, dtype=np.float32).reshape(-1, 1), num_frames
    )
    object_ground_contact = pad_or_truncate(
        np.asarray(object_ground_contact, dtype=np.float32).reshape(-1, 1), num_frames
    )

    motion_data = {
        "fps": float(fps),
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "world_body_pos": world_body_pos.cpu().numpy(),
        "world_body_orient": world_body_orient[..., [3, 0, 1, 2]].cpu().numpy(),
        "local_body_pos": local_body_pos.cpu().numpy(),
        "local_body_orient": local_body_orient[..., [3, 0, 1, 2]].cpu().numpy(),
        "link_body_list": kinematics.body_names,
        "dof_names": retarget.robot_motor_names,
        "object_pos": object_pos,
        "object_rot": object_rot_wxyz,
        "contact_sequence": np.round(object_contact).astype(np.int8),
        "ground_contact_sequence": np.round(object_ground_contact).astype(np.int8),
    }

    if not contact_points_o_local:
        return motion_data

    try:
        optimized_object_pos, optimized_object_vel = optimize_object_traj_from_motion(
            motion_data=motion_data,
            contact_link_names=list(contact_points_o_local.keys()),
            local_offsets=[
                contact_points_o_local[name] for name in contact_points_o_local.keys()
            ],
            speed_thresh=object_speed_thresh,
        )
    except Exception as exc:
        print(f"[WARN] Failed to retarget object trajectory: {exc}")
        return motion_data

    motion_data = dict(motion_data)
    motion_data["object_pos"] = optimized_object_pos
    motion_data["object_vel"] = optimized_object_vel
    return motion_data


def build_projection_motion_data_with_gmr(
    *,
    input_path: str,
    robot: str,
    gender: str,
    tgt_fps: int = 60,
    device: str,
    min_object_height: float,
    contact_points_o_local: Optional[Dict[str, np.ndarray]],
    object_speed_thresh: float,
) -> Tuple[Dict[str, Any], str]:

    source_data, source_name = _extract_source_data(
        input_path=input_path,
        gender=gender,
        tgt_fps=tgt_fps,
        min_object_height=min_object_height,
    )

    motion_data = _build_motion_data_from_source_frames(
        smplx_frames=source_data["smplx_frames"],
        fps=source_data["fps"],
        object_pos=source_data["object_pos"],
        object_rot_wxyz=source_data["object_rot"],
        object_contact=source_data["object_contact"],
        object_ground_contact=source_data["object_ground_contact"],
        robot=robot,
        actual_human_height=source_data["actual_human_height"],
        device=device,
        contact_points_o_local=contact_points_o_local or {},
        object_speed_thresh=object_speed_thresh,
    )
    return motion_data, source_name
