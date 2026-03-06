import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from smplx.joint_names import JOINT_NAMES
from general_motion_retargeting.utils.intermimic import (
    build_intermimic_pose_body,
    load_intermimic_hoi_2d,
    parse_intermimic_hoi,
)

import general_motion_retargeting.utils.lafan_vendor.utils as utils

GYM_TO_SMPL_IDX = torch.tensor([
    0, 1, 5, 10, 2, 6, 11, 3, 7, 12, 4, 8, 13, 15, 34, 14,
    16, 35, 17, 36, 18, 37, 9, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 38, 39, 40, 41, 42, 43,
    44, 45, 46, 47, 48, 49, 50, 51, 52
])

def load_smpl_file(smpl_file):
    smpl_data = np.load(smpl_file, allow_pickle=True)
    return smpl_data

def load_smplx_file(smplx_file, smplx_body_model_path):
    smplx_data = np.load(smplx_file, allow_pickle=True)
    body_model = smplx.create(
        smplx_body_model_path,
        "smplx",
        gender=str(smplx_data["gender"]),
        use_pca=False,
    )
    # print(smplx_data["pose_body"].shape)
    # print(smplx_data["betas"].shape)
    # print(smplx_data["root_orient"].shape)
    # print(smplx_data["trans"].shape)
    
    num_frames = smplx_data["pose_body"].shape[0]
    smplx_output = body_model(
        betas=torch.tensor(smplx_data["betas"]).float().view(1, -1), # (16,)
        global_orient=torch.tensor(smplx_data["root_orient"]).float(), # (N, 3)
        body_pose=torch.tensor(smplx_data["pose_body"]).float(), # (N, 63)
        transl=torch.tensor(smplx_data["trans"]).float(), # (N, 3)
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        # expression=torch.zeros(num_frames, 10).float(),
        return_full_pose=True,
    )
    
    if len(smplx_data["betas"].shape)==1:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0]
    else:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0, 0]
    
    return smplx_data, body_model, smplx_output, human_height


def load_smplx_data(smplx_data, smplx_body_model_path):
    body_model = smplx.create(
        smplx_body_model_path,
        "smplx",
        gender=str(smplx_data["gender"]),
        use_pca=False,
    )

    num_frames = smplx_data["pose_body"].shape[0]
    smplx_output = body_model(
        betas=torch.tensor(smplx_data["betas"]).float().view(1, -1),
        global_orient=torch.tensor(smplx_data["root_orient"]).float(),
        body_pose=torch.tensor(smplx_data["pose_body"]).float(),
        transl=torch.tensor(smplx_data["trans"]).float(),
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        return_full_pose=True,
    )

    if len(smplx_data["betas"].shape) == 1:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0]
    else:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0, 0]

    obj_data = None
    if "obj_state" in smplx_data:
        obj_data = {
            "pos": smplx_data["obj_state"][:, :3],
            "rot": smplx_data["obj_state"][:, 3:6],
            "contact": smplx_data["obj_contact"][:, :],
        }

    return body_model, smplx_output, human_height, obj_data


def convert_skillmimic_to_smplx(input_path, gender="neutral"):
    smpl_data = torch.load(input_path, weights_only=False).to(torch.device("cpu")).numpy()
    data_dict = {
        "trans": smpl_data[:, 0:3],
        "poses": smpl_data[:, 3:],
        "mocap_framerate": np.array(60),
    }

    if "betas" in data_dict:
        betas = data_dict["betas"]
        if betas.shape == (10,):
            data_dict["betas"] = np.concatenate([betas, np.zeros(6, dtype=betas.dtype)])
        elif betas.shape not in [(16,), (1, 16)]:
            raise ValueError(f"Unexpected betas shape: {betas.shape}.")
    else:
        data_dict["betas"] = np.zeros(16, dtype=np.float32)

    data_dict["mocap_frame_rate"] = data_dict.pop("mocap_framerate")

    poses = data_dict["poses"]
    reindex_map = 3 + torch.cat(
        [3 * GYM_TO_SMPL_IDX[:, None] + i for i in range(3)], dim=1
    ).flatten()
    poses[:, 3:(54 * 3)] = poses[:, reindex_map]

    data_dict["obj_state"] = smpl_data[:, 324:330]
    data_dict["obj_contact"] = smpl_data[:, 336:337]
    data_dict["root_orient"] = poses[:, :3]
    data_dict["pose_body"] = poses[:, 6:69]
    data_dict["gender"] = np.array(gender)

    del data_dict["poses"]
    return data_dict

def convert_intermimic_to_smplx(input_path, gender="neutral"):
    hoi_2d = load_intermimic_hoi_2d(input_path)
    parsed = parse_intermimic_hoi(hoi_2d)

    root_orient = (
        R.from_quat(parsed["root_rot_xyzw"])
        * R.from_matrix(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32))
    ).as_rotvec().astype(np.float32) # Reorder from InterMimic y-up to SMPL-X z-up convention
    pose_body = build_intermimic_pose_body(parsed)
    obj_rot = (
        R.from_quat(parsed["obj_rot_xyzw"]) 
        * R.from_matrix(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32))
    ).as_rotvec().astype(np.float32)
    obj_state = np.concatenate([parsed["obj_pos"], obj_rot], axis=1).astype(np.float32)

    return {
        "gender": np.array(gender),
        "betas": np.zeros(16, dtype=np.float32),
        "mocap_frame_rate": np.array(30.0, dtype=np.float32), # Fixed frame rate for InterMimic data since it's not explicitly provided
        "trans": parsed["root_pos"].astype(np.float32),
        "root_orient": root_orient,
        "pose_body": pose_body,
        "obj_state": obj_state,
        "obj_contact": parsed["contact_obj"].astype(np.int32),
        "contact_human": parsed["contact_human"].astype(np.int32),
    }


def load_gvhmr_pred_file(gvhmr_pred_file, smplx_body_model_path):
    gvhmr_pred = torch.load(gvhmr_pred_file)
    smpl_params_global = gvhmr_pred['smpl_params_global']
    # print(smpl_params_global['body_pose'].shape)
    # print(smpl_params_global['betas'].shape)
    # print(smpl_params_global['global_orient'].shape)
    # print(smpl_params_global['transl'].shape)
    
    betas = np.pad(smpl_params_global['betas'][0], (0,6))
    
    # correct rotations
    # rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    # rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    
    # smpl_params_global['body_pose'] = smpl_params_global['body_pose'] @ rotation_matrix
    # smpl_params_global['global_orient'] = smpl_params_global['global_orient'] @ rotation_quat
    
    smplx_data = {
        'pose_body': smpl_params_global['body_pose'].numpy(),
        'betas': betas,
        'root_orient': smpl_params_global['global_orient'].numpy(),
        'trans': smpl_params_global['transl'].numpy(),
        "mocap_frame_rate": torch.tensor(30),
    }

    body_model = smplx.create(
        smplx_body_model_path,
        "smplx",
        gender="neutral",
        use_pca=False,
    )
    
    num_frames = smpl_params_global['body_pose'].shape[0]
    smplx_output = body_model(
        betas=torch.tensor(smplx_data["betas"]).float().view(1, -1), # (16,)
        global_orient=torch.tensor(smplx_data["root_orient"]).float(), # (N, 3)
        body_pose=torch.tensor(smplx_data["pose_body"]).float(), # (N, 63)
        transl=torch.tensor(smplx_data["trans"]).float(), # (N, 3)
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        # expression=torch.zeros(num_frames, 10).float(),
        return_full_pose=True,
    )
    
    if len(smplx_data['betas'].shape)==1:
        human_height = 1.66 + 0.1 * smplx_data['betas'][0]
    else:
        human_height = 1.66 + 0.1 * smplx_data['betas'][0, 0]
    
    return smplx_data, body_model, smplx_output, human_height


def get_smplx_data(smplx_data, body_model, smplx_output, curr_frame):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    global_orient = smplx_output.global_orient[curr_frame].squeeze()
    full_body_pose = smplx_output.full_pose[curr_frame].reshape(-1, 3)
    joints = smplx_output.joints[curr_frame].detach().numpy().squeeze()
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents

    result = {}
    joint_orientations = []
    for i, joint_name in enumerate(joint_names):
        if i == 0:
            rot = R.from_rotvec(global_orient)
        else:
            rot = joint_orientations[parents[i]] * R.from_rotvec(
                full_body_pose[i].squeeze()
            )
        joint_orientations.append(rot)
        result[joint_name] = (joints[i], rot.as_quat(scalar_first=True))

  
    return result


def slerp(rot1, rot2, t):
    """Spherical linear interpolation between two rotations."""
    # Convert to quaternions
    q1 = rot1.as_quat()
    q2 = rot2.as_quat()
    
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Compute dot product
    dot = np.sum(q1 * q2)
    
    # If the dot product is negative, slerp won't take the shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # If the inputs are too close, linearly interpolate
    if dot > 0.9995:
        return R.from_quat(q1 + t * (q2 - q1))
    
    # Perform SLERP
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    q = s0 * q1 + s1 * q2
    
    return R.from_quat(q)

def get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30, object_data=None):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    src_fps = smplx_data["mocap_frame_rate"].item()
    frame_skip = int(src_fps / tgt_fps)
    num_frames = smplx_data["pose_body"].shape[0]
    global_orient = smplx_output.global_orient.squeeze()
    full_body_pose = smplx_output.full_pose.reshape(num_frames, -1, 3)
    joints = smplx_output.joints.detach().numpy().squeeze()
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents

    has_object = object_data is not None
    if has_object:
        obj_pos = np.asarray(object_data["pos"])            # (T,3)
        obj_rot = np.asarray(object_data["rot"])            # (T,3) rotvec or (T,4) quat (wxyz)
        obj_contact = np.asarray(object_data["contact"]).reshape(-1)  # (T,) binary
        assert obj_pos.shape[0] == num_frames and obj_rot.shape[0] == num_frames and obj_contact.shape[0] == num_frames

    if tgt_fps < src_fps:
        # perform fps alignment with proper interpolation
        new_num_frames = num_frames // frame_skip
        
        # Create time points for interpolation
        original_time = np.arange(num_frames)
        target_time = np.linspace(0, num_frames-1, new_num_frames)
        
        # Interpolate global orientation using SLERP
        global_orient_interp = []
        for i in range(len(target_time)):
            t = target_time[i]
            idx1 = int(np.floor(t))
            idx2 = min(idx1 + 1, num_frames - 1)
            alpha = t - idx1
            
            rot1 = R.from_rotvec(global_orient[idx1])
            rot2 = R.from_rotvec(global_orient[idx2])
            interp_rot = slerp(rot1, rot2, alpha)
            global_orient_interp.append(interp_rot.as_rotvec())
        global_orient = np.stack(global_orient_interp, axis=0)
        
        # Interpolate full body pose using SLERP
        full_body_pose_interp = []
        for i in range(full_body_pose.shape[1]):  # For each joint
            joint_rots = []
            for j in range(len(target_time)):
                t = target_time[j]
                idx1 = int(np.floor(t))
                idx2 = min(idx1 + 1, num_frames - 1)
                alpha = t - idx1
                
                rot1 = R.from_rotvec(full_body_pose[idx1, i])
                rot2 = R.from_rotvec(full_body_pose[idx2, i])
                interp_rot = slerp(rot1, rot2, alpha)
                joint_rots.append(interp_rot.as_rotvec())
            full_body_pose_interp.append(np.stack(joint_rots, axis=0))
        full_body_pose = np.stack(full_body_pose_interp, axis=1)
        
        # Interpolate joint positions using linear interpolation
        joints_interp = []
        for i in range(joints.shape[1]):  # For each joint
            for j in range(3):  # For each coordinate
                interp_func = interp1d(original_time, joints[:, i, j], kind='linear')
                joints_interp.append(interp_func(target_time))
        joints = np.stack(joints_interp, axis=1).reshape(new_num_frames, -1, 3)

        # --- Interpolate object (if provided) ---
        if has_object:
            # positions: linear
            obj_pos_interp = []
            for j in range(3):
                interp_func = interp1d(original_time, obj_pos[:, j], kind='linear')
                obj_pos_interp.append(interp_func(target_time))
            obj_pos = np.stack(obj_pos_interp, axis=1)  # (new_T,3)

            # rotations: SLERP between neighbors
            obj_rot_interp_quat = []
            is_quat = (obj_rot.shape[-1] == 4)
            if is_quat:
                def to_R(k): return R.from_quat(obj_rot[k], scalar_first=True)
            else:
                def to_R(k): return R.from_rotvec(obj_rot[k])
            for t in target_time:
                idx1 = int(np.floor(t))
                idx2 = min(idx1 + 1, num_frames - 1)
                alpha = t - idx1
                rot1 = to_R(idx1)
                rot2 = to_R(idx2)
                interp_rot = slerp(rot1, rot2, alpha)
                obj_rot_interp_quat.append(interp_rot.as_quat(scalar_first=True))
            obj_rot = np.stack(obj_rot_interp_quat, axis=0)  # (new_T,4)

            # contact: nearest-neighbor (keeps binary)
            contact_interp_func = interp1d(original_time, obj_contact.astype(float), kind='nearest')
            obj_contact = contact_interp_func(target_time).astype(np.int32).reshape(-1, 1)  # (new_T,1)

        aligned_fps = len(global_orient) / num_frames * src_fps
    else:
        aligned_fps = tgt_fps
        # Keep original object data if any (no resampling)
        if has_object:
            if obj_rot.shape[-1] == 3:
                # keep orientation as quaternion for output consistency
                obj_rot = R.from_rotvec(obj_rot).as_quat(scalar_first=True)
            obj_contact = obj_contact.astype(np.int32).reshape(-1, 1)  # ensure int

    smplx_data_frames = []
    for curr_frame in range(len(global_orient)):
        result = {}
        single_global_orient = global_orient[curr_frame]
        single_full_body_pose = full_body_pose[curr_frame]
        single_joints = joints[curr_frame]
        joint_orientations = []
        for i, joint_name in enumerate(joint_names):
            if i == 0:
                rot = R.from_rotvec(single_global_orient)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(
                    single_full_body_pose[i].squeeze()
                )
            joint_orientations.append(rot)
            result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True))
        smplx_data_frames.append(result)

    # --- Build object frames aligned with SMPL frames (optional) ---
    object_frames = None
    if has_object:
        # Ensure lengths match (in case of rounding)
        T = len(smplx_data_frames)
        if obj_pos.shape[0] != T:
            obj_pos = obj_pos[:T]
            obj_rot = obj_rot[:T]
            obj_contact = obj_contact[:T]  # NEW
        object_frames = [(obj_pos[k], obj_rot[k], obj_contact[k]) for k in range(T)]  # NEW

    return smplx_data_frames, aligned_fps, object_frames



def get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    src_fps = smplx_data["mocap_frame_rate"].item()
    frame_skip = int(src_fps / tgt_fps)
    num_frames = smplx_data["pose_body"].shape[0]
    global_orient = smplx_output.global_orient.squeeze()
    full_body_pose = smplx_output.full_pose.reshape(num_frames, -1, 3)
    joints = smplx_output.joints.detach().numpy().squeeze()
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents
    
    if tgt_fps < src_fps:
        # perform fps alignment with proper interpolation
        new_num_frames = num_frames // frame_skip
        
        # Create time points for interpolation
        original_time = np.arange(num_frames)
        target_time = np.linspace(0, num_frames-1, new_num_frames)
        
        # Interpolate global orientation using SLERP
        global_orient_interp = []
        for i in range(len(target_time)):
            t = target_time[i]
            idx1 = int(np.floor(t))
            idx2 = min(idx1 + 1, num_frames - 1)
            alpha = t - idx1
            
            rot1 = R.from_rotvec(global_orient[idx1])
            rot2 = R.from_rotvec(global_orient[idx2])
            interp_rot = slerp(rot1, rot2, alpha)
            global_orient_interp.append(interp_rot.as_rotvec())
        global_orient = np.stack(global_orient_interp, axis=0)
        
        # Interpolate full body pose using SLERP
        full_body_pose_interp = []
        for i in range(full_body_pose.shape[1]):  # For each joint
            joint_rots = []
            for j in range(len(target_time)):
                t = target_time[j]
                idx1 = int(np.floor(t))
                idx2 = min(idx1 + 1, num_frames - 1)
                alpha = t - idx1
                
                rot1 = R.from_rotvec(full_body_pose[idx1, i])
                rot2 = R.from_rotvec(full_body_pose[idx2, i])
                interp_rot = slerp(rot1, rot2, alpha)
                joint_rots.append(interp_rot.as_rotvec())
            full_body_pose_interp.append(np.stack(joint_rots, axis=0))
        full_body_pose = np.stack(full_body_pose_interp, axis=1)
        
        # Interpolate joint positions using linear interpolation
        joints_interp = []
        for i in range(joints.shape[1]):  # For each joint
            for j in range(3):  # For each coordinate
                interp_func = interp1d(original_time, joints[:, i, j], kind='linear')
                joints_interp.append(interp_func(target_time))
        joints = np.stack(joints_interp, axis=1).reshape(new_num_frames, -1, 3)
        
        aligned_fps = len(global_orient) / num_frames * src_fps
    else:
        aligned_fps = tgt_fps
        
    smplx_data_frames = []
    for curr_frame in range(len(global_orient)):
        result = {}
        single_global_orient = global_orient[curr_frame]
        single_full_body_pose = full_body_pose[curr_frame]
        single_joints = joints[curr_frame]
        joint_orientations = []
        for i, joint_name in enumerate(joint_names):
            if i == 0:
                rot = R.from_rotvec(single_global_orient)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(
                    single_full_body_pose[i].squeeze()
                )
            joint_orientations.append(rot)
            result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True))


        smplx_data_frames.append(result)
        
    # add correct rotations
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    for result in smplx_data_frames:
        for joint_name in result.keys():
            orientation = utils.quat_mul(rotation_quat, result[joint_name][1])
            position = result[joint_name][0] @ rotation_matrix.T
            result[joint_name] = (position, orientation)
            

    return smplx_data_frames, aligned_fps
