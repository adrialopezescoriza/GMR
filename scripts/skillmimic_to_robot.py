import os
import argparse
import pathlib
from threading import local
import time
import numpy as np
from sympy import im
from tqdm import tqdm
import torch
import smplx
from general_motion_retargeting import torch_utils 
import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast
from general_motion_retargeting import optimize_object_traj_from_motion

SKILLMIMIC_INDEX_MAP = torch.tensor([0, 1, 5, 10, 2, 6, 11, 3, 7, 12, 4, 8, 13, 15, 34, 14,
                                        16, 35, 17, 36, 18, 37, 9, 19, 20, 21, 22, 23, 24, 25,
                                        26, 27, 28, 29, 30, 31, 32, 33, 38, 39, 40, 41, 42, 43,
                                        44, 45, 46, 47, 48, 49, 50, 51, 52])

MIN_BALL_HEIGHT = 0.13  # Minimum height to consider ball in contact with ground
CONTACT_LINK_NAME = {
    "unitree_g1": "right_rubber_hand",
    "unitree_h1_2_with_hands": "right_hand_link",
}
LOCAL_LINK_OFSET_DICT = {
    "unitree_g1": np.array([0.08, 0.12, 0.03]),
    "unitree_h1_2_with_hands": np.array([0.08, 0.12, 0.03]),
}

def load_smplx_data(smplx_data, smplx_body_model_path):
    body_model = smplx.create(
        smplx_body_model_path,
        "smplx",
        gender=str(smplx_data["gender"]),
        use_pca=False,
    )
    
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
        return_full_pose=True,
    )
    
    if len(smplx_data["betas"].shape)==1:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0]
    else:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0, 0]
    if "obj_state" in smplx_data:
        obj_data = {
            "pos": smplx_data["obj_state"][:, :3],
            "rot": smplx_data["obj_state"][:, 3:6],
            "contact": smplx_data["obj_contact"][:, :],
        }
    else:
        obj_data = None

    return body_model, smplx_output, human_height, obj_data

def convert_skillmimic_to_smplx(input_path, gender='neutral'):
    # Load SkillMimic data
    smpl_data = torch.load(input_path).to(torch.device("cpu")).numpy()
    data_dict = {"trans": smpl_data[:, 0:3], "poses": smpl_data[:, 3:], "mocap_framerate": np.array(60)}

    # Handle betas padding for SMPL-X (pad from 10 to 16 if necessary)
    if 'betas' in data_dict:
        betas = data_dict['betas']
        if betas.shape == (10,):
            data_dict['betas'] = np.concatenate([betas, np.zeros(6, dtype=betas.dtype)])
            print(f"Padded betas from 10 to 16 for {input_path}")
        elif betas.shape not in [(16,), (1, 16)]:
            raise ValueError(f"Unexpected betas shape: {betas.shape}. Expected (10,), (16,), or (1,16) for padding to SMPL-X.")
    else:
        data_dict['betas'] = np.zeros(16, dtype=np.float32)
        print(f"Added default betas for {input_path}")

    # Handle mocap_frame_rate variations
    if 'mocap_framerate' in data_dict:
        data_dict['mocap_frame_rate'] = data_dict.pop('mocap_framerate')
        print(f"Renamed 'mocap_framerate' to 'mocap_frame_rate' for {input_path}")

    if 'poses' not in data_dict:
        raise ValueError("Input file does not contain 'poses' key. Is this an SMPL file?")

    poses = data_dict['poses']

    # Reorder pose parameters from SkillMimic format to SMPL-X format
    reindex_map = 3 + torch.cat([3 * SKILLMIMIC_INDEX_MAP[:, None] + i for i in range(3)], dim=1).flatten()
    poses[:,3:(54*3)] = poses[:,reindex_map]
    
    # Get object state (position + rotation as 6D) if available
    data_dict['obj_state'] = smpl_data[:, 324:330]  # From https://github.com/wyhuai/SkillMimic/blob/main/skillmimic/utils/motion_data_handler.py
    data_dict['obj_contact'] = smpl_data[:, 336:337]  # From https://github.com/wyhuai/SkillMimic/blob/main/skillmimic/utils/motion_data_handler.py


    # Map to SMPL-X format
    data_dict['root_orient'] = poses[:, :3]
    data_dict['pose_body'] = poses[:, 6:69]  # 21 joints x 3 = 63, ignoring SMPL hand poses (no pelvis)

    # Ensure gender is set
    if 'gender' not in data_dict:
        data_dict['gender'] = np.array(gender)

    # Remove original poses key
    del data_dict['poses']

    # Save as SMPL-X npz
    return data_dict

def convert_smplx_to_robot(smplx_data, args):
    HERE = pathlib.Path(__file__).parent
    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

    # Load SMPLX trajectory
    body_model, smplx_output, actual_human_height, obj_data = load_smplx_data(smplx_data, SMPLX_FOLDER)
    
    # align fps
    tgt_fps = 60
    smplx_data_frames, aligned_fps, object_frames = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps, object_data=obj_data)

    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )
    
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []
    
    # Start the viewer
    i = 0

    while True:
        if args.loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time

        # retarget
        qpos = retarget.retarget(smplx_data_frames[i])
        qpos_list.append(qpos)

    root_pos = np.array([qpos[:3] for qpos in qpos_list])
    root_rot = np.array([qpos[3:7] for qpos in qpos_list])
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])
    
    # obtain local body pos
    kinematics_model = KinematicsModel(retarget.xml_file, device="cuda:0")
    world_body_pos, world_body_orient = kinematics_model.forward_kinematics(
        root_pos=torch.from_numpy(root_pos).to(device="cuda:0", dtype=torch.float32), 
        root_rot=torch.from_numpy(root_rot)[...,[1,2,3,0]].to(device="cuda:0", dtype=torch.float32), 
        dof_pos=torch.from_numpy(dof_pos).to(device="cuda:0", dtype=torch.float32)
    )

    local_body_pos, local_body_orient = kinematics_model.forward_kinematics(
        root_pos=torch.zeros(root_pos.shape).to(device="cuda:0", dtype=torch.float32),
        root_rot=torch.zeros(root_rot.shape).to(device="cuda:0", dtype=torch.float32) + torch.tensor([0.0, 0.0, 0.0, 1.0]).to(device="cuda:0", dtype=torch.float32),  # identity quat
        dof_pos=torch.from_numpy(dof_pos).to(device="cuda:0", dtype=torch.float32)
    )


    body_names = kinematics_model.body_names
    world_body_orient = world_body_orient[..., [3,0,1,2]].cpu().numpy()  # to quat scalar first
    world_body_pos = world_body_pos.cpu().numpy()

    local_body_orient = local_body_orient[..., [3,0,1,2]].cpu().numpy()  # to quat scalar first
    local_body_pos = local_body_pos.cpu().numpy()
    
    motion_data = {
        "fps": aligned_fps, # motion frequency
        "root_pos": root_pos, # root position in world frame (N, 3)
        "root_rot": root_rot, # root rotation (quaternion) in world frame (N, 4)
        "dof_pos": dof_pos, # robot joint angles (rad) (N, num_dof)
        "world_body_pos": world_body_pos, # local body positions (N, num_bodies, 3)
        "world_body_orient": world_body_orient, # local body orientations (quaternion) (N, num_bodies, 4)
        "local_body_pos": local_body_pos, # local body positions (N, num_bodies, 3)
        "local_body_orient": local_body_orient, # local body orientations
        "link_body_list": body_names, # body names corresponding to local_body_pos
        "dof_names": retarget.robot_motor_names, # robot joint names in order
    }

    if object_frames is not None:
        obj_pos = np.array([object_frames[k][0] for k in range(len(object_frames))])
        obj_rot = np.array([object_frames[k][1] for k in range(len(object_frames))])
        obj_contact = np.array([object_frames[k][2] for k in range(len(object_frames))])
        obj_ground_contact = (obj_pos[:, 2:] <= MIN_BALL_HEIGHT).astype(np.int8)
        motion_data["object_pos"] = obj_pos
        motion_data["object_rot"] = obj_rot
        motion_data["contact_sequence"] = obj_contact
        motion_data["ground_contact_sequence"] = obj_ground_contact

        if args.retarget_dynamic_object:
            optimized_obj_pos, optimized_obj_vel = optimize_object_traj_from_motion(
                motion_data=motion_data,
                contact_link_name=CONTACT_LINK_NAME.get(args.robot, "right_rubber_hand"),
                local_offset=np.array(LOCAL_LINK_OFSET_DICT.get(args.robot, np.array([0.0, 0.0, 0.0]))),
                speed_thresh=0.05,
            )
            motion_data["object_pos"] = optimized_obj_pos
            object_frames = [(optimized_obj_pos[k], object_frames[k][1], object_frames[k][2]) for k in range(len(object_frames))]
            
    if args.save_path is not None:
        import pickle

        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    # Video
    motion_folder = str(pathlib.Path(args.save_path).with_suffix('')).split('/',1)[1]
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=aligned_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=f"videos/{motion_folder}/{args.robot}_{args.input_file.split('/')[-1].split('.')[0]}.mp4")
    for i, qpos in enumerate(qpos_list):
        body_poses = {k: (motion_data["world_body_pos"][i, j], motion_data["world_body_orient"][i, j]) for j, k in enumerate(body_names)}
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=body_poses,
            object_data=object_frames[i],
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=True,
            rate_limit=args.rate_limit,
        )
            
      
    
    robot_motion_viewer.close()


def process_folder(src_folder, tgt_folder, args):
    for filename in tqdm(os.listdir(src_folder)):
        if filename.endswith('.pt'):
            args.input_file = os.path.join(src_folder, filename)
            args.save_path = os.path.join(tgt_folder, filename.replace('.pt', '.pkl'))
            smplx_data = convert_skillmimic_to_smplx(args.input_file, args.gender)
            convert_smplx_to_robot(smplx_data, args)
        elif os.path.isdir(os.path.join(src_folder, filename)):
            new_src_folder = os.path.join(src_folder, filename)
            new_tgt_folder = os.path.join(tgt_folder, filename)
            process_folder(new_src_folder, new_tgt_folder, args)
        else:
            print(f"Skipping non-.pt file: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SMPL SkillMimic motion data to Robot format.")
    parser.add_argument("--src_folder", type=str, help="Source directory of SMPL .pt files")
    parser.add_argument("--tgt_folder", type=str, help="Target directory for .pkl files")
    parser.add_argument("--input_file", type=str, help="Single input SMPL .pt file")
    parser.add_argument("--save_path", type=str, help="Single output .pkl file")
    parser.add_argument("--loop", default=False, action="store_true", help="Loop the motion.")
    parser.add_argument("--record_video", default=False, action="store_true", help="Record the video.")
    parser.add_argument("--rate_limit", default=False, action="store_true", help="Limit the rate of the retargeted robot motion to keep the same as the human motion.")
    parser.add_argument("--retarget_dynamic_object", default=False, action="store_true", help="Retarget the dynamic object trajectory based on the retargeted robot motion.")
    parser.add_argument("--gender", type=str, default="neutral", choices=["male", "female", "neutral"],
                        help="Gender for SMPL-X model if not present in file.")
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2", "unitree_h1_2_with_hands",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openloong", "tienkung", "smplx_humanoid"],
        default="unitree_g1",
    )
    args = parser.parse_args()

    if args.src_folder and args.tgt_folder:
        process_folder(args.src_folder, args.tgt_folder, args)
    elif args.input_file:
        smplx_data = convert_skillmimic_to_smplx(args.input_file, args.gender)
        if args.save_path is None:
            args.save_path = args.input_file.replace('.pt', '.pkl')
        convert_smplx_to_robot(smplx_data, args)
    else:
        parser.print_help()