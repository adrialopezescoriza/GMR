import os
import argparse
import pathlib
import time
import numpy as np
from tqdm import tqdm
import torch

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.utils.smpl import (
    get_smplx_data_offline_fast,
    load_smplx_data,
    convert_skillmimic_to_smplx,
)
from general_motion_retargeting import optimize_object_traj_from_motion

MIN_BALL_HEIGHT = 0.13  # Minimum height to consider ball in contact with ground
CONTACT_POINTS_O_LOCAL = {
    # "left_rubber_hand": np.array([0.07, -0.11, 0.05]),
    "right_rubber_hand": np.array([0.07, 0.11, 0.05]),
}

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
                contact_link_names=list(CONTACT_POINTS_O_LOCAL.keys()),
                local_offsets=[CONTACT_POINTS_O_LOCAL[name] for name in CONTACT_POINTS_O_LOCAL.keys()],
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
    parser.add_argument("--src_folder", type=str, default="data/skillmimic/run", help="Source directory of SkillMimic .pt files")
    parser.add_argument("--tgt_folder", type=str, default="data/g1_skillmimic/run/", help="Target directory for projected .pkl files")
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
