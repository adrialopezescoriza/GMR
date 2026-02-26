import os
import time
import argparse
import pathlib
from typing import Optional

import numpy as np
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin
import torch
import pickle

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer, optimize_object_traj_from_motion
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.params import ROBOT_XML_DICT
from general_motion_retargeting.rot_utils import (
    rot_from_quat_wxyz,
    quat_conjugate_wxyz,
    quat_mul_wxyz,
    axis_angle_from_quat_wxyz,
    finite_difference_velocities,
    pad_or_truncate,
    pad_or_truncate_1d,
)
from general_motion_retargeting.utils.smpl import (
    get_smplx_data_offline_fast,
    load_smplx_data,
    convert_skillmimic_to_smplx,
)


URDF_PATH = "assets/unitree_g1/g1_29dof_w_hands.urdf"
BASE_BODY_NAME = "pelvis"

LINK_TRACKING_WEIGHTS_W = {
    "head_link": 1.0,
    "left_ankle_roll_link": 100.0,
    "right_ankle_roll_link": 100.0,
}

LINK_TRACKING_WEIGHTS_B = {
    "torso_link": 1.0,
    "left_rubber_hand": 1.0,
    "right_rubber_hand": 1.0,
    "right_wrist_yaw_link": 1.0,
    "left_wrist_yaw_link": 1.0,
    "right_elbow_link": 1.0,
    "left_elbow_link": 1.0,
    "right_shoulder_yaw_link": 1.0,
    "left_shoulder_yaw_link": 1.0,
    "left_knee_link": 1.0,
    "right_knee_link": 1.0,
}

JOINT_TRACKING_WEIGHTS = {
    # set weights here if you want specific joint tracking
}

CONTACT_POINTS_O_LOCAL = {
    # "left_rubber_hand": np.array([0.07, -0.11, 0.05]),
    "right_rubber_hand": np.array([0.07, 0.11, 0.05]),
}

POSE_TRACKING_WEIGHT = 1.0
LINK_POS_TRACKING_WEIGHT = 1.0
LINK_B_ORIENT_TRACKING_WEIGHT = 1.0
LINK_W_ORIENT_TRACKING_WEIGHT = 1.0

KINEMATIC_CONSISTENCY_WEIGHT = 1.0
VELOCITY_REG_WEIGHT = 5e-4

BALL_RADIUS = 0.12
MIN_BALL_HEIGHT = 0.13
BALL_GRAVITY = np.array([0.0, 0.0, -9.81])
BALL_SPEED_THRESH = 0.05
BALL_CONTACT_WEIGHT = 500.0
BALL_VEL_SMOOTH_W = 10.0
BALL_GROUND_SMOOTH_W = 200.0
BALL_GROUND_Z_MIN = 0.11
BALL_GROUND_Z_MAX = 0.13
BALL_GROUND_VZ_ABS = 0.05


def build_motion_data_from_skillmimic(
    input_path: str,
    robot: str,
    gender: str,
    tgt_fps: int = 60,
    device: Optional[str] = None,
) -> dict:
    smplx_folder = pathlib.Path(__file__).parent / ".." / "assets" / "body_models"
    smplx_data = convert_skillmimic_to_smplx(input_path, gender)
    body_model, smplx_output, actual_human_height, obj_data = load_smplx_data(
        smplx_data, smplx_folder
    )

    smplx_frames, aligned_fps, object_frames = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps, object_data=obj_data
    )
    if object_frames is None:
        raise ValueError("SkillMimic file has no object data; cannot optimize ball.")

    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=robot,
    )

    qpos_list = [retarget.retarget(frame) for frame in smplx_frames]

    root_pos = np.array([qpos[:3] for qpos in qpos_list])
    root_rot = np.array([qpos[3:7] for qpos in qpos_list])
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    kin = KinematicsModel(retarget.xml_file, device=device)
    world_body_pos, world_body_orient = kin.forward_kinematics(
        root_pos=torch.from_numpy(root_pos).to(device=device, dtype=torch.float32),
        root_rot=torch.from_numpy(root_rot)[..., [1, 2, 3, 0]].to(device=device, dtype=torch.float32),
        dof_pos=torch.from_numpy(dof_pos).to(device=device, dtype=torch.float32),
    )
    local_body_pos, local_body_orient = kin.forward_kinematics(
        root_pos=torch.zeros(root_pos.shape).to(device=device, dtype=torch.float32),
        root_rot=(
            torch.zeros(root_rot.shape).to(device=device, dtype=torch.float32)
            + torch.tensor([0.0, 0.0, 0.0, 1.0]).to(device=device, dtype=torch.float32)
        ),
        dof_pos=torch.from_numpy(dof_pos).to(device=device, dtype=torch.float32),
    )

    body_names = kin.body_names
    world_body_orient = world_body_orient[..., [3, 0, 1, 2]].cpu().numpy()
    world_body_pos = world_body_pos.cpu().numpy()
    local_body_orient = local_body_orient[..., [3, 0, 1, 2]].cpu().numpy()
    local_body_pos = local_body_pos.cpu().numpy()

    obj_pos = np.array([object_frames[k][0] for k in range(len(object_frames))])
    obj_rot = np.array([object_frames[k][1] for k in range(len(object_frames))])
    obj_contact = np.array([object_frames[k][2] for k in range(len(object_frames))])
    obj_ground_contact = (obj_pos[:, 2:] <= MIN_BALL_HEIGHT).astype(np.int8)

    motion_data = {
        "fps": aligned_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "world_body_pos": world_body_pos,
        "world_body_orient": world_body_orient,
        "local_body_pos": local_body_pos,
        "local_body_orient": local_body_orient,
        "link_body_list": body_names,
        "dof_names": retarget.robot_motor_names,
        "object_pos": obj_pos,
        "object_rot": obj_rot,
        "contact_sequence": obj_contact,
        "ground_contact_sequence": obj_ground_contact,
    }

    try:
        optimized_obj_pos, optimized_obj_vel = optimize_object_traj_from_motion(
            motion_data=motion_data,
            contact_link_names=list(CONTACT_POINTS_O_LOCAL.keys()),
            local_offsets=[CONTACT_POINTS_O_LOCAL[name] for name in CONTACT_POINTS_O_LOCAL.keys()],
            speed_thresh=BALL_SPEED_THRESH,
        )
    except Exception as exc:
        print(f"[WARN] Failed to retarget ball trajectory: {exc}")
        return motion_data

    motion_data = dict(motion_data)
    motion_data["object_pos"] = optimized_obj_pos
    motion_data["object_vel"] = optimized_obj_vel
    return motion_data


class PinocchioCasadiRobot:
    def __init__(self, urdf_path, package_dirs=None):
        if package_dirs is None:
            package_dirs = [os.path.dirname(urdf_path)]

        root_joint = pin.JointModelFreeFlyer()
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path, package_dirs, root_joint
        )
        self.data = self.model.createData()
        self.nq = self.model.nq
        self.nv = self.model.nv

        self.q_lower = self.model.lowerPositionLimit.copy()
        self.q_upper = self.model.upperPositionLimit.copy()
        self.v_abs = self.model.velocityLimit.copy()

        self.joint_q_indices = []
        self.joint_v_indices = []
        for jid, joint in enumerate(self.model.joints):
            if jid <= 1:
                continue
            if joint.nq == 1 and joint.nv == 1:
                self.joint_q_indices.append(joint.idx_q)
                self.joint_v_indices.append(joint.idx_v)
        self.joint_q_indices = np.array(self.joint_q_indices, dtype=int)
        self.joint_v_indices = np.array(self.joint_v_indices, dtype=int)

        ff_joint = self.model.joints[1]
        self.base_q_start = ff_joint.idx_q
        self.base_q_size = ff_joint.nq

        self.base_body_name = BASE_BODY_NAME
        self.body_names = [f.name for f in self.model.frames]

        self.dof_name_to_index = {}
        for jid, joint in enumerate(self.model.joints):
            name = self.model.names[jid]
            if jid <= 1:
                continue
            if joint.nq == 1:
                self.dof_name_to_index[name] = joint.idx_q

        self.cmodel = cpin.Model(self.model)
        q_sym = ca.SX.sym("q", self.cmodel.nq, 1)
        data_sym = self.cmodel.createData()
        cpin.forwardKinematics(self.cmodel, data_sym, q_sym)
        cpin.updateFramePlacements(self.cmodel, data_sym)

        self.fk_world_pos = {}
        self.fk_world_rot = {}
        self.fk_rel_pos = {}
        self.fk_rel_rot = {}

        base_frame_id = self.cmodel.getFrameId(self.base_body_name)
        for frame_id in range(self.cmodel.nframes):
            frame = self.cmodel.frames[frame_id]
            name = frame.name
            oMf = data_sym.oMf[frame_id]
            p_WB = oMf.translation
            R_WB = oMf.rotation
            self.fk_world_pos[name] = ca.Function(f"fk_wp_{name}", [q_sym], [p_WB])
            self.fk_world_rot[name] = ca.Function(f"fk_wr_{name}", [q_sym], [R_WB])

            oMf_base = data_sym.oMf[base_frame_id]
            base_to_body = oMf_base.inverse() * oMf
            p_BB = base_to_body.translation
            R_BB = base_to_body.rotation
            self.fk_rel_pos[(self.base_body_name, name)] = ca.Function(
                f"fk_bp_{name}", [q_sym], [p_BB]
            )
            self.fk_rel_rot[(self.base_body_name, name)] = ca.Function(
                f"fk_br_{name}", [q_sym], [R_BB]
            )

        self.q0 = pin.neutral(self.model)

    def world_pos(self, body_name, q):
        return self.fk_world_pos[body_name](q)

    def world_rotmat(self, body_name, q):
        return self.fk_world_rot[body_name](q)

    def rel_pos(self, base_name, body_name, q):
        return self.fk_rel_pos[(base_name, body_name)](q)

    def rel_rotmat(self, base_name, body_name, q):
        return self.fk_rel_rot[(base_name, body_name)](q)

    def compute_kinematics(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)


class PinocchioContactProjector:
    def __init__(self, urdf_path):
        self.robot = PinocchioCasadiRobot(urdf_path)
        self.base_body_name = BASE_BODY_NAME

    def project_motion(self, motion_data: dict) -> dict:
        fps = float(motion_data["fps"])
        dt = 1.0 / fps

        root_pos = np.asarray(motion_data["root_pos"])
        root_rot = np.asarray(motion_data["root_rot"])
        dof_pos = np.asarray(motion_data["dof_pos"])
        obj_pos = np.asarray(motion_data["object_pos"])
        obj_rot = np.asarray(motion_data["object_rot"])

        dof_names = list(motion_data["dof_names"])
        body_names = list(motion_data["link_body_list"])
        ref_world_body_pos = np.asarray(motion_data["world_body_pos"])
        ref_world_body_orient = np.asarray(motion_data["world_body_orient"])
        ref_local_body_pos = np.asarray(motion_data["local_body_pos"])
        ref_local_body_orient = np.asarray(motion_data["local_body_orient"])
        contact_seq = np.asarray(motion_data["contact_sequence"]).reshape(-1)
        ground_contact_seq = np.asarray(
            motion_data.get(
                "ground_contact_sequence",
                (obj_pos[:, 2:] <= MIN_BALL_HEIGHT).astype(np.int8),
            )
        ).reshape(-1)

        N, _ = dof_pos.shape
        nq = self.robot.nq
        nv = self.robot.nv
        dof_idx_map = self.robot.dof_name_to_index
        dof_col_index = {name: i for i, name in enumerate(dof_names)}

        obj_pos = pad_or_truncate(obj_pos, N)
        obj_rot = pad_or_truncate(obj_rot, N)
        contact_seq = pad_or_truncate_1d(contact_seq, N)
        ground_contact_seq = pad_or_truncate_1d(ground_contact_seq, N)

        R_base_ref = [rot_from_quat_wxyz(root_rot[t]) for t in range(N)]

        R_world_ref = np.zeros(ref_world_body_orient.shape[:-1] + (3, 3))
        for t in range(N):
            for i in range(len(body_names)):
                R_world_ref[t, i] = rot_from_quat_wxyz(ref_world_body_orient[t, i])

        R_local_ref = np.zeros(ref_local_body_orient.shape[:-1] + (3, 3))
        for t in range(N):
            for i in range(len(body_names)):
                R_local_ref[t, i] = rot_from_quat_wxyz(ref_local_body_orient[t, i])

        body_contact_flags = (contact_seq != 0)
        ground_contact_flags = (ground_contact_seq != 0)
        v_ref = finite_difference_velocities(obj_pos, dt)
        speed_ref = np.linalg.norm(v_ref, axis=1)
        g_vec_dm = ca.DM(BALL_GRAVITY)
        zero3 = ca.DM.zeros(3, 1)

        contact_offsets_dm = {
            name: ca.DM(offset.reshape(3, 1))
            for name, offset in CONTACT_POINTS_O_LOCAL.items()
        }

        opti = ca.Opti()
        q_traj = opti.variable(N, nq)
        v_traj = opti.variable(N, nv)
        ball_pos = opti.variable(N, 3)
        ball_vel = opti.variable(N, 3)

        total_cost = 0
        base_pos_idx = self.robot.base_q_start
        base_quat_idx = self.robot.base_q_start + 3

        for t in range(N):
            q_t = q_traj[t, :].T
            q_base_pos = q_t[base_pos_idx: base_pos_idx + 3]
            q_base_quat = q_t[base_quat_idx: base_quat_idx + 4]
            q_joints = q_t[7:]
            v_joints = v_traj[t, 6:].T

            for idx_q, idx_v in zip(self.robot.joint_q_indices, self.robot.joint_v_indices):
                q_lo = float(self.robot.q_lower[idx_q])
                q_hi = float(self.robot.q_upper[idx_q])
                if np.isfinite(q_lo) and np.isfinite(q_hi):
                    opti.subject_to(q_t[idx_q] >= q_lo)
                    opti.subject_to(q_t[idx_q] <= q_hi)

                v_lim = float(self.robot.v_abs[idx_v])
                if np.isfinite(v_lim) and v_lim > 0:
                    opti.subject_to(v_traj[t, idx_v] >= -v_lim)
                    opti.subject_to(v_traj[t, idx_v] <= v_lim)

            if t > 0:
                q_prev_joints = q_traj[t - 1, 7:].T
                v_fd = (q_joints - q_prev_joints) / dt
                # Cost: penalize mismatch between finite-diff joint velocity and v_traj.
                total_cost += KINEMATIC_CONSISTENCY_WEIGHT * ca.sumsqr(v_fd - v_joints)

                q_prev_base_pos = q_traj[t - 1, base_pos_idx: base_pos_idx + 3].T
                v_fd_base = (q_base_pos - q_prev_base_pos) / dt
                v_joints_base = v_traj[t, 0:3].T
                opti.subject_to(v_fd_base == v_joints_base)

                q_prev_base_quat = q_traj[t - 1, base_quat_idx: base_quat_idx + 4].T
                q_conj = quat_conjugate_wxyz(q_prev_base_quat)
                q_rel = quat_mul_wxyz(q_conj, q_base_quat)
                axis_angle = axis_angle_from_quat_wxyz(q_rel)
                v_fd_base_orient = axis_angle / dt
                v_joints_base_orient = v_traj[t, 3:6].T
                opti.subject_to(v_fd_base_orient == v_joints_base_orient)

            R_WB = self.robot.world_rotmat(self.base_body_name, q_t)
            # Cost: keep base orientation close to the reference.
            total_cost += POSE_TRACKING_WEIGHT * 1e-4 * ca.sumsqr(R_WB - ca.DM(R_base_ref[t]))

            if t > 1:
                v_prev = v_traj[t - 1, :].T
                v_curr = v_traj[t, :].T
                a_prev = (v_curr - v_prev) / dt
                v_prev2 = v_traj[t - 2, :].T
                a_prev2 = (v_prev - v_prev2) / dt
                da = a_prev - a_prev2
                # Cost: smooth accelerations (jerk penalty).
                total_cost += VELOCITY_REG_WEIGHT * ca.sumsqr(da)

            for j_name, w in JOINT_TRACKING_WEIGHTS.items():
                if j_name in dof_idx_map and j_name in dof_col_index:
                    idx_q = dof_idx_map[j_name]
                    col = dof_col_index[j_name]
                    q_ref = dof_pos[t, col]
                    q_curr = q_t[idx_q]
                    # Cost: track selected joint angles.
                    total_cost += w * (q_curr - q_ref) ** 2

            for link_name, w in LINK_TRACKING_WEIGHTS_W.items():
                if link_name in body_names and link_name in self.robot.fk_world_pos:
                    idx = body_names.index(link_name)
                    p_WB_ref = ref_world_body_pos[t, idx]
                    p_WB = self.robot.world_pos(link_name, q_t)
                    # Cost: track link world position.
                    total_cost += w * LINK_POS_TRACKING_WEIGHT * ca.sumsqr(p_WB - p_WB_ref)
                    R_WB = self.robot.world_rotmat(link_name, q_t)
                    # Cost: track link world orientation.
                    total_cost += w * LINK_W_ORIENT_TRACKING_WEIGHT * ca.sumsqr(
                        R_WB - ca.DM(R_world_ref[t, idx])
                    )

            for link_name, w in LINK_TRACKING_WEIGHTS_B.items():
                key = (self.base_body_name, link_name)
                if link_name in body_names and key in self.robot.fk_rel_pos:
                    idx = body_names.index(link_name)
                    R_BB = self.robot.rel_rotmat(self.base_body_name, link_name, q_t)
                    # Cost: track link orientation relative to the base.
                    total_cost += w * LINK_B_ORIENT_TRACKING_WEIGHT * ca.sumsqr(
                        R_BB - ca.DM(R_local_ref[t, idx])
                    )

            p_ball_t = ball_pos[t, :].T
            v_ball_t = ball_vel[t, :].T

            if body_contact_flags[t]:
                for name, p_BC_DM in contact_offsets_dm.items():
                    if name not in self.robot.fk_world_pos:
                        continue
                    p_WB = self.robot.world_pos(name, q_t)
                    R_WB = self.robot.world_rotmat(name, q_t)
                    p_WC_des = p_WB + R_WB @ p_BC_DM
                    # Cost: keep ball at contact point during contact frames.
                    total_cost += BALL_CONTACT_WEIGHT * ca.sumsqr(p_ball_t - p_WC_des)

            if ground_contact_flags[t]:
                opti.subject_to(p_ball_t[2] <= BALL_GROUND_Z_MAX)
                opti.subject_to(p_ball_t[2] >= BALL_GROUND_Z_MIN)
                opti.subject_to(v_ball_t[2] <= BALL_GROUND_VZ_ABS)
                opti.subject_to(v_ball_t[2] >= -BALL_GROUND_VZ_ABS)

        for k in range(N - 1):
            p_k = ball_pos[k, :].T
            v_k = ball_vel[k, :].T
            p_next = ball_pos[k + 1, :].T
            v_next = ball_vel[k + 1, :].T

            opti.subject_to(p_next == p_k + dt * v_k)

            is_contact_k = body_contact_flags[k] or ground_contact_flags[k]
            is_contact_k1 = body_contact_flags[k + 1] or ground_contact_flags[k + 1]

            if (not is_contact_k) and (not is_contact_k1):
                opti.subject_to(v_next == v_k + dt * g_vec_dm)
            else:
                dv_z = v_next[2] - v_k[2]
                # Cost: smooth vertical velocity while in contact.
                total_cost += BALL_VEL_SMOOTH_W * ca.sumsqr(dv_z)

            dv_xy = v_next[0:2] - v_k[0:2]
            # Cost: smooth horizontal velocity changes.
            total_cost += BALL_GROUND_SMOOTH_W * ca.sumsqr(dv_xy)

            if speed_ref[k] > BALL_SPEED_THRESH:
                v_ref_k = ca.DM(v_ref[k])
                denom = (ca.norm_2(v_k) * ca.norm_2(v_ref_k) + 1e-8)
                cos_sim = (v_k.T @ v_ref_k) / denom
                # Cost: align ball velocity direction with reference when moving.
                total_cost += (1.0 - cos_sim)
            else:
                opti.subject_to(v_k == zero3)

        v_last = ball_vel[N - 1, :].T
        if speed_ref[N - 1] > BALL_SPEED_THRESH:
            v_ref_last = ca.DM(v_ref[N - 1])
            denom_last = (ca.norm_2(v_last) * ca.norm_2(v_ref_last) + 1e-8)
            cos_sim_last = (v_last.T @ v_ref_last) / denom_last
            # Cost: align final ball velocity direction with reference when moving.
            total_cost += (1.0 - cos_sim_last)
        else:
            opti.subject_to(v_last == zero3)

        opti.minimize(total_cost)

        q_init = np.zeros((N, nq))
        v_init = np.zeros((N, nv))
        for t in range(N):
            x, y, z = root_pos[t]
            w_ref, qx_ref, qy_ref, qz_ref = root_rot[t]
            base_idx = self.robot.base_q_start
            q_init[t, base_idx:base_idx + 3] = np.array([x, y, z])
            q_init[t, base_idx + 3:base_idx + 7] = np.array([qx_ref, qy_ref, qz_ref, w_ref])
            for i, j_name in enumerate(dof_names):
                if j_name in dof_idx_map:
                    idx_q = dof_idx_map[j_name]
                    q_init[t, idx_q] = np.clip(
                        dof_pos[t, i], self.robot.q_lower[idx_q], self.robot.q_upper[idx_q]
                    )

        for t in range(1, N):
            v_init[t, :3] = np.clip(
                (q_init[t, :3] - q_init[t - 1, :3]) / dt,
                -self.robot.v_abs[:3],
                self.robot.v_abs[:3],
            )
            q_prev_quat = q_init[t - 1, 3:7]
            q_curr_quat = q_init[t, 3:7]
            q_conj = quat_conjugate_wxyz(ca.DM(q_prev_quat.reshape(4, 1)))
            q_rel = quat_mul_wxyz(q_conj, ca.DM(q_curr_quat.reshape(4, 1)))
            axis_angle = axis_angle_from_quat_wxyz(q_rel)
            v_init[t, 3:6] = np.clip(
                (axis_angle / dt).full().flatten(),
                -self.robot.v_abs[3:6],
                self.robot.v_abs[3:6],
            )
            v_init[t, 6:] = np.clip(
                (q_init[t, 7:] - q_init[t - 1, 7:]) / dt,
                -self.robot.v_abs[6:],
                self.robot.v_abs[6:],
            )
        v_init[0, :] = v_init[1, :]

        opti.set_initial(q_traj, q_init)
        opti.set_initial(v_traj, v_init)
        ball_vel_init = motion_data.get("object_vel")
        if ball_vel_init is not None:
            ball_vel_init = pad_or_truncate(ball_vel_init, N)
        else:
            ball_vel_init = v_ref

        opti.set_initial(ball_pos, obj_pos)
        opti.set_initial(ball_vel, ball_vel_init)

        opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": False})

        print(f"[INFO] Solving global optimization (Vars: {N * (nq + nv)})...")
        t0 = time.time()
        try:
            sol = opti.solve()
        except RuntimeError as e:
            print(f"[ERROR] Solver failed: {e}")
            print("[INFO] Returning original motion data.")
            return motion_data
        print(f"[INFO] Solver time: {time.time() - t0:.2f} s")

        q_sol = sol.value(q_traj)
        ball_pos_sol = sol.value(ball_pos)
        ball_vel_sol = sol.value(ball_vel)

        new_dof_pos = np.zeros_like(dof_pos)
        new_world_body_pos = np.zeros_like(ref_world_body_pos)
        new_world_body_orient = np.zeros_like(ref_world_body_orient)
        new_local_body_pos = np.zeros_like(ref_local_body_pos)
        new_local_body_orient = np.zeros_like(ref_local_body_orient)
        new_root_pos = np.zeros_like(root_pos)
        new_root_rot = np.zeros_like(root_rot)

        base_frame_id = self.robot.model.getFrameId(self.base_body_name)
        for t in range(N):
            q_t = q_sol[t, :]
            self.robot.compute_kinematics(q_t)

            oMf_base = self.robot.data.oMf[base_frame_id]
            new_root_pos[t] = np.array(oMf_base.translation).reshape(3,)
            R_WBase = oMf_base.rotation
            quat_base = pin.Quaternion(np.array(R_WBase))
            quat_base.normalize()
            new_root_rot[t] = np.array([quat_base.w, quat_base.x, quat_base.y, quat_base.z])

            for i, j_name in enumerate(dof_names):
                if j_name in dof_idx_map:
                    idx_q = dof_idx_map[j_name]
                    new_dof_pos[t, i] = q_t[idx_q]
                else:
                    new_dof_pos[t, i] = dof_pos[t, i]

            for i, b_name in enumerate(body_names):
                if b_name not in self.robot.body_names:
                    continue
                f_id = self.robot.model.getFrameId(b_name)
                oMf_body = self.robot.data.oMf[f_id]

                new_world_body_pos[t, i] = np.array(oMf_body.translation).reshape(3,)
                R_WB = oMf_body.rotation
                quat_WB = pin.Quaternion(np.array(R_WB))
                quat_WB.normalize()
                new_world_body_orient[t, i] = np.array(
                    [quat_WB.w, quat_WB.x, quat_WB.y, quat_WB.z]
                )

                base_to_body = oMf_base.inverse() * oMf_body
                new_local_body_pos[t, i] = np.array(base_to_body.translation).reshape(3,)
                R_BB = base_to_body.rotation
                quat_BB = pin.Quaternion(np.array(R_BB))
                quat_BB.normalize()
                new_local_body_orient[t, i] = np.array(
                    [quat_BB.w, quat_BB.x, quat_BB.y, quat_BB.z]
                )

        new_motion_data = dict(motion_data)
        new_motion_data.update(
            {
                "fps": fps,
                "root_pos": new_root_pos,
                "root_rot": new_root_rot,
                "dof_pos": new_dof_pos,
                "world_body_pos": new_world_body_pos,
                "world_body_orient": new_world_body_orient,
                "local_body_pos": new_local_body_pos,
                "local_body_orient": new_local_body_orient,
                "object_pos": ball_pos_sol,
                "object_vel": ball_vel_sol,
                "object_rot": obj_rot,
                "contact_sequence": contact_seq.reshape(-1, 1),
                "ground_contact_sequence": ground_contact_seq.reshape(-1, 1),
            }
        )

        return new_motion_data


def record_motion_video(motion_data: dict, robot: str, video_path: str):
    fps = float(motion_data["fps"])
    root_pos = np.asarray(motion_data["root_pos"])
    root_rot = np.asarray(motion_data["root_rot"])
    dof_pos = np.asarray(motion_data["dof_pos"])
    # Recompute FK so all links (e.g., toes) have valid orientations.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robot_xml = str(ROBOT_XML_DICT[robot])
    kinematics_model = KinematicsModel(robot_xml, device=device)
    world_body_pos, world_body_orient = kinematics_model.forward_kinematics(
        root_pos=torch.from_numpy(root_pos).to(device=device, dtype=torch.float32),
        root_rot=torch.from_numpy(root_rot)[..., [1, 2, 3, 0]].to(device=device, dtype=torch.float32),
        dof_pos=torch.from_numpy(dof_pos).to(device=device, dtype=torch.float32),
    )
    world_body_pos = world_body_pos.cpu().numpy()
    world_body_orient = world_body_orient[..., [3, 0, 1, 2]].cpu().numpy()
    body_names = kinematics_model.body_names
    obj_pos = np.asarray(motion_data.get("object_pos")) if "object_pos" in motion_data else None
    obj_rot = np.asarray(motion_data.get("object_rot")) if "object_rot" in motion_data else None
    contact_seq = np.asarray(motion_data.get("contact_sequence")).reshape(-1) if "contact_sequence" in motion_data else None

    viewer = RobotMotionViewer(
        robot_type=robot,
        motion_fps=fps,
        record_video=True,
        video_path=video_path,
    )
    try:
        for t in range(root_pos.shape[0]):
            object_data = None
            if obj_pos is not None and obj_rot is not None:
                contact = 0.0
                if contact_seq is not None and t < contact_seq.shape[0]:
                    contact = float(contact_seq[t])
                object_data = (obj_pos[t], obj_rot[t], contact)
            human_motion_data = {
                name: (world_body_pos[t, i], world_body_orient[t, i])
                for i, name in enumerate(body_names)
            }
            viewer.step(
                root_pos=root_pos[t],
                root_rot=root_rot[t],
                dof_pos=dof_pos[t],
                human_motion_data=human_motion_data,
                show_human_body_name=True,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                object_data=object_data,
                rate_limit=False,
            )
    finally:
        viewer.close()


def process_skillmimic_file(
    projector: PinocchioContactProjector,
    in_path: str,
    out_path: str,
    robot: str,
    gender: str,
    tgt_fps: int,
    device: Optional[str],
    record_video: bool,
):
    motion_data = build_motion_data_from_skillmimic(
        input_path=in_path,
        robot=robot,
        gender=gender,
        tgt_fps=tgt_fps,
        device=device,
    )
    fixed_motion = projector.project_motion(motion_data)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(fixed_motion, f)
    print(f"[INFO] Saved projected motion data to: {out_path}")
    if record_video:
        out_stem = str(pathlib.Path(out_path).with_suffix(""))
        if "/" in out_stem:
            motion_folder = out_stem.split("/", 1)[1]
        else:
            motion_folder = out_stem
        in_stem = os.path.splitext(os.path.basename(in_path))[0]
        record_video_path = f"videos/{motion_folder}/{robot}_{in_stem}.mp4"
        record_motion_video(fixed_motion, robot, record_video_path)
    return out_path


def process_skillmimic_folder(
    projector: PinocchioContactProjector,
    src_folder: str,
    tgt_folder: str,
    robot: str,
    gender: str,
    tgt_fps: int,
    device: Optional[str],
    record_video: bool,
):
    for root, _, files in os.walk(src_folder):
        rel_root = os.path.relpath(root, src_folder)
        out_root = os.path.join(tgt_folder, rel_root)
        for fname in sorted(files):
            if not fname.endswith(".pt"):
                continue
            in_path = os.path.join(root, fname)
            stem, _ = os.path.splitext(fname)
            out_path = os.path.join(out_root, stem + "_projected.pkl")
            print(f"[INFO] Processing: {in_path}")
            process_skillmimic_file(
                projector=projector,
                in_path=in_path,
                out_path=out_path,
                robot=robot,
                gender=gender,
                tgt_fps=tgt_fps,
                device=device,
                record_video=record_video,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retarget SkillMimic -> robot and run joint robot+ball optimization."
    )
    parser.add_argument("--src_folder", type=str, default="data/skillmimic/run", help="Source directory of SkillMimic .pt files")
    parser.add_argument("--tgt_folder", type=str, default="data/g1_skillmimic/run/projected_with_object", help="Target directory for projected .pkl files")
    parser.add_argument("--input_file", type=str, help="Single input SkillMimic .pt file")
    parser.add_argument("--save_path", type=str, help="Single output .pkl file")
    parser.add_argument("--tgt_fps", type=int, default=60, help="Target FPS for retargeting")
    parser.add_argument("--gender", type=str, default="neutral", choices=["male", "female", "neutral"])
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
            "unitree_h1_2_with_hands", "booster_t1", "booster_t1_29dof", "stanford_toddy",
            "fourier_n1", "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro",
            "berkeley_humanoid_lite", "booster_k1", "pnd_adam_lite", "openloong",
            "tienkung", "smplx_humanoid",
        ],
        default="unitree_g1",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device for kinematics")
    parser.add_argument("--record_video", action="store_true", default=True, help="Record a MuJoCo video for each output")
    parser.add_argument("--urdf_path", type=str, default=URDF_PATH, help="Robot URDF path")
    args = parser.parse_args()

    projector = PinocchioContactProjector(urdf_path=args.urdf_path)

    if args.src_folder and args.tgt_folder:
        process_skillmimic_folder(
            projector=projector,
            src_folder=args.src_folder,
            tgt_folder=args.tgt_folder,
            robot=args.robot,
            gender=args.gender,
            tgt_fps=args.tgt_fps,
            device=args.device,
            record_video=args.record_video,
        )
    elif args.input_file:
        save_path = args.save_path or (os.path.splitext(args.input_file)[0] + "_projected.pkl")
        process_skillmimic_file(
            projector=projector,
            in_path=args.input_file,
            out_path=save_path,
            robot=args.robot,
            gender=args.gender,
            tgt_fps=args.tgt_fps,
            device=args.device,
            record_video=args.record_video,
        )
    else:
        parser.print_help()
