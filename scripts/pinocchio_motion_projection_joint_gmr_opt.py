import os
import time
import json
import argparse
import pathlib
from typing import Optional, List, Dict, Any

import numpy as np
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin
import torch

from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf

from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast
from general_motion_retargeting.params import IK_CONFIG_DICT
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
    load_smplx_data,
    convert_skillmimic_to_smplx,
)


# Joint optimization of: (1) GMR retargeting targets, (2) robot motion, (3) ball trajectory.
# This script loads SkillMimic .pt, builds GMR target costs, and solves one Opti problem.

URDF_PATH = "assets/unitree_g1/g1_29dof_w_hands.urdf"
BASE_BODY_NAME = "pelvis"

# Match the link tracking sets/weights used in the other projection scripts.
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

# Local offsets from contact link frame to ball center.
CONTACT_POINTS_O_LOCAL = {
    "left_rubber_hand": np.array([0.07, -0.11, 0.05]),
    "right_rubber_hand": np.array([0.07, 0.11, 0.05]),
}

# Optimization weights / tunables
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

# Foot constraints / tuning
FOOT_GROUND_Z_MIN = 0.0
FOOT_STATIC_SPEED_THRESH = 0.02
FOOT_ORIENT_WEIGHT = 50.0
FOOT_FRAME_KEYWORDS = ["foot", "toe", "ankle", "heel"]


def is_foot_frame(name: str) -> bool:
    name_l = name.lower()
    return any(key in name_l for key in FOOT_FRAME_KEYWORDS)


# Build per-frame GMR target positions/orientations from IK config.
# This mirrors GMR's scaling + offset logic but as explicit cost targets.
def build_gmr_targets(
    smplx_frames: List[Dict[str, Any]],
    ik_config: Dict[str, Any],
    actual_human_height: float,
) -> Dict[str, Any]:
    # Scale human data to match the height assumption in the IK config.
    ratio = actual_human_height / ik_config["human_height_assumption"]
    scale_table = {k: v * ratio for k, v in ik_config["human_scale_table"].items()}
    root_name = ik_config["human_root_name"]
    ground = ik_config["ground_height"] * np.array([0.0, 0.0, 1.0])

    tables = []
    if ik_config.get("use_ik_match_table1", False):
        tables.append(ik_config["ik_match_table1"])
    if ik_config.get("use_ik_match_table2", False):
        tables.append(ik_config["ik_match_table2"])

    N = len(smplx_frames)
    all_targets = []

    for table in tables:
        for robot_frame, entry in table.items():
            human_body, pos_w, rot_w, pos_offset, rot_offset = entry
            if (
                pos_w == 0
                and rot_w == 0
                and robot_frame not in LINK_TRACKING_WEIGHTS_W
                and robot_frame not in LINK_TRACKING_WEIGHTS_B
                and not is_foot_frame(robot_frame)
            ):
                continue
            if human_body not in smplx_frames[0]:
                continue
            if human_body not in scale_table or root_name not in scale_table:
                continue

            pos_refs = np.zeros((N, 3))
            rot_refs = np.zeros((N, 3, 3))
            for t in range(N):
                human = smplx_frames[t]
                root_pos, _ = human[root_name]
                scaled_root = scale_table[root_name] * root_pos

                pos, quat = human[human_body]
                if human_body == root_name:
                    scaled_pos = scaled_root
                    scaled_quat = quat
                else:
                    # Scale in the human-root local frame then re-center.
                    scaled_pos = (pos - root_pos) * scale_table[human_body] + scaled_root
                    scaled_quat = quat

                # Apply IK-config offsets in the (updated) body frame.
                pos_offset_local = np.array(pos_offset) - ground
                rot_offset_quat = np.array(rot_offset)
                updated_quat = quat_mul_wxyz(scaled_quat, rot_offset_quat)
                pos_ref = scaled_pos + rot_from_quat_wxyz(updated_quat).dot(pos_offset_local)

                pos_refs[t] = pos_ref
                rot_refs[t] = rot_from_quat_wxyz(updated_quat)

            all_targets.append(
                {
                    "frame": robot_frame,
                    "pos_w": float(pos_w),
                    "rot_w": float(rot_w),
                    "pos_ref": pos_refs,
                    "rot_ref": rot_refs,
                }
            )

    base_target = next(
        (t for t in all_targets if t["frame"] == ik_config["robot_root_name"]), None
    )
    track_targets = []
    foot_targets = []

    for t in all_targets:
        frame = t["frame"]
        if is_foot_frame(frame):
            foot_targets.append(
                {"frame": frame, "pos_ref": t["pos_ref"], "rot_ref": t["rot_ref"]}
            )

        if frame in LINK_TRACKING_WEIGHTS_W:
            w = LINK_TRACKING_WEIGHTS_W[frame]
            track_targets.append(
                {
                    "frame": frame,
                    "pos_w": float(w),
                    "rot_w": float(w),
                    "pos_ref": t["pos_ref"],
                    "rot_ref": t["rot_ref"],
                    "rel_to_base": False,
                }
            )
        elif frame in LINK_TRACKING_WEIGHTS_B:
            w = LINK_TRACKING_WEIGHTS_B[frame]
            if base_target is not None:
                R_base = base_target["rot_ref"]
                R_rel = np.einsum("tij,tjk->tik", np.transpose(R_base, (0, 2, 1)), t["rot_ref"])
            else:
                R_rel = t["rot_ref"]
            track_targets.append(
                {
                    "frame": frame,
                    "pos_w": 0.0,
                    "rot_w": float(w),
                    "pos_ref": t["pos_ref"],
                    "rot_ref": R_rel,
                    "rel_to_base": True,
                }
            )

    return {
        "targets": track_targets,
        "foot_targets": foot_targets,
        "all_targets": all_targets,
        "root_name": root_name,
        "scale_table": scale_table,
    }


# Load SkillMimic, run SMPL-X forward pass, and extract object/contact data.
def build_motion_from_skillmimic(
    input_path: str,
    gender: str,
    tgt_fps: int,
) -> Dict[str, Any]:
    # Load SMPL-X data and align to target FPS; also returns object frames.
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

    obj_pos = np.array([object_frames[k][0] for k in range(len(object_frames))])
    obj_rot = np.array([object_frames[k][1] for k in range(len(object_frames))])
    obj_contact = np.array([object_frames[k][2] for k in range(len(object_frames))])
    obj_ground_contact = (obj_pos[:, 2:] <= MIN_BALL_HEIGHT).astype(np.int8)

    return {
        "smplx_frames": smplx_frames,
        "fps": aligned_fps,
        "obj_pos": obj_pos,
        "obj_rot": obj_rot,
        "obj_contact": obj_contact,
        "obj_ground_contact": obj_ground_contact,
        "actual_human_height": actual_human_height,
    }


# Pinocchio robot wrapper with CasADi FK for all frames + Meshcat viewer.
class PinocchioCasadiRobot:
    _shared_viz = None
    _shared_viz_urdf = None

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

        if (
            PinocchioCasadiRobot._shared_viz is None
            or PinocchioCasadiRobot._shared_viz_urdf != urdf_path
        ):
            viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
            viz.initViewer(open=False)
            viz.loadViewerModel()
            viz.viewer["ball"].set_object(
                g.Sphere(BALL_RADIUS), g.MeshLambertMaterial(color=0xFF8000)
            )
            PinocchioCasadiRobot._shared_viz = viz
            PinocchioCasadiRobot._shared_viz_urdf = urdf_path

        self.viz = PinocchioCasadiRobot._shared_viz
        self.ball_path = "ball"

    def world_pos(self, body_name, q):
        return self.fk_world_pos[body_name](q)

    def world_rotmat(self, body_name, q):
        return self.fk_world_rot[body_name](q)

    def compute_kinematics(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)


# Joint optimization over robot q/v and ball position/velocity.
class JointOptimizer:
    def __init__(self, urdf_path):
        self.robot = PinocchioCasadiRobot(urdf_path)

    def optimize(
        self,
        targets: List[Dict[str, Any]],
        foot_targets: List[Dict[str, Any]],
        obj_pos: np.ndarray,
        obj_rot: np.ndarray,
        contact_seq: np.ndarray,
        ground_contact_seq: np.ndarray,
        fps: float,
        base_frame: str = BASE_BODY_NAME,
        base_init_pos: Optional[np.ndarray] = None,
        base_init_quat: Optional[np.ndarray] = None,
    ) -> dict:
        # Horizon length and timestep.
        N = targets[0]["pos_ref"].shape[0] if targets else obj_pos.shape[0]
        dt = 1.0 / float(fps)

        # If caller passed base init, use it; otherwise zeros.
        if base_init_pos is None:
            base_init_pos = np.zeros((N, 3))
        if base_init_quat is None:
            base_init_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (N, 1))

        # Align all sequences to the same horizon length.
        contact_seq = pad_or_truncate_1d(contact_seq, N)
        ground_contact_seq = pad_or_truncate_1d(ground_contact_seq, N)
        obj_pos = pad_or_truncate(obj_pos, N)
        obj_rot = pad_or_truncate(obj_rot, N)
        base_init_pos = pad_or_truncate(base_init_pos, N)
        base_init_quat = pad_or_truncate(base_init_quat, N)
        for target in targets:
            target["pos_ref"] = pad_or_truncate(target["pos_ref"], N)
            target["rot_ref"] = pad_or_truncate(target["rot_ref"], N)

        # Precompute foot static masks.
        for target in foot_targets:
            target["pos_ref"] = pad_or_truncate(target["pos_ref"], N)
            target["rot_ref"] = pad_or_truncate(target["rot_ref"], N)
            ref_vel = finite_difference_velocities(target["pos_ref"], dt)
            speed = np.linalg.norm(ref_vel, axis=1)
            target["static_mask"] = speed < FOOT_STATIC_SPEED_THRESH

        # Ball reference velocities and contact flags.
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

        # Decision variables: robot q/v per frame + ball pos/vel per frame.
        opti = ca.Opti()
        nq = self.robot.nq
        nv = self.robot.nv
        q_traj = opti.variable(N, nq)
        v_traj = opti.variable(N, nv)
        ball_pos = opti.variable(N, 3)
        ball_vel = opti.variable(N, 3)

        total_cost = 0
        base_pos_idx = self.robot.base_q_start
        base_quat_idx = self.robot.base_q_start + 3

        # Per-frame robot and ball costs/constraints.
        for t in range(N):
            q_t = q_traj[t, :].T
            q_base_pos = q_t[base_pos_idx: base_pos_idx + 3]
            q_base_quat = q_t[base_quat_idx: base_quat_idx + 4]
            q_joints = q_t[7:]
            v_joints = v_traj[t, 6:].T

            # Joint limits and velocity limits from URDF.
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
                # Kinematic consistency between q and v (finite differences).
                q_prev_joints = q_traj[t - 1, 7:].T
                v_fd = (q_joints - q_prev_joints) / dt
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

            if t > 1:
                # Smooth accelerations (jerk regularization).
                v_prev = v_traj[t - 1, :].T
                v_curr = v_traj[t, :].T
                a_prev = (v_curr - v_prev) / dt
                v_prev2 = v_traj[t - 2, :].T
                a_prev2 = (v_prev - v_prev2) / dt
                da = a_prev - a_prev2
                total_cost += VELOCITY_REG_WEIGHT * ca.sumsqr(da)

            # GMR target costs (position/orientation) per matched frame.
            for target in targets:
                frame = target["frame"]
                if frame not in self.robot.fk_world_pos:
                    continue
                p_ref = target["pos_ref"][t]
                R_ref = target["rot_ref"][t]
                if target["pos_w"] > 0:
                    p_WB = self.robot.world_pos(frame, q_t)
                    total_cost += target["pos_w"] * ca.sumsqr(p_WB - ca.DM(p_ref))
                if target["rot_w"] > 0:
                    if target.get("rel_to_base", False):
                        R_base = self.robot.world_rotmat(base_frame, q_t)
                        R_WB = self.robot.world_rotmat(frame, q_t)
                        R_rel = R_base.T @ R_WB
                        total_cost += target["rot_w"] * ca.sumsqr(R_rel - ca.DM(R_ref))
                    else:
                        R_WB = self.robot.world_rotmat(frame, q_t)
                        total_cost += target["rot_w"] * ca.sumsqr(R_WB - ca.DM(R_ref))

            # Foot constraints: stay above ground, keep orientation, and lock when reference is static.
            for target in foot_targets:
                frame = target["frame"]
                if frame not in self.robot.fk_world_pos:
                    continue
                p_ref = target["pos_ref"][t]
                R_ref = target["rot_ref"][t]
                p_WB = self.robot.world_pos(frame, q_t)
                R_WB = self.robot.world_rotmat(frame, q_t)
                opti.subject_to(p_WB[2] >= FOOT_GROUND_Z_MIN)
                if target["static_mask"][t]:
                    opti.subject_to(p_WB[:2] == ca.DM(p_ref[:2]))
                total_cost += FOOT_ORIENT_WEIGHT * ca.sumsqr(R_WB - ca.DM(R_ref))

            p_ball_t = ball_pos[t, :].T
            v_ball_t = ball_vel[t, :].T

            # Ball contact cost with all contact points.
            if body_contact_flags[t]:
                for name, p_BC_DM in contact_offsets_dm.items():
                    if name not in self.robot.fk_world_pos:
                        continue
                    p_WB = self.robot.world_pos(name, q_t)
                    R_WB = self.robot.world_rotmat(name, q_t)
                    p_WC_des = p_WB + R_WB @ p_BC_DM
                    total_cost += BALL_CONTACT_WEIGHT * ca.sumsqr(p_ball_t - p_WC_des)

            # Ball-ground constraints when contact is active.
            if ground_contact_flags[t]:
                opti.subject_to(p_ball_t[2] <= BALL_GROUND_Z_MAX)
                opti.subject_to(p_ball_t[2] >= BALL_GROUND_Z_MIN)
                opti.subject_to(v_ball_t[2] <= BALL_GROUND_VZ_ABS)
                opti.subject_to(v_ball_t[2] >= -BALL_GROUND_VZ_ABS)

        # Ball dynamics + velocity alignment costs across frames.
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
                total_cost += BALL_VEL_SMOOTH_W * ca.sumsqr(dv_z)

            dv_xy = v_next[0:2] - v_k[0:2]
            total_cost += BALL_GROUND_SMOOTH_W * ca.sumsqr(dv_xy)

            if speed_ref[k] > BALL_SPEED_THRESH:
                v_ref_k = ca.DM(v_ref[k])
                denom = (ca.norm_2(v_k) * ca.norm_2(v_ref_k) + 1e-8)
                cos_sim = (v_k.T @ v_ref_k) / denom
                total_cost += (1.0 - cos_sim)
            else:
                opti.subject_to(v_k == zero3)

        v_last = ball_vel[N - 1, :].T
        if speed_ref[N - 1] > BALL_SPEED_THRESH:
            v_ref_last = ca.DM(v_ref[N - 1])
            denom_last = (ca.norm_2(v_last) * ca.norm_2(v_ref_last) + 1e-8)
            cos_sim_last = (v_last.T @ v_ref_last) / denom_last
            total_cost += (1.0 - cos_sim_last)
        else:
            opti.subject_to(v_last == zero3)

        # Final objective.
        opti.minimize(total_cost)

        # Initial guess for robot q/v and ball states.
        q_init = np.tile(self.robot.q0, (N, 1))
        for t in range(N):
            q_init[t, base_pos_idx: base_pos_idx + 3] = base_init_pos[t]
            w, x, y, z = base_init_quat[t]
            q_init[t, base_pos_idx + 3: base_pos_idx + 7] = np.array([x, y, z, w])

        v_init = np.zeros((N, nv))
        for t in range(1, N):
            v_init[t, :3] = (q_init[t, :3] - q_init[t - 1, :3]) / dt
            q_prev_quat = q_init[t - 1, 3:7]
            q_curr_quat = q_init[t, 3:7]
            q_conj = quat_conjugate_wxyz(ca.DM(q_prev_quat.reshape(4, 1)))
            q_rel = quat_mul_wxyz(q_conj, ca.DM(q_curr_quat.reshape(4, 1)))
            axis_angle = axis_angle_from_quat_wxyz(q_rel)
            v_init[t, 3:6] = (axis_angle / dt).full().flatten()
            v_init[t, 6:] = (q_init[t, 7:] - q_init[t - 1, 7:]) / dt
        v_init[0, :] = v_init[1, :]

        opti.set_initial(q_traj, q_init)
        opti.set_initial(v_traj, v_init)
        opti.set_initial(ball_pos, obj_pos)
        opti.set_initial(ball_vel, v_ref)

        # Solve NLP.
        opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": False})

        print(f"[INFO] Solving global optimization (Vars: {N * (nq + nv)})...")
        t0 = time.time()
        try:
            sol = opti.solve()
        except RuntimeError as e:
            print(f"[ERROR] Solver failed: {e}")
            return None
        print(f"[INFO] Solver time: {time.time() - t0:.2f} s")

        q_sol = sol.value(q_traj)
        ball_pos_sol = sol.value(ball_pos)
        ball_vel_sol = sol.value(ball_vel)

        # Extract optimized kinematics via numeric FK.
        new_world_body_pos = np.zeros((N, len(self.robot.body_names), 3))
        new_world_body_orient = np.zeros((N, len(self.robot.body_names), 4))
        new_local_body_pos = np.zeros((N, len(self.robot.body_names), 3))
        new_local_body_orient = np.zeros((N, len(self.robot.body_names), 4))
        new_root_pos = np.zeros((N, 3))
        new_root_rot = np.zeros((N, 4))
        new_dof_pos = np.zeros((N, len(self.robot.dof_name_to_index)))

        dof_names = [
            name for name, _ in sorted(self.robot.dof_name_to_index.items(), key=lambda kv: kv[1])
        ]
        base_frame_id = self.robot.model.getFrameId(self.robot.base_body_name)

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
                idx_q = self.robot.dof_name_to_index[j_name]
                new_dof_pos[t, i] = q_t[idx_q]

            for i, b_name in enumerate(self.robot.body_names):
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

        # Visualize final trajectory in Meshcat (robot + ball).
        print("[INFO] Visualizing optimized trajectory in Meshcat...")
        for t in range(N):
            self.robot.viz.display(q_sol[t, :])
            if obj_rot is not None:
                w, x, y, z = obj_rot[t]
                T = tf.quaternion_matrix([x, y, z, w])
                T[0:3, 3] = ball_pos_sol[t]
                self.robot.viz.viewer[self.robot.ball_path].set_transform(T)
            if t % 2 == 0:
                time.sleep(0.02)

        # Pack output in motion_data format.
        return {
            "fps": float(fps),
            "root_pos": new_root_pos,
            "root_rot": new_root_rot,
            "dof_pos": new_dof_pos,
            "world_body_pos": new_world_body_pos,
            "world_body_orient": new_world_body_orient,
            "local_body_pos": new_local_body_pos,
            "local_body_orient": new_local_body_orient,
            "link_body_list": self.robot.body_names,
            "dof_names": dof_names,
            "object_pos": ball_pos_sol,
            "object_vel": ball_vel_sol,
            "object_rot": obj_rot,
            "contact_sequence": contact_seq.reshape(-1, 1),
            "ground_contact_sequence": ground_contact_seq.reshape(-1, 1),
        }


# Estimate base pose initialization from GMR root target; fallback to scaled human root.
def compute_base_init_from_target(
    targets: List[Dict[str, Any]],
    robot_root_name: str,
    smplx_frames: List[Dict[str, Any]],
    human_root_name: str,
    scale_table: Dict[str, float],
) -> Optional[Dict[str, np.ndarray]]:
    for target in targets:
        if target["frame"] == robot_root_name:
            pos_ref = target["pos_ref"]
            rot_ref = target["rot_ref"]
            quat_ref = np.zeros((rot_ref.shape[0], 4))
            for t in range(rot_ref.shape[0]):
                quat = pin.Quaternion(np.array(rot_ref[t]))
                quat.normalize()
                quat_ref[t] = np.array([quat.w, quat.x, quat.y, quat.z])
            return {"pos": pos_ref, "quat": quat_ref}

    # Fallback: use scaled human root directly
    if not smplx_frames or human_root_name not in smplx_frames[0]:
        return None
    N = len(smplx_frames)
    root_pos = np.zeros((N, 3))
    root_quat = np.zeros((N, 4))
    scale = scale_table.get(human_root_name, 1.0)
    for t in range(N):
        pos, quat = smplx_frames[t][human_root_name]
        root_pos[t] = scale * pos
        root_quat[t] = quat
    return {"pos": root_pos, "quat": root_quat}


# End-to-end processing of one SkillMimic file.
def process_one(input_path: str, output_path: str, robot: str, gender: str, tgt_fps: int, urdf_path: str):
    # 1) Load SkillMimic and SMPL-X frames + object data.
    motion = build_motion_from_skillmimic(input_path, gender, tgt_fps)

    # 2) Build GMR target costs from IK config.
    ik_path = IK_CONFIG_DICT["smplx"][robot]
    with open(ik_path, "r") as f:
        ik_config = json.load(f)

    gmr = build_gmr_targets(
        smplx_frames=motion["smplx_frames"],
        ik_config=ik_config,
        actual_human_height=motion["actual_human_height"],
    )

    # 3) Initialize base from root target (or scaled human root).
    targets = gmr["targets"]
    foot_targets = gmr["foot_targets"]
    base_init = compute_base_init_from_target(
        gmr["all_targets"],
        ik_config["robot_root_name"],
        motion["smplx_frames"],
        gmr["root_name"],
        gmr["scale_table"],
    )

    # 4) Joint optimization: robot motion + ball trajectory + GMR costs.
    optimizer = JointOptimizer(urdf_path=urdf_path)
    result = optimizer.optimize(
        targets=targets,
        foot_targets=foot_targets,
        obj_pos=motion["obj_pos"],
        obj_rot=motion["obj_rot"],
        contact_seq=motion["obj_contact"],
        ground_contact_seq=motion["obj_ground_contact"],
        fps=motion["fps"],
        base_frame=ik_config["robot_root_name"],
        base_init_pos=None if base_init is None else base_init["pos"],
        base_init_quat=None if base_init is None else base_init["quat"],
    )
    if result is None:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        import pickle
        pickle.dump(result, f)
    print(f"[INFO] Saved: {output_path}")


# Batch process a folder tree of SkillMimic .pt files.
def process_folder(src_folder: str, tgt_folder: str, robot: str, gender: str, tgt_fps: int, urdf_path: str):
    for root, _, files in os.walk(src_folder):
        rel_root = os.path.relpath(root, src_folder)
        out_root = os.path.join(tgt_folder, rel_root)
        for fname in sorted(files):
            if not fname.endswith(".pt"):
                continue
            in_path = os.path.join(root, fname)
            stem, _ = os.path.splitext(fname)
            out_path = os.path.join(out_root, stem + "_joint_projected.pkl")
            print(f"[INFO] Processing: {in_path}")
            process_one(in_path, out_path, robot, gender, tgt_fps, urdf_path)


if __name__ == "__main__":
    # CLI to process a single file or folder tree.
    parser = argparse.ArgumentParser(
        description="Jointly optimize GMR retargeting, robot motion, and ball trajectory."
    )
    parser.add_argument("--src_folder", type=str, default="data/skillmimic/pick_40", help="Source directory of SkillMimic .pt files")
    parser.add_argument("--tgt_folder", type=str, default="data/g1_skillmimic/pick_40/projected_full", help="Target directory for projected .pkl files")
    parser.add_argument("--input_file", type=str, help="Single input SkillMimic .pt file")
    parser.add_argument("--save_path", type=str, help="Single output .pkl file")
    parser.add_argument("--tgt_fps", type=int, default=60, help="Target FPS for retargeting")
    parser.add_argument("--gender", type=str, default="neutral", choices=["male", "female", "neutral"])
    parser.add_argument(
        "--robot",
        choices=list(IK_CONFIG_DICT["smplx"].keys()),
        default="unitree_g1",
    )
    parser.add_argument("--urdf_path", type=str, default=URDF_PATH, help="Robot URDF path")
    args = parser.parse_args()

    if args.src_folder and args.tgt_folder:
        process_folder(args.src_folder, args.tgt_folder, args.robot, args.gender, args.tgt_fps, args.urdf_path)
    elif args.input_file:
        save_path = args.save_path or (os.path.splitext(args.input_file)[0] + "_joint_projected.pkl")
        process_one(args.input_file, save_path, args.robot, args.gender, args.tgt_fps, args.urdf_path)
    else:
        parser.print_help()
