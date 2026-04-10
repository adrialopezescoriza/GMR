import os
import time
import argparse
import pathlib
import contextlib
from typing import Optional

import numpy as np
import casadi as ca
import pinocchio as pin
import meshcat.transformations as tf
import torch
import pickle

from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.object_model_utils import (
    get_object_motion_defaults,
    resolve_object_model_path,
)
from general_motion_retargeting.pinocchio_model import PinocchioCasadiRobot
from general_motion_retargeting.projection_data_utils import (
    build_projection_motion_data_with_gmr,
)
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

def _compute_object_contact_points_world_by_frame(
    motion_data: dict,
    num_frames: int,
) -> list[dict[str, np.ndarray]]:
    contact_link_names = list(motion_data.get("contact_link_names", []))
    fixed_contact_points_object = motion_data.get("fixed_contact_points_in_object_frame")
    if fixed_contact_points_object is None or not contact_link_names:
        return []

    contact_points_object = np.asarray(fixed_contact_points_object, dtype=np.float32)
    if contact_points_object.size == 0:
        return []
    contact_points_object = contact_points_object.reshape(-1, 3)
    if contact_points_object.shape[0] != len(contact_link_names):
        return []

    if "object_pos" not in motion_data or "object_rot" not in motion_data:
        return []
    object_pos = pad_or_truncate(np.asarray(motion_data["object_pos"], dtype=np.float32), num_frames)
    object_rot = pad_or_truncate(np.asarray(motion_data["object_rot"], dtype=np.float32), num_frames)

    contact_points_world_by_frame = []
    for t in range(num_frames):
        R_world_object = rot_from_quat_wxyz(object_rot[t])
        frame_contact_points = {}
        for c, link_name in enumerate(contact_link_names):
            frame_contact_points[link_name] = (
                object_pos[t] + R_world_object @ contact_points_object[c]
            )
        contact_points_world_by_frame.append(frame_contact_points)
    return contact_points_world_by_frame


URDF_PATH = "assets/unitree_g1/g1_29dof_w_hands.urdf"
OBJECT_MODEL_PATH = "assets/objects/basketball.urdf"
AUTO_OBJECT_FROM_PATH = True
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

POSE_TRACKING_WEIGHT = 1.0
LINK_POS_TRACKING_WEIGHT = 1.0
LINK_B_ORIENT_TRACKING_WEIGHT = 1.0
LINK_W_ORIENT_TRACKING_WEIGHT = 1.0

KINEMATIC_CONSISTENCY_WEIGHT = 1.0
VELOCITY_REG_WEIGHT = 5e-4

BALL_RADIUS = 0.12
OBJECT_GRAVITY = np.array([0.0, 0.0, -9.81])
OBJECT_SPEED_THRESH = 0.05
OBJECT_CONTACT_WEIGHT = 500.0
OBJECT_VEL_SMOOTH_W = 10.0
OBJECT_GROUND_SMOOTH_W = 200.0
OBJECT_GROUND_VZ_ABS = 0.05


class PinocchioContactProjector:
    def __init__(
        self,
        urdf_path,
        object_model_path,
        min_object_height,
        anchor_links_smplx=None,
        contact_links=None,
        optimize_object=True,
    ):
        self.robot = PinocchioCasadiRobot(
            urdf_path,
            object_model_path=object_model_path,
            base_body_name=BASE_BODY_NAME,
            object_radius=BALL_RADIUS,
        )
        self.base_body_name = BASE_BODY_NAME
        self.min_object_height = float(min_object_height)
        self.anchor_links_smplx = anchor_links_smplx or []
        self.contact_links = contact_links or []
        self.optimize_object = bool(optimize_object)

    def project_motion(self, motion_data: dict) -> dict:
        # Stage 2: build and solve one global trajectory optimization over the
        # whole sequence. The GMR output in `motion_data` is used as the
        # reference/initial guess, and this step adjusts the robot trajectory
        # (and optionally the object trajectory) to reduce artifacts while
        # staying close to the retargeted motion.
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
                (obj_pos[:, 2:] <= self.min_object_height).astype(np.int8),
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
        # Enforce hard ground constraints only when the object is effectively static.
        ground_contact_hard_flags = ground_contact_flags & (speed_ref <= OBJECT_SPEED_THRESH)
        g_vec_dm = ca.DM(OBJECT_GRAVITY)
        zero3 = ca.DM.zeros(3, 1)

        contact_points_object_dm = {}
        contact_link_names = list(motion_data.get("contact_link_names", []))
        fixed_contact_points_object = np.asarray(
            motion_data.get("fixed_contact_points_in_object_frame", []),
            dtype=np.float32,
        )
        if fixed_contact_points_object.size > 0:
            fixed_contact_points_object = fixed_contact_points_object.reshape(-1, 3)
            if fixed_contact_points_object.shape[0] == len(contact_link_names):
                contact_points_object_dm = {
                    name: ca.DM(fixed_contact_points_object[i].reshape(3, 1))
                    for i, name in enumerate(contact_link_names)
                }
        R_world_object_dm = [ca.DM(rot_from_quat_wxyz(obj_rot[t])) for t in range(N)]

        # Main optimization variables:
        # - q_traj: robot configuration at every frame
        # - v_traj: robot velocity at every frame
        # - object_pos/object_vel: optional object trajectory variables
        opti = ca.Opti()
        q_traj = opti.variable(N, nq)
        v_traj = opti.variable(N, nv)
        object_pos = opti.variable(N, 3) if self.optimize_object else None
        object_vel = opti.variable(N, 3) if self.optimize_object else None

        total_cost = 0
        base_pos_idx = self.robot.base_q_start
        base_quat_idx = self.robot.base_q_start + 3

        # Per-frame objective/constraint assembly:
        # - enforce joint/velocity limits
        # - keep velocities consistent with finite differences of q
        # - track selected link poses from the GMR reference
        # - penalize robot/object contact mismatch
        # - optionally constrain the object during static ground contact
        for t in range(N):
            q_t = q_traj[t, :].T
            q_base_pos = q_t[base_pos_idx: base_pos_idx + 3]
            q_base_quat_xyzw = q_t[base_quat_idx: base_quat_idx + 4]
            q_base_quat = ca.vertcat(
                q_base_quat_xyzw[3],
                q_base_quat_xyzw[0],
                q_base_quat_xyzw[1],
                q_base_quat_xyzw[2],
            )
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

                q_prev_base_quat_xyzw = q_traj[t - 1, base_quat_idx: base_quat_idx + 4].T
                q_prev_base_quat = ca.vertcat(
                    q_prev_base_quat_xyzw[3],
                    q_prev_base_quat_xyzw[0],
                    q_prev_base_quat_xyzw[1],
                    q_prev_base_quat_xyzw[2],
                )
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

            if self.optimize_object:
                p_object_t = object_pos[t, :].T
                v_object_t = object_vel[t, :].T
                
                if ground_contact_hard_flags[t]:
                    opti.subject_to(p_object_t[2] <= (obj_pos[t, 2] + 0.01))
                    opti.subject_to(p_object_t[2] >= (obj_pos[t, 2] - 0.01))
                    opti.subject_to(v_object_t[2] <= OBJECT_GROUND_VZ_ABS)
                    opti.subject_to(v_object_t[2] >= -OBJECT_GROUND_VZ_ABS)
            else:
                p_object_t = ca.DM(obj_pos[t, :].reshape(3, 1))
                v_object_t = ca.DM(v_ref[t, :].reshape(3, 1))

            if body_contact_flags[t]:
                for name, p_OC_DM in contact_points_object_dm.items():
                    if name not in self.robot.fk_world_pos:
                        continue
                    p_WB = self.robot.world_pos(name, q_t)  # contact-link world position
                    p_WC_from_object = p_object_t + R_world_object_dm[t] @ p_OC_DM  # contact point world position from object
                    total_cost += OBJECT_CONTACT_WEIGHT * ca.sumsqr(p_WC_from_object - p_WB)

        # Sequence-level object dynamics terms:
        # - enforce position integration p_{k+1} = p_k + dt * v_k
        # - encourage ballistic motion in free flight
        # - smooth velocities during contact
        # - keep velocity direction aligned with the reference when moving
        if self.optimize_object:
            for k in range(N - 1):
                p_k = object_pos[k, :].T
                v_k = object_vel[k, :].T
                p_next = object_pos[k + 1, :].T
                v_next = object_vel[k + 1, :].T

                opti.subject_to(p_next == p_k + dt * v_k)

                is_contact_k = body_contact_flags[k] or ground_contact_flags[k]
                is_contact_k1 = body_contact_flags[k + 1] or ground_contact_flags[k + 1]

                if (not is_contact_k) and (not is_contact_k1):
                    # Free flight: penalize deviation from ballistic trajectory under gravity.
                    a_free = (v_next - v_k) / dt - g_vec_dm
                    total_cost += OBJECT_VEL_SMOOTH_W * ca.sumsqr(a_free)
                    # opti.subject_to(v_next == v_k + dt * g_vec_dm)
                else:
                    dv_z = v_next[2] - v_k[2]
                    # Cost: smooth vertical velocity while in contact.
                    total_cost += OBJECT_VEL_SMOOTH_W * ca.sumsqr(dv_z)

                dv_xy = v_next[0:2] - v_k[0:2]
                # Cost: smooth horizontal velocity changes.
                total_cost += OBJECT_VEL_SMOOTH_W * ca.sumsqr(dv_xy)

                if speed_ref[k] > OBJECT_SPEED_THRESH:
                    v_ref_k = ca.DM(v_ref[k])
                    denom = (ca.norm_2(v_k) * ca.norm_2(v_ref_k) + 1e-8)
                    cos_sim = (v_k.T @ v_ref_k) / denom
                    # Cost: align ball velocity direction with reference when moving.
                    total_cost += (1.0 - cos_sim)
                else:
                    opti.subject_to(v_k == zero3)

            v_last = object_vel[N - 1, :].T
            if speed_ref[N - 1] > OBJECT_SPEED_THRESH:
                v_ref_last = ca.DM(v_ref[N - 1])
                denom_last = (ca.norm_2(v_last) * ca.norm_2(v_ref_last) + 1e-8)
                cos_sim_last = (v_last.T @ v_ref_last) / denom_last
                # Cost: align final ball velocity direction with reference when moving.
                total_cost += (1.0 - cos_sim_last)
            else:
                opti.subject_to(v_last == zero3)

        opti.minimize(total_cost)

        # Stage 3: initialize the solver from the GMR reference motion so IPOPT
        # starts from a sensible trajectory instead of from scratch.
        ## Initialize optimization with reference motion.
        q_init = np.zeros((N, nq))
        v_init = np.zeros((N, nv))
        base_idx = self.robot.base_q_start
        for t in range(N):
            x, y, z = root_pos[t]
            w_ref, qx_ref, qy_ref, qz_ref = root_rot[t]
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
                (q_init[t, base_idx:base_idx + 3] - q_init[t - 1, base_idx:base_idx + 3]) / dt,
                -self.robot.v_abs[:3],
                self.robot.v_abs[:3],
            )
            q_prev_quat = q_init[t - 1, base_idx + 3:base_idx + 7][[3, 0, 1, 2]]
            q_curr_quat = q_init[t, base_idx + 3:base_idx + 7][[3, 0, 1, 2]]
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
        if N > 1:
            v_init[0, :] = v_init[1, :]

        opti.set_initial(q_traj, q_init)
        opti.set_initial(v_traj, v_init)
        object_vel_init = motion_data.get("object_vel")
        if object_vel_init is not None:
            object_vel_init = pad_or_truncate(object_vel_init, N)
        else:
            object_vel_init = v_ref

        if self.optimize_object:
            opti.set_initial(object_pos, obj_pos)
            opti.set_initial(object_vel, object_vel_init)

        opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": False})

        # Stage 4: solve the nonlinear program once for the full trajectory.
        print(f"[INFO] Solving gradient-based optimization (Vars: {N * (nq + nv)})...")
        t0 = time.time()
        try:
            sol = opti.solve()
        except RuntimeError as e:
            print(f"[ERROR] Solver failed: {e}")
            print("[INFO] Returning original motion data.")
            return motion_data
        print(f"[INFO] Solver time: {time.time() - t0:.2f} s")

        # Stage 5: unpack the optimized decision variables and convert the
        # optimized robot state back into the saved motion-data representation.
        q_sol = sol.value(q_traj)
        if self.optimize_object:
            object_pos_sol = sol.value(object_pos)
            object_vel_sol = sol.value(object_vel)
        else:
            object_pos_sol = obj_pos
            object_vel_sol = object_vel_init

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

        # Stage 6: visualize the optimized result in Meshcat for inspection.
        # --------- Visualization of the optimized trajectory ----------
        print("[INFO] Visualizing optimized trajectory in Meshcat...")
        for t in range(N):
            self.robot.viz.display(q_sol[t, :])

            # Ball transform (you use [w,x,y,z] -> Meshcat wants [x,y,z,w])
            w, x, y, z = obj_rot[t]
            T = tf.quaternion_matrix([x, y, z, w])
            T[0:3, 3] = object_pos_sol[t]
            self.robot.viz.viewer[self.robot.ball_path].set_transform(T)

            if t % 2 == 0:
                time.sleep(0.02)


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
                "object_pos": object_pos_sol,
                "object_vel": object_vel_sol,
                "object_rot": obj_rot,
                "contact_sequence": contact_seq.reshape(-1, 1),
                "ground_contact_sequence": ground_contact_seq.reshape(-1, 1),
            }
        )

        return new_motion_data


def record_motion_video(
    motion_data: dict,
    robot: str,
    video_path: str,
    object_model_path: Optional[str] = None,
):
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
    contact_points_world_by_frame = _compute_object_contact_points_world_by_frame(
        motion_data=motion_data,
        num_frames=root_pos.shape[0],
    )

    viewer = RobotMotionViewer(
        robot_type=robot,
        motion_fps=fps,
        record_video=True,
        video_path=video_path,
        object_model_path=object_model_path,
    )
    try:
        for t in range(root_pos.shape[0]):
            contact = float(contact_seq[t]) if (contact_seq is not None and t < contact_seq.shape[0]) else 0.0
            object_data = None
            if obj_pos is not None and obj_rot is not None:
                object_data = (obj_pos[t], obj_rot[t], contact)
            human_motion_data = {
                name: (world_body_pos[t, i], world_body_orient[t, i])
                for i, name in enumerate(body_names)
            }
            object_contact_points_world = None
            if t < len(contact_points_world_by_frame):
                object_contact_points_world = contact_points_world_by_frame[t]
            viewer.step(
                root_pos=root_pos[t],
                root_rot=root_rot[t],
                dof_pos=dof_pos[t],
                human_motion_data=human_motion_data,
                show_human_body_name=True,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                object_data=object_data,
                object_contact_points_world=object_contact_points_world,
                rate_limit=True,
            )
    finally:
        viewer.close()


def process_file(
    projector: PinocchioContactProjector,
    in_path: str,
    out_path: str,
    robot: str,
    gender: str,
    tgt_fps: int,
    device: str,
    record_video: bool,
    object_model_path: Optional[str] = None,
):
    # Stage 1: build the initial projection data with GMR.
    # This loads the source motion, retargets it frame-by-frame to the robot,
    # computes FK reference poses, and may also run an object-only optimization
    # to clean up the object trajectory before the global projection stage.
    with open(os.devnull, "w") as _null, contextlib.redirect_stdout(_null), contextlib.redirect_stderr(_null):
        motion_data, source_tag = build_projection_motion_data_with_gmr(
            input_path=in_path,
            robot=robot,
            gender=gender,
            tgt_fps=tgt_fps,
            device=device,
            min_object_height=projector.min_object_height,
            anchor_links_smplx=projector.anchor_links_smplx,
            contact_links=projector.contact_links,
            object_speed_thresh=OBJECT_SPEED_THRESH,
        )
    print(f"[INFO] Loaded motion source '{source_tag}' from '{in_path}'.")

    # Stage 2: run the whole-body Pinocchio/CasADi projection that refines the
    # retargeted motion and optionally co-optimizes the object trajectory.
    fixed_motion = projector.project_motion(motion_data)
    # fixed_motion = motion_data
    if object_model_path is not None:
        fixed_motion = dict(fixed_motion)
        fixed_motion["object_model_path"] = object_model_path

    # Stage 3: save the optimized motion for later playback / training / export.
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(fixed_motion, f)
    print(f"[INFO] Saved projected motion data to: {out_path}")
    if record_video:
        # Stage 4: optionally render a video of the final optimized trajectory.
        out_stem = str(pathlib.Path(out_path).with_suffix(""))
        if "/" in out_stem:
            motion_folder = out_stem.split("/", 1)[1]
        else:
            motion_folder = out_stem
        in_stem = os.path.splitext(os.path.basename(in_path))[0]
        record_video_path = f"videos/{motion_folder}/{robot}_{in_stem}.mp4"
        record_motion_video(
            fixed_motion,
            robot,
            record_video_path,
            object_model_path=object_model_path or fixed_motion.get("object_model_path"),
        )
    return out_path


def process_folder(
    projector_cache: dict,
    src_folder: str,
    tgt_folder: str,
    robot: str,
    gender: str,
    tgt_fps: int,
    device: str,
    record_video: bool,
    urdf_path: str,
    default_object_model_path: str,
    optimize_object: bool,
):
    for root, _, files in os.walk(src_folder):
        rel_root = os.path.relpath(root, src_folder)
        out_root = os.path.join(tgt_folder, rel_root)
        for fname in sorted(files):
            if not (fname.endswith(".pt") or fname.endswith(".pkl")):
                continue
            in_path = os.path.join(root, fname)
            stem, _ = os.path.splitext(fname)
            out_path = os.path.join(out_root, stem + "_projected.pkl")
            print(f"[INFO] Processing: {in_path}")

            object_model_path = resolve_object_model_path(
                motion_path=in_path,
                configured_object_model_path=default_object_model_path,
                auto_from_motion_path=AUTO_OBJECT_FROM_PATH,
                default_object_model_path=default_object_model_path,
            )
            if object_model_path not in projector_cache:
                print(f"[INFO] Using object model: {object_model_path}")
                object_defaults = get_object_motion_defaults(object_model_path)
                projector_cache[object_model_path] = PinocchioContactProjector(
                    urdf_path=urdf_path,
                    object_model_path=object_model_path,
                    min_object_height=object_defaults["min_object_height"],
                    anchor_links_smplx=object_defaults["anchor_links_smplx"],
                    contact_links=object_defaults["contact_links"],
                    optimize_object=optimize_object,
                )
            projector = projector_cache[object_model_path]
            print(f"projector: {projector}")
            print(f"in_path: {in_path}")
            print(f"out_path: {out_path}")
            process_file(
                projector=projector,
                in_path=in_path,
                out_path=out_path,
                robot=robot,
                gender=gender,
                tgt_fps=tgt_fps,
                device=device,
                record_video=record_video,
                object_model_path=object_model_path,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retarget SMPL motion data with GMR and run joint robot+object optimization."
    )
    parser.add_argument("--src_folder", type=str, default="data/skillmimic/run", help="Source directory of SMPL motion files")
    parser.add_argument("--tgt_folder", type=str, default="data/skillmimic/run/projected_with_object", help="Target directory for projected .pkl files")
    parser.add_argument("--input_file", type=str, help="Single input SMPL motion file")
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
    parser.add_argument("--record_video", action="store_true", default=True, help="Record a MuJoCo video for each output")
    parser.add_argument("--urdf_path", type=str, default=URDF_PATH, help="Robot URDF path")
    parser.add_argument(
        "--object_model_path",
        type=str,
        default=OBJECT_MODEL_PATH,
        help="Optional external object visual model (.urdf/.xml/.obj/.stl/.dae).",
    )
    parser.add_argument(
        "--optimize_object",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include object position/velocity variables and object-specific constraints in the NLP.",
    )
    args = parser.parse_args()

    projector_cache = {}

    if args.src_folder and args.tgt_folder:
        process_folder(
            projector_cache=projector_cache,
            src_folder=args.src_folder,
            tgt_folder=args.tgt_folder,
            robot=args.robot,
            gender=args.gender,
            tgt_fps=args.tgt_fps,
            device="cuda" if torch.cuda.is_available() else "cpu",
            record_video=args.record_video,
            urdf_path=args.urdf_path,
            default_object_model_path=args.object_model_path,
            optimize_object=args.optimize_object,
        )
    elif args.input_file:
        object_model_path = resolve_object_model_path(
            motion_path=args.input_file,
            configured_object_model_path=args.object_model_path,
            auto_from_motion_path=AUTO_OBJECT_FROM_PATH,
            default_object_model_path=args.object_model_path,
        )
        if object_model_path not in projector_cache:
            print(f"[INFO] Using object model: {object_model_path}")
            object_defaults = get_object_motion_defaults(object_model_path)
            projector_cache[object_model_path] = PinocchioContactProjector(
                urdf_path=args.urdf_path,
                object_model_path=object_model_path,
                min_object_height=object_defaults["min_object_height"],
                anchor_links_smplx=object_defaults["anchor_links_smplx"],
                contact_links=object_defaults["contact_links"],
                optimize_object=args.optimize_object,
            )
        projector = projector_cache[object_model_path]

        save_path = args.save_path or (os.path.splitext(args.input_file)[0] + "_projected.pkl")
        process_file(
            projector=projector,
            in_path=args.input_file,
            out_path=save_path,
            robot=args.robot,
            gender=args.gender,
            tgt_fps=args.tgt_fps,
            device="cuda" if torch.cuda.is_available() else "cpu",
            record_video=args.record_video,
            object_model_path=object_model_path,
        )
    else:
        parser.print_help()
