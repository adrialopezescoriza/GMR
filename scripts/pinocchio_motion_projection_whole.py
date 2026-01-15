import os
import time
import numpy as np
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf
import os
import pickle

URDF_PATH = "assets/unitree_g1/g1_29dof_w_hands.urdf"
OBJECT_URDF_PATH = "assets/objects/basketball.urdf"
BASE_JOINT_NAME = "pelvis"
BASE_BODY_NAME = "pelvis"
OBJECT_BODY_NAME = "ball_link"

LINK_TRACKING_WEIGHTS_W = {
    "head_link": 1.0,
    # "pelvis": 1.0,
    "left_ankle_roll_link": 100.0,
    "right_ankle_roll_link": 100.0,
    # "left_ankle_pitch_link": 10.0,
    # "right_ankle_pitch_link": 10.0,
    # "left_knee_link": 1.0,
    # "right_knee_link": 1.0,
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
}

STATIONARY_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]
STATIONARY_SPEED_THRESH = 0.02  # [m/s] tune (e.g., 0.02–0.10)

JOINT_TRACKING_WEIGHTS = {
    # # 'left_hip_pitch_joint': 1.0, 
    # # 'left_hip_roll_joint': 1.0, 
    # 'left_hip_yaw_joint': 1.0, 
    # 'left_knee_joint': 1.0, 
    # 'left_ankle_pitch_joint': 1.0, 
    # 'left_ankle_roll_joint': 1.0, 
    # # 'right_hip_pitch_joint': 1.0, 
    # # 'right_hip_roll_joint': 1.0, 
    # 'right_hip_yaw_joint': 1.0, 
    # 'right_knee_joint': 1.0, 
    # 'right_ankle_pitch_joint': 1.0, 
    # 'right_ankle_roll_joint': 1.0, 
    # 'waist_yaw_joint': 1.0, 
    # 'waist_roll_joint': 1.0, 
    # 'waist_pitch_joint': 1.0, 
    # 'left_shoulder_pitch_joint': 1.0, 
    # 'left_shoulder_roll_joint': 1.0, 
    # 'left_shoulder_yaw_joint': 1.0, 
    # 'left_elbow_joint': 1.0, 
    # 'left_wrist_roll_joint': 1.0, 
    # 'left_wrist_pitch_joint': 1.0, 
    # 'left_wrist_yaw_joint': 1.0, 
    # 'right_shoulder_pitch_joint': 1.0, 
    # 'right_shoulder_roll_joint': 1.0, 
    # 'right_shoulder_yaw_joint': 1.0, 
    # 'right_elbow_joint': 1.0, 
    # 'right_wrist_roll_joint': 1.0, 
    # 'right_wrist_pitch_joint': 1.0, 
    # 'right_wrist_yaw_joint': 1.0,
}

CONTACT_POINTS_O_LOCAL = {
    "left_rubber_hand":  np.array([0.07, -0.11, 0.05]),
    "right_rubber_hand": np.array([0.07, 0.11, 0.05]),
} # object center position in hand frame

BALL_RADIUS = 0.12

# Axis-aligned boxes in each hand's *local frame*:
# center: box center in hand frame
# half_size: half-length along each axis [hx, hy, hz]
HAND_COLLISION_BOX = {
    "left_rubber_hand": {
        "center":    np.array([0.07, 0.08, 0.0]),   # tune as needed
        "half_size": np.array([0.15,  0.16, 0.15]),  # box extends ± these
    },
    "right_rubber_hand": {
        "center":    np.array([0.07, -0.08, 0.0]),
        "half_size": np.array([0.15, 0.16, 0.15]),
    },
}

POSE_TRACKING_WEIGHT = 1.0
LINK_POS_TRACKING_WEIGHT = 1.0
LINK_B_ORIENT_TRACKING_WEIGHT = 1.0
LINK_W_ORIENT_TRACKING_WEIGHT = 1.0
CONTACT_POS_COST_WEIGHT = 50.0

KINEMATIC_CONSISTENCY_WEIGHT = 1.0
VELOCITY_REG_WEIGHT = 5e-4


# -------------------------------------------------------------------
# Small utilities
# -------------------------------------------------------------------
def rot_from_quat_wxyz(q):
    """
    Quaternion q = [w, x, y, z] -> 3x3 rotation matrix (NumPy).
    """
    w, x, y, z = q
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    R = np.array([
        [ww + xx - yy - zz, 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     ww - xx + yy - zz, 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     ww - xx - yy + zz]
    ])
    return R

def obj_point_world(obj_pos, obj_quat, p_OQ):
    """
    Compute world position of a point Q on the object:
      p_WQ = p_WO + R_WO * p_OQ
    obj_pos: (3,) np
    obj_quat: (4,) np [w,x,y,z] (your motion_data convention)
    p_OQ: (3,) np in object frame
    """
    R = rot_from_quat_wxyz(obj_quat)
    return obj_pos + R.dot(p_OQ)

def body_to_object_center_world(body_pos_W, body_quat_W, p_BC):
    """
    Compute world position of the object center given:
      - body_pos_W: (3,) np, position of the contact body in world frame
      - body_quat_W: (4,) np [w,x,y,z], orientation of the contact body in world
      - p_BC: (3,) np, object center position expressed in the body frame

    Returns:
        p_WC: (3,) np, object center in world frame
    """
    R_WB = rot_from_quat_wxyz(body_quat_W)
    return body_pos_W + R_WB.dot(p_BC)


def quat_wxyz_from_pin(q_pin):
    """
    Convert quaternion from Pinocchio convention [qx,qy,qz,qw]
    to [qw,qx,qy,qz].
    q_pin: 4x1 MX/SX
    """
    qx = q_pin[0]
    qy = q_pin[1]
    qz = q_pin[2]
    qw = q_pin[3]
    return ca.vertcat(qw, qx, qy, qz)


def quat_conjugate_wxyz(q):
    """Conjugate of quaternion [qw,qx,qy,qz]."""
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    return ca.vertcat(qw, -qx, -qy, -qz)


def quat_mul_wxyz(q1, q2):
    """
    Hamilton product q = q1 * q2, both in [qw,qx,qy,qz] convention.
    q1, q2: 4x1 MX/SX
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return ca.vertcat(w, x, y, z)


def axis_angle_from_quat_wxyz(q, eps=1e-6):
    """
    Axis-angle vector from quaternion [qw,qx,qy,qz].
    Returns a 3x1 MX/SX whose direction is the axis and magnitude is the angle.
    """
    # Ensure shortest representation (qw >= 0)
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    sign = ca.if_else(qw < 0, -1.0, 1.0)
    qw = sign * qw
    qx = sign * qx
    qy = sign * qy
    qz = sign * qz

    mag = ca.sqrt(qx*qx + qy*qy + qz*qz)
    half_angle = ca.atan2(mag, qw)
    angle = 2.0 * half_angle

    sin_half = ca.sin(half_angle)

    # sin(theta/2)/theta with Taylor series near 0
    sin_over_angle = ca.if_else(
        ca.fabs(angle) > eps,
        sin_half / angle,
        0.5 - (angle*angle) / 48.0
    )

    ax = qx / sin_over_angle
    ay = qy / sin_over_angle
    az = qz / sin_over_angle

    return ca.vertcat(ax, ay, az)

def so3_angle_cost(R_ref_DM: ca.DM, R_sym) -> ca.MX:
    """
    Squared geodesic angle between:
      - R_ref_DM: 3x3 DM (numeric reference rotation)
      - R_sym   : 3x3 MX/SX (symbolic rotation from FK)

    Uses: R_err = R_ref^T * R_sym
          cos(theta) = (trace(R_err) - 1) / 2, clamped to [-1,1]
          cost = theta^2
    """
    R_err = R_ref_DM.T @ R_sym  # 3x3

    tr = ca.trace(R_err)
    cos_theta = (tr - 1.0) / 2.0

    # Clamp to [-1,1] to avoid acos domain errors
    cos_theta = ca.fmax(-1.0, ca.fmin(1.0, cos_theta))

    return 1 - cos_theta**2


# -------------------------------------------------------------------
# Pinocchio + CasADi robot model wrapper
# -------------------------------------------------------------------

class PinocchioCasadiRobot:
    def __init__(self, urdf_path, package_dirs=None):
        if package_dirs is None:
            package_dirs = [os.path.dirname(urdf_path)]

        root_joint = pin.JointModelFreeFlyer()
        self.model, self.collision_model, self.visual_model = \
            pin.buildModelsFromUrdf(urdf_path, package_dirs, root_joint)
        self.data = self.model.createData()

        self.nq = self.model.nq
        self.nv = self.model.nv

        # ----------------------------
        # Joint limits from URDF / Pinocchio model
        # ----------------------------
        self.q_lower = self.model.lowerPositionLimit.copy()
        self.q_upper = self.model.upperPositionLimit.copy()
        self.v_abs   = self.model.velocityLimit.copy()   # absolute (symmetric): |v| <= v_abs

        # Indices in q / v corresponding to *1-DoF joints only* (matches dof_name_to_index)
        self.joint_q_indices = []
        self.joint_v_indices = []

        for jid, joint in enumerate(self.model.joints):
            if jid <= 1:  # universe + freeflyer
                continue
            if joint.nq == 1 and joint.nv == 1:
                self.joint_q_indices.append(joint.idx_q)
                self.joint_v_indices.append(joint.idx_v)

        self.joint_q_indices = np.array(self.joint_q_indices, dtype=int)
        self.joint_v_indices = np.array(self.joint_v_indices, dtype=int)

        # Free-flyer joint is usually joint 1
        ff_joint = self.model.joints[1]
        self.base_q_start = ff_joint.idx_q   # typically 0
        self.base_q_size  = ff_joint.nq      # 7 = [x,y,z,qx,qy,qz,qw]

        self.base_body_name = BASE_BODY_NAME
        self.body_names = [f.name for f in self.model.frames]

        # Map joint names to position index (for single-DoF joints only)
        self.dof_name_to_index = {}
        for jid, joint in enumerate(self.model.joints):
            name = self.model.names[jid]
            # Skip "universe" and root free-flyer
            if jid <= 1:
                continue
            if joint.nq == 1:
                self.dof_name_to_index[name] = joint.idx_q

        # Build CasADi model
        self.cmodel = cpin.Model(self.model)

        # CasADi configuration symbol (SX is fine and simpler)
        q_sym = ca.SX.sym("q", self.cmodel.nq, 1)

        # Symbolic data and FK
        data_sym = self.cmodel.createData()
        cpin.forwardKinematics(self.cmodel, data_sym, q_sym)
        cpin.updateFramePlacements(self.cmodel, data_sym)

        # Build CasADi FK functions for all frames
        self.fk_world_pos = {}
        self.fk_world_rot = {}
        self.fk_rel_pos = {}
        self.fk_rel_rot = {}

        base_frame_id = self.cmodel.getFrameId(self.base_body_name)

        for frame_id in range(self.cmodel.nframes):
            frame = self.cmodel.frames[frame_id]
            name = frame.name

            # --- World pose of this frame ---
            oMf = data_sym.oMf[frame_id]  # SE3 (symbolic)
            p_WB = oMf.translation        # 3x1 SX
            R_WB = oMf.rotation           # 3x3 SX

            self.fk_world_pos[name] = ca.Function(
                f"fk_wp_{name}", [q_sym], [p_WB]
            )
            self.fk_world_rot[name] = ca.Function(
                f"fk_wr_{name}", [q_sym], [R_WB]
            )

            # --- Pose relative to base frame ---
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

        # Default config
        self.q0 = pin.neutral(self.model)

        # Visualization
        self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
        self.viz.initViewer(open=False)
        self.viz.loadViewerModel()

        # Add a ball to Meshcat viewer
        self.ball_path = "ball"
        self.viz.viewer[self.ball_path].set_object(
            g.Sphere(BALL_RADIUS), g.MeshLambertMaterial(color=0xFF8000)
        )

    # Simple wrappers
    def world_pos(self, body_name, q):
        return self.fk_world_pos[body_name](q)

    def world_rotmat(self, body_name, q):
        return self.fk_world_rot[body_name](q)

    def rel_pos(self, base_name, body_name, q):
        return self.fk_rel_pos[(base_name, body_name)](q)

    def rel_rotmat(self, base_name, body_name, q):
        return self.fk_rel_rot[(base_name, body_name)](q)

    # Numeric FK (standard Pinocchio) for final extraction
    def compute_kinematics(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)


# -------------------------------------------------------------------
# CasADi-based global optimization
# -------------------------------------------------------------------

class PinocchioContactProjector:
    def __init__(self, urdf_path):
        self.robot = PinocchioCasadiRobot(urdf_path)
        self.base_body_name = BASE_BODY_NAME

    def project_motion(self, motion_data: dict) -> dict:
        # --------- Extract data ----------
        fps = float(motion_data["fps"])
        dt = 1.0 / fps

        root_pos = np.asarray(motion_data["root_pos"])      # (N, 3)
        root_rot = np.asarray(motion_data["root_rot"])      # (N, 4) [w,x,y,z]
        dof_pos  = np.asarray(motion_data["dof_pos"])       # (N, num_dof)
        obj_pos  = np.asarray(motion_data["object_pos"])    # (N, 3)
        obj_rot  = np.asarray(motion_data["object_rot"])    # (N, 4) [w,x,y,z]

        dof_names             = list(motion_data["dof_names"])
        body_names            = list(motion_data["link_body_list"])
        ref_world_body_pos    = np.asarray(motion_data["world_body_pos"])
        ref_world_body_orient = np.asarray(motion_data["world_body_orient"])
        ref_local_body_pos    = np.asarray(motion_data["local_body_pos"])
        ref_local_body_orient = np.asarray(motion_data["local_body_orient"])
        contact_seq           = np.asarray(motion_data["contact_sequence"])

        N, num_dof = dof_pos.shape
        nq = self.robot.nq
        nv = self.robot.nv
        dof_idx_map = self.robot.dof_name_to_index
        dof_col_index = {name: i for i, name in enumerate(dof_names)}

        # --- Precompute reference rotation matrices from quaternions (numeric) ---

        # Base (root) world rotation reference
        R_base_ref = [rot_from_quat_wxyz(root_rot[t]) for t in range(N)]

        # World link rotations reference (shape: [N, n_bodies, 3, 3])
        R_world_ref = np.zeros(ref_world_body_orient.shape[:-1] + (3, 3))
        for t in range(N):
            for i in range(len(body_names)):
                q_wxyz = ref_world_body_orient[t, i]  # [w,x,y,z], already "correct format"
                R_world_ref[t, i] = rot_from_quat_wxyz(q_wxyz)

        # Local (base-relative) link rotations reference
        R_local_ref = np.zeros(ref_local_body_orient.shape[:-1] + (3, 3))
        for t in range(N):
            for i in range(len(body_names)):
                q_wxyz = ref_local_body_orient[t, i]
                R_local_ref[t, i] = rot_from_quat_wxyz(q_wxyz)

        # --------- Build CasADi Opti problem ----------
        opti = ca.Opti()
        q_traj = opti.variable(N, nq)   # full configuration per frame
        v_traj = opti.variable(N, nv)   # velocities (not dynamically linked, just smoothed)

        total_cost = 0

        # Base indices in Pinocchio: [x,y,z,qx,qy,qz,qw]
        base_pos_idx  = self.robot.base_q_start
        base_quat_idx = self.robot.base_q_start + 3

        ################ Per-frame constraints and costs ################
        for t in range(N):
            q_t = q_traj[t, :].T  # (nq,1) MX

            q_base_pos  = q_t[base_pos_idx : base_pos_idx+3]
            q_base_quat = q_t[base_quat_idx : base_quat_idx+4]
            q_joints   = q_t[7:]
            v_joints   = v_traj[t, 6:].T

            # Joint position + velocity limits from URDF
            for idx_q, idx_v in zip(self.robot.joint_q_indices, self.robot.joint_v_indices):
                # Position limits
                q_lo = float(self.robot.q_lower[idx_q])
                q_hi = float(self.robot.q_upper[idx_q])

                # Some URDF joints may be "continuous" and show +/- inf.
                # Skip if not finite.
                if np.isfinite(q_lo) and np.isfinite(q_hi):
                    opti.subject_to(q_t[idx_q] >= q_lo)
                    opti.subject_to(q_t[idx_q] <= q_hi)

                # Velocity limits (absolute bound)
                v_lim = float(self.robot.v_abs[idx_v])
                if np.isfinite(v_lim) and v_lim > 0:
                    opti.subject_to(v_traj[t, idx_v] >= -v_lim)
                    opti.subject_to(v_traj[t, idx_v] <=  v_lim)


            # Kinematic consistency constraint (FK vs finite diff velocities)
            if t > 0:
                q_prev_joints = q_traj[t-1, 7:].T
                v_fd = (q_joints - q_prev_joints) / dt
                total_cost += KINEMATIC_CONSISTENCY_WEIGHT * ca.sumsqr(v_fd - v_joints)

                q_prev_base_pos = q_traj[t-1, base_pos_idx : base_pos_idx+3].T
                v_fd_base = (q_base_pos - q_prev_base_pos) / dt
                v_joints_base = v_traj[t, 0:3].T

                # Constraint (with numerical stability in mind)
                opti.subject_to(v_fd_base == v_joints_base)
                # Cost
                #total_cost += KINEMATIC_CONSISTENCY_WEIGHT * ca.sumsqr(v_fd_base - v_joints_base)

                # Constraint on base orientation via quaternions
                q_prev_base_quat = q_traj[t-1, base_quat_idx : base_quat_idx+4].T
                q_conj = quat_conjugate_wxyz(q_prev_base_quat)
                q_rel = quat_mul_wxyz(q_conj, q_base_quat)  # relative rotation
                axis_angle = axis_angle_from_quat_wxyz(q_rel)
                v_fd_base_orient = axis_angle / dt
                v_joints_base_orient = v_traj[t, 3:6].T
                opti.subject_to(v_fd_base_orient == v_joints_base_orient)

                
            # --- Base position constraint (tight bounding box) ---
            # for i in range(3):
            #     opti.subject_to(q_base_pos[i] >= root_pos[t, i] - 0.5)
            #     opti.subject_to(q_base_pos[i] <= root_pos[t, i] + 0.5)

            # --- Base orientation cost ---
            R_WB = self.robot.world_rotmat(self.base_body_name, q_t)
            total_cost += POSE_TRACKING_WEIGHT * 1e-4 * ca.sumsqr(
                R_WB - ca.DM(R_base_ref[t])
            )

            # --- Penalize heavy acceleration changes ---
            if t > 1:
                v_prev = v_traj[t-1, :].T
                v_curr = v_traj[t, :].T
                a_prev = (v_curr - v_prev) / dt

                v_prev2 = v_traj[t-2, :].T
                a_prev2 = (v_prev - v_prev2) / dt

                da = a_prev - a_prev2
                total_cost += VELOCITY_REG_WEIGHT * ca.sumsqr(da)

            # --- Joint position tracking (only for selected joints) ---
            for j_name, w in JOINT_TRACKING_WEIGHTS.items():
                # Need both: a Pinocchio index for q, and a column in dof_pos
                if j_name in dof_idx_map and j_name in dof_col_index:
                    idx_q = dof_idx_map[j_name]          # index in full q
                    col   = dof_col_index[j_name]        # column in dof_pos[t, :]
                    q_ref   = dof_pos[t, col]            # scalar reference
                    q_curr  = q_t[idx_q]                 # MX scalar
                    total_cost += w * (q_curr - q_ref) ** 2


            # --- Global link tracking (world pos & orient) ---
            for link_name, w in LINK_TRACKING_WEIGHTS_W.items():
                if link_name in body_names and link_name in self.robot.fk_world_pos:
                    idx = body_names.index(link_name)
                    p_WB_ref = ref_world_body_pos[t, idx]

                    p_WB = self.robot.world_pos(link_name, q_t)
                    total_cost += w * LINK_POS_TRACKING_WEIGHT * ca.sumsqr(p_WB - p_WB_ref)

                    R_WB = self.robot.world_rotmat(link_name, q_t)
                    total_cost += w * LINK_W_ORIENT_TRACKING_WEIGHT * ca.sumsqr(
                        R_WB - ca.DM(R_world_ref[t, idx])
                    )

            # --- Local (base-relative) link tracking (only orient) ---
            for link_name, w in LINK_TRACKING_WEIGHTS_B.items():
                key = (self.base_body_name, link_name)
                if link_name in body_names and key in self.robot.fk_rel_pos:
                    idx = body_names.index(link_name)
                    p_BB_ref = ref_local_body_pos[t, idx]

                    p_BB = self.robot.rel_pos(self.base_body_name, link_name, q_t)
                    # total_cost += w * LINK_POS_TRACKING_WEIGHT * ca.sumsqr(p_BB - p_BB_ref)

                    R_BB = self.robot.rel_rotmat(self.base_body_name, link_name, q_t)
                    total_cost += w * LINK_B_ORIENT_TRACKING_WEIGHT * ca.sumsqr(
                        R_BB - ca.DM(R_local_ref[t, idx])
                    )

            # --- Contact tracking (hands to ball) ---
            for name in CONTACT_POINTS_O_LOCAL.keys():
                # Only active when contact_seq says "in contact" and we have an offset
                if name not in CONTACT_POINTS_O_LOCAL:
                    continue
                if name not in self.robot.fk_world_pos:
                    continue

                # World pose of the *optimized* contact body
                p_WB = self.robot.world_pos(name, q_t)        # 3x1 MX
                R_WB = self.robot.world_rotmat(name, q_t)     # 3x3 MX

                # Object center offset expressed in body frame (constant)
                p_BC = CONTACT_POINTS_O_LOCAL[name]           # np.array shape (3,)
                p_BC_DM = ca.DM(p_BC.reshape(3, 1))           # 3x1 DM

                # Predicted object center in world from the optimized body pose
                p_WC_pred = p_WB + R_WB @ p_BC_DM             # 3x1 MX

                # Desired object center from the reference data
                p_WC_des = ca.DM(obj_pos[t].reshape(3, 1))    # 3x1 DM

                # Contact position cost
                if contact_seq[t] > 0.5:
                    total_cost += CONTACT_POS_COST_WEIGHT * ca.sumsqr(p_WC_pred - p_WC_des)
                # elif not np.any(contact_seq[t:t+3] > 0.5):
                #     box_def = HAND_COLLISION_BOX.get(name, None)
                #     if box_def is None:
                #         # no box specified for this hand → skip
                #         continue

                #     center_B = ca.DM(box_def["center"].reshape(3, 1))      # 3x1
                #     half_B   = ca.DM(box_def["half_size"].reshape(3, 1))   # 3x1

                #     # Ball center in hand frame
                #     delta_W = p_WC_des - p_WB               # 3x1 MX
                #     p_B = R_WB.T @ delta_W                  # 3x1 MX

                #     # Distance from point p_B to axis-aligned box in hand frame
                #     # centered at 'center_B' with half-sizes 'half_B':
                #     #   d = || max(0, |p_B - center_B| - half_B) ||
                #     diff_local  = p_B - center_B           # 3x1
                #     abs_diff = ca.fabs(diff_local)      # 3x1
                #     zero_vec = ca.DM.zeros(3, 1)
                #     excess = ca.fmax(zero_vec, abs_diff - half_B)  # 3x1 MX

                #     dist_box_sq = ca.sumsqr(excess)          # scalar MX ≥ 0

                #     # Hard inequality on squared distance
                #     opti.subject_to(dist_box_sq >= (BALL_RADIUS+0.01)**2)

                    

        # Set objective
        opti.minimize(total_cost)

        # --------- Initial guess ----------
        q_init = np.zeros((N, nq))
        v_init = np.zeros((N, nv))

        for t in range(N):
            # Base pose in Pinocchio format: [x,y,z,qx,qy,qz,qw]
            x, y, z = root_pos[t]
            w_ref, qx_ref, qy_ref, qz_ref = root_rot[t]
            base_idx = self.robot.base_q_start

            q_init[t, base_idx:base_idx+3] = np.array([x, y, z])
            q_init[t, base_idx+3:base_idx+7] = np.array([qx_ref, qy_ref, qz_ref, w_ref])

            # Joints
            for i, j_name in enumerate(dof_names):
                if j_name in dof_idx_map:
                    idx_q = dof_idx_map[j_name]
                    q_init[t, idx_q] = np.clip(dof_pos[t, i], self.robot.q_lower[idx_q], self.robot.q_upper[idx_q])

        # velocities via finite diff (joints only)
        for t in range(1, N):
            v_init[t, :3] = np.clip((q_init[t, :3] - q_init[t-1, :3]) / dt, -self.robot.v_abs[:3], self.robot.v_abs[:3])
            # Orientation velocities (base)
            q_prev_quat = q_init[t-1, 3:7]
            q_curr_quat = q_init[t, 3:7]
            q_conj = quat_conjugate_wxyz(ca.DM(q_prev_quat.reshape(4,1)))
            q_rel = quat_mul_wxyz(q_conj, ca.DM(q_curr_quat.reshape(4,1)))
            axis_angle = axis_angle_from_quat_wxyz(q_rel)
            v_init[t, 3:6] = np.clip((axis_angle / dt).full().flatten(), -self.robot.v_abs[3:6], self.robot.v_abs[3:6])
            # Joint velocities
            v_init[t, 6:] = np.clip((q_init[t, 7:] - q_init[t-1, 7:]) / dt, -self.robot.v_abs[6:], self.robot.v_abs[6:])
        v_init[0, :] = v_init[1, :]

        opti.set_initial(q_traj, q_init)
        opti.set_initial(v_traj, v_init)

        # Solver options
        opts = {"ipopt.print_level": 0, "print_time": False}
        opti.solver("ipopt", opts)

        print(f"[INFO] Solving global optimization (Vars: {N*(nq+nv)})...")
        t0 = time.time()
        try:
            sol = opti.solve()
        except RuntimeError as e:
            print(f"[ERROR] Solver failed: {e}")
            print("[INFO] Returning original motion data.")
            return motion_data
        t1 = time.time()
        print("[INFO] Solver finished.")
        print(f"[INFO] Solver time: {t1 - t0:.2f} s")

        q_sol = sol.value(q_traj)

        # --------- Extract new motion_data via numeric FK ----------
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

            # base pose (world)
            oMf_base = self.robot.data.oMf[base_frame_id]
            new_root_pos[t] = np.array(oMf_base.translation).reshape(3,)
            R_WBase = oMf_base.rotation
            quat_base = pin.Quaternion(np.array(R_WBase))
            quat_base.normalize()
            new_root_rot[t] = np.array(
                [quat_base.w, quat_base.x, quat_base.y, quat_base.z]
            )

            # joints
            for i, j_name in enumerate(dof_names):
                if j_name in dof_idx_map:
                    idx_q = dof_idx_map[j_name]
                    new_dof_pos[t, i] = q_t[idx_q]
                else:
                    new_dof_pos[t, i] = dof_pos[t, i]

            # world & local body poses
            for i, b_name in enumerate(body_names):
                if b_name not in self.robot.body_names:
                    continue

                f_id = self.robot.model.getFrameId(b_name)
                oMf_body = self.robot.data.oMf[f_id]

                # world
                new_world_body_pos[t, i] = np.array(oMf_body.translation).reshape(3,)
                R_WB = oMf_body.rotation
                quat_WB = pin.Quaternion(np.array(R_WB))
                quat_WB.normalize()
                new_world_body_orient[t, i] = np.array(
                    [quat_WB.w, quat_WB.x, quat_WB.y, quat_WB.z]
                )

                # local (base-relative)
                base_to_body = oMf_base.inverse() * oMf_body
                new_local_body_pos[t, i] = np.array(base_to_body.translation).reshape(3,)
                R_BB = base_to_body.rotation
                quat_BB = pin.Quaternion(np.array(R_BB))
                quat_BB.normalize()
                new_local_body_orient[t, i] = np.array(
                    [quat_BB.w, quat_BB.x, quat_BB.y, quat_BB.z]
                )

        # --------- Visualization of the optimized trajectory ----------
        print("[INFO] Visualizing optimized trajectory in Meshcat...")
        for t in range(N):
            self.robot.viz.display(q_sol[t, :])

            # Ball transform (you use [w,x,y,z] -> Meshcat wants [x,y,z,w])
            w, x, y, z = obj_rot[t]
            T = tf.quaternion_matrix([x, y, z, w])
            T[0:3, 3] = obj_pos[t]
            self.robot.viz.viewer[self.robot.ball_path].set_transform(T)

            if t % 2 == 0:
                time.sleep(0.02)

        # --------- Build output dict ----------
        new_motion_data = dict(motion_data)
        new_motion_data["fps"] = fps
        new_motion_data["root_pos"] = new_root_pos
        new_motion_data["root_rot"] = new_root_rot
        new_motion_data["dof_pos"] = new_dof_pos
        new_motion_data["world_body_pos"] = new_world_body_pos
        new_motion_data["world_body_orient"] = new_world_body_orient
        new_motion_data["local_body_pos"] = new_local_body_pos
        new_motion_data["local_body_orient"] = new_local_body_orient

        return new_motion_data


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def process_one_file(projector: PinocchioContactProjector, in_path: str, out_dir: str):
    """
    Runs the projection for a single .pkl file and saves it to out_dir
    as <name>_projected.pkl. Returns the output path.
    """
    # Load motion data
    motion_data = np.load(in_path, allow_pickle=True)

    # Run projection
    fixed_motion_data = projector.project_motion(motion_data)

    # Build output path in the projected folder
    base_name = os.path.basename(in_path)          # e.g. "001_007pickle_pick_006.pkl"
    stem, ext = os.path.splitext(base_name)        # ("001_007pickle_pick_006", ".pkl")
    out_path = os.path.join(out_dir, stem + "_projected" + ext)

    # Save projected file
    with open(out_path, "wb") as f:
        pickle.dump(fixed_motion_data, f)

    print(f"[INFO] Saved projected motion data to: {out_path}")
    return out_path


if __name__ == "__main__":
    urdf_path = URDF_PATH
    folder = "data/g1_skillmimic/pick_40"
    projected_folder = os.path.join(folder, "projected")

    # Make sure the projected folder exists
    os.makedirs(projected_folder, exist_ok=True)

    # Build the projector once (expensive part)
    projector = PinocchioContactProjector(urdf_path=urdf_path)

    # Loop over all .pkl files in the folder
    for fname in sorted(os.listdir(folder)):
        path = os.path.join(folder, fname)

        # only process .pkl files in the main folder (not subfolders)
        if (not fname.endswith(".pkl")) or fname.endswith("_projected.pkl"):
            continue

        print(f"[INFO] Processing: {path}")
        process_one_file(projector, path, projected_folder)
