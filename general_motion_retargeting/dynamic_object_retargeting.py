
import numpy as np
import casadi as ca
from scipy.spatial.transform import Rotation as R

# ------------------------ helpers ------------------------ #

def finite_difference_velocities(positions: np.ndarray, dt: float) -> np.ndarray:
    """
    Simple finite-difference velocities for a (T, 3) position array.
    Central differences for interior points, forward/backward for endpoints.
    """
    T, D = positions.shape
    vel = np.zeros_like(positions)

    if T < 2:
        return vel

    # Central differences for interior points
    vel[1:-1] = (positions[2:] - positions[:-2]) / (2.0 * dt)

    # Forward / backward for endpoints
    vel[0] = (positions[1] - positions[0]) / dt
    vel[-1] = (positions[-1] - positions[-2]) / dt

    return vel


def build_and_solve_ball_optimization(
    p_ref: np.ndarray,
    v_ref: np.ndarray,
    body_contact_flags: np.ndarray,
    ground_contact_flags: np.ndarray,
    link_pos_world_offset: np.ndarray,
    fps: int,
    g_vec: np.ndarray = np.array([0.0, 0.0, -9.81]),
    speed_thresh: float = 0.05,  # threshold for "ball is moving"
):
    """
    Build and solve the CasADi NLP for the ball trajectory.

    - Always enforce kinematics: p_{k+1} = p_k + dt * v_k
    - Enforce ballistic velocity dynamics only in free flight:
        v_{k+1} = v_k + dt * g   if no contact at k or k+1
    - If ||v_ref[k]|| <= speed_thresh:
        * No cosine cost at step k
        * Enforce v_k = 0
      Otherwise:
        * Use cosine-difference cost between v_k and v_ref[k]
    - Contact constraints (unchanged):
        * Body contact: p = link_pos + hand_offset
        * Ground contact: p_z <= 0.12, v_z = 0
    """

    T = p_ref.shape[0]
    dt = 1.0 / float(fps)
    nx = 6  # [px, py, pz, vx, vy, vz]

    # Decision variables: X in R^{nx x T}
    X = ca.MX.sym("X", nx, T)

    eq_constraints = []   # g_eq(x) = 0
    ineq_constraints = [] # g_ineq(x) <= 0

    J = 0.0
    g_vec_ca = ca.DM(g_vec)

    body_contact_flags = np.asarray(body_contact_flags, dtype=bool)
    ground_contact_flags = np.asarray(ground_contact_flags, dtype=bool)

    # ---------------- dynamics + cost ---------------- #
    for k in range(T - 1):
        x_k     = X[:, k]
        x_next  = X[:, k + 1]

        p_k     = x_k[0:3]
        v_k     = x_k[3:6]
        p_next  = x_next[0:3]
        v_next  = x_next[3:6]

        # 1) KINEMATICS: always enforce p_{k+1} = p_k + dt * v_k
        eq_constraints.append(p_next - (p_k + dt * v_k))

        # 2) VELOCITY dynamics: only in free flight
        is_contact_k  = body_contact_flags[k]  or ground_contact_flags[k]
        is_contact_k1 = body_contact_flags[k+1] or ground_contact_flags[k+1]

        if (not is_contact_k) and (not is_contact_k1):
            # Free motion: v_{k+1} = v_k + dt * g
            eq_constraints.append(v_next - (v_k + dt * g_vec_ca))
        else:
            # If not free motion, z velocity smoothing cost
            vel_smooth_w = 10.0  # tune 10–200 depending on severity
            dv_z = v_next[2] - v_k[2]       # (vz_next - vz_k)
            J = J + vel_smooth_w * ca.sumsqr(dv_z)
        
        # Penalize horizontal velocity changes
        ground_smooth_w = 200.0  # tune 50–500 depending on severity
        dv_xy = v_next[0:2] - v_k[0:2]       # (vx_next - vx_k, vy_next - vy_k)
        J = J + ground_smooth_w * ca.sumsqr(dv_xy)

        # 3) Cost / constraints on velocity depending on ref speed
        v_ref_k_np = v_ref[k]               # (3,) numpy
        speed_ref_k = np.linalg.norm(v_ref_k_np)

        if speed_ref_k > speed_thresh:
            # ball is moving: use cosine-difference cost
            v_ref_k = ca.DM(v_ref_k_np)
            denom = (ca.norm_2(v_k) * ca.norm_2(v_ref_k) + 1e-8)
            cos_sim = (v_k.T @ v_ref_k) / denom
            J = J + (1.0 - cos_sim)
        else:
            # ball is effectively static: no cost, enforce v_k = 0
            eq_constraints.append(v_k)

    # ---- last step: same logic on v_T-1 ----
    v_last = X[3:6, -1]
    v_ref_last_np = v_ref[-1]
    speed_ref_last = np.linalg.norm(v_ref_last_np)

    if speed_ref_last > speed_thresh:
        v_ref_last = ca.DM(v_ref_last_np)
        denom_last = (ca.norm_2(v_last) * ca.norm_2(v_ref_last) + 1e-8)
        cos_sim_last = (v_last.T @ v_ref_last) / denom_last
        J = J + (1.0 - cos_sim_last)
    else:
        # static at last frame
        eq_constraints.append(v_last)

    # ---------------- contact constraints and cost ---------------- #

    body_contact_weight = 500.0  # you can tune this (50–2000)

    for k in range(T):
        x_k = X[:, k]
        p_k = x_k[0:3]
        v_k = x_k[3:6]

        if body_contact_flags[k]:
            link_p_k = ca.DM(link_pos_world_offset[k])  # (3,)
            desired_p = link_p_k
            diff = p_k - desired_p
            J = J + body_contact_weight * ca.sumsqr(diff)

        if ground_contact_flags[k]:
            # GROUND contact:
            #   p_z >= 0.1, p_z <= 0.14
            #   v_z >= -0.05, v_z <= 0.05
            p_z = p_k[2]
            v_z = v_k[2]
            ineq_constraints.append(p_z - 0.14)   # p_z <= 0.14
            ineq_constraints.append(-p_z + 0.10)  # p_z >= 0.10
            ineq_constraints.append(v_z - 0.05)    # v_z <= 0.05
            ineq_constraints.append(-v_z - 0.05)   # v_z >= -0.05

    # ---------------- stack constraints & solve ---------------- #

    g_list = []
    lbg_list = []
    ubg_list = []

    # Equalities: g_eq(x) = 0
    for c in eq_constraints:
        g_list.append(c)
        n_c = int(c.size1())
        lbg_list.append(np.zeros(n_c))
        ubg_list.append(np.zeros(n_c))

    # Inequalities: g_ineq(x) <= 0  ->  lbg = -BIG, ubg = 0
    BIG = 1e6
    for c in ineq_constraints:
        g_list.append(c)
        n_c = int(c.size1())
        lbg_list.append(-BIG * np.ones(n_c))
        ubg_list.append(np.zeros(n_c))

    if len(g_list) > 0:
        g_all = ca.vertcat(*g_list)
        lbg = np.concatenate(lbg_list)
        ubg = np.concatenate(ubg_list)
    else:
        g_all = ca.DM([])
        lbg = np.array([])
        ubg = np.array([])

    nlp = {
        "x": ca.vec(X),
        "f": J,
        "g": g_all,
    }

    opts = {
        "ipopt.print_level": 5,
        "ipopt.max_iter": 2000,
        "print_time": False,
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    # Initial guess: reference trajectory
    x0 = np.zeros((nx, T))
    x0[0:3, :] = p_ref.T
    x0[3:6, :] = v_ref.T
    x0_vec = x0.flatten(order="F")

    sol = solver(x0=x0_vec, lbg=lbg, ubg=ubg)

    x_opt_vec = np.array(sol["x"]).squeeze()
    x_opt = x_opt_vec.reshape((nx, T), order="F")

    p_opt = x_opt[0:3, :].T  # (T, 3)
    v_opt = x_opt[3:6, :].T  # (T, 3)

    return p_opt, v_opt

def optimize_object_traj_from_motion(
    motion_data: dict,
    contact_link_name: str = "right_rubber_hand",
    local_offset: np.ndarray = np.array([0.0, 0.0, 0.0]),
    speed_thresh: float = 0.05,
):
    """
    Given a motion_data dict, build the inputs for `build_and_solve_ball_optimization`
    and return optimized object positions and velocities.

    motion_data keys expected:
        - "fps"
        - "world_body_pos"         : (T_body, N_links, 3)
        - "link_body_list"         : list of length N_links
        - "object_pos"             : (T_ball, 3)
        - "contact_sequence"       : (T_ball, 1) or (T_ball,)
        - "ground_contact_sequence": (T_ball, 1) or (T_ball,)

    Returns:
        p_opt: (T_ball, 3) optimized object positions
        v_opt: (T_ball, 3) optimized object velocities
    """
    # --- Extract basics ---
    fps = int(motion_data["fps"])
    p_ref = np.asarray(motion_data["object_pos"], dtype=float)  # (T_ball, 3)
    contact_seq = np.asarray(motion_data["contact_sequence"]).reshape(-1)
    ground_contact_seq = np.asarray(motion_data["ground_contact_sequence"]).reshape(-1)

    body_contact_flags = (contact_seq != 0)
    ground_contact_flags = (ground_contact_seq != 0)

    world_body_pos = np.asarray(motion_data["world_body_pos"], dtype=float)  # (T_body, N_links, 3)
    world_body_orient = np.asarray(motion_data["world_body_orient"], dtype=float) # (T_body, N_links, 4)
    link_names = list(motion_data["link_body_list"])

    # --- Choose link index for body contact ---
    if contact_link_name in link_names:
        link_idx = link_names.index(contact_link_name)
    else:
        # Fallback: last link if not found
        print(
            f"[WARN] Link '{contact_link_name}' not found in link_body_list. "
            f"Available links include e.g. {link_names[:5]}... Using last link instead."
        )
        link_idx = len(link_names) - 1

    T_ball = p_ref.shape[0]
    T_body, N_links, _ = world_body_pos.shape

    # --- Build link_pos aligned with object trajectory length ---
    link_pos = np.zeros((T_ball, 3), dtype=float)
    link_orient = np.zeros((T_ball, 4), dtype=float)

    T_min = min(T_ball, T_body)
    link_pos[:T_min] = world_body_pos[:T_min, link_idx, :]
    link_orient[:T_min] = world_body_orient[:T_min, link_idx, :]

    # Repeat last pose if object is longer than body sequence
    if T_ball > T_body:
        link_pos[T_body:] = world_body_pos[T_body - 1, link_idx, :]
        link_orient[T_body:] = world_body_orient[T_body - 1, link_idx, :]

    # --- Convert link offset into world frame each timestep ---
    world_offset = np.zeros((T_ball, 3), dtype=float)
    for t in range(T_ball):
        q = link_orient[t,[1,2,3,0]]  # quaternion (x, y, z, w)
        rot = R.from_quat(q)
        world_offset[t] = rot.apply(local_offset)

    # Final desired link positions = hand position + rotated offset
    link_pos_world_offset = link_pos + world_offset

    # --- Reference velocities from positions ---
    dt = 1.0 / float(fps)
    v_ref = finite_difference_velocities(p_ref, dt)

    # --- Call the optimizer ---
    p_opt, v_opt = build_and_solve_ball_optimization(
        p_ref=p_ref,
        v_ref=v_ref,
        body_contact_flags=body_contact_flags,
        ground_contact_flags=ground_contact_flags,
        link_pos_world_offset=link_pos_world_offset,
        fps=fps,
        speed_thresh=speed_thresh,
    )

    return p_opt, v_opt

