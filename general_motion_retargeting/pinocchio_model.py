import os
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

import casadi as ca
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
from pinocchio.visualize import MeshcatVisualizer


def _parse_float_list(raw: Optional[str], expected: int, default: List[float]) -> List[float]:
    if raw is None:
        values = list(default)
    else:
        values = [float(v) for v in raw.strip().split()]
    if len(values) < expected:
        values.extend([0.0] * (expected - len(values)))
    return values[:expected]


def _resolve_asset_path(model_path: str, asset_ref: str) -> str:
    if os.path.isabs(asset_ref):
        return asset_ref
    return os.path.normpath(os.path.join(os.path.dirname(model_path), asset_ref))


def _material_from_rgba(
    rgba: Optional[List[float]],
    default_object_color: int,
) -> g.MeshLambertMaterial:
    if rgba is None:
        return g.MeshLambertMaterial(color=default_object_color)
    r = max(0.0, min(1.0, rgba[0]))
    gg = max(0.0, min(1.0, rgba[1]))
    b = max(0.0, min(1.0, rgba[2]))
    a = max(0.0, min(1.0, rgba[3]))
    color = (int(r * 255) << 16) | (int(gg * 255) << 8) | int(b * 255)
    return g.MeshLambertMaterial(color=color, opacity=a, transparent=(a < 0.999))


def _origin_transform(xyz: List[float], rpy: List[float]) -> np.ndarray:
    T = tf.euler_matrix(rpy[0], rpy[1], rpy[2], axes="sxyz")
    T[0:3, 3] = np.asarray(xyz)
    return T


def _with_scale(T: np.ndarray, scale: List[float]) -> np.ndarray:
    S = np.eye(4)
    S[0, 0], S[1, 1], S[2, 2] = scale[0], scale[1], scale[2]
    return T @ S


def _mesh_geometry_from_file(mesh_path: str):
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext == ".obj" and hasattr(g, "ObjMeshGeometry"):
        return g.ObjMeshGeometry.from_file(mesh_path)
    if ext == ".stl" and hasattr(g, "StlMeshGeometry"):
        return g.StlMeshGeometry.from_file(mesh_path)
    if ext == ".dae" and hasattr(g, "DaeMeshGeometry"):
        return g.DaeMeshGeometry.from_file(mesh_path)
    raise ValueError(f"Unsupported mesh extension for Meshcat: {ext}")


def _load_object_visual_urdf(
    model_path: str,
    object_radius: float,
    default_object_color: int,
) -> Tuple[object, g.MeshLambertMaterial, np.ndarray]:
    root = ET.parse(model_path).getroot()
    visual = root.find(".//link/visual")
    if visual is None:
        raise ValueError("No <visual> found in URDF.")

    origin = visual.find("origin")
    xyz = _parse_float_list(
        None if origin is None else origin.attrib.get("xyz"), 3, [0.0, 0.0, 0.0]
    )
    rpy = _parse_float_list(
        None if origin is None else origin.attrib.get("rpy"), 3, [0.0, 0.0, 0.0]
    )
    local_tf = _origin_transform(xyz, rpy)

    geom_tag = visual.find("geometry")
    if geom_tag is None:
        raise ValueError("No <geometry> found under <visual> in URDF.")

    sphere = geom_tag.find("sphere")
    box = geom_tag.find("box")
    cylinder = geom_tag.find("cylinder")
    mesh = geom_tag.find("mesh")
    if sphere is not None:
        radius = float(sphere.attrib["radius"])
        shape = g.Sphere(radius)
    elif box is not None:
        size = _parse_float_list(box.attrib.get("size"), 3, [2 * object_radius] * 3)
        shape = g.Box(size)
    elif cylinder is not None:
        radius = float(cylinder.attrib["radius"])
        length = float(cylinder.attrib["length"])
        shape = g.Cylinder(length, radius)
    elif mesh is not None:
        mesh_ref = mesh.attrib.get("filename") or mesh.attrib.get("file")
        if mesh_ref is None:
            raise ValueError("URDF mesh has no filename/file attribute.")
        mesh_path = _resolve_asset_path(model_path, mesh_ref)
        shape = _mesh_geometry_from_file(mesh_path)
        scale = _parse_float_list(mesh.attrib.get("scale"), 3, [1.0, 1.0, 1.0])
        local_tf = _with_scale(local_tf, scale)
    else:
        raise ValueError("URDF visual geometry must be sphere/box/cylinder/mesh.")

    rgba = None
    material = visual.find("material")
    if material is not None:
        color = material.find("color")
        if color is not None:
            rgba = _parse_float_list(color.attrib.get("rgba"), 4, [1.0, 0.5, 0.0, 1.0])
    return shape, _material_from_rgba(rgba, default_object_color), local_tf


def _load_object_visual_mjcf(
    model_path: str,
    object_radius: float,
    default_object_color: int,
) -> Tuple[object, g.MeshLambertMaterial, np.ndarray]:
    root = ET.parse(model_path).getroot()
    compiler = root.find("compiler")
    angle_unit = "degree" if compiler is None else compiler.attrib.get(
        "angle", "degree"
    ).lower()

    mesh_files = {}
    mesh_scales = {}
    for mesh in root.findall(".//asset/mesh"):
        name = mesh.attrib.get("name")
        mesh_file = mesh.attrib.get("file")
        if name and mesh_file:
            mesh_files[name] = _resolve_asset_path(model_path, mesh_file)
            mesh_scales[name] = _parse_float_list(
                mesh.attrib.get("scale"), 3, [1.0, 1.0, 1.0]
            )

    geom = root.find(".//worldbody//geom")
    if geom is None:
        geom = root.find(".//geom")
    if geom is None:
        raise ValueError("No <geom> found in XML.")

    geom_type = geom.attrib.get("type", "sphere").lower()
    size = _parse_float_list(
        geom.attrib.get("size"), 3, [object_radius, object_radius, object_radius]
    )

    local_tf = np.eye(4)
    pos = _parse_float_list(geom.attrib.get("pos"), 3, [0.0, 0.0, 0.0])
    quat = geom.attrib.get("quat")
    euler = geom.attrib.get("euler")
    if quat is not None:
        q = _parse_float_list(quat, 4, [1.0, 0.0, 0.0, 0.0])
        local_tf = tf.quaternion_matrix([q[1], q[2], q[3], q[0]])
    elif euler is not None:
        e = _parse_float_list(euler, 3, [0.0, 0.0, 0.0])
        if angle_unit == "degree":
            e = [np.deg2rad(v) for v in e]
        local_tf = tf.euler_matrix(e[0], e[1], e[2], axes="sxyz")
    local_tf[0:3, 3] = np.asarray(pos)

    if geom_type == "sphere":
        shape = g.Sphere(size[0])
    elif geom_type == "box":
        # MuJoCo box uses half-sizes.
        shape = g.Box([2.0 * size[0], 2.0 * size[1], 2.0 * size[2]])
    elif geom_type in ("cylinder", "capsule"):
        half_len = size[1] if len(size) > 1 else size[0]
        shape = g.Cylinder(2.0 * half_len, size[0])
    elif geom_type == "mesh":
        mesh_name = geom.attrib.get("mesh")
        if not mesh_name or mesh_name not in mesh_files:
            raise ValueError(f"MuJoCo mesh '{mesh_name}' not found in <asset>.")
        shape = _mesh_geometry_from_file(mesh_files[mesh_name])
        local_tf = _with_scale(local_tf, mesh_scales.get(mesh_name, [1.0, 1.0, 1.0]))
    else:
        raise ValueError(f"Unsupported MuJoCo geom type: {geom_type}")

    rgba = _parse_float_list(geom.attrib.get("rgba"), 4, [1.0, 0.5, 0.0, 1.0])
    return shape, _material_from_rgba(rgba, default_object_color), local_tf


def load_object_visual(
    object_model_path: Optional[str],
    object_radius: float = 0.12,
    default_object_color: int = 0xFF8000,
) -> Tuple[object, g.MeshLambertMaterial, np.ndarray, Optional[str]]:
    if object_model_path is None:
        return (
            g.Sphere(object_radius),
            g.MeshLambertMaterial(color=default_object_color),
            np.eye(4),
            None,
        )

    model_path = os.path.abspath(object_model_path)
    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".urdf":
        geom, mat, local_tf = _load_object_visual_urdf(
            model_path, object_radius, default_object_color
        )
    elif ext == ".xml":
        geom, mat, local_tf = _load_object_visual_mjcf(
            model_path, object_radius, default_object_color
        )
    elif ext in (".obj", ".stl", ".dae"):
        geom = _mesh_geometry_from_file(model_path)
        mat = g.MeshLambertMaterial(color=default_object_color)
        local_tf = np.eye(4)
    else:
        raise ValueError(f"Unsupported object model extension: {ext}")
    return geom, mat, local_tf, model_path


class PinocchioCasadiRobot:
    _shared_viz = None
    _shared_viz_key = None

    def __init__(
        self,
        urdf_path: str,
        object_model_path: Optional[str] = None,
        base_body_name: str = "pelvis",
        object_radius: float = 0.12,
        default_object_color: int = 0xFF8000,
        package_dirs: Optional[List[str]] = None,
    ):
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

        self.base_body_name = base_body_name
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

        object_geom = None
        object_mat = None
        object_local_tf = np.eye(4)
        object_model_abs = None
        
        # Visual of the object
        object_geom, object_mat, object_local_tf, object_model_abs = load_object_visual(
            object_model_path=object_model_path,
            object_radius=object_radius,
            default_object_color=default_object_color,
        )

        shared_key = (
            os.path.abspath(urdf_path),
            object_model_abs,
            self.base_body_name,
            float(object_radius),
            int(default_object_color),
        )
        if (
            PinocchioCasadiRobot._shared_viz is None
            or PinocchioCasadiRobot._shared_viz_key != shared_key
        ):
            viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
            viz.initViewer(open=False)
            viz.loadViewerModel()
            viz.viewer["object"].set_object(object_geom, object_mat)
            PinocchioCasadiRobot._shared_viz = viz
            PinocchioCasadiRobot._shared_viz_key = shared_key

        self.viz = PinocchioCasadiRobot._shared_viz
        self.ball_path = "object"
        self.ball_local_tf = object_local_tf

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
