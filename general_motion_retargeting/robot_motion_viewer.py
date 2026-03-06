import os
import time
import xml.etree.ElementTree as ET
import mujoco as mj
import mujoco.viewer as mjv
import imageio
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from loop_rate_limiters import RateLimiter
import numpy as np
from rich import print

def _parse_float_list(raw, n, default):
    if raw is None:
        vals = list(default)
    else:
        vals = [float(x) for x in raw.strip().split()]
    if len(vals) < n:
        vals.extend([0.0] * (n - len(vals)))
    return vals[:n]


def _resolve_mesh_path(model_path, mesh_ref):
    if mesh_ref is None:
        return None
    if os.path.isabs(mesh_ref) and os.path.isfile(mesh_ref):
        return mesh_ref

    base = os.path.dirname(os.path.abspath(model_path))
    candidates = [
        os.path.join(base, mesh_ref),
        os.path.join(base, "..", mesh_ref),
        os.path.join(base, "..", "..", mesh_ref),
        os.path.join(os.getcwd(), mesh_ref),
        os.path.join(os.getcwd(), "assets", mesh_ref),
    ]
    for c in candidates:
        c = os.path.normpath(c)
        if os.path.isfile(c):
            return c
    return None


def _mesh_half_extents_from_obj(mesh_path, scale=None):
    if mesh_path is None:
        return None
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext != ".obj":
        return None
    verts = []
    try:
        with open(mesh_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith("v "):
                    continue
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    except Exception:
        return None

    if not verts:
        return None
    v = np.asarray(verts, dtype=float)
    if scale is not None:
        v = v * np.asarray(scale, dtype=float).reshape(1, 3)
    vmin = v.min(axis=0)
    vmax = v.max(axis=0)
    half = 0.5 * (vmax - vmin)
    if np.any(half <= 1e-6):
        return None
    return half


def _load_object_draw_spec(object_model_path):
    default_spec = {"kind": "sphere", "size": np.array([0.12, 0.0, 0.0]), "name": "ball"}
    if object_model_path is None:
        return default_spec

    path = os.path.abspath(object_model_path)
    if not os.path.isfile(path):
        return default_spec

    ext = os.path.splitext(path)[1].lower()
    name = os.path.splitext(os.path.basename(path))[0]

    try:
        root = ET.parse(path).getroot()
        if ext == ".urdf":
            visual = root.find(".//link/visual")
            if visual is None:
                return default_spec
            geom = visual.find("geometry")
            if geom is None:
                return default_spec
            sphere = geom.find("sphere")
            box = geom.find("box")
            cylinder = geom.find("cylinder")
            mesh = geom.find("mesh")
            if sphere is not None:
                r = float(sphere.attrib.get("radius", "0.12"))
                return {"kind": "sphere", "size": np.array([r, 0.0, 0.0]), "name": name}
            if box is not None:
                full = np.asarray(_parse_float_list(box.attrib.get("size"), 3, [0.24, 0.24, 0.24]), dtype=float)
                return {"kind": "box", "size": 0.5 * full, "name": name}
            if cylinder is not None:
                radius = float(cylinder.attrib.get("radius", "0.12"))
                length = float(cylinder.attrib.get("length", "0.24"))
                return {"kind": "cylinder", "size": np.array([radius, 0.5 * length, 0.0]), "name": name}
            if mesh is not None:
                mesh_ref = mesh.attrib.get("filename") or mesh.attrib.get("file")
                mesh_scale = _parse_float_list(mesh.attrib.get("scale"), 3, [1.0, 1.0, 1.0])
                mesh_path = _resolve_mesh_path(path, mesh_ref)
                half = _mesh_half_extents_from_obj(mesh_path, mesh_scale)
                if half is not None:
                    return {"kind": "box", "size": half, "name": name}
            return default_spec

        if ext == ".xml":
            geom = root.find(".//worldbody//geom")
            if geom is None:
                geom = root.find(".//geom")
            if geom is None:
                return default_spec
            gtype = geom.attrib.get("type", "sphere").lower()
            size = np.asarray(_parse_float_list(geom.attrib.get("size"), 3, [0.12, 0.12, 0.12]), dtype=float)
            if gtype == "sphere":
                return {"kind": "sphere", "size": np.array([size[0], 0.0, 0.0]), "name": name}
            if gtype == "box":
                return {"kind": "box", "size": np.array([size[0], size[1], size[2]]), "name": name}
            if gtype in ("cylinder", "capsule"):
                half_len = size[1] if size.shape[0] > 1 else size[0]
                return {"kind": gtype, "size": np.array([size[0], half_len, 0.0]), "name": name}
            if gtype == "mesh":
                mesh_name = geom.attrib.get("mesh")
                mesh_ref = None
                for mesh in root.findall(".//asset/mesh"):
                    if mesh.attrib.get("name") == mesh_name:
                        mesh_ref = mesh.attrib.get("file")
                        mesh_scale = _parse_float_list(mesh.attrib.get("scale"), 3, [1.0, 1.0, 1.0])
                        mesh_path = _resolve_mesh_path(path, mesh_ref)
                        half = _mesh_half_extents_from_obj(mesh_path, mesh_scale)
                        if half is not None:
                            return {"kind": "box", "size": half, "name": name}
                        break
            return default_spec
    except Exception:
        return default_spec

    return default_spec


def _object_geom_params(spec, data):
    """Return (gtype, size, pos, mat, rgba) for the kinematic object."""
    if data is None:
        return None
    pos, quat_wxyz, contact = data

    kind = spec.get("kind", "sphere")
    size = np.asarray(spec.get("size", np.array([0.12, 0.0, 0.0])), dtype=float)
    if kind == "sphere":
        gtype = mj.mjtGeom.mjGEOM_SPHERE
        draw_size = np.array([size[0], 0.0, 0.0])
    elif kind == "box":
        gtype = mj.mjtGeom.mjGEOM_BOX
        draw_size = np.array([size[0], size[1], size[2]])
    elif kind == "cylinder":
        gtype = mj.mjtGeom.mjGEOM_CYLINDER
        draw_size = np.array([size[0], size[1], 0.0])
    elif kind == "capsule":
        gtype = mj.mjtGeom.mjGEOM_CAPSULE
        draw_size = np.array([size[0], size[1], 0.0])
    else:
        gtype = mj.mjtGeom.mjGEOM_SPHERE
        draw_size = np.array([0.12, 0.0, 0.0])
    
    rgba = np.array([0.9, 0.2, 0.2, 1.0]) if contact else np.array([0.2, 0.9, 0.2, 0.5])

    # SciPy expects (x, y, z, w); user passes scalar-first (w, x, y, z)
    mat = R.from_quat(quat_wxyz, scalar_first=True).as_matrix().reshape(-1)
    return gtype, draw_size, pos, mat, rgba

def draw_object(viewer, spec, data):
    """Draw a single primitive at the desired pose in the on-screen viewer."""
    params = _object_geom_params(spec, data)
    if params is None:
        return
    gtype, size, pos, mat, rgba = params
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mj.mjv_initGeom(geom, type=gtype, size=size, pos=pos, mat=mat, rgba=rgba)
    viewer.user_scn.ngeom += 1


def draw_frame(
    pos,
    mat,
    v,
    size,
    joint_name=None,
    orientation_correction=R.from_euler("xyz", [0, 0, 0]),
    pos_offset=np.array([0, 0, 0]),
):
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for i in range(3):
        geom = v.user_scn.geoms[v.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, 0.01],
            pos=pos + pos_offset,
            mat=mat.flatten(),
            rgba=rgba_list[i],
        )
        if joint_name is not None:
            geom.label = joint_name  # 这里赋名字
        fix = orientation_correction.as_matrix()
        mj.mjv_connector(
            v.user_scn.geoms[v.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=pos + pos_offset,
            to=pos + pos_offset + size * (mat @ fix)[:, i],
        )
        v.user_scn.ngeom += 1

class RobotMotionViewer:
    def __init__(self,
                robot_type,
                camera_follow=True,
                motion_fps=30,
                transparent_robot=0,
                # video recording
                record_video=False,
                video_path=None,
                video_width=640,
                video_height=480,
                keyboard_callback=None,
                object_model_path=None,
                ):
        
        self.robot_type = robot_type
        self.xml_path = ROBOT_XML_DICT[robot_type]
        self.model = mj.MjModel.from_xml_path(str(self.xml_path))
        self.data = mj.MjData(self.model)
        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]
        mj.mj_step(self.model, self.data)
        
        self.motion_fps = motion_fps
        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
        self.camera_follow = camera_follow
        self.record_video = record_video
        self.object_draw_spec = _load_object_draw_spec(object_model_path)
        print(
            f"[viewer] object draw spec: {self.object_draw_spec['name']} ({self.object_draw_spec['kind']})"
        )


        self.viewer = mjv.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False, 
            key_callback=keyboard_callback
            )      

        self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = transparent_robot
        
        if self.record_video:
            assert video_path is not None, "Please provide video path for recording"
            self.video_path = video_path
            video_dir = os.path.dirname(self.video_path)
            
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            self.mp4_writer = imageio.get_writer(self.video_path, fps=self.motion_fps)
            print(f"Recording video to {self.video_path}")
            
            # Initialize renderer for video recording
            self.renderer = mj.Renderer(self.model, height=video_height, width=video_width)
        
    def step(self, 
            # robot data
            root_pos, root_rot, dof_pos, 
            # human data
            human_motion_data=None, 
            show_human_body_name=False,
            # scale for human point visualization
            human_point_scale=0.1,
            # human pos offset add for visualization    
            human_pos_offset=np.array([0.0, 0.0, 0]),
            object_data=None,
            # rate limit
            rate_limit=True, 
            follow_camera=True,
            ):
        """
        by default visualize robot motion.
        also support visualize human motion by providing human_motion_data, to compare with robot motion.
        
        human_motion_data is a dict of {"human body name": (3d global translation, 3d global rotation)}.

        if rate_limit is True, the motion will be visualized at the same rate as the motion data.
        else, the motion will be visualized as fast as possible.
        """
        
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_rot # quat need to be scalar first! for mujoco
        self.data.qpos[7:] = dof_pos
        
        mj.mj_forward(self.model, self.data)
        
        if follow_camera:
            self.viewer.cam.lookat = self.data.xpos[self.model.body(self.robot_base).id]
            self.viewer.cam.distance = self.viewer_cam_distance
            self.viewer.cam.elevation = -10  # 正面视角，轻微向下看
            # self.viewer.cam.azimuth = 180    # 正面朝向机器人
        
        if human_motion_data is not None or object_data is not None:
            # Clean custom geometry
            self.viewer.user_scn.ngeom = 0
        if object_data is not None:
            draw_object(self.viewer, self.object_draw_spec, object_data)
            draw_frame(
                object_data[0],
                R.from_quat(object_data[1], scalar_first=True).as_matrix(),
                self.viewer,
                0.8,
            )
        # Draw the task targets for reference
        if human_motion_data is not None:
            for human_body_name, (pos, rot) in human_motion_data.items():
                draw_frame(
                    pos,
                    R.from_quat(rot, scalar_first=True).as_matrix(),
                    self.viewer,
                    human_point_scale,
                    pos_offset=human_pos_offset,
                    joint_name=human_body_name if show_human_body_name else None
                )

        self.viewer.sync()
        if rate_limit is True:
            self.rate_limiter.sleep()

        if self.record_video:
            # Use renderer for proper offscreen rendering
            self.renderer.update_scene(self.data, camera=self.viewer.cam)

            # Also draw the kinematic object into the renderer's scene (so it shows in video)
            if object_data is not None:
                params = _object_geom_params(self.object_draw_spec, object_data)
                if params is not None:
                    gtype, size, pos, mat, rgba = params
                    rscene = self.renderer.scene
                    rgeom = rscene.geoms[rscene.ngeom]
                    mj.mjv_initGeom(rgeom, type=gtype, size=size, pos=pos, mat=mat, rgba=rgba)
                    rscene.ngeom += 1

            img = self.renderer.render()
            self.mp4_writer.append_data(img)
    
    def close(self):
        self.viewer.close()
        time.sleep(0.5)
        if self.record_video:
            self.mp4_writer.close()
            print(f"Video saved to {self.video_path}")
