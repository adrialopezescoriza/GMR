import os
import time
import tempfile
import atexit
import signal
import xml.etree.ElementTree as ET
import mujoco as mj
import mujoco.viewer as mjv
import imageio
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from loop_rate_limiters import RateLimiter
import numpy as np
from rich import print

_RUNTIME_XML_PATHS = set()
_CLEANUP_HOOKS_INSTALLED = False


def _cleanup_runtime_xml_paths():
    for p in list(_RUNTIME_XML_PATHS):
        try:
            os.remove(p)
        except OSError:
            pass
        _RUNTIME_XML_PATHS.discard(p)


def _install_runtime_xml_cleanup_hooks():
    global _CLEANUP_HOOKS_INSTALLED
    if _CLEANUP_HOOKS_INSTALLED:
        return
    _CLEANUP_HOOKS_INSTALLED = True
    atexit.register(_cleanup_runtime_xml_paths)

    for sig in (signal.SIGINT, signal.SIGTERM):
        prev = signal.getsignal(sig)

        def _handler(signum, frame, _prev=prev):
            _cleanup_runtime_xml_paths()
            if callable(_prev):
                _prev(signum, frame)
            elif _prev == signal.SIG_DFL:
                raise SystemExit(128 + signum)
            # SIG_IGN: intentionally do nothing after cleanup.

        signal.signal(sig, _handler)


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
        raise ValueError(f"Missing mesh filename in URDF visual geometry: {model_path}")
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
    raise FileNotFoundError(f"Cannot resolve mesh path '{mesh_ref}' from '{model_path}'")


def _rpy_to_quat_wxyz(rpy_xyz):
    quat_xyzw = R.from_euler("xyz", rpy_xyz).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)


def _inject_urdf_object_into_robot_xml(robot_xml_path, object_model_path):
    """
    Merge a URDF visual object into a temporary robot MJCF as a mocap body.
    Returns (temp_xml_path, object_body_name).
    """
    if object_model_path is None:
        raise ValueError("object_model_path is required and must point to a URDF file.")
    object_path = os.path.abspath(object_model_path)
    if not os.path.isfile(object_path):
        raise FileNotFoundError(f"Object URDF not found: {object_path}")
    if not object_path.lower().endswith(".urdf"):
        raise ValueError(f"Object model must be a URDF file, got: {object_path}")

    robot_tree = ET.parse(robot_xml_path)
    robot_root = robot_tree.getroot()

    obj_root = ET.parse(object_path).getroot()
    visual = obj_root.find(".//link/visual")
    if visual is None:
        raise ValueError(f"No <visual> found in URDF: {object_path}")
    geom = visual.find("geometry")
    if geom is None:
        raise ValueError(f"No <geometry> under <visual> in URDF: {object_path}")

    origin = visual.find("origin")
    xyz = _parse_float_list(None if origin is None else origin.attrib.get("xyz"), 3, [0.0, 0.0, 0.0])
    rpy = _parse_float_list(None if origin is None else origin.attrib.get("rpy"), 3, [0.0, 0.0, 0.0])
    quat_wxyz = _rpy_to_quat_wxyz(rpy)
    pos_str = f"{xyz[0]} {xyz[1]} {xyz[2]}"
    quat_str = f"{quat_wxyz[0]} {quat_wxyz[1]} {quat_wxyz[2]} {quat_wxyz[3]}"

    rgba = [0.8, 0.8, 0.8, 1.0]
    material = visual.find("material")
    if material is not None:
        color = material.find("color")
        if color is not None:
            rgba = _parse_float_list(color.attrib.get("rgba"), 4, rgba)
    rgba_str = f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"

    asset = robot_root.find("asset")
    if asset is None:
        asset = ET.SubElement(robot_root, "asset")
    worldbody = robot_root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"No <worldbody> in robot XML: {robot_xml_path}")

    object_tag = os.path.splitext(os.path.basename(object_path))[0]
    object_body_name = f"runtime_object_{object_tag}"
    object_mesh_name = f"runtime_object_mesh_{object_tag}"

    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": object_body_name,
            "mocap": "true",
            "pos": "0 0 0",
            "quat": "1 0 0 0",
        },
    )

    geom_attrs = {
        "pos": pos_str,
        "quat": quat_str,
        "rgba": rgba_str,
        "contype": "0",
        "conaffinity": "0",
        "group": "1",
    }

    sphere = geom.find("sphere")
    box = geom.find("box")
    cylinder = geom.find("cylinder")
    mesh = geom.find("mesh")
    if sphere is not None:
        radius = float(sphere.attrib.get("radius", "0.12"))
        geom_attrs.update({"type": "sphere", "size": f"{radius}"})
    elif box is not None:
        full = _parse_float_list(box.attrib.get("size"), 3, [0.24, 0.24, 0.24])
        half = [0.5 * full[0], 0.5 * full[1], 0.5 * full[2]]
        geom_attrs.update({"type": "box", "size": f"{half[0]} {half[1]} {half[2]}"})
    elif cylinder is not None:
        radius = float(cylinder.attrib.get("radius", "0.12"))
        length = float(cylinder.attrib.get("length", "0.24"))
        geom_attrs.update({"type": "cylinder", "size": f"{radius} {0.5 * length}"})
    elif mesh is not None:
        mesh_ref = mesh.attrib.get("filename") or mesh.attrib.get("file")
        mesh_path = _resolve_mesh_path(object_path, mesh_ref)
        mesh_scale = _parse_float_list(mesh.attrib.get("scale"), 3, [1.0, 1.0, 1.0])
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": object_mesh_name,
                "file": mesh_path,
                "scale": f"{mesh_scale[0]} {mesh_scale[1]} {mesh_scale[2]}",
            },
        )
        geom_attrs.update({"type": "mesh", "mesh": object_mesh_name})
    else:
        raise ValueError(f"Unsupported URDF visual geometry in: {object_path}")

    ET.SubElement(body, "geom", geom_attrs)

    robot_xml_abs = os.path.abspath(robot_xml_path)
    robot_xml_dir = os.path.dirname(robot_xml_abs)
    fd, tmp_xml_path = tempfile.mkstemp(
        prefix="runtime_robot_with_object_",
        suffix=".xml",
        dir=robot_xml_dir if os.path.isdir(robot_xml_dir) else None,
    )
    os.close(fd)
    robot_tree.write(tmp_xml_path, encoding="utf-8", xml_declaration=True)
    return tmp_xml_path, object_body_name


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
        self.object_mocap_id = None
        self.runtime_model_xml_path = None
        self.runtime_object_body_name = None

        if object_model_path is None:
            raise ValueError("object_model_path is required and must be a URDF.")

        runtime_xml, runtime_body_name = _inject_urdf_object_into_robot_xml(
            str(self.xml_path), object_model_path
        )
        _install_runtime_xml_cleanup_hooks()
        _RUNTIME_XML_PATHS.add(runtime_xml)
        self.model = mj.MjModel.from_xml_path(runtime_xml)
        self.runtime_model_xml_path = runtime_xml
        self.runtime_object_body_name = runtime_body_name
        self.data = mj.MjData(self.model)
        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]
        mj.mj_step(self.model, self.data)

        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, self.runtime_object_body_name)
        if body_id < 0:
            raise RuntimeError(f"Runtime object body not found in model: {self.runtime_object_body_name}")
        mocap_id = int(self.model.body_mocapid[body_id])
        if mocap_id < 0:
            raise RuntimeError(f"Runtime object body is not mocap: {self.runtime_object_body_name}")
        self.object_mocap_id = mocap_id
        geom_id = int(self.model.body_geomadr[body_id])
        self.object_geom_id = geom_id if geom_id >= 0 else None
        self.object_rgba_default = (
            self.model.geom_rgba[self.object_geom_id].copy() if self.object_geom_id is not None else None
        )
        
        self.motion_fps = motion_fps
        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
        self.camera_follow = camera_follow
        self.record_video = record_video
        print(f"[viewer] runtime URDF object loaded in MuJoCo model: {self.runtime_object_body_name}")


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

        if object_data is not None:
            self.data.mocap_pos[self.object_mocap_id] = object_data[0]
            self.data.mocap_quat[self.object_mocap_id] = object_data[1]
            if self.object_geom_id is not None:
                contact = bool(object_data[2])
                if contact:
                    self.model.geom_rgba[self.object_geom_id] = np.array([0.95, 0.2, 0.2, 1.0], dtype=float)
                elif self.object_rgba_default is not None:
                    self.model.geom_rgba[self.object_geom_id] = self.object_rgba_default
        
        mj.mj_forward(self.model, self.data)
        
        if follow_camera:
            self.viewer.cam.lookat = self.data.xpos[self.model.body(self.robot_base).id]
            self.viewer.cam.distance = self.viewer_cam_distance
            self.viewer.cam.elevation = -10  # 正面视角，轻微向下看
            # self.viewer.cam.azimuth = 180    # 正面朝向机器人
        
        if human_motion_data is not None:
            # Clean custom geometry
            self.viewer.user_scn.ngeom = 0
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

            img = self.renderer.render()
            self.mp4_writer.append_data(img)
    
    def close(self):
        self.viewer.close()
        time.sleep(0.5)
        if self.record_video:
            self.mp4_writer.close()
            print(f"Video saved to {self.video_path}")
        if self.runtime_model_xml_path is not None:
            try:
                os.remove(self.runtime_model_xml_path)
            except OSError:
                pass
            _RUNTIME_XML_PATHS.discard(self.runtime_model_xml_path)
