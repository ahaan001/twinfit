"""MJCF builders for the synthetic scenes.

`mechanism_xml` produces the ground-truth model.xml (mechanism only, DynamicsModel rules).
`scene_xml` adds the textured board, the backdrop, the camera and the light for rendering.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from twinfit.sim.replay import DEFAULT_TIMESTEP
from twinfit.synth.scenes import BOARD_CENTER, BOARD_H, BOARD_W, PartSpec, SceneSpec

BACKDROP_GAP_M = 0.002
BACKDROP_HALF_M = 3.0
# A plane geom faces its local +z; this quaternion (180 degrees about x) turns +z into -z_W so the
# board and backdrop face the camera. The texture then appears unmirrored (verified by detection).
FACE_CAMERA_QUAT = "0 1 0 0"


def fmt(values) -> str:
    return " ".join(repr(float(v)) for v in np.atleast_1d(np.asarray(values, dtype=float)))


def box_inertia(mass: float, size) -> tuple[float, float, float]:
    """Principal moments of a uniform box about its centre: I_xx = m (y^2 + z^2) / 12 etc."""
    x, y, z = size
    return (mass * (y * y + z * z) / 12, mass * (x * x + z * z) / 12, mass * (x * x + y * y) / 12)


def _add_part(parent_el: ET.Element, part: PartSpec, materials: bool) -> ET.Element:
    body = ET.SubElement(parent_el, "body", name=part.part_id, pos=fmt(part.body_pos))
    if tuple(part.body_quat) != (1.0, 0.0, 0.0, 0.0):
        body.set("quat", fmt(part.body_quat))
    j = part.joint
    if j is not None:
        attrs = {
            "name": j.joint_id,
            "type": "hinge" if j.type == "revolute" else "slide",
            "axis": fmt(j.axis),
            "pos": "0 0 0",
            "damping": repr(float(j.damping)),
            "frictionloss": repr(float(j.frictionloss)),
            "armature": repr(float(j.armature)),
            "stiffness": repr(float(j.stiffness)),
            "springref": repr(float(j.springref)),
        }
        if j.range is not None:
            attrs["limited"] = "true"
            attrs["range"] = fmt(j.range)
        ET.SubElement(body, "joint", **attrs)
    ixx, iyy, izz = box_inertia(part.mass_kg, part.size)
    ET.SubElement(
        body,
        "inertial",
        pos=fmt(part.geom_pos),
        mass=repr(float(part.mass_kg)),
        fullinertia=fmt([ixx, iyy, izz, 0.0, 0.0, 0.0]),
    )
    geom = ET.SubElement(
        body,
        "geom",
        name=f"geom_{part.part_id}",
        type="box",
        size=fmt(part.half),
        pos=fmt(part.geom_pos),
        contype="0",
        conaffinity="0",
    )
    if materials and part.speckle:
        geom.set("material", f"mat_{part.part_id}")
    else:
        geom.set("rgba", fmt(list(part.color) + [1.0]))
    return body


def _add_tree(worldbody: ET.Element, spec: SceneSpec, materials: bool) -> None:
    by_id = {p.part_id: p for p in spec.parts}
    elements: dict[str, ET.Element] = {}
    for part in spec.parts:  # parts are listed parents before children
        parent_el = worldbody if part.parent is None else elements[part.parent]
        if part.parent is not None and part.parent not in by_id:
            raise ValueError(f"{part.part_id}: unknown parent {part.parent}")
        elements[part.part_id] = _add_part(parent_el, part, materials)


def _root(spec: SceneSpec, gravity_W) -> tuple[ET.Element, ET.Element]:
    root = ET.Element("mujoco", model=spec.name)
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false")
    ET.SubElement(
        root, "option", gravity=fmt(gravity_W), timestep=repr(DEFAULT_TIMESTEP), integrator="RK4"
    )
    worldbody = ET.SubElement(root, "worldbody")
    return root, worldbody


def _to_string(root: ET.Element) -> str:
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def mechanism_xml(spec: SceneSpec, gravity_W) -> str:
    root, worldbody = _root(spec, gravity_W)
    _add_tree(worldbody, spec, materials=False)
    return _to_string(root)


def scene_xml(
    spec: SceneSpec,
    gravity_W,
    board_png: str,
    cam_pos,
    cam_xyaxes,
    fovy_deg: float,
    light_dir,
    width: int,
    height: int,
) -> str:
    root, worldbody = _root(spec, gravity_W)
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(width), offheight=str(height))
    ET.SubElement(
        visual, "headlight", ambient="0.45 0.45 0.45", diffuse="0.35 0.35 0.35", specular="0 0 0"
    )
    ET.SubElement(visual, "quality", shadowsize="0")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="tex_board", type="2d", file=board_png)
    ET.SubElement(
        asset,
        "material",
        name="mat_board",
        texture="tex_board",
        texrepeat="1 1",
        texuniform="false",
        specular="0",
        shininess="0",
        reflectance="0",
    )
    for part in spec.parts:
        if part.speckle:
            ET.SubElement(
                asset,
                "texture",
                name=f"tex_{part.part_id}",
                type="cube",
                builtin="flat",
                mark="random",
                random="0.3",
                rgb1=fmt(part.color),
                markrgb="0 0 0",
                width="128",
                height="128",
            )
            ET.SubElement(
                asset,
                "material",
                name=f"mat_{part.part_id}",
                texture=f"tex_{part.part_id}",
                texuniform="true",
                texrepeat="4 4",
                specular="0.05",
                shininess="0.05",
            )
    ET.SubElement(
        worldbody,
        "light",
        directional="true",
        dir=fmt(light_dir),
        pos=fmt(-np.asarray(light_dir)),
        diffuse="0.55 0.55 0.55",
        specular="0 0 0",
        castshadow="false",
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="board",
        type="plane",
        size=fmt([BOARD_W / 2, BOARD_H / 2, 0.01]),
        pos=fmt(BOARD_CENTER),
        quat=FACE_CAMERA_QUAT,
        material="mat_board",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="backdrop",
        type="plane",
        size=fmt([BACKDROP_HALF_M, BACKDROP_HALF_M, 0.01]),
        pos=fmt(BOARD_CENTER + np.array([0.0, 0.0, BACKDROP_GAP_M])),
        quat=FACE_CAMERA_QUAT,
        rgba="0.55 0.55 0.55 1",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        worldbody,
        "camera",
        name="cam",
        pos=fmt(cam_pos),
        xyaxes=fmt(cam_xyaxes),
        fovy=repr(float(fovy_deg)),
    )
    _add_tree(worldbody, spec, materials=True)
    return _to_string(root)
