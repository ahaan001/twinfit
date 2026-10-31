"""The commented metadata.yaml template for real clips.

The text is written by hand rather than with yaml.dump so the comments survive. The template
itself must validate as ClipMetadata, which `write_metadata_template` checks after writing.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from twinfit import paths
from twinfit.contracts.models import ClipMetadata

TEMPLATE_TEXT = """\
# Twinfit clip metadata, contract 1.0. Copy this file into ~/twinfit-data/real/<clip_id>/ and
# fill in every field. Lengths are in metres and times in seconds. The recording protocol is in
# docs/TEST_DATA_PROTOCOL.md; the fields below follow it step by step.
contract_version: "1.0"

# <mechanism>_<A|B>_<nn>, identical to the clip folder name. A is Take A (part markers on),
# B is Take B (markerless).
clip_id: pendulum_ruler_A_01

# Always "real" for recorded clips. Generated clips use "synthetic".
source: real

# One of: pendulum_ruler, double_pendulum, lid_hinge, desk_lamp, slider.
mechanism: pendulum_ruler

# Filename of the video inside this clip folder. Replace CLIP.MOV with the actual filename of the
# file you copied here (AirDrop or cable, never a messaging app).
video: CLIP.MOV

# Name of the camera setting used for this clip and for its calibration video, for example rear
# main camera at 1x, landscape, 1080p, 240 fps, AE/AF locked. Changing any setting needs a new
# camera_id and a new calibration video.
camera_id: iphone_1x_1080p240

# Written by `twinfit perception calibrate` (Milestone 1) from
# ~/twinfit-data/real/_calibration/<camera_id>/calib.mov. Replace <camera_id> with the value of
# camera_id above. The path is relative to this clip folder.
intrinsics: ../_calibration/<camera_id>/camera_intrinsics.json

# The printed ChArUco board from ~/twinfit-data/printables/charuco_board_letter.pdf.
board:
  type: charuco
  dictionary: DICT_5X5_100
  squares_x: 7
  squares_y: 5
  # Measured: with a mm ruler, measure across 5 squares of the printed board and divide by 5.
  # Replace the nominal 0.035 with your measurement.
  square_length_m: 0.035
  # Measured: one black marker square inside the board, measured the same way. Replace the
  # nominal 0.026 with your measurement.
  marker_length_m: 0.026

# vertical_arrow_up: backing stands upright, printed UP arrow on top, top edge level within 1 deg
# (pendulums, lamp, lid seen from the side, slider).
# horizontal_face_up: board lies flat with the printed face up (a door seen from above).
board_orientation: vertical_arrow_up

# planar for every protocol mechanism: the parts move in a plane parallel to the board.
# spatial only if the motion leaves that plane.
motion_model: planar

# Distance from the printed board surface to the camera-facing surface of the moving part, in
# metres. Positive if the part's surface is closer to the camera than the board, negative if it
# is farther away.
plane_offset_m: 0.012

# Leave null. It is derived from board_orientation: -9.81 * up_W for vertical_arrow_up and
# -9.81 * normal_toward_camera_W for horizontal_face_up.
gravity_W: null

# One entry per part. fixed: true for a base part that never moves, false for a moving part,
# null if unknown. mass_kg from the kitchen scale, weighed before assembly; null if not weighed.
# Add a base entry (fixed: true) only when there is a separate base part, as for lid_hinge.
parts_hint:
  - label: ruler
    fixed: false
    mass_kg: 0.031

# Take A only: the DICT_4X4_50 markers from part_markers_letter.pdf. For Take B (markerless)
# replace the whole block with `part_markers: null`.
part_markers:
  dictionary: DICT_4X4_50
  # Measured: one black square of a printed part marker, in metres. Replace the nominal 0.030
  # with your measurement.
  marker_length_m: 0.030
  # Marker id taped on each part, keyed by the label used in parts_hint.
  assignments:
    ruler: 0

# One [release, settle] pair per release, in seconds from the start of the video, read from the
# video timeline. release: the hand lets go. settle: the motion has nearly stopped. Use one
# release per take. Leave the list empty ([]) for kinematics-only clips such as lid_hinge.
free_motion_segments_s:
  - [3.0, 18.0]

# false for real clips: there is no ground-truth dynamics model.
dynamics_gt: false

# Anything unusual: lighting, board tilt, occlusions, which speckle patches went where.
notes: ""

# Free-form additions outside the contract. Leave {} unless a milestone asks for something.
extras: {}
"""


def write_metadata_template(out_dir: Path | None = None) -> Path:
    out = Path(out_dir) if out_dir is not None else paths.data_dir() / "real" / "_template"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metadata.yaml"
    path.write_text(TEMPLATE_TEXT)
    ClipMetadata.model_validate(yaml.safe_load(path.read_text()))
    return path
