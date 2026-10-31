# Test data protocol

How to print, mount, calibrate, record and file the real clips. This is the "Test data protocol"
section of docs/RUNBOOK.md, followed by the board axes that every real clip relies on.

## Where things live
- Printables (three PDFs, US Letter landscape, print at 100%): `~/twinfit-data/printables/`
  - `charuco_board_letter.pdf`, `part_markers_letter.pdf`, `speckle_letter.pdf`
- Metadata template: `~/twinfit-data/real/_template/metadata.yaml`
- Calibration videos and intrinsics: `~/twinfit-data/real/_calibration/<camera_id>/`
- Clips: `~/twinfit-data/real/<clip_id>/`

Regenerate the printables and the template with `twinfit printables`, or from Python with
`twinfit.printables.write_printables()` and `twinfit.printables.write_metadata_template()`.

## Equipment
- Phone held on a tripod or clamp. A stack of books works, as long as it cannot move.
- Printer (US Letter).
- Stiff backing, such as foam board or corrugated cardboard, about 40 × 50 cm.
- Tape, scissors, and a ruler with mm marks.
- A phone level app.
- Pushpins and brad paper fasteners.
- Two cardboard strips, or a ruler.
- A kitchen scale (optional, but valuable).

## Camera settings
Use identical settings for the calibration video and every clip:
- Rear main camera at 1x, landscape orientation, 1080p.
- The highest frame rate you have, at least 120 fps. 240 fps slow motion is ideal.
- Lock focus and exposure. On iPhone, long-press the screen for AE/AF Lock.
- Turn off stabilization or Action mode if your phone allows it.
- No zoom. HDR video off if possible.

Give this setting a camera_id, for example `iphone_1x_1080p240`. If you change any setting, record a new calibration under a new camera_id.

## Printing and measuring
1. Print the three PDFs from `~/twinfit-data/printables/` at 100% / Actual size, not "Fit to page".
2. Check the 100 mm scale bar on each page with your ruler.
3. ChArUco board: measure across 5 squares and divide by 5. That is `square_length_m`. Measure one black marker square inside the board the same way; that is `marker_length_m`.
4. Part markers: measure one black square. That is the part `marker_length_m`.
5. Tape or glue the board flat onto the backing, with no bubbles or curl. Leave space on the backing above the printed board for pivots.

## Calibration video
Record once per camera_id.
1. Fix the phone in place. Move the board slowly by hand for 45 to 60 s.
2. Cover every part of the frame, including all four corners.
3. Tilt the board up to about 45° left, right, up and down.
4. Vary the distance, from near (the board fills the frame) to far (about a quarter of the frame).
5. Move slowly to avoid blur.

Save the file as `~/twinfit-data/real/_calibration/<camera_id>/calib.mov` (or `.mp4`).

## Mounting the board
- **Vertical mechanisms** (pendulums, lamp, lid seen from the side, slider): stand the backing upright. Put the printed UP arrow on top, and level the board's top edge to within 1° using the level app. Metadata: `board_orientation: vertical_arrow_up`.
- **Horizontal mechanisms** (a door seen from above): lay the board flat with the printed face up. Metadata: `board_orientation: horizontal_face_up`.
- **Camera:**
  - 0.5 to 1.0 m from the board, facing it within about 20° of perpendicular.
  - The whole board and the full motion range must be in frame.
  - Keep the board mostly unblocked.

## Mechanisms
Record a Take A and a Take B for each mechanism. Items 1 and 2 are the minimum. Aim for all five.
1. **pendulum_ruler.** A 30 cm ruler or stiff cardboard strip with a hole near one end.
   - Pin it through the hole into the backing, just above the printed board, so it swings in front of the board.
   - Weigh it first and record `mass_kg`.
   - No separate base part.
2. **double_pendulum.** Two cardboard strips, about 20 cm and 15 cm, joined by a brad fastener.
   - Pin the top strip like the pendulum.
   - Weigh each strip before assembling.
3. **lid_hinge.** A laptop lid or a hinged box lid, seen from the side.
   - Stand the board vertically beside it, parallel to the lid's side face.
   - Kinematics only, so free-motion segments may be empty.
4. **desk_lamp** (if you have a jointed lamp). Arms moving in a vertical plane, with the board behind and parallel to that plane. Release it from a displaced pose.
5. **slider.** A small block or toy car sliding or rolling down a straight ruler.
   - Tape the ruler across the backing at 20 to 30°.
   - Release from the top.

## Preparing each mechanism
1. **Speckle.** Cut patches from `speckle_letter.pdf` and tape at least two onto the camera-facing side of every moving part, spread apart. Matte surfaces only: no glossy tape over the speckle.
2. **Plane offset.** Measure `plane_offset_m`, the distance from the printed board surface to the camera-facing surface of the moving part, in metres.
   - Positive if the part's surface is closer to the camera than the board.
   - Negative if it is farther away.
3. **Take A (ground truth).** Tape one part marker flat on each moving part's camera-facing surface, away from the speckle. Do the same on a separate base part, if there is one. Record each marker ID per part under `part_markers.assignments`.
4. **Take B (markerless).** Remove the part markers and keep everything else identical.

## Recording a take
1. Record 15 to 25 s per take.
2. Start with 2 s of stillness, with all parts at rest.
3. Move the part or parts to the start pose by hand, hold for 1 s, then release cleanly.
4. After release, keep your hands out of the frame and out of the motion plane.
5. Let the motion run until it has nearly stopped. Use one release per take.
6. From the video timeline, note the release time and the time it settles. Record these as `free_motion_segments_s: [[release, settle]]`.

## Filing a clip
1. Create `~/twinfit-data/real/<clip_id>/`. The clip_id format is `<mechanism>_<A|B>_<nn>`, for example `pendulum_ruler_A_01`.
2. Put the original video file in the folder.
3. Copy `~/twinfit-data/real/_template/metadata.yaml` into the folder and fill in every field. The comments in the template explain each one.
4. Transfer videos with AirDrop or a cable, keeping the original quality. Never use messaging apps; they recompress the video and drop the frame rate.

## Privacy
These clips may be published. Use a plain background, with no faces, documents, screens or addresses in frame.

## Board axes
The world frame W is the ChArUco board frame as OpenCV defines it, with z = 0 on the printed
face. The directions below were measured by detecting corners in the generated board image
(`twinfit.contracts.conventions.board_axes_in_image`), and tests/setup/test_conventions.py fails
if any of them flips.

Hold the board so the printed UP arrow is on top. Then:
- Printed +x runs to the right.
- Printed +y runs down the page.
- +z points into the printed face, away from the viewer.
- `up_W = (0, -1, 0)`.
- `normal_toward_camera_W = (0, 0, -1)`.
- `gravity_W = (0, 9.81, 0)` for `vertical_arrow_up` and `gravity_W = (0, 0, 9.81)` for
  `horizontal_face_up`. Leave `gravity_W: null` in metadata.yaml and it is derived from
  `board_orientation`.
- `plane_offset_m` is positive toward the camera, along `normal_toward_camera_W`: a part surface
  closer to the camera than the board has a positive offset, one farther away has a negative one.

The origin of W is the board corner at the top left when the UP arrow is on top, and the first
interior chessboard corner is at (0.035, 0.035, 0) with the nominal square length.
