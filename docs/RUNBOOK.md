# Twinfit runbook (Ahaan)

Claude Code does the engineering. Your part is to:
- finish the prerequisites
- start each session
- capture test data
- review the reports
- record and submit the demo

Plan on about 3 hours a day.

## 1. Prerequisites (about 30 minutes, today)

1. **Meta Model API.** Go to https://dev.meta.ai/platform.
   - Billing: add a card. A temporary pre-authorization hold is placed on it.
   - Spend limits, then Email notifications: set an alert at $150.
   - API keys: create a new key named `twinfit` and copy it immediately. The secret of the old `Playground` key cannot be retrieved.
2. **Store the key outside every repository.**
   ```
   mkdir -p ~/.config/twinfit
   nano ~/.config/twinfit/.env
   ```
   Type these two lines, then save:
   ```
   MODEL_API_KEY=<your new key>
   TWINFIT_BUDGET_USD=120
   ```
   Then run `chmod 600 ~/.config/twinfit/.env`.
3. **GitHub CLI.** Run `brew install gh` if you don't have it, then `gh auth login`. Choose GitHub.com, HTTPS, and log in with the browser.
4. **Project folder.** Create `~/twinfit` and put these seven files in it:
   - 00_setup.md
   - 01_m1_perception.md
   - 02_m2_identification.md
   - 03_m3_verify_agent.md
   - 04_integrate_and_submit.md
   - RUNBOOK.md
   - your hackathon context file, renamed to `hackathon_context.md`

## 2. Order of work

1. **Setup.** Run `cd ~/twinfit && claude`, then say: `Read 00_setup.md and carry it out exactly.`
   - Allow uv, git, gh and pytest commands for the session when asked.
   - Wait for the "Setup report". It ends with your next actions.
2. **Print and calibrate** while setup finishes. Section 5 has the details.
3. **Milestones, in parallel.** Open three terminal tabs, start Claude Code in each, and give it the matching instruction:

   | Tab | Commands | Instruction |
   |---|---|---|
   | 1 | `cd ~/twinfit-m1 && claude` | `Read prompts/01_m1_perception.md and carry it out exactly.` |
   | 2 | `cd ~/twinfit-m2 && claude` | `Read prompts/02_m2_identification.md and carry it out exactly.` |
   | 3 | `cd ~/twinfit-m3 && claude` | `Read prompts/03_m3_verify_agent.md and carry it out exactly.` |

   The milestones are independent. Each one builds against the setup's stubs and synthetic data, so any order works.
   - If your Claude plan limits make three parallel sessions impractical, run them one at a time: M1, then M3, then M2. M1 carries the most external API risk, and M3 is the other Meta-dependent milestone.
4. **Record test clips** while the milestones run. Section 5 has the details. M1 picks up real clips whenever they appear in `~/twinfit-data/real/`.
5. **Review each milestone report.** To request changes, reply in the same session: `Continue: <what to fix>. Keep every constraint in the prompt.`
6. **Integrate.** Run `cd ~/twinfit && claude`, then say: `Read prompts/04_integrate_and_submit.md and carry it out exactly.` It asks before publishing any of your clips.
7. **Record and submit.** Follow `docs/DEMO_SCRIPT.md`, then the checklist in `docs/SUBMISSION.md`.

## 3. Schedule

| Day | You (about 3 h) | Claude Code |
|---|---|---|
| Sat Oct 31 | Prerequisites. Start setup. Print and measure the PDFs. Record the calibration video. | Setup |
| Sun Nov 1 | Start M1, M2 and M3. Build the pendulum and the double pendulum. Record Take A and Take B of each. | M1, M2, M3 |
| Mon Nov 2 | Record the lid, lamp and slider clips. Read the milestone reports and send fixes. | Milestones finish; M1 runs real clips |
| Tue Nov 3 | Run integration. Review the demo assets. Record the 3-minute video. Meta #office-hours on Discord, 10 to 11 AM ET, if API problems remain (last full support day). | Integration |
| Wed Nov 4 | Buffer. Submit by 5 PM ET (the hard deadline is 9 PM ET). Post on X if you want to compete for Most Viral. | Fixes only |

## 4. Budget

You have $150 in credits. Code enforces a hard cap of $120:

| Stage | Cap |
|---|---|
| Setup | $1 |
| M1 | $45 |
| M2 | $0 |
| M3 | $25 |
| Integration | $40 |

Check spend any time with `cd ~/twinfit && uv run twinfit cost`.

## 5. Test data protocol

### Equipment
- Phone held on a tripod or clamp. A stack of books works, as long as it cannot move.
- Printer (US Letter).
- Stiff backing, such as foam board or corrugated cardboard, about 40 × 50 cm.
- Tape, scissors, and a ruler with mm marks.
- A phone level app.
- Pushpins and brad paper fasteners.
- Two cardboard strips, or a ruler.
- A kitchen scale (optional, but valuable).

### Camera settings
Use identical settings for the calibration video and every clip:
- Rear main camera at 1x, landscape orientation, 1080p.
- The highest frame rate you have, at least 120 fps. 240 fps slow motion is ideal.
- Lock focus and exposure. On iPhone, long-press the screen for AE/AF Lock.
- Turn off stabilization or Action mode if your phone allows it.
- No zoom. HDR video off if possible.

Give this setting a camera_id, for example `iphone_1x_1080p240`. If you change any setting, record a new calibration under a new camera_id.

### Printing and measuring
1. Print the three PDFs from `~/twinfit-data/printables/` at 100% / Actual size, not "Fit to page".
2. Check the 100 mm scale bar on each page with your ruler.
3. ChArUco board: measure across 5 squares and divide by 5. That is `square_length_m`. Measure one black marker square inside the board the same way; that is `marker_length_m`.
4. Part markers: measure one black square. That is the part `marker_length_m`.
5. Tape or glue the board flat onto the backing, with no bubbles or curl. Leave space on the backing above the printed board for pivots.

### Calibration video
Record once per camera_id.
1. Fix the phone in place. Move the board slowly by hand for 45 to 60 s.
2. Cover every part of the frame, including all four corners.
3. Tilt the board up to about 45° left, right, up and down.
4. Vary the distance, from near (the board fills the frame) to far (about a quarter of the frame).
5. Move slowly to avoid blur.

Save the file as `~/twinfit-data/real/_calibration/<camera_id>/calib.mov` (or `.mp4`).

### Mounting the board
- **Vertical mechanisms** (pendulums, lamp, lid seen from the side, slider): stand the backing upright. Put the printed UP arrow on top, and level the board's top edge to within 1° using the level app. Metadata: `board_orientation: vertical_arrow_up`.
- **Horizontal mechanisms** (a door seen from above): lay the board flat with the printed face up. Metadata: `board_orientation: horizontal_face_up`.
- **Camera:**
  - 0.5 to 1.0 m from the board, facing it within about 20° of perpendicular.
  - The whole board and the full motion range must be in frame.
  - Keep the board mostly unblocked.

### Mechanisms
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

### Preparing each mechanism
1. **Speckle.** Cut patches from `speckle_letter.pdf` and tape at least two onto the camera-facing side of every moving part, spread apart. Matte surfaces only: no glossy tape over the speckle.
2. **Plane offset.** Measure `plane_offset_m`, the distance from the printed board surface to the camera-facing surface of the moving part, in metres.
   - Positive if the part's surface is closer to the camera than the board.
   - Negative if it is farther away.
3. **Take A (ground truth).** Tape one part marker flat on each moving part's camera-facing surface, away from the speckle. Do the same on a separate base part, if there is one. Record each marker ID per part under `part_markers.assignments`.
4. **Take B (markerless).** Remove the part markers and keep everything else identical.

### Recording a take
1. Record 15 to 25 s per take.
2. Start with 2 s of stillness, with all parts at rest.
3. Move the part or parts to the start pose by hand, hold for 1 s, then release cleanly.
4. After release, keep your hands out of the frame and out of the motion plane.
5. Let the motion run until it has nearly stopped. Use one release per take.
6. From the video timeline, note the release time and the time it settles. Record these as `free_motion_segments_s: [[release, settle]]`.

### Filing a clip
1. Create `~/twinfit-data/real/<clip_id>/`. The clip_id format is `<mechanism>_<A|B>_<nn>`, for example `pendulum_ruler_A_01`.
2. Put the original video file in the folder.
3. Copy `~/twinfit-data/real/_template/metadata.yaml` into the folder and fill in every field. The comments in the template explain each one.
4. Transfer videos with AirDrop or a cable, keeping the original quality. Never use messaging apps; they recompress the video and drop the frame rate.

### Privacy
These clips may be published. Use a plain background, with no faces, documents, screens or addresses in frame.
