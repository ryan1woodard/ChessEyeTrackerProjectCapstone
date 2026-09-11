# Eye Tracker

Webcam-based gaze estimation for chess study. The application watches your
webcam and your screen, estimates where on the monitor you are looking, works
out whether that is the chessboard (and which square), a clock, somewhere else,
or away from the screen entirely, and records the result as a session you can
review afterwards.

It is a **passive observer**. It never clicks, moves pieces, reads game state,
injects scripts, calls private APIs or automates anything on chess.com. It only
looks at your webcam and at the pixels already on your screen.

---

## What it does

- Tracks face, eyes and irises with MediaPipe, plus head orientation.
- Maps those features to screen coordinates through a per-user calibration.
- Reports a confidence score with every estimate, and marks estimates invalid
  when they should not be trusted.
- Draws an optional red dot at the estimated gaze position, in a click-through
  overlay that does not interfere with anything you are doing.
- Builds a **live heatmap** while you play, toggleable at any time, showing a
  soft cloud over the screen and dwell time per chess square.
- Finds the chessboard visually, or lets you select it by hand.
- Maps gaze to individual squares in algebraic notation.
- Distinguishes blinks from looking away, and looking away *from the board*
  from looking away *from the screen*.
- Aggregates everything into events, not per-frame rows, and stores them in
  SQLite.
- Shows a session dashboard with a timeline, most-viewed squares, a chessboard
  heatmap and a full-screen gaze heatmap.
- Exports sessions to CSV and JSON.
- Runs in the system tray while you play.

## What it cannot do

Read this before you draw conclusions from the numbers.

- **A webcam is not an eye tracker.** Expect roughly 1-3 degrees of visual
  angle in good conditions, which on a typical monitor is 50-150 pixels. On a
  600 px chessboard the squares are 75 px, so individual squares are at the
  edge of what is resolvable. Neighbouring-square confusion is normal.
- Accuracy degrades as soon as you move. Calibration ties the model to one head
  position, one seating distance and one monitor. Shift your chair and you
  should recalibrate.
- Glasses, strong backlighting, dim rooms and off-axis webcams all reduce
  accuracy, sometimes drastically.
- Board detection is a convenience, not a guarantee. Unusual themes, heavy
  overlays or partially scrolled boards will defeat it. Manual selection is
  always exact and takes three seconds.
- Board orientation cannot be read from the game, because the application does
  not read the game. Set it yourself in Settings.

The application reports its calibration error and per-frame confidence
precisely so you can judge how much to trust it. Treat the output as an
estimate with error bars, not a measurement.

---

## Requirements

- Windows 10 or 11 (it also runs on Linux and macOS, but the camera backend and
  overlay behaviour are tuned for Windows)
- Python 3.11 or newer
- A webcam, ideally mounted at the top centre of the monitor you calibrate on

## Checking your setup

If anything misbehaves, run this before anything else:

```
python check_setup.py
```

It reports your Python version, every dependency, which MediaPipe backend is
available, and which cameras and monitors were found.

### A note on MediaPipe versions

MediaPipe changed its packaging mid-2024:

- **up to 0.10.14** the `mediapipe.solutions` API ships its models inside the
  wheel, so face tracking works entirely offline;
- **from about 0.10.30** `solutions` was removed, and the replacement Tasks API
  needs a ~3.8 MB `face_landmarker.task` model file.

`requirements.txt` pins 0.10.14 wherever a wheel exists, so the normal install
needs no download at all. On Python 3.13+, where no 0.10.14 wheel is published,
a newer MediaPipe is installed and the model is fetched once on first run. That
download is the only network access anywhere in the application; it transfers a
model file and uploads nothing. To stay strictly offline, download
`face_landmarker.task` yourself into `assets/`.

## Installation

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

On Linux or macOS use `source .venv/bin/activate` instead.

## Running

```
python app.py                 # normal start
python app.py --debug         # debug logging, camera preview, debug overlay
python app.py --calibrate     # start tracking and calibrate immediately
python app.py --no-overlay    # never show the gaze dot this run
python app.py --camera 1      # override the webcam index
python app.py --monitor 2     # track the second monitor
```

---

## The workflow

### 1. Calibrate

Press **Start Tracking**, then **Calibrate**. A full-screen window shows 13
targets one at a time. Look directly at each dot until its green ring fills.

**Let your head move gently while you do this.** Keep your eyes locked on the
dot, but drift your head a little -- left and right, nearer and further. This
is counter-intuitive and it matters more than anything else in this document.

A calibration recorded with the head clamped still teaches the tracker nothing
about how head movement and eye movement trade off, so the moment you shift in
your chair the estimate degrades badly. Measured against the simulator, an 8
degree head turn costs **313 px** of error after a still-head calibration and
**39 px** after a head-varied one. Sit at your normal playing distance.

Press `R` during a target to redo it, or `Esc` to cancel.

At the end you get a report:

```
Average error: 74 px
Median error:  61 px
Worst error:   183 px

Quality: GOOD
```

These figures come from **leave-one-point-out cross-validation**: for each
calibration target the model is refitted without it and then asked to predict
it. That means the error you are shown estimates accuracy at screen positions
the model has never seen, which is what you actually care about. A training-set
error would look far better and mean nothing.

If quality comes back POOR you are offered **Calibrate Again** or **Use
Anyway**.

### 2. Test the red dot

This is the test that matters. Tick **Gaze Dot** and look around the screen.
The dot should follow you within roughly a square's width, and it should fade
when confidence drops.

If it does not track well:

- improve your lighting, especially from the front;
- move the webcam to the top centre of the monitor;
- sit closer, roughly 50-70 cm;
- recalibrate without moving between calibration and testing.

Do not go on to the chess features until the dot behaves.

### 3. Find the board

Open chess.com, then press **Detect Chessboard**. If the green outline lands on
the board you are done. If not, press **Select Chessboard Manually** and drag a
box around it; the selection snaps to a square, and Shift allows free-form.

Set **Board orientation** in Settings to match the side you are playing.

### 3b. Quick Recentre

If the dot develops a steady offset after a long session -- everything reads
slightly high, or slightly left -- press **Quick Recentre**. One dot appears in
the middle of the screen; look at it for three seconds and the measured bias is
subtracted from subsequent estimates.

This corrects a constant offset, which is what slouching and shifting produce.
It cannot fix a mapping that has changed shape, and it says so rather than
silently applying a bad correction: if the samples disagree too much, or the
offset is implausibly large, it declines and asks for a full calibration.

### 4. Watch the live heatmap (optional)

Tick **Live Heatmap** to build a map of where you are looking as you play. The
dropdown beneath it chooses what is drawn:

| Mode | What you see |
| --- | --- |
| Screen + Board | Both layers at once (default) |
| Screen only | A soft cloud over the whole monitor |
| Board only | Each square shaded by dwell time, labelled in seconds |

**Clear** resets it, which is useful for looking at a single position rather
than a whole game. The map also resets automatically when a new session starts;
turn that off with `overlay.heatmap_reset_per_session` in the config.

The overlay is click-through like the gaze dot, so you can keep playing with it
on. It is rendered a few times a second rather than every frame, so leaving it
enabled costs very little.

Two settings are worth knowing about in the config file:

- `overlay.heatmap_half_life_s` (default `0`, meaning never fade). Set it to
  something like `60` and the map fades older data, so it reflects recent
  attention rather than the whole session.
- `overlay.heatmap_opacity` (default `0.55`) if it obscures the board too much.

The grid is deliberately coarse. Gaze estimates carry 50-150 px of error, so
resolving the heatmap more finely than that would imply precision the data does
not have.

### 5. Play

Minimise the window. Tracking continues in the tray. Play normally.

### 6. Review

Press **Stop Tracking**, then **View Sessions** for the timeline, the square
heatmap and the statistics. Export to CSV or JSON from the same screen.

---

## Webcam positioning

```
        Webcam
           o
           |
       +---+---+
       |       |
       | Face  |
       |       |
       +-------+
```

- Top centre of the monitor you calibrate on, as close to the screen edge as
  possible.
- Face centred in the frame, roughly 50-70 cm away.
- Even, front-facing light. Avoid a bright window behind you.
- Tilt glasses slightly if reflections cover your eyes.
- Recalibrate after moving your chair, your monitor or the camera.

---

## Privacy

Everything happens on your computer.

- Webcam frames are analysed in memory and discarded immediately.
- No webcam video is written to disk, ever.
- Screen captures are used only to locate the board, in memory, and discarded.
- No screenshots or recordings are saved.
- Nothing is uploaded. The tracking engine makes no network connections and
  needs no account.
- Only derived numbers are stored: gaze coordinates, aggregated events,
  statistics and calibration model parameters.

Local data lives in `data/sessions/eye_tracker.db`, `data/calibrations/*.json`
and `logs/eye_tracker.log`. Deleting those removes everything.

Heatmaps are reconstructed from stored gaze coordinates, not from saved images.

---

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| Calibration always POOR, dot reads "off screen" | Fixed in this version (Windows display-scaling bug). Run `python check_setup.py`; if it reports scaling, just recalibrate. |
| Accuracy decays as you shift position | Recalibrate, letting your head drift gently during each dot. Use Quick Recentre for a steady offset. |
| Dot jumps around while you stare | Set Smoothing to High in Settings. If it persists, improve lighting: noisy landmarks are the root cause. |
| A blink knocks the estimate off | Should no longer happen; raise "Hold estimate after a blink" in Settings if it does. |
| "module 'mediapipe' has no attribute 'solutions'" | MediaPipe 0.10.30+ removed that API. Run `pip install -r requirements.txt --upgrade`, or let the app download the Tasks model on first run. |
| Camera preview stays "off" | The tracking worker failed to start; the real error is in the status bar and `logs/eye_tracker.log`. The preview only shows frames while the worker is alive. |
| "No webcam detected" | Connect a webcam and restart. Check no other app holds it. |
| "Camera opened but returned no frames" | Another application (Teams, Zoom, OBS) is using it. Close it. |
| Status stuck on "Face not detected" | Improve lighting; make sure your face fills a reasonable part of the frame. |
| Status stuck on "Not calibrated" | Run Calibrate. Uncalibrated frames produce no gaze estimate at all. |
| "Tracking confidence is low" | Front-facing light, sit closer, reduce head rotation, recalibrate. |
| Dot lags behind your eyes | Set Smoothing to Low in Settings. |
| Dot jitters | Set Smoothing to High. |
| Dot is offset consistently | Recalibrate without moving afterwards. |
| Squares flicker between neighbours | Increase Square dwell in Settings. |
| Board not detected | Use Select Chessboard Manually. |
| Squares are mirrored | Change Board orientation in Settings. |
| Low FPS | Reduce capture resolution to 640x480 in Settings. |

Logs are in `logs/eye_tracker.log`. They contain numbers only, never images.

---

## Architecture

```
app.py                     entry point, CLI, wiring
src/
  gui/                     PySide6: main window, calibration, board selector,
                           settings, session dashboard, tray, first run
  tracking/                camera, MediaPipe landmarks, head pose, features,
                           gaze model, smoothing, calibration, worker thread
  screen/                  capture, chessboard detection, regions, classifier,
                           screen worker thread
  events/                  attention states, state machine, event aggregation
  storage/                 SQLite with migrations, CSV/JSON export
  visualization/           click-through gaze overlay, live heatmap,
                           debug rendering
  utils/                   config, logging, geometry
tests/                     161 tests, no hardware required
```

The pipeline:

```
WEBCAM -> FACE LANDMARKS -> EYE + IRIS + HEAD FEATURES -> CALIBRATION MODEL
       -> GAZE X/Y -> SMOOTHING + CONFIDENCE -> SCREEN REGION
       -> CHESSBOARD / SQUARE -> ATTENTION STATE -> EVENT AGGREGATION
       -> DATABASE -> SESSION ANALYSIS
```

Each stage is a separate class and each is independently testable.

### Threading

Three threads, communicating only through Qt queued signals:

| Thread | Work | Rate |
| --- | --- | --- |
| GUI | UI, classification, event bookkeeping, database writes | event-driven |
| `TrackerWorker` | camera capture, MediaPipe, gaze model, smoothing | 20-30 Hz |
| `ScreenWorker` | screen capture and board detection | ~2 Hz |

Neither camera work nor screen capture ever touches the GUI event loop. Board
detection is cached and re-run periodically rather than per frame, because the
board only moves when you move the window.

### Design decisions worth knowing

**One coordinate space, chosen deliberately.** Everything -- calibration
targets, gaze estimates, regions, squares, the overlay -- uses Qt *logical*
pixels. Screen capture converts to physical pixels only at the moment of
grabbing. Mixing the two is catastrophic rather than subtle: at 150% Windows
scaling a model calibrated in physical pixels predicts positions 1.5x too
large, so nearly every estimate lands off screen.

**Features use fixed nominal scales, not observed variance.** Dividing each
feature by its standard deviation across the calibration samples is the
textbook move and it is wrong here. A user sitting still produces almost no
variation in `face_scale`, so dividing by that tiny number amplifies posture
noise enormously; leaning in slightly then throws the estimate hundreds of
pixels. The features are already physically normalised, so fixed scales are
both meaningful and stable.

**The one-standard-error rule picks the regularisation.** Rather than the alpha
with the best cross-validated score, the strongest alpha within one standard
error of the best is chosen. With only 13 points the minimum is noisy, and the
simpler model extrapolates far better outside the calibrated region -- which is
where a gaze tracker spends most of its time.

**Calibration deliberately samples head movement.** Because the camera sees eye
rotation relative to the *head*, head rotation and eye rotation trade off
against each other: the same iris offset points at different screen positions
depending on where the head is. A calibration recorded at one fixed head pose
contains no information about that trade-off, so it cannot compensate for it.
Sampling a range of head positions during calibration is what makes the tracker
survive the user shifting in their chair.

**Both the inputs and the output are filtered.** Smoothing only the output is
too late: the degree-2 polynomial amplifies input noise first, and amplified
noise cannot be removed afterwards without adding visible lag. Filtering the
iris and head features on the way in cuts peak-to-peak wobble from 98 px to
42 px while a cross-screen saccade still settles in about 130 ms. The feature
filters use a much larger One Euro `beta` than the output filter, because
feature values are two orders of magnitude smaller than pixel coordinates and a
pixel-sized `beta` would never let the filter open up during a saccade.

**Blinks hold rather than update.** MediaPipe keeps reporting iris landmarks
while the lid is closed; they are simply wrong. Feeding them to the model throws
the estimate and, worse, poisons the smoothing filter so the error outlives the
blink by a second or more. During a closure, and for a short recovery window
after it, the last good estimate is held and nothing reaches the filters.

**Predictions are clamped.** A polynomial extrapolates without limit, so one
bad frame can produce a coordinate in the millions and poison the smoothing
filter for seconds. Estimates are clamped to the screen plus a margin, with an
`out_of_bounds` flag so "looking away" stays distinguishable from "clamped".

**Ridge regression, not deep learning.** With 13 calibration points there are
13 genuinely independent observations. A large model would interpolate them
perfectly and generalise terribly. The model is a degree-2 polynomial ridge
regression whose regularisation strength is chosen by leave-one-point-out CV.

**Features are scale- and roll-invariant.** Every eye measurement is divided by
that eye's own corner-to-corner width, and iris offsets are expressed in the
eye's local frame rather than image axes. Moving closer to the camera or
tilting your head therefore does not shift the features. Vertical offsets are
normalised by eye *width*, not height, because eye height collapses during a
blink.

**One Euro filter for smoothing.** A moving average would trade jitter for lag.
The One Euro filter filters hard when the signal is slow and relaxes as speed
rises, so fixations are steady while saccades still land quickly.

**Every transition is debounced.** A candidate state must persist before it is
committed, which is why a single noisy frame cannot manufacture a look-away and
why blinks pass through invisibly: a blink is simply not long enough to clear
the threshold.

**Events, not frames.** Looking at e4 for 1.4 seconds is one database row, not
forty.

**The board detector samples square borders, not centres.** The centre of a
square is exactly where the piece stands. Sampling the border ring and reducing
with a median means a full starting position barely dents the detection score.

---

## Testing

```
pip install pytest
python -m pytest
```

161 tests, none of which need a webcam, a screen or a network. Synthetic
landmarks, synthetic boards and synthetic observation streams stand in for
hardware. The GUI tests run against Qt's offscreen platform:

```
# Linux/macOS
QT_QPA_PLATFORM=offscreen python -m pytest

# Windows PowerShell
$env:QT_QPA_PLATFORM="offscreen"; python -m pytest
```

Coverage includes screen-coordinate conversion, chess square mapping in both
orientations, board detection scoring under piece occlusion, region
classification and priority, state transitions and debounce, event aggregation,
look-away durations, calibration fitting and outlier rejection, model
serialisation, configuration loading, database migrations and export formats,
plus an end-to-end integration test that runs the whole pipeline from features
to exported JSON.

## Packaging

```
pip install pyinstaller
pyinstaller eye_tracker.spec
```

The output lands in `dist/EyeTracker/`. The spec file exists because MediaPipe
ships its models as package data that PyInstaller does not find on its own; a
plain `pyinstaller app.py` produces an executable that fails at first frame.

For a single file, add `--onefile`, but expect a slow first start: the archive
unpacks to a temporary directory on every launch.

## Extending it

The interfaces are deliberately narrow:

- `GazeEstimator.estimate(features) -> GazeResult` is all the rest of the
  system knows about gaze estimation. Implement it against a hardware eye
  tracker and nothing downstream changes.
- `RegionManager` holds generic named rectangles, so tracking attention on a
  different application is a matter of supplying a different region set.
- The board detector, the state machine and the storage layer are equally
  replaceable.

Reasonable next steps: correlating gaze with move times, a richer region
editor, per-opening heatmap comparison, and export to standard eye-tracking
research formats.

## Licence

MIT. See `LICENSE`.
#   C h e s s E y e T r a c k e r P r o j e c t C a p s t o n e  
 