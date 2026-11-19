# Keypad recording scripts

This is a local recording tool for a controlled keypad experiment. Use test sequences, not an active access code.

## Single-key press/release calibration

The calibration plan is ready: **200 trials, 20 presses per digit**, arranged in
20 randomized blocks of ten. Each block contains every digit once. This dataset
has its own plan, manifest, and audio folder, separate from the four-digit PINs.

From the `cs687-padpadtatck` folder, run:

```powershell
python scripts/record.py serve --dataset calibration
```

Open **http://localhost:8766/** directly in Chrome or Edge and allow microphone
access. Set the session ID, keypad ID, and actual microphone distance. Keep the
keypad and microphone fixed, the cover open, and the normal beep enabled.

For each trial:

1. Press **Q**. Keep still during the one-second WAIT period.
2. At the green **PRESS** cue, press the displayed digit **once** and fully release
   it. Do not press ENTER or any other key. Keep still afterward.
3. Recording stops automatically after **four seconds**. Listen to the take.
4. Press **E** to save, or **R** to discard it and retry the same digit with Q.

W allows an early stop after 3.2 seconds. Takes shorter than three seconds are
rejected. There are no spoken or audible timing cues. If filming for press/release
annotation, show the finger, key, and on-screen trial ID; do not speak during takes.
You can pause between trials or blocks and resume at the next unsaved trial.
Choose a new session ID when starting a new recording session; it is metadata.

Files are saved under `scripts/calibration/`: `plan.csv` (C0001–C0200),
`manifest.csv` (digit, block, repetition, recording metadata), and `audio/*.wav`.
The plan and empty manifest are already initialized. **Do not run init again.**

```powershell
python scripts/record.py audit --dataset calibration
```

The audit reports per-digit counts, completed blocks, and basic audio flags.
Calibration samples are not automatically included in PIN training or its test set.

## Continue recording four-digit PINs

From the `cs687-padpadtatck` folder, run:

```powershell
python scripts/record.py serve
```

The app always picks the next unsaved PIN trial from the existing plan. You can
stop and resume between groups. Change the Session ID when starting a new session.

Open `http://localhost:8765/` on the same computer. The browser asks for microphone access. Enter the session and keypad IDs; the microphone distance defaults to 20 cm. For each prompted PIN, record one complete entry, listen to it, then Save or Retry. The browser creates mono 16-bit PCM WAV files. The app stores them in `scripts/data/audio/` and labels in `scripts/data/manifest.csv`. Restarting the server resumes at the next unsaved trial.

Keyboard shortcuts: **Q** starts recording, **W** stops, **E** saves the take, and **R** retries (discards the take so you can press Q again). Shortcuts only trigger enabled buttons and are inactive while editing a field. Holding a key does not repeat the action.

The plan, manifest, and saved WAV recordings are part of this Git repository. WAV files use Git LFS. Recording and saving a take updates the local files; commit and push the dataset when you are ready to share a batch. Check recordings for accidental background speech before pushing to the public repository.

Use a microphone that can stay fixed relative to the keypad. Keep the keypad detached from any garage controller and use only dummy PINs. For a new session, change the Session ID before recording. Record a few trials and inspect the audio before completing the pilot.

If the browser reports that microphone permission was denied by the system, open `http://localhost:8765/` directly in Chrome or Edge rather than an embedded browser. Allow microphone access for the site. In Windows, check Settings > Privacy & security > Microphone and enable Microphone access and Let desktop apps access your microphone. Reload the page and try Start recording again.

Run `python scripts/record.py audit` to see progress, digit balance, and basic quiet/clipping flags. The flags do not replace listening to a sample of recordings. Do not run `init` again on an existing dataset; it preserves existing recordings and refuses to overwrite the plan.

The PIN plan has 500 unique four-digit strings. Every group of 50 is balanced
by digit position: each digit appears five times in each position per group.
