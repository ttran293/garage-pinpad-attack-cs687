# Keypad recording scripts

This is a local recording tool for a controlled keypad experiment. Use test sequences, not an active access code.

## Continue recording the expanded plan

From the `cs687-padpadtatck` folder, run:

```powershell
python scripts/record.py serve
```

The first 50 trials are complete. The plan now contains 250 trials, and the app
continues at T0051. The next 200 trials are arranged in four balanced groups of
50. You can stop and resume between groups; the app always picks the next
unsaved trial. Change the Session ID when starting a new recording session.

Open `http://localhost:8765/` on the same computer. The browser asks for microphone access. Enter the session and keypad IDs; the microphone distance defaults to 20 cm. For each prompted PIN, record one complete entry, listen to it, then Save or Retry. The browser creates mono 16-bit PCM WAV files. The app stores them in `scripts/data/audio/` and labels in `scripts/data/manifest.csv`. Restarting the server resumes at the next unsaved trial.

The plan, manifest, and saved WAV recordings are part of this Git repository. WAV files use Git LFS. Recording and saving a take updates the local files; commit and push the dataset when you are ready to share a batch. Check recordings for accidental background speech before pushing to the public repository.

Use a microphone that can stay fixed relative to the keypad. Keep the keypad detached from any garage controller and use only dummy PINs. For a new session, change the Session ID before recording. Record a few trials and inspect the audio before completing the pilot.

If the browser reports that microphone permission was denied by the system, open `http://localhost:8765/` directly in Chrome or Edge rather than an embedded browser. Allow microphone access for the site. In Windows, check Settings > Privacy & security > Microphone and enable Microphone access and Let desktop apps access your microphone. Reload the page and try Start recording again.

Run `python scripts/record.py audit` to see progress, digit balance, and basic quiet/clipping flags. The flags do not replace listening to a sample of recordings. Do not run `init` again on an existing dataset; it preserves existing recordings and refuses to overwrite the plan.

The full plan has 250 unique four-digit strings. Every group of 50 is balanced
by digit position: each digit appears five times in each position per group.
