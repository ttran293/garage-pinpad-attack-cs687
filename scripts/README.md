# Keypad recording scripts

This is a local recording tool for a controlled keypad experiment. Use test sequences, not an active access code.

## Start a 50-sequence pilot

From the `cs687-padpadtatck` folder, run:

```powershell
python scripts/record.py init --count 50
python scripts/record.py serve
```

Open `http://localhost:8765/` on the same computer. The browser asks for microphone access. Enter the session and keypad IDs; the microphone distance defaults to 20 cm. For each prompted PIN, record one complete entry, listen to it, then Save or Retry. The browser creates mono 16-bit PCM WAV files. The app stores them in `scripts/data/audio/` and labels in `scripts/data/manifest.csv`. Restarting the server resumes at the next unsaved trial.

The plan, manifest, and saved WAV recordings are part of this Git repository. WAV files use Git LFS. Recording and saving a take updates the local files; commit and push the dataset when you are ready to share a batch. Check recordings for accidental background speech before pushing to the public repository.

Use a microphone that can stay fixed relative to the keypad. Keep the keypad detached from any garage controller and use only dummy PINs. For a new session, change the Session ID before recording. Record a few trials and inspect the audio before completing the pilot.

Run `python scripts/record.py audit` to see progress, digit balance, and basic quiet/clipping flags. The flags do not replace listening to a sample of recordings. Do not run `init` again on an existing dataset; it preserves existing recordings and refuses to overwrite the plan.

The first pilot plan is balanced by digit position. It contains 50 unique four-digit strings. A larger dataset plan should be created after reviewing the pilot, without overwriting its files.
