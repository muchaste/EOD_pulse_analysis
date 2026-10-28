# EOD_pulse_analysis
Analysis pipeline for extraction and analysis of pulse-type electric organ discharges from recordings

## Layout
- `pulse_functions.py`, `parameter_gui.py`: shared library and parameter GUIs (detection, extraction, events, plots, tracking helpers).
- `Uganda_2023_eodlogger/`: pipeline for weakly electric fish (multiphasic pulses, differential channels): gain correction, extraction (`02_*`), event annotation, fish tracking and tuning (`04_*`), waveform clustering (`06_*`). Master copy of the extraction scripts.
- `Suriname_2024_eodlogger/`: electric eel (monophasic, envelope detection) extraction, event subsetting and dataset README. Finished; frozen.
- `Helpers/`, `lab/`, `Pre_AI_code/`: helper tools, lab-specific scripts, legacy code.
- `.github/`: instructions and handoff notes (`.github/handoff/HANDOFF.md`); not tracked by git.
