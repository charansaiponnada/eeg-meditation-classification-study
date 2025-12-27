import matplotlib
matplotlib.use('Qt5Agg')

import mne

raw = mne.io.read_raw_bdf(
    r'ds001787\sub-001\ses-01\eeg\sub-001_ses-01_task-meditation_eeg.bdf',
    preload=True
)

print(raw.info)

raw.plot(duration=5, n_channels=80)
