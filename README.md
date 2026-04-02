# Human-in-the-Loop EMG Demo

A browser-based EMG demo for collecting signals, training a simple model, and updating it with live user feedback. It supports a demo signal or OpenBCI Ganglion input, extracts basic features, and runs real-time predictions for relax, light grip, and tight grip. Relax and tight grip work reliably, while light grip is still in progress and less consistent.

## How to Use

Start the demo signal or connect a Ganglion device. Collect samples for each label (relax, light grip, tight grip), then train the model. Once trained, the app will begin making live predictions. Use the feedback buttons to confirm or correct predictions to improve performance.

## How It Works

Incoming EMG data is cleaned, smoothed, and converted into simple features (RMS, MAV, waveform length). These features are used to train a small neural network in the browser. The model then classifies new data in real time, and user feedback is added back into the dataset to refine it.
