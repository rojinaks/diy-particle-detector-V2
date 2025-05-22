import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

FS = 44100
BLOCKSIZE = 1024
THRESHOLD = 0.1  # Schwelle, um "Signal" von Rauschen zu unterscheiden

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(np.zeros(BLOCKSIZE))
ax.set_ylim(-1, 1)
ax.set_title("Live-Signal")
ax.set_xlabel("Sample")
ax.set_ylabel("Amplitude")

def callback(indata, frames, time, status):
    signal = indata[:, 0]
    line.set_ydata(signal)
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Trigger-Erkennung
    if np.max(np.abs(signal)) > THRESHOLD:
        print("📡 Signal erkannt!")

stream = sd.InputStream(callback=callback, channels=1, samplerate=FS, blocksize=BLOCKSIZE)
with stream:
    print("Live-Oszilloskop läuft. Mit STRG+C beenden.")
    while True:
        pass
