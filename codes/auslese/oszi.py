# Python-Oszilloskop mit Trigger, Screenshot, CSV-Export, Mittelungspanel, FFT-Ansicht

import tkinter as tk
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import pandas as pd

FS = 44100
BLOCKSIZE = 1024
GAIN = 1000  # Umrechnung in mV

class OsziApp:
    def __init__(self, master):
        self.master = master
        master.title("Python-Oszilloskop")

        self.trigger_level = tk.DoubleVar(value=0)
        self.y_range = tk.DoubleVar(value=500)
        self.x_range = tk.DoubleVar(value=BLOCKSIZE / FS)
        self.running = True
        self.triggered_once = False
        self.last_max = 0

        self.current_data = None
        self.freeze_data = None
        self.freeze_t = None
        self.trigger_line = None

        self.accumulated_signals = []
        self.showing_fft = False

        ctrl_frame = tk.Frame(master)
        ctrl_frame.pack(side="top", fill="x")

        main_frame = tk.Frame(master)
        main_frame.pack(fill="both", expand=True)

        plot_frame = tk.Frame(main_frame)
        plot_frame.pack(side="left", fill="both", expand=True)

        info_frame = tk.Frame(main_frame)
        info_frame.pack(side="right", fill="y")

        tk.Label(ctrl_frame, text="Trigger [mV]").grid(row=0, column=0)
        tk.Scale(ctrl_frame, variable=self.trigger_level, from_=-1000, to=1000,
                 orient="horizontal", resolution=10, length=200).grid(row=0, column=1)

        tk.Label(ctrl_frame, text="Y-Achse [±mV]").grid(row=0, column=2)
        tk.Scale(ctrl_frame, variable=self.y_range, from_=100, to=2000,
                 orient="horizontal", resolution=100, length=200).grid(row=0, column=3)

        tk.Label(ctrl_frame, text="X-Achse [s]").grid(row=0, column=4)
        tk.Scale(ctrl_frame, variable=self.x_range, from_=0.01, to=0.1,
                 orient="horizontal", resolution=0.01, length=200).grid(row=0, column=5)

        self.toggle_btn = tk.Button(ctrl_frame, text="⏸️ Stopp", command=self.toggle_running)
        self.toggle_btn.grid(row=0, column=6, padx=10)

        self.fft_btn = tk.Button(ctrl_frame, text="🔁 Zeit/Frequenz", command=self.toggle_fft)
        self.fft_btn.grid(row=0, column=7, padx=10)

        self.fig, self.ax = plt.subplots(figsize=(6, 3), dpi=100)
        self.setup_plot()

        self.line, = self.ax.plot([], [], color='yellow', linewidth=1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.min_label = tk.Label(info_frame, text="Min: ---", font=("Courier", 12))
        self.min_label.pack(pady=5)
        self.max_label = tk.Label(info_frame, text="Max: ---", font=("Courier", 12))
        self.max_label.pack(pady=5)

        tk.Button(info_frame, text="🔽 Min finden", command=self.show_min).pack(pady=5)
        tk.Button(info_frame, text="🔼 Max finden", command=self.show_max).pack(pady=5)
        tk.Button(info_frame, text="📸 Screenshot", command=self.save_screenshot).pack(pady=5)
        tk.Button(info_frame, text="💾 CSV speichern", command=self.save_csv).pack(pady=5)

        self.avg_label = tk.Label(info_frame, text="μ-Mittelung: 0 / 10", font=("Courier", 12))
        self.avg_label.pack(pady=5)
        tk.Button(info_frame, text="➕ Signal hinzufügen", command=self.add_signal).pack(pady=5)
        tk.Button(info_frame, text="🧮 Mittelwert anzeigen", command=self.plot_average).pack(pady=5)
        tk.Button(info_frame, text="📃 Mittelwert speichern", command=self.save_average_csv).pack(pady=5)

        self.stream = sd.InputStream(callback=self.audio_callback, channels=1, samplerate=FS, blocksize=BLOCKSIZE)
        self.stream.start()
        self.update_plot()

    def setup_plot(self):
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        self.ax.tick_params(colors='lightgray')
        self.ax.spines['bottom'].set_color('lightgray')
        self.ax.spines['left'].set_color('lightgray')
        self.ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
        self.ax.set_xlabel("Zeit [s]", color='lightgray')
        self.ax.set_ylabel("Spannung [mV]", color='lightgray')

    def toggle_running(self):
        self.running = not self.running
        self.toggle_btn.config(text="▶️ Start" if not self.running else "⏸️ Stopp")
        if self.running:
            self.triggered_once = False
            self.last_max = 0

    def toggle_fft(self):
        self.showing_fft = not self.showing_fft

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        if self.running:
            self.current_data = indata[:, 0] * GAIN

    def update_plot(self):
        if self.current_data is not None:
            samples = len(self.current_data)
            t = np.linspace(0, samples / FS, samples)
            data = self.current_data
            trigger = self.trigger_level.get()

            if trigger != 0 and self.running:
                peak = np.max(np.abs(data))
                if peak >= abs(trigger) and (not self.triggered_once or peak >= self.last_max):
                    self.running = False
                    self.toggle_btn.config(text="▶️ Start")
                    self.triggered_once = True
                    self.last_max = peak
                    self.freeze_data = data.copy()
                    self.freeze_t = t.copy()

            if not self.running:
                data = self.freeze_data
                t = self.freeze_t

            self.ax.clear()
            self.setup_plot()

            if self.showing_fft:
                freqs = np.fft.rfftfreq(len(data), 1/FS)
                spectrum = np.abs(np.fft.rfft(data))
                self.ax.set_xlabel("Frequenz [Hz]", color='lightgray')
                self.ax.set_ylabel("Amplitude", color='lightgray')
                self.ax.plot(freqs, spectrum, color='lime')
            else:
                self.ax.plot(t, data, color='yellow')
                self.ax.set_xlim(0, self.x_range.get())
                self.ax.set_ylim(-self.y_range.get(), self.y_range.get())
                if self.trigger_line is not None:
                    try:
                        self.trigger_line.remove()
                    except Exception:
                        pass
                if trigger != 0:
                    self.trigger_line = self.ax.axhline(trigger, color="red", linestyle="--", linewidth=0.8)

            self.canvas.draw()

        self.master.after(50, self.update_plot)

    def get_active_data(self):
        return self.freeze_data if self.freeze_data is not None else self.current_data

    def show_min(self):
        data = self.get_active_data()
        if data is not None:
            self.min_label.config(text=f"Min: {np.min(data):.1f} mV")

    def show_max(self):
        data = self.get_active_data()
        if data is not None:
            self.max_label.config(text=f"Max: {np.max(data):.1f} mV")

    def save_screenshot(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.fig.savefig(f"oszi_{ts}.png", dpi=150, facecolor=self.fig.get_facecolor())

    def save_csv(self):
        data = self.get_active_data()
        if data is None:
            return
        t = np.linspace(0, len(data) / FS, len(data))
        df = pd.DataFrame({"Zeit [s]": t, "Spannung [mV]": data})
        ts = time.strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"oszi_{ts}.csv", index=False)

    def add_signal(self):
        data = self.get_active_data()
        if data is not None:
            self.accumulated_signals.append(data.copy())
            self.avg_label.config(text=f"μ-Mittelung: {len(self.accumulated_signals)} / 10")

    def plot_average(self):
        if not self.accumulated_signals:
            return
        avg = np.mean(np.vstack(self.accumulated_signals), axis=0)
        t = np.linspace(0, len(avg) / FS, len(avg))
        self.ax.clear()
        self.setup_plot()
        self.ax.plot(t, avg, color='cyan', linewidth=1.5, label="Mittelwert")
        self.ax.set_xlim(0, self.x_range.get())
        self.ax.set_ylim(-self.y_range.get(), self.y_range.get())
        self.ax.legend(loc='upper right')
        self.canvas.draw()

    def save_average_csv(self):
        if not self.accumulated_signals:
            return
        avg = np.mean(np.vstack(self.accumulated_signals), axis=0)
        t = np.linspace(0, len(avg) / FS, len(avg))
        df = pd.DataFrame({"Zeit [s]": t, "gemittelte Spannung [mV]": avg})
        ts = time.strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"oszi_average_{ts}.csv", index=False)

root = tk.Tk()
oszi = OsziApp(root)
root.mainloop()