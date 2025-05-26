import tkinter as tk
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import pandas as pd

FS = 44100
BLOCKSIZE = 4096
GAIN = 1000  # Umrechnung in mV

class OsziApp:
    def __init__(self, master):
        self.master = master
        master.title("Python-Oszilloskop")

        self.trigger_level = tk.DoubleVar(value=0)
        self.y_range = tk.DoubleVar(value=500)
        self.x_range = tk.DoubleVar(value=0.002)  # ±1 ms
        self.trigger_direction = tk.StringVar(value="rising")

        self.running = True
        self.triggered_once = False
        self.last_max = 0

        self.current_data = None
        self.freeze_data = None
        self.freeze_t = None

        self.accumulated_signals = []
        self.showing_fft = False
        self.showing_average = False
        self.avg_data = None
        self.avg_time = None

        self.count_rate = 0
        self.total_counts = 0
        self.last_count_time = time.time()

        ctrl_frame = tk.Frame(master)
        ctrl_frame.pack(side="top", fill="x")
        main_frame = tk.Frame(master)
        main_frame.pack(fill="both", expand=True)
        plot_frame = tk.Frame(main_frame)
        plot_frame.pack(side="left", fill="both", expand=True)
        info_frame = tk.Frame(main_frame)
        info_frame.pack(side="right", fill="y")
        status_frame = tk.Frame(master)
        status_frame.pack(side="bottom", fill="x")
        self.status_label = tk.Label(status_frame, text="Status: LIVE", font=("Courier", 10), anchor="w")
        self.status_label.pack(fill="x")

        tk.Label(ctrl_frame, text="Trigger [mV]").grid(row=0, column=0)
        tk.Scale(ctrl_frame, variable=self.trigger_level, from_=-1000, to=1000,
                 orient="horizontal", resolution=10, length=200).grid(row=0, column=1)
        tk.Label(ctrl_frame, text="Y-Achse [±mV]").grid(row=0, column=2)
        tk.Scale(ctrl_frame, variable=self.y_range, from_=100, to=2000,
                 orient="horizontal", resolution=100, length=200).grid(row=0, column=3)
        tk.Label(ctrl_frame, text="X-Achse [±s]").grid(row=0, column=4)
        tk.Scale(ctrl_frame, variable=self.x_range, from_=0.0001, to=0.01,
                 orient="horizontal", resolution=0.0001, length=200).grid(row=0, column=5)
        self.toggle_btn = tk.Button(ctrl_frame, text="⏸️ Stopp", command=self.toggle_running)
        self.toggle_btn.grid(row=0, column=6, padx=10)
        self.fft_btn = tk.Button(ctrl_frame, text="🔁 Zeit/Frequenz", command=self.toggle_fft)
        self.fft_btn.grid(row=0, column=7, padx=10)
        tk.OptionMenu(ctrl_frame, self.trigger_direction, "rising", "falling").grid(row=0, column=8)

        self.fig, self.ax = plt.subplots(figsize=(6, 3), dpi=100)
        self.setup_plot()
        self.line, = self.ax.plot([], [], color='yellow', linewidth=1)
        self.scaling_text = self.ax.text(0.01, 0.95, '', transform=self.ax.transAxes,
                                         color='lightgray', fontsize=9, verticalalignment='top')
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.min_label = tk.Label(info_frame, text="Min: ---", font=("Courier", 12))
        self.min_label.pack(pady=5)
        self.max_label = tk.Label(info_frame, text="Max: ---", font=("Courier", 12))
        self.max_label.pack(pady=5)
        tk.Button(info_frame, text="⬇️ Min finden", command=self.show_min).pack(pady=5)
        tk.Button(info_frame, text="⬆️ Max finden", command=self.show_max).pack(pady=5)
        tk.Button(info_frame, text="📸 Screenshot", command=self.save_screenshot).pack(pady=5)
        tk.Button(info_frame, text="📀 CSV speichern", command=self.save_csv).pack(pady=5)

        self.avg_label = tk.Label(info_frame, text="μ-Mittelung: 0 / 10", font=("Courier", 12))
        self.avg_label.pack(pady=5)
        tk.Button(info_frame, text="➕ Signal hinzufügen", command=self.add_signal).pack(pady=5)
        tk.Button(info_frame, text="⟨⟩ Mittelwert anzeigen", command=self.plot_average).pack(pady=5)
        tk.Button(info_frame, text="📊 Mittelwert speichern", command=self.save_average_csv).pack(pady=5)

        self.count_label = tk.Label(info_frame, text="Zählrate: 0 /s", font=("Courier", 12))
        self.count_label.pack(pady=5)

        self.stream = sd.InputStream(device=1, callback=self.audio_callback,
                                     channels=1, samplerate=FS, blocksize=BLOCKSIZE)
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
        self.showing_average = False
        self.toggle_btn.config(text="▶️ Start" if not self.running else "⏸️ Stopp")
        self.status_label.config(text="Status: LIVE", fg="lime", font=("Courier", 10, "bold"))
        if self.running:
            self.triggered_once = False
            self.last_max = 0
            if self.current_data is not None:
                self.freeze_data = self.current_data.copy()

    def toggle_fft(self):
        self.showing_fft = not self.showing_fft
        self.showing_average = False
        self.status_label.config(text="Status: FREQUENZ", fg="orchid", font=("Courier", 10, "bold"))

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        if self.running:
            self.current_data = indata[:, 0] * GAIN
            now = time.time()
            above_trigger = np.sum(self.current_data > self.trigger_level.get())
            if above_trigger > 0:
                self.total_counts += 1

            if now - self.last_count_time >= 1.0:
                self.count_rate = self.total_counts
                self.total_counts = 0
                self.last_count_time = now
                self.count_label.config(text=f"Zählrate: {self.count_rate} /s")

            trigger = self.trigger_level.get()
            direction = self.trigger_direction.get()
            data = self.current_data
            prev = data[:-1]
            curr = data[1:]

            if direction == "rising":
                trigger_indices = np.where((prev < trigger) & (curr >= trigger))[0]
            else:
                trigger_indices = np.where((prev > trigger) & (curr <= trigger))[0]

            if trigger != 0 and not self.triggered_once and trigger_indices.size > 0:
                trig_idx = trigger_indices[0]
                n_samples = int(self.x_range.get() * FS)
                half = n_samples // 2
                start = trig_idx - half
                end = trig_idx + half

                if start < 0:
                    pad = abs(start)
                    data = np.pad(data, (pad, 0), mode='constant')
                    start = 0
                    end += pad
                elif end > len(data):
                    pad = end - len(data)
                    data = np.pad(data, (0, pad), mode='constant')

                self.freeze_data = data[start:end].copy()
                self.freeze_t = np.linspace(-self.x_range.get() / 2, self.x_range.get() / 2, n_samples)
                self.triggered_once = True
                self.running = False
                self.toggle_btn.config(text="▶️ Start")
                self.status_label.config(text="Status: PAUSE", fg="orange", font=("Courier", 10, "bold"))
            elif trigger == 0:
                self.triggered_once = False
                self.freeze_data = None  # reset freeze when trigger is off)

    def update_plot(self):
        xlim = self.x_range.get()
        ylim = self.y_range.get()

        if self.showing_average and self.avg_data is not None:
            self.ax.clear()
            self.setup_plot()
            self.ax.plot(self.avg_time, self.avg_data, color='cyan', linewidth=1.5, label="Mittelwert")
            self.ax.set_xlim(-xlim / 2, xlim / 2)
            self.ax.set_ylim(-ylim, ylim)
            self.ax.legend(loc='upper right')
            self.scaling_text.set_text(f"{xlim * 1000:.1f} ms/div, {ylim / 5:.0f} mV/div")
            self.canvas.draw()
        else:
            data = self.freeze_data if self.freeze_data is not None else self.current_data
            if data is not None:
                n_samples = int(xlim * FS)
                if len(data) > n_samples:
                    center = len(data) // 2
                    data = data[center - n_samples // 2:center + n_samples // 2]
                t = np.linspace(-xlim / 2, xlim / 2, len(data))
                self.ax.clear()
                self.setup_plot()
                if self.showing_fft:
                    freqs = np.fft.rfftfreq(len(data), 1 / FS)
                    spectrum = np.abs(np.fft.rfft(data))
                    self.ax.set_xlabel("Frequenz [Hz]", color='lightgray')
                    self.ax.set_ylabel("Amplitude", color='lightgray')
                    spectrum_db = 20 * np.log10(spectrum + 1e-12)  # dB-Skala zur besseren Unterscheidung
                    self.ax.plot(freqs, spectrum_db, color='lime')
                    self.ax.set_ylabel("Amplitude [dB]", color='lightgray')

                    # einfache automatische Rauschfilterung anzeigen
                    noise_floor = np.median(spectrum_db)
                    peak_value = np.max(spectrum_db)
                    if peak_value - noise_floor > 10:
                        print("✅ Deutliches Signal im Frequenzbereich")
                    else:
                        print("⚠️ FFT zeigt nur Rauschen")
                else:
                    self.ax.plot(t, data, color='yellow')
                    self.ax.set_xlim(-xlim / 2, xlim / 2)
                    self.ax.set_ylim(-ylim, ylim)
                    if self.trigger_level.get() != 0:
                        self.ax.axhline(self.trigger_level.get(), color="red", linestyle="--", linewidth=0.8)
                self.scaling_text.set_text(f"{xlim * 1000:.1f} ms/div, {ylim / 5:.0f} mV/div")
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
        t = np.linspace(-self.x_range.get() / 2, self.x_range.get() / 2, len(data))
        df = pd.DataFrame({"Zeit [s]": t, "Spannung [mV]": data})
        ts = time.strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"oszi_{ts}.csv", index=False)

    def add_signal(self):
        data = self.get_active_data()
        if data is not None:
            if len(self.accumulated_signals) >= 10:
                self.accumulated_signals.pop(0)
            self.accumulated_signals.append(data.copy())
            self.avg_label.config(text=f"μ-Mittelung: {len(self.accumulated_signals)} / 10")

    def plot_average(self):
        if not self.accumulated_signals:
            return
        min_len = min(len(sig) for sig in self.accumulated_signals)
        trimmed = [s[:min_len] for s in self.accumulated_signals]
        avg = np.mean(trimmed, axis=0)
        xlim = self.x_range.get()
        t = np.linspace(-xlim / 2, xlim / 2, len(avg))
        self.avg_data = avg
        self.avg_time = t
        self.showing_average = True
        self.status_label.config(text="Status: MITTELWERT", fg="cyan", font=("Courier", 10, "bold"))

    def save_average_csv(self):
        if not self.accumulated_signals:
            return
        min_len = min(len(sig) for sig in self.accumulated_signals)
        trimmed = [s[:min_len] for s in self.accumulated_signals]
        avg = np.mean(trimmed, axis=0)
        xlim = self.x_range.get()
        t = np.linspace(-xlim / 2, xlim / 2, len(avg))
        df = pd.DataFrame({"Zeit [s]": t, "gemittelte Spannung [mV]": avg})
        ts = time.strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"oszi_average_{ts}.csv", index=False)

root = tk.Tk()
oszi = OsziApp(root)
root.mainloop()

