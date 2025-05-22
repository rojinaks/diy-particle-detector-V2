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

        # GUI-Rahmen
        ctrl_frame = tk.Frame(master)
        ctrl_frame.pack(side="top", fill="x")

        main_frame = tk.Frame(master)
        main_frame.pack(fill="both", expand=True)

        plot_frame = tk.Frame(main_frame)
        plot_frame.pack(side="left", fill="both", expand=True)

        info_frame = tk.Frame(main_frame)
        info_frame.pack(side="right", fill="y")

        # Steuerung (horizontal)
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

        # Plot (Oszilloskop-Style)
        self.fig, self.ax = plt.subplots(figsize=(6, 3), dpi=100)
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        self.ax.tick_params(colors='lightgray')
        self.ax.spines['bottom'].set_color('lightgray')
        self.ax.spines['left'].set_color('lightgray')
        self.ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

        self.line, = self.ax.plot([], [], color='yellow', linewidth=1)
        self.trigger_line = None
        self.ax.set_xlabel("Zeit [s]", color='lightgray')
        self.ax.set_ylabel("Spannung [mV]", color='lightgray')

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Info-Panel rechts
        self.min_label = tk.Label(info_frame, text="Min: ---", font=("Courier", 12))
        self.min_label.pack(pady=10)
        self.max_label = tk.Label(info_frame, text="Max: ---", font=("Courier", 12))
        self.max_label.pack(pady=10)

        tk.Button(info_frame, text="🔽 Min finden", command=self.show_min).pack(pady=5)
        tk.Button(info_frame, text="🔼 Max finden", command=self.show_max).pack(pady=5)
        tk.Button(info_frame, text="📸 Screenshot", command=self.save_screenshot).pack(pady=10)
        tk.Button(info_frame, text="💾 Daten als CSV", command=self.save_csv).pack(pady=5)

        # Audio starten
        self.stream = sd.InputStream(callback=self.audio_callback,
                                     channels=1, samplerate=FS,
                                     blocksize=BLOCKSIZE)
        self.stream.start()

        self.update_plot()

    def toggle_running(self):
        self.running = not self.running
        self.toggle_btn.config(text="▶️ Start" if not self.running else "⏸️ Stopp")
        if self.running:
            self.triggered_once = False
            self.last_max = 0

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        if self.running:
            self.current_data = indata[:, 0] * GAIN

    def update_plot(self):
        if self.current_data is not None:
            samples = len(self.current_data)
            t = np.linspace(0, samples / FS, samples)
            trigger = self.trigger_level.get()

            # Trigger
            if trigger != 0 and self.running:
                peak = np.max(np.abs(self.current_data))
                if peak >= abs(trigger) and (not self.triggered_once or peak >= self.last_max):
                    self.running = False
                    self.toggle_btn.config(text="▶️ Start")
                    self.triggered_once = True
                    self.last_max = peak
                    self.freeze_data = self.current_data.copy()
                    self.freeze_t = t.copy()
                    print(f"📌 Trigger ausgelöst bei: {peak:.1f} mV")

            # Datenquelle wählen
            if self.running:
                ydata = self.current_data
                xdata = t
            else:
                ydata = self.freeze_data
                xdata = self.freeze_t

            self.line.set_data(xdata, ydata)
            self.ax.set_xlim(0, self.x_range.get())
            self.ax.set_ylim(-self.y_range.get(), self.y_range.get())

            # Triggerlinie aktualisieren
            if self.trigger_line:
                self.trigger_line.remove()
            if trigger != 0:
                self.trigger_line = self.ax.axhline(trigger, color="red", linestyle="--", linewidth=0.8)

            self.canvas.draw()

        self.master.after(50, self.update_plot)

    def get_active_data(self):
        if self.freeze_data is not None:
            return self.freeze_data
        elif self.current_data is not None:
            return self.current_data
        else:
            return None

    def show_min(self):
        data = self.get_active_data()
        if data is not None:
            val = np.min(data)
            self.min_label.config(text=f"Min: {val:.1f} mV")
        else:
            self.min_label.config(text="Min: ---")

    def show_max(self):
        data = self.get_active_data()
        if data is not None:
            val = np.max(data)
            self.max_label.config(text=f"Max: {val:.1f} mV")
        else:
            self.max_label.config(text="Max: ---")

    def save_screenshot(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"oszi_{timestamp}.png"
        self.fig.savefig(filename, dpi=150, facecolor=self.fig.get_facecolor())
        print(f"📸 Screenshot gespeichert: {filename}")

    def save_csv(self):
        data = self.get_active_data()
        if data is None:
            print("⚠️ Keine Daten vorhanden.")
            return

        samples = len(data)
        t = np.linspace(0, samples / FS, samples)
        df = pd.DataFrame({"Zeit [s]": t, "Spannung [mV]": data})

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"oszi_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"💾 CSV gespeichert: {filename}")

# App starten
root = tk.Tk()
app = OsziApp(root)
root.mainloop()
