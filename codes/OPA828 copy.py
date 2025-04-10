import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("detector_test2100.txt", sep="\t")  

df["V(output)"] *= 1000  

min_v = df["V(output)"].min()
max_v = df["V(output)"].max()
diff = max_v - min_v
print(f"Gesamt: {diff:.6f} mV")
print(f"Minimale Spannung: {min_v:.6f} mV")
print(f"Maximale Spannung: {max_v:.6f} mV")

plt.figure(figsize=(8, 5))
plt.plot(df["time"], df["V(output)"], marker='o', linestyle='-', markersize=2, color='purple', label=f"Min: {min_v:.2f} mV\nMax: {max_v:.2f} mV")
plt.xlabel("Time (s)")
plt.ylabel("V(output) [mV]") 

plt.title("Detektor Daten Plot in mV mit OPA828")
plt.grid(True)

plt.legend(loc="upper right")  # Legende in die obere rechte Ecke setzen

plt.show()
