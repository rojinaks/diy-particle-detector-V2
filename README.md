# 🎛️ DIY Particle Detector v2.3

**Ein kompakter Teilchendetektor für Bildungs- und Outreach-Zwecke**, inspiriert vom DIY-particle-detector von Oliver Keller, weiterentwickelt am Physikalischen Institut der Universität Bonn.

---

### Dieses Repository enthält die Hardware- und Softwarebasis für einen DIY-Teilchendetektors

## ✨ Features

- SMD-basiertes PCB-Design  
- Python-Oszilloskop zur Signalvisualisierung 
- Dokumentierte Tests und Messergebnisse (Raumladungszone, Signal/noise Analyse, Triggerverhalten)

---

## 📦 Inhalt

- `/hardware` → KiCad-Designs und Schaltpläne  
- `/codes` → Python-Auslesetool mit GUI, weitere messungen (Dioden Messung (Raumladungszone), Opamp vergleiche, Signal/noise Analyse)
- `/archives` → Weitere KiCad-Designs für vorherige Versionen  
- `/spice` → LTspice simulationen für verschiedene opamps


---

## Python Oszilloskop

```bash
  git clone https://github.com/rojinaks/diy-particle-detector-V2.git
  cd diy-particle-detector-V2/codes/auslese
  python oscilloscope.py



