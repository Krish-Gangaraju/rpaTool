# RPA Post-Processing Tool

Welcome to the **RPA Post-Processing Tool**! This Streamlit application automates the analysis and visualization of RPA (Rubber Process Analyzer) data for various test types. This guide will walk you through:

- 📥 **Installation & Launch**
- 🖥️ **User Interface Overview**
- 🧪 **Test Modes & Their Outputs**
- 📊 **Graph Interface**
- 🔑 **Key Values Tab**
- 📂 **Data Interface**
- ⚙️ **Advanced Settings & Customization**

---

## 📥 Installation & Launch

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-org/rpa-post-processing-tool.git
   cd rpa-post-processing-tool
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**
   ```bash
   streamlit run app.py
   ```

4. **Open in Browser**
   Navigate to `http://localhost:8501` to access the tool.

---

## 🖥️ User Interface Overview

The app is divided into three main sections, accessible via tabs:

1. **Graph Interface**: Interactive plotting of your test data.
2. **Key Values**: Automatically computed metrics and thresholds.
3. **Data Interface**: Raw and cleaned data preview and export.

At the top, you can select the **Test Mode** from a dropdown:

- **Cure Test**
- **Scorch Test**
- **Dynamic Test**
- **IVE Test**
- **Temperature Sweep**
- **Plastequiv Test**
- **Indus - Plastequiv Test**

Each mode accepts one or more `.erp` files for batch processing.

---

## 🧪 Test Modes & Their Outputs

Below is a breakdown of each test mode, including what is plotted in the Graph Interface and which key values are computed.

### 1. **Cure Test**

- **Plots**:
  - **Sp (Torque)**: Smoothed torque curve (rolling window 3).
  - **Gp (Modulus)**: Smoothed modulus curve (rolling window 3).
  - **Alpha**: Degree of cure (Sp / Sp<sub>max</sub>).

- **Key Values**:
  - **Total Time**: Full duration of the test [min].
  - **Max / Min Sp**: Peak and baseline torque [dNm].
  - **Sp Range**: Difference between max and min [dNm].
  - **Threshold Times (TS2, TC30, TC50, …)**: Times to reach 2%, 30%, 50%, …, 100% of Sp<sub>max</sub>.
  - **Cure Law Table**: Sp at a user-defined time and its percentage of Sp<sub>max</sub>.

### 2. **Scorch Test**

- **Plots**:
  - **Sp, Gp, Alpha** (same smoothing as Cure but with window=5).

- **Key Values**:
  - **Min Sp**: Minimum torque value [dNm].
  - **T0**: Time at which Scorch begins (min Sp).
  - **T5, T35**: Times to reach 5× and 35× the min Sp.

### 3. **Dynamic Test** (Strain Sweep)

- **Plots**:
  - **Metrics**: TanDelta, G′ & G″, G′, G″, η′ (Np), η″ (Npp), η<sub>s</sub> (Ns).
  - **Phases**: “Both”, “Go” (increasing strain), “Return” (decreasing strain).
  - **X-axis**: Strain [%] (log scale).

- **Key Values**:
  - **Max TanDelta**: Peak damping in Go/Return.
  - **T10, T20, T50**: TanDelta values at 5%, 10%, 25% strain.
  - **Crossover Strain**: Strain where G′ = G″.

### 4. **IVE Test** (Frequency Sweep)

- **Plots**:
  - **Metrics**: G′ & G″, G′, G″, η′, η″, η<sub>s</sub>, TanDelta.
  - **X-axis Options**: Frequency [Hz] or Angular Velocity [rad/s] (log scale).
  - **Y-axis Scale**: Linear or log depending on metric.

- **Key Values**:
  - **IVE Ratio**: G′/G″ at lowest frequency.
  - **Crossover Frequency**: Frequency where G′ = G″.

### 5. **Temperature Sweep**

- **Plots**:
  - **Metrics**: Same as IVE (Gp & Gpp, etc.) vs Temperature [°C].

- **Key Values**:
  - **Crossover Temperature**: Temperature where G′ = G″.

### 6. **Plastequiv Test**

- **Plots**:
  - **Sp (Torque)** vs Time [s], with 3‑point smoothing and preserved endpoints.

- **Key Values**:
  - **Final Viscosity (Np)**: Last non‑NaN η′ value [kPa·s].

### 7. **Indus - Plastequiv Test**

- **Plots**:
  - **Ss (Phase Shift)** vs Time and **Sp & Spp** vs Time in side‑by‑side panels.

- **Key Values**:
  - **Ss Max & Final**: Peak and end values of phase shift.
  - **Overshoot**: Ss<sub>max</sub>/Ss<sub>final</sub>.
  - **Sp Max & Final**: Peak and end torque.
  - **Sp‑Spp Crossover Time**: Time when Sp = Spp.

---

## 📊 Graph Interface - Common Controls

- **Metric Selection**: Radio buttons to pick the plotted variable.
- **Legend Label**: Choose between Filename or custom Mix nicknames (Mix1, Mix2…).
- **Select All**: Quickly toggle all uploaded files.
- **Custom Title**: Edit the plot title or accept the default.
- **Grid Lines**: Toggle major gridlines On/Off.
- **Download**: Save any plot as a high‑resolution PNG.

---

## 🔑 Key Values Tab

Provides **automatically computed tables** of important metrics, with interactive inputs where relevant:

- **Time / Threshold sliders** for Cure law evaluation.
- **Phase radio** for Dynamic Test splitting.
- All tables support **sorting**, **resizing**, and **horizontal scrolling**.

---

## 📂 Data Interface Tab

A preview of the **cleaned raw data** (first ~22–27 columns):

- For **IVE** tests, an extra column for Angular Frequency.
- **Scroll** and **filter** within the table to inspect any value.

---

## ⚙️ Advanced Settings & Customization

- **Smoothing Windows**: Hardcoded per test but can be tweaked in the source.
- **Header Detection**: Robust search for each RPA file format.
- **Error Handling**: Clear messages for missing/invalid headers.
- **Caching**: Results are cached via `@st.cache_data` for speed on repeated runs.

---

## 💡 Tips & Best Practices

- **Consistent Filenames**: Use clear `.erp` names; the tool strips the extension automatically.
- **Batch Uploads**: Drag & drop multiple files to compare mixes side‑by‑side.
- **Review Raw Data**: Always check the Data Interface to confirm correct parsing.

---

> Your feedback is valuable! Feel free to open issues or submit pull requests for new features or tests.

---

<div align="center">
✨ **Happy Testing!** ✨
</div>
