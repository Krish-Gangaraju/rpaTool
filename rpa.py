import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator, LogFormatter, FuncFormatter, ScalarFormatter, LogFormatterMathtext
from io import StringIO
import streamlit as st
import io
import base64
import re


# ——— Custom CSS for larger tabs & panels ———
st.set_page_config(page_title="RPA Post-Processing Tool", layout="wide")

st.markdown("""
<style>
.reportview-container .main .block-container {padding-left: 20rem; padding-right: 20rem; }
div[role="tab"] { font-size: 3.25rem !important; padding: 1rem 1.5rem !important; }
div[role="tab"] > button { min-height: 3rem !important; }
div[role="tabpanel"] { min-height: 500px !important; }
</style>
""", unsafe_allow_html=True)



def _read_test_temp(lines, search_header):
    search_header = (
        "Time,Strain,Torque,Torque,Torque,Modulus,Modulus,Modulus,Compl,Compl,Compl,"
        "Visc,Visc,Visc,GenericB,Temp,Temp,Temp,Pressure,Force,Reserve1,Reserve2"
    )
    idx = next((i for i, L in enumerate(lines) if L.strip() == search_header), None)
    if idx is None or idx < 7:
        return None
    temp_line = lines[idx - 10].strip()
    parts = [p.strip() for p in temp_line.split(',') if p.strip() != ""]
    try:
        return float(parts[-1])
    except:
        return "0"



# ——— Cure low-level loader ———
def _load_raw_df_cure(buffer, col_names):
    raw = buffer.readlines() if hasattr(buffer, "readlines") else open(buffer, 'rb').readlines()
    lines = [
        L.decode('utf-8', errors='replace') if isinstance(L, (bytes, bytearray)) else L
        for L in raw
    ]
    search_header = (
        "Time,Strain,Torque,Torque,Torque,Modulus,Modulus,Modulus,Compl,Compl,Compl,"
        "Visc,Visc,Visc,GenericB,Temp,Temp,Temp,Pressure,Force,Reserve1,Reserve2"
    )
    idx = next((i for i,L in enumerate(lines) if L.strip() == search_header), None)
    if idx is None:
        raise ValueError("Header not found in file.")
    data_str = "".join(lines[idx+1:])

    temp = _read_test_temp(lines, search_header)
    return pd.read_csv(StringIO(data_str), names=col_names), temp



# ——— Dynamic low-level loader ———
def _load_raw_df_dynamic(buffer, col_names):
    raw = buffer.readlines() if hasattr(buffer, "readlines") else open(buffer, "rb").readlines()
    lines = [L.decode("utf-8", errors="replace") if isinstance(L, (bytes,bytearray)) else L for L in raw]
    search_header = (
        "GenericA,GenericA,Time,Temp,Temp,Strain,Freq,Strain,Temp,,Torque,Torque,Torque,"
        "Modulus,Modulus,Modulus,Compl,Compl,Compl,Visc,Visc,Visc,"
        "GenericB,Shear,Reserve1,Reserve2,Pressure"
    )
    idx = next((i for i,L in enumerate(lines) if L.strip() == search_header), None)
    if idx is None:
        raise ValueError("Dynamic header not found in file.")

    # ---- new block starts here ----
    data_lines = lines[idx+1:]
    clean_lines = []
    for line in data_lines:
        first_tok = line.strip().split(',',1)[0]
        try:
            float(first_tok)
        except:
            break
        clean_lines.append(line)
    data_str = "".join(clean_lines)
    # ---- new block ends here ----

    temp = _read_test_temp(lines, search_header)
    return pd.read_csv(StringIO(data_str), names=col_names), temp



# ——— IVE low-level loader ———
def _load_raw_df_ive(buffer, col_names):
    raw = buffer.readlines() if hasattr(buffer, "readlines") else open(buffer, "rb").readlines()
    lines = [
        L.decode("utf-8", errors="replace") if isinstance(L, (bytes, bytearray)) else L
        for L in raw
    ]
    search_header = (
        "GenericA,GenericA,Time,Temp,Temp,Strain,Freq,Strain,Temp,,Torque,"
        "Torque,Torque,Modulus,Modulus,Modulus,Compl,Compl,Compl,Visc,Visc,Visc,"
        "GenericB,Shear,Reserve1,Reserve2,Pressure"
    )
    idx = next((i for i, L in enumerate(lines) if L.strip() == search_header), None)
    if idx is None:
        raise ValueError("IVE header not found in file.")

    # ---- new block starts here ----
    data_lines = lines[idx+1:]
    clean_lines = []
    for line in data_lines:
        first_tok = line.strip().split(",", 1)[0]
        try:
            float(first_tok)
        except:
            break
        clean_lines.append(line)
    data_str = "".join(clean_lines)
    # ---- new block ends here ----

    temp = _read_test_temp(lines, search_header)
    return pd.read_csv(StringIO(data_str), names=col_names), temp



# ——— Plastequiv low-level loader ———
def _load_raw_df_plastequiv(buffer, col_names):
    raw = buffer.readlines() if hasattr(buffer, "readlines") else open(buffer, 'rb').readlines()
    lines = [
        L.decode('utf-8', errors='replace') if isinstance(L, (bytes, bytearray)) else L
        for L in raw
    ]
    search_header = (
        "Time,Strain,Torque,Torque,Torque,Modulus,Modulus,Modulus,Compl,Compl,Compl,"
        "Visc,Visc,Visc,GenericB,Temp,Temp,Temp,Pressure,Force,Reserve1,Reserve2"
    )
    idx = next((i for i,L in enumerate(lines) if L.strip() == search_header), None)
    if idx is None:
        raise ValueError("Header not found in file.")
    # extract data lines starting 1 row after header
    data_lines = lines[idx+1:]
    # find first all‐blank row
    end = next(
        (i for i, line in enumerate(data_lines)
         if all(not p.strip() for p in line.split(','))),
        len(data_lines)
    )
    data_str = "".join(data_lines[:end])

    temp = _read_test_temp(lines, search_header)
    return pd.read_csv(StringIO(data_str), names=col_names), temp



# ——— Stress Decay low-level loader ———
def _load_raw_df_stressdecay(buffer, col_names):
    raw = buffer.readlines() if hasattr(buffer, "readlines") else open(buffer, 'rb').readlines()
    lines = [
        L.decode('utf-8', errors='replace') if isinstance(L, (bytes, bytearray)) else L
        for L in raw
    ]
    search_header = ("Time,Torque,Strain,Modulus")
    idx = next((i for i,L in enumerate(lines) if L.strip() == search_header), None)
    if idx is None:
        raise ValueError("Header not found in file.")
    # extract data lines starting 1 row after header
    data_lines = lines[idx+2:]
    # find first all‐blank row
    end = next(
        (i for i, line in enumerate(data_lines)
         if all(not p.strip() for p in line.split(','))),
        len(data_lines)
    )
    data_str = "".join(data_lines[:end])

    temp = _read_test_temp(lines, search_header)
    return pd.read_csv(StringIO(data_str), names=col_names), temp






# ——— 1) Cure-test cleaner ———
@st.cache_data
def clean_cure_file(buffer):
    col_names = [
        "Time","Strain","Sp","Spp","Ss","Gp","Gpp","Gs",
        "Jp","Jpp","Js","Np","Npp","Ns","TDelt",
        "UTemp","LTemp","Temp","Pressure","Force","Reserve1","Reserve2"
    ]
    df, temp = _load_raw_df_cure(buffer, col_names)
    # add Gp/Sp and smoothed
    df['Gp'] = pd.to_numeric(df['Gp'], errors='coerce')
    df['Sp'] = pd.to_numeric(df['Sp'],  errors='coerce')
    df['Gp_smooth'] = df['Gp'].rolling(window=3, center=True).mean()
    df['Sp_smooth'] = df['Sp'].rolling(window=3, center=True).mean()
    alpha = df['Sp'] / df['Sp'].max()
    df.insert(loc=5, column='Alpha', value=alpha)
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    return df, temp



# ——— 2) Scorch-test cleaner ———
@st.cache_data
def clean_scorch_file(buffer):
    # identical format to Cure
    col_names = [
        "Time","Strain","Sp","Spp","Ss","Gp","Gpp","Gs",
        "Jp","Jpp","Js","Np","Npp","Ns","TDelt",
        "UTemp","LTemp","Temp","Pressure","Force","Reserve1","Reserve2"
    ]
    df, temp = _load_raw_df_cure(buffer, col_names)
    df['Gp'] = pd.to_numeric(df['Gp'], errors='coerce')
    df['Sp'] = pd.to_numeric(df['Sp'],  errors='coerce')
    df['Gp_smooth'] = df['Gp'].rolling(window=5, center=True).mean()
    df['Sp_smooth'] = df['Sp'].rolling(window=5, center=True).mean()
    alpha = df['Sp'] / df['Sp'].max()
    df.insert(loc=5, column='Alpha', value=alpha)
    df['Alpha'] = df['Alpha'].rolling(window=4, center=True).mean()
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    return df, temp



# ——— 3) Dynamic-test cleaner ———
@st.cache_data
def clean_dynamic_file(buffer):
    col_names = [
        "Cond","Stat","Time","UTemp","LTemp","Strain","Freq","SStrain","Temp","dummy",
        "Sp","Spp","Ss","Gp","Gpp","Gs","Jp","Jpp","Js",
        "Np","Npp","Ns","TDelt","Shear","Reserve1","Reserve2","Pressure"
    ]
    df, temp = _load_raw_df_dynamic(buffer, col_names)

    # convert key columns to numeric
    df['Strain']   = pd.to_numeric(df['Strain'],   errors='coerce')
    df['UTemp']   = pd.to_numeric(df['UTemp'],   errors='coerce')

    df['Gp']       = pd.to_numeric(df['Gp'],       errors='coerce')
    df['Gpp']      = pd.to_numeric(df['Gpp'],      errors='coerce')
    df['Np']       = pd.to_numeric(df['Np'],       errors='coerce')
    df['Npp']      = pd.to_numeric(df['Npp'],      errors='coerce')
    df['Ns']       = pd.to_numeric(df['Ns'],       errors='coerce')
    df['TanDelta'] = pd.to_numeric(df['TDelt'],    errors='coerce')


    # apply 3-point centered rolling smoothing
    df['Gp_smooth']       = df['Gp'].rolling(window=3, center=True).mean()
    df['Gpp_smooth']      = df['Gpp'].rolling(window=3, center=True).mean()
    df['Np_smooth']       = df['Np'].rolling(window=3, center=True).mean()
    df['Npp_smooth']      = df['Npp'].rolling(window=3, center=True).mean()
    df['Ns_smooth']       = df['Ns'].rolling(window=3, center=True).mean()
    df['TanDelta_smooth'] = df['TanDelta'].rolling(window=3, center=True).mean()

    return df, temp



# ——— 4) IVE-test cleaner ———
@st.cache_data
def clean_ive_file(buffer):
    col_names = [
        "Cond","Stat","Time","UTemp","LTemp","Strain","Freq","SStrain","Temp","dummy",
        "Sp","Spp","Ss","Gp","Gpp","Gs","Jp","Jpp","Js","Np","Npp","Ns","TDelt",
        "Shear","Reserve1","Reserve2","Pressure"
    ]
    df, temp = _load_raw_df_ive(buffer, col_names)

    # convert and smooth Np & Ns
    df['Strain'] = pd.to_numeric(df['Strain'], errors='coerce')
    df['Gp'] = pd.to_numeric(df['Gp'], errors='coerce')
    df['Gpp'] = pd.to_numeric(df['Gpp'], errors='coerce')
    df['Np']       = pd.to_numeric(df['Np'],       errors='coerce')
    df['Npp']      = pd.to_numeric(df['Npp'],      errors='coerce')
    df['Ns']      = pd.to_numeric(df['Ns'],       errors='coerce')
    df['TanDelta'] = pd.to_numeric(df['TDelt'],    errors='coerce')

    df['Gp_smooth'] = df['Gp'].rolling(window=3, center=True).mean()
    df['Gpp_smooth'] = df['Gpp'].rolling(window=3, center=True).mean()
    df['Np_smooth']       = df['Np'].rolling(window=3, center=True).mean()
    df['Npp_smooth']      = df['Npp'].rolling(window=3, center=True).mean()
    df['Ns_smooth']       = df['Ns'].rolling(window=3, center=True).mean()
    df['TanDelta_smooth'] = df['TanDelta'].rolling(window=3, center=True).mean()
    # ensure time is numeric
    df['Freq'] = pd.to_numeric(df['Freq'], errors='coerce')
    return df, temp



# ——— 5) Plastequiv-test cleaner ———
@st.cache_data
def clean_plastequiv_file(buffer):
    col_names = [
        "Time","Strain","Sp","Spp","Ss","Gp","Gpp","Gs",
        "Jp","Jpp","Js","Np","Npp","Ns","TDelt",
        "UTemp","LTemp","Temp","Pressure","Force","Reserve1","Reserve2"
    ]
    df, temp = _load_raw_df_plastequiv(buffer, col_names)

    # — convert to numeric —
    df["Strain"] = pd.to_numeric(df["Strain"], errors="coerce")
    df["Sp"]     = pd.to_numeric(df["Sp"],     errors="coerce")
    df["Spp"]    = pd.to_numeric(df["Spp"],    errors="coerce")
    df["Ss"]     = pd.to_numeric(df["Ss"],     errors="coerce")
    df["Time"]   = pd.to_numeric(df["Time"],   errors="coerce")

    # — 3‐point centered rolling for each —
    df["Sp_smooth"]  = df["Sp"].rolling(window=3, center=True).mean()
    df["Spp_smooth"] = df["Spp"].rolling(window=3, center=True).mean()
    df["Ss_smooth"]  = df["Ss"].rolling(window=3, center=True).mean()

    # — overwrite the first two rows with the raw values —
    #    use .iloc to target position 0 and 1
    df.loc[df.index[:2], "Sp_smooth"]  = df.loc[df.index[:2], "Sp"]
    df.loc[df.index[:2], "Spp_smooth"] = df.loc[df.index[:2], "Spp"]
    df.loc[df.index[:2], "Ss_smooth"]  = df.loc[df.index[:2], "Ss"]

    return df, temp



# ——— 6) Stress Decay cleaner ———
@st.cache_data
def clean_stressdecay_file(buffer):
    col_names = ["Time", "Torque", "Strain", "Modulus"]
    df, temp = _load_raw_df_stressdecay(buffer, col_names)

    # — convert to numeric —
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["Torque"] = pd.to_numeric(df["Torque"],     errors="coerce")
    df["Strain"] = pd.to_numeric(df["Strain"],    errors="coerce")
    df["Modulus"] = pd.to_numeric(df["Modulus"],     errors="coerce")

    # — 3‐point centered rolling for each —
    df["Torque_smooth"]  = df["Torque"].rolling(window=3, center=True).mean()
    df["Modulus_smooth"] = df["Modulus"].rolling(window=3, center=True).mean()

    # — overwrite the first two rows with the raw values —
    #    use .iloc to target position 0 and 1
    df.loc[df.index[:2], "Torque_smooth"]  = df.loc[df.index[:2], "Torque"]
    df.loc[df.index[:2], "Modulus_smooth"] = df.loc[df.index[:2], "Modulus"]

    return df, temp





# ——— UI ———
st.title("RPA Post-Processing Tool")
modes = ["Help", "Cure Test", "Scorch Test", "Dynamic Test", "IVE Test", "Temperature Sweep", "Plastequiv Test", "Indus - Plastequiv Test", "Indus - Stress Decay"]
display_names = {
    "Dynamic Test": "Dynamic Test/Strain Sweep",
    "IVE Test": "Frequency Sweep (includes IVE Test)",
    "Help"  : "Help & User Manual",
}
mode = st.selectbox("Choose mode:", modes, format_func=lambda key: display_names.get(key, key))

if mode == "Help":
    st.subheader("Help & User Manual")
    st.markdown("""
    Welcome to the **RPA Post-Processing Tool**! This Streamlit application automates the analysis and visualization of RPA (Rubber Process Analyzer) data for various test types. This guide will walk you through:
    - 🖥️ **User Interface Overview**
    - 🧪 **Test Modes & Their Outputs**
    - 📊 **Graph Interface**
    - 🔑 **Key Values Tab**
    - 📂 **Data Interface**
    - ⚙️ **Advanced Settings & Customization**


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

    """)
    st.stop()



# per-mode uploader
key_map = {
    "Cure Test":   "uploader_cure",
    "Scorch Test": "uploader_scorch",
    "Dynamic Test": "uploader_dynamic",
    "IVE Test": "uploader_ive",
    "Plastequiv Test":  "uploader_plastequiv",
    "Indus - Plastequiv Test": "uploader_indus_plastequiv",
    "Temperature Sweep": "uploader_temp_sweep",
    "Indus - Stress Decay": "uploader_indus_stress_decay"
}

labels = {
    "Cure Test":      "Upload Cure-test .erp files",
    "Scorch Test":    "Upload Scorch-test .erp files",
    "Dynamic Test":   "Upload dynamic-test .erp files",
    "IVE Test": "Upload IVE-test .erp files",
    "Plastequiv Test":     "Upload Plastequiv-test .erp files",
    "Indus - Plastequiv Test": "Upload Plastequiv-test .erp files",
    "Temperature Sweep":  "Upload Temperature Sweep .erp files",
    "Indus - Stress Decay": "Upload Indus Stress Decay .erp files"
}

uploaded = st.file_uploader(labels[mode], type=['erp'], accept_multiple_files=True, key=key_map[mode])

if not uploaded:
    st.info("📂 Please upload one or more files to continue.")
    st.stop()

# process files
processed = {}      # will hold { filename: (df, temp) }
for f in uploaded:
    try:
        if mode == "Cure Test":
            df, temp = clean_cure_file(f)
        elif mode == "Scorch Test":
            df, temp = clean_scorch_file(f)
        elif mode == "Dynamic Test" or mode == "Temperature Sweep":
            df, temp = clean_dynamic_file(f)
        elif mode == "IVE Test":
            df, temp = clean_ive_file(f)
        elif mode == "Plastequiv Test" or mode == "Indus - Plastequiv Test":
            df, temp = clean_plastequiv_file(f)
        elif mode == "Indus - Stress Decay":
            df, temp = clean_stressdecay_file(f)
        processed[f.name] = (df, temp)
    except Exception as e:
        st.error(f"⚠️ Failed **{f.name}**: {e}")

if not processed:
    st.stop()

# three‐tab interface
tab_graph, tab_key, tab_data = st.tabs(["Graph Interface", "Key Values", "Data Interface"])





# — Styling constants for all RPA plots —
TITLE_FS        = 7
LABEL_FS        = 6
TICK_FS         = 5
LEGEND_FS       = 4.5
LEGEND_TITLE_FS = 5.5
LINEWIDTH       = 1.5

# — Graph Interface —
with tab_graph:
    st.subheader(f"{mode}")

    if mode == "Cure Test":
        opts    = ["Sp", "Gp", "Alpha"]
        x_axis  = "Time"
        x_label = "Time [min]"

        # Controls
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            metric = st.radio("Metric", opts, horizontal=True)
        with col2:
            legend_choice = st.radio("Legend label:", ["Filename", "Nickname"], horizontal=True)

        # File selection
        select_all = st.checkbox("Select All", value=True)
        to_plot = [name for name in sorted(processed)
                   if st.checkbox(re.sub(r'(?i)\.erp$', '', name), value=select_all, key=f"cb_{mode}_{name}")]


        if not to_plot:
            st.info("Select at least one file to plot.")
        else:
            # Prepare
            palette   = plt.get_cmap('tab20').colors
            color_map = {n: palette[i % len(palette)] for i, n in enumerate(sorted(processed))}
            nicknames = {n: f"Mix{i+1}" for i, n in enumerate(sorted(processed))}
            temp      = next(iter(processed.values()))[1]
            max_t     = max(df['Time'].max() for df, _ in processed.values())
            time_lb   = f"{max_t:.0f}"
            temp_lb   = f"{temp:.0f}"

            # Plot
            fig, ax = plt.subplots(figsize=(3.5, 3.5), constrained_layout=True)
            ax.set_box_aspect(1)
            for name in to_plot:
                df, _ = processed[name]
                # strip .erp extension for filename legend
                clean_name = re.sub(r'(?i)\.erp$', '', name)
                lbl = clean_name if legend_choice == "Filename" else nicknames[name]
                if metric == "Sp":
                    y = df['Sp_smooth']
                    unit = "[dNm]"
                elif metric == "Gp":
                    y = df['Gp_smooth']
                    unit = "[kPa]"
                else:  # Alpha
                    y = df['Alpha']
                    unit = ""

                ax.plot(df[x_axis], y, color=color_map[name], linewidth=LINEWIDTH, label=lbl)
                

            default_title = f"RPA - Cure Test {temp_lb}°C/{time_lb}min - {metric} vs {x_axis}"
            col_title, col_grid = st.columns([1, 1], gap="large")
            with col_title:
                custom_title = st.text_input("**Custom plot title (leave blank for default):**", value=default_title, key="cure_custom_title")
            with col_grid:
                grid_choice = st.radio("Grid lines:", ["On", "Off"], horizontal=True)
                show_grid = (grid_choice == "On")
            plot_title   = custom_title if custom_title else default_title
            ax.set_title(plot_title, fontsize=TITLE_FS)
            if show_grid:
                ax.grid(which="major", linestyle="-", linewidth=0.5)
            else:
                ax.grid(False)


            ax.set_xlabel(x_label, fontsize=LABEL_FS)
            ax.set_ylabel(f"{metric} {unit}", fontsize=LABEL_FS)
            ax.tick_params(axis="both", labelsize=TICK_FS)
            ax.set_xlim(left=0); ax.set_ylim(bottom=0)
            leg = ax.legend(title="Mixes", fontsize=LEGEND_FS, title_fontsize=LEGEND_TITLE_FS, loc="lower right", frameon=True, edgecolor='black')
            leg.get_frame().set_linewidth(0.5)

            st.pyplot(fig, use_container_width=False)
            buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=300); buf.seek(0)
            st.download_button("Download plot as PNG", data=buf, file_name="rpa_cure_plot.png", mime="image/png")


    elif mode == "Scorch Test":
        opts    = ["Sp", "Gp", "Alpha"]
        x_axis  = "Time"
        x_label = "Time [min]"
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            metric = st.radio("Metric", opts, horizontal=True)
        with col2:
            legend_choice = st.radio("Legend label:", ["Filename", "Nickname"], horizontal=True)
        select_all = st.checkbox("Select All", value=True)
        to_plot = [name for name in sorted(processed)
                   if st.checkbox(re.sub(r'(?i)\.erp$', '', name), value=select_all, key=f"cb_{mode}_{name}")]
        if not to_plot:
            st.info("Select at least one file to plot.")
        else:
            palette   = plt.get_cmap('tab20').colors
            color_map = {n: palette[i % len(palette)] for i, n in enumerate(sorted(processed))}
            nicknames = {n: f"Mix{i+1}" for i, n in enumerate(sorted(processed))}
            temp      = next(iter(processed.values()))[1]
            max_t     = max(df['Time'].max() for df, _ in processed.values())
            time_lb   = f"{max_t:.0f}"
            temp_lb   = f"{temp:.0f}"

            fig, ax = plt.subplots(figsize=(3.5, 3.5), constrained_layout=True)
            ax.set_box_aspect(1)
            for name in to_plot:
                df, _ = processed[name]
                clean_name = re.sub(r'(?i)\.erp$', '', name)
                lbl = clean_name if legend_choice == "Filename" else nicknames[name]
                if metric == "Sp": 
                    y = df['Sp_smooth']
                    unit = "[dNm]"
                elif metric == "Gp": 
                    y = df['Gp_smooth']
                    unit = "[MPa]"
                else:                
                    y = df['Alpha']
                    unit = ""
                ax.plot(df[x_axis], y, color=color_map[name], linewidth=LINEWIDTH, label=lbl)

            default_title = f"RPA - Scorch Test {temp_lb}°C/{time_lb}min - {metric} vs {x_axis}"
            col_title, col_grid = st.columns([1, 1], gap="large")
            with col_title:
                custom_title = st.text_input("**Custom plot title (leave blank for default):**", value=default_title, key="cure_custom_title")
            with col_grid:
                grid_choice = st.radio("Grid lines:", ["On", "Off"], horizontal=True)
                show_grid = (grid_choice == "On")
            plot_title   = custom_title if custom_title else default_title
            ax.set_title(plot_title, fontsize=TITLE_FS)
            if show_grid:
                ax.grid(which="major", linestyle="-", linewidth=0.5)
            else:
                ax.grid(False)

            ax.set_xlabel(x_label, fontsize=LABEL_FS);  ax.set_ylabel(f"{metric} {unit}", fontsize=LABEL_FS)
            ax.tick_params(axis="both", labelsize=TICK_FS); ax.set_xlim(0); ax.set_ylim(0)
            leg = ax.legend(title="Mixes", fontsize=LEGEND_FS, title_fontsize=LEGEND_TITLE_FS, loc="lower right", frameon=True, edgecolor='black'); leg.get_frame().set_linewidth(0.5)

            st.pyplot(fig, use_container_width=False)
            buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=300); buf.seek(0)
            st.download_button("Download plot as PNG", data=buf, file_name="rpa_scorch_plot.png", mime="image/png")


    elif mode == "Dynamic Test":
        opts    = ["TanDelta", "Gp & Gpp", "Gp", "Gpp", "Np", "Npp", "Ns"]
        x_axis  = "Strain"
        x_label = "Strain [%]"
        col1, col2, col3 = st.columns([1, 1, 1], gap="large")
        with col1:
            metric = st.radio("Metric", opts, horizontal=True)
        with col2:
            phase = st.radio("Phase", ["Both", "Go", "Return"], horizontal=True)
        with col3:
            legend_choice = st.radio("Legend label:", ["Filename", "Nickname"], horizontal=True)

        select_all = st.checkbox("Select All", value=True)
        to_plot = [name for name in sorted(processed)
                   if st.checkbox(re.sub(r'(?i)\.erp$', '', name), value=select_all, key=f"cb_{mode}_{name}")]
        if not to_plot:
            st.info("Select at least one file to plot.")
        else:
            palette   = plt.get_cmap('tab20').colors
            color_map = {n: palette[i % len(palette)] for i, n in enumerate(sorted(processed))}
            nicknames = {n: f"Mix{i+1}" for i, n in enumerate(sorted(processed))}
            temp      = next(iter(processed.values()))[1]
            temp_lb   = f"{temp:.0f}"
            lo_str    = min(pd.to_numeric(df['Strain'], errors='coerce').min() for df, _ in processed.values())
            hi_str    = max(pd.to_numeric(df['Strain'], errors='coerce').max() for df, _ in processed.values())
            range_lb  = f"{lo_str:.0f}-{hi_str:.0f}"

            fig, ax = plt.subplots(figsize=(4.5, 4.5), constrained_layout=True)
            ax.set_box_aspect(1)
            if phase != "Both":
                # will slice inside loop
                pass

            for name in to_plot:
                df, _ = processed[name]
                clean_name = re.sub(r'(?i)\.erp$', '', name)
                lbl = clean_name if legend_choice == "Filename" else nicknames[name]
                if phase != "Both":
                    peak = df[x_axis].idxmax()
                    df = df.iloc[:peak+1] if phase == "Go" else df.iloc[peak:]

                if metric == "Gp & Gpp":
                    ax.plot(df[x_axis], df['Gp_smooth'],  color=color_map[name], linewidth=LINEWIDTH, label=f"{lbl} Gp")
                    ax.plot(df[x_axis], df['Gpp_smooth'], color=color_map[name], linewidth=LINEWIDTH, linestyle="--", label=f"{lbl} Gpp")
                    unit = "[kPa]"
                else:
                    if metric in ("Gp", "Gpp"):
                        y = df[f"{metric}_smooth"]
                        unit = "[kPa]"
                    elif metric in ("Np", "Npp", "Ns"):
                        y = df[f"{metric}_smooth"]
                        unit = "[Pa·s]"
                    else:
                        y = df["TanDelta_smooth"]
                        unit = ""
                    ax.plot(df[x_axis], y, color=color_map[name], linewidth=LINEWIDTH, label=lbl)


            default_title = f"RPA - Strain Sweep {range_lb} at {temp_lb}°C - {metric} vs {x_axis}"
            col_title, col_grid = st.columns([1, 1], gap="large")
            with col_title:
                custom_title = st.text_input("**Custom plot title (leave blank for default):**", value=default_title, key="cure_custom_title")
            with col_grid:
                grid_choice = st.radio("Grid lines:", ["On", "Off"], horizontal=True)
                show_grid = (grid_choice == "On")
            plot_title   = custom_title if custom_title else default_title
            ax.set_title(plot_title, fontsize=TITLE_FS)
            if show_grid:
                ax.grid(which="major", linestyle="-", linewidth=0.5)
            else:
                ax.grid(False)

            ax.set_xscale("log"); ax.set_xticks([1,10,100]); ax.set_xticklabels([str(t) for t in [1,10,100]], fontsize=TICK_FS)
            ax.set_xlabel(x_label, fontsize=LABEL_FS); ax.set_ylabel(f"{metric} {unit}", fontsize=LABEL_FS)
            ax.tick_params(axis="both", labelsize=TICK_FS)
            leg = ax.legend(title="Mixes", fontsize=LEGEND_FS, title_fontsize=LEGEND_TITLE_FS, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True, edgecolor='black')
            leg.get_frame().set_linewidth(0.5)

            st.pyplot(fig, use_container_width=False)
            buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=300); buf.seek(0)
            st.download_button("Download plot as PNG", data=buf, file_name="rpa_dynamic_plot.png", mime="image/png")


    elif mode == "IVE Test":
        x_axis = "Freq"
        opts   = ["Gp & Gpp", "Gp", "Gpp", "Np", "Npp", "Ns", "TanDelta"]

        # — Controls —
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            metric = st.radio("Y-Axis", opts, horizontal=True)
        with col2:
            axis_type = st.radio("X-Axis:", ["Frequency", "Angular"], horizontal=True)

        # — File selection —
        select_all = st.checkbox("Select All", value=True)
        to_plot = [
            name for name in sorted(processed)
            if st.checkbox(re.sub(r'(?i)\.erp$', '', name), value=select_all, key=f"cb_{mode}_{name}")]
        if not to_plot:
            st.info("Select at least one file to plot.")
        else:
            # — Prep colors & nicknames —
            palette   = plt.get_cmap('tab20').colors
            names     = sorted(processed.keys())
            color_map = {n: palette[i % len(palette)] for i, n in enumerate(names)}
            nicknames = {n: f"Mix{i+1}" for i, n in enumerate(names)}
            temp      = next(iter(processed.values()))[1]
            temp_lb   = f"{temp:.0f}"

            # — Figure setup —
            fig, ax = plt.subplots(figsize=(4.7, 4.7), constrained_layout=True)
            ax.set_box_aspect(1)
            ax.set_xscale("log")

            # — X-axis setup —
            base_ticks = [0.001, 0.01, 0.1, 1, 10, 100]
            if axis_type == "Frequency":
                get_x    = lambda df: df[x_axis]
                x_label  = "Frequency [Hz]"
                x_ticks  = [0.001, 0.01, 0.1, 1, 10, 100]
                x_fmt    = [str(t) for t in x_ticks]
            else:
                get_x    = lambda df: 2*np.pi * df[x_axis]
                x_label  = "Angular Velocity [rad/s]"
                x_ticks  = [0.01, 0.1, 1, 10, 100, 1000]
                x_fmt    = [str(t) for t in x_ticks]

            ax.set_xlim(min(x_ticks), max(x_ticks))
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_fmt, fontsize=TICK_FS)

            # — Y-axis setup (before plotting) —
            if metric in ("Gp & Gpp", "Gp", "Gpp"):
                ax.set_yscale("log")
                ax.set_ylim(0.001, 1)
                ax.set_yticks([0.001, 0.01, 0.1, 1])
                ax.set_yticklabels([str(t) for t in [0.001, 0.01, 0.1, 1]], fontsize=TICK_FS)
                unit = "[MPa]"

            elif metric in ("Np", "Npp", "Ns"):
                ax.set_yscale("log", base=2)
                ax.margins(y=0.03)
                ax.yaxis.set_major_locator(LogLocator(base=2, subs=[1]))
                ax.yaxis.set_major_formatter(LogFormatterMathtext(base=2))
                unit = "[Pa·s]"

            else: # TanDelta
                ax.set_yscale("linear")
                ax.margins(y=0.03)
                ax.ticklabel_format(style='plain', axis='y')
                unit = ""

            # — Gridlines —
            ax.grid(which="major", linestyle="-", linewidth=0.5)

            # — Plot each mix —
            for name in to_plot:
                df, _      = processed[name]
                clean_name = re.sub(r'(?i)\.erp$', '', name)
                lbl        = clean_name

                xvals = get_x(df)
                if metric == "Gp & Gpp":
                    ax.plot(xvals, df["Gp"],  color=color_map[name], linewidth=LINEWIDTH, label=f"{lbl} Gp")
                    ax.plot(xvals, df["Gpp"], color=color_map[name], linewidth=LINEWIDTH, linestyle="--", label=f"{lbl} Gpp")
                else:
                    ycol = metric if metric != "TanDelta" else "TanDelta"
                    ax.plot(xvals, df[ycol], color=color_map[name], linewidth=LINEWIDTH, label=lbl)

            # — Custom title & grid toggle —
            default_title = f"RPA - Frequency Sweep {temp_lb}°C - {metric} vs {axis_type}"
            col_title, col_grid = st.columns([2,1], gap="large")
            with col_title:
                custom_title = st.text_input("Custom plot title (leave blank for default):", value=default_title,key="ive_custom_title")
            with col_grid:
                grid_choice = st.radio("Grid lines:", ["On", "Off"], horizontal=True)
                if grid_choice == "On":
                    ax.grid(True)
                else:
                    ax.grid(False)

            plot_title = custom_title or default_title
            ax.set_title(plot_title, fontsize=TITLE_FS)

            # — Axis labels & legend —
            ax.set_xlabel(x_label, fontsize=LABEL_FS)
            ax.set_ylabel(f"{metric} {unit}", fontsize=LABEL_FS)
            ax.tick_params(axis="both", labelsize=TICK_FS)

            leg = ax.legend(title="Mixes", fontsize=LEGEND_FS, title_fontsize=LEGEND_TITLE_FS, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True, edgecolor='black')
            leg.get_frame().set_linewidth(0.5)

            # — Render + download —
            st.pyplot(fig, use_container_width=False)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300)
            buf.seek(0)
            st.download_button("Download plot as PNG", data=buf, file_name="rpa_freqsweep_plot.png", mime="image/png")


    elif mode == "Temperature Sweep":
        opts    = ["TanDelta", "Gp & Gpp", "Gp", "Gpp", "Np", "Npp", "Ns"]
        x_axis  = "UTemp"
        x_label = "Temperature [°C]"
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            metric = st.radio("Metric", opts, horizontal=True)
        with col2:
            legend_choice = st.radio("Legend label:", ["Filename", "Nickname"], horizontal=True)
        select_all = st.checkbox("Select All", value=True)
        to_plot = [name for name in sorted(processed)
                   if st.checkbox(re.sub(r'(?i)\.erp$', '', name), value=select_all, key=f"cb_{mode}_{name}")]
        if not to_plot:
            st.info("Select at least one file to plot.")
        else:
            palette   = plt.get_cmap('tab20').colors
            color_map = {n: palette[i % len(palette)] for i, n in enumerate(sorted(processed))}
            nicknames = {n: f"Mix{i+1}" for i, n in enumerate(sorted(processed))}
            temp_lb   = f"{next(iter(processed.values()))[1]:.0f}"

            fig, ax = plt.subplots(figsize=(4.5, 4.5), constrained_layout=True)
            ax.set_box_aspect(1)
            for name in to_plot:
                df, _ = processed[name]
                clean_name = re.sub(r'(?i)\.erp$', '', name)
                lbl = clean_name if legend_choice == "Filename" else nicknames[name]
                if metric == "Gp & Gpp":
                    ax.plot(df[x_axis], df['Gp'],  color=color_map[name], linewidth=LINEWIDTH, label=f"{lbl} Gp")
                    ax.plot(df[x_axis], df['Gpp'], color=color_map[name], linewidth=LINEWIDTH, linestyle="--", label=f"{lbl} Gpp")
                    unit = "[MPa]"
                else:
                    if metric in ("Gp", "Gpp"):
                        y = df[f"{metric}"]
                        unit = "[MPa]"
                    elif metric in ("Sp"):
                        y = df[f"{metric}"]
                        unit = "[dNm]"
                    elif metric in ("Np", "Npp", "Ns"):
                        y = df[f"{metric}"]
                        unit = "[Pa·s]"
                    else:
                        y = df["TanDelta_smooth"]
                        unit = ""
                    ax.plot(df[x_axis], y, color=color_map[name], linewidth=LINEWIDTH, label=lbl)

            default_title = f"Temp Sweep {temp_lb}°C - {metric} vs Temperature"
            col_title, col_grid = st.columns([1, 1], gap="large")
            with col_title:
                custom_title = st.text_input("**Custom plot title (leave blank for default):**", value=default_title, key="cure_custom_title")
            with col_grid:
                grid_choice = st.radio("Grid lines:", ["On", "Off"], horizontal=True)
                show_grid = (grid_choice == "On")
            plot_title   = custom_title if custom_title else default_title
            ax.set_title(plot_title, fontsize=TITLE_FS)

            
            if show_grid:
                ax.grid(which="major", linestyle="-", linewidth=0.5)
            else:
                ax.grid(False)

            ax.set_xlabel(x_label, fontsize=LABEL_FS); ax.set_ylabel(f"{metric} {unit}", fontsize=LABEL_FS)
            ax.tick_params(axis="both", labelsize=TICK_FS)
            leg = ax.legend(title="Mixes", fontsize=LEGEND_FS, title_fontsize=LEGEND_TITLE_FS, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True, edgecolor='black')
            leg.get_frame().set_linewidth(0.5)

            st.pyplot(fig, use_container_width=False)
            buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=300); buf.seek(0)
            st.download_button("Download plot as PNG", data=buf, file_name="rpa_tempsweep_plot.png", mime="image/png")


    elif mode == "Plastequiv Test":
        metric = "Sp"  
        x_axis  = "Time"
        x_label = "Time [sec]"

        # File pickers
        select_all = st.checkbox("Select All", value=True)
        to_plot = [
            name for name in sorted(processed)
            if st.checkbox(re.sub(r'(?i)\.erp$', '', name), value=select_all, key=f"cb_{mode}_{name}")
        ]
        if not to_plot:
            st.info("Select at least one file to plot.")
        else:
            # Prep
            palette   = plt.get_cmap('tab20').colors
            color_map = {n: palette[i % len(palette)] for i, n in enumerate(sorted(processed))}
            nicknames = {n: f"Mix{i+1}" for i, n in enumerate(sorted(processed))}
            temp_lb   = f"{next(iter(processed.values()))[1]:.0f}"

            # Plot
            fig, ax = plt.subplots(figsize=(3.5, 3.5), constrained_layout=True)
            ax.set_box_aspect(1)
            for name in to_plot:
                df, _ = processed[name]
                clean_name = re.sub(r'(?i)\.erp$', '', name)
                lbl = clean_name
                ax.plot(df[x_axis], df["Sp_smooth"], color=color_map[name], linewidth=LINEWIDTH, label=lbl)

            default_title = f"RPA - Plastequiv Test {temp_lb}°C - Sp vs {x_axis}"
            col_title, col_grid = st.columns([1, 1], gap="large")
            with col_title:
                custom_title = st.text_input("**Custom plot title (leave blank for default):**", value=default_title, key="cure_custom_title")
            with col_grid:
                grid_choice = st.radio("Grid lines:", ["On", "Off"], horizontal=True)
                show_grid = (grid_choice == "On")
            plot_title   = custom_title if custom_title else default_title
            ax.set_title(plot_title, fontsize=TITLE_FS)
            if show_grid:
                ax.grid(which="major", linestyle="-", linewidth=0.5)
            else:
                ax.grid(False)

            ax.set_xlabel(x_label, fontsize=LABEL_FS)
            ax.set_ylabel("Sp [dNm]", fontsize=LABEL_FS)
            ax.tick_params(axis="both", labelsize=TICK_FS)
            ax.set_xlim(left=0)
            leg = ax.legend(title="Mixes", fontsize=LEGEND_FS, title_fontsize=LEGEND_TITLE_FS, loc="upper right", frameon=True, edgecolor='black')
            leg.get_frame().set_linewidth(0.5)

            st.pyplot(fig, use_container_width=False)
            buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=300); buf.seek(0)
            st.download_button("Download plot as PNG", data=buf, file_name="rpa_plastequiv_plot.png", mime="image/png")


    elif mode == "Indus - Plastequiv Test":
        x_axis  = "Time"
        x_label = "Time [sec]"


        # File pickers
        select_all = st.checkbox("Select All", value=True)
        to_plot = [
            name for name in sorted(processed)
            if st.checkbox(re.sub(r'(?i)\.erp$', '', name), value=select_all, key=f"cb_{mode}_{name}")
        ]
        if not to_plot:
            st.info("Select at least one file to plot.")
        else:
            palette   = plt.get_cmap('tab20').colors
            color_map = {n: palette[i % len(palette)] for i, n in enumerate(sorted(processed))}
            nicknames = {n: f"Mix{i+1}" for i, n in enumerate(sorted(processed))}

            col_grid, _ = st.columns([1, 1], gap="large")
            with col_grid:
                grid_choice = st.radio("Grid lines:", ["On", "Off"], horizontal=True)
                show_grid = (grid_choice == "On")
            
            # Two‐panel: Ss on left, Sp & Spp on right
            colL, colR = st.columns(2, gap="large")

            with colL:
                fig1, ax1 = plt.subplots(figsize=(3.5, 3.5), constrained_layout=True)
                ax1.set_box_aspect(1)
                for name in to_plot:
                    df, _ = processed[name]
                    clean_name = re.sub(r'(?i)\.erp$', '', name)
                    lbl = clean_name
                    ax1.plot(df[x_axis], df["Ss_smooth"], color=color_map[name], linewidth=LINEWIDTH, label=lbl)

                default_title1 = "Plastequiv Test - Ss vs Time"
                custom_title1 = st.text_input("**Custom plot title (leave blank for default):**", value=default_title1, key="cure_custom_title1")
                plot_title1   = custom_title1 if custom_title1 else default_title1
                ax1.set_title(plot_title1, fontsize=TITLE_FS)

                ax1.set_xlabel(x_label, fontsize=LABEL_FS)
                ax1.set_ylabel("Ss [dNm]", fontsize=LABEL_FS)
                ax1.tick_params(axis="both", labelsize=TICK_FS)
                ax1.set_xlim(left=0)

                if show_grid:
                    ax1.grid(which="major", linestyle="-", linewidth=0.5)
                else:
                    ax1.grid(False)

                leg1 = ax1.legend(title="Mixes", fontsize=LEGEND_FS, title_fontsize=LEGEND_TITLE_FS)
                leg1.get_frame().set_linewidth(0.5)
                st.pyplot(fig1, use_container_width=True)
                buf1 = io.BytesIO(); fig1.savefig(buf1, format="png", dpi=300); buf1.seek(0)
                st.download_button("Download plot as PNG", data=buf1, file_name="rpa_plastequivSs_plot.png", mime="image/png")

            with colR:
                fig2, ax2 = plt.subplots(figsize=(3.5, 3.5), constrained_layout=True)
                ax2.set_box_aspect(1)
                for name in to_plot:
                    df, _ = processed[name]
                    clean_name = re.sub(r'(?i)\.erp$', '', name)
                    lbl = clean_name
                    ax2.plot(df[x_axis], df["Sp_smooth"], color=color_map[name], linewidth=LINEWIDTH, label=f"{lbl} Sp")
                    ax2.plot(df[x_axis], df["Spp_smooth"], color=color_map[name], linewidth=LINEWIDTH, linestyle="--", label=f"{lbl} Spp")

                default_title2 = "Plastequiv Test - Sp & Spp vs Time"
                custom_title2 = st.text_input("**Custom plot title (leave blank for default):**", value=default_title2, key="cure_custom_title2")
                plot_title2   = custom_title2 if custom_title2 else default_title2
                ax2.set_title(plot_title2, fontsize=TITLE_FS)
                ax2.set_xlim(left=0)
 
                ax2.set_xlabel(x_label, fontsize=LABEL_FS)
                ax2.set_ylabel("Sp & Spp [dNm]", fontsize=LABEL_FS)
                ax2.tick_params(axis="both", labelsize=TICK_FS)
                if show_grid:
                    ax2.grid(which="major", linestyle="-", linewidth=0.5)
                else:
                    ax2.grid(False)
                leg2 = ax2.legend(title="Mixes", fontsize=LEGEND_FS, title_fontsize=LEGEND_TITLE_FS)
                leg2.get_frame().set_linewidth(0.5)
                st.pyplot(fig2, use_container_width=True)
                buf2 = io.BytesIO(); fig2.savefig(buf2, format="png", dpi=300); buf2.seek(0)
                st.download_button("Download plot as PNG", data=buf2, file_name="rpa_plastequivSpSpp_plot.png", mime="image/png")


    elif mode == "Indus - Stress Decay":

        # File selection (same style as Cure Test)
        select_all = st.checkbox("Select All", value=True)
        to_plot = [
            name for name in sorted(processed)
            if st.checkbox(re.sub(r'(?i)\.txt$|\.erp$|\.csv$', '', name), value=select_all, key=f"cb_{mode}_{name}")
        ]
        if not to_plot:
            st.info("Select at least one file to plot.")
            st.stop()

        # prepare a consistent color map
        palette   = plt.get_cmap("tab20").colors
        color_map = {n: palette[i % len(palette)] for i, n in enumerate(sorted(processed))}

        col1, col2 = st.columns([1, 1], gap="large")

        # —— LEFT PANEL: Time vs Torque (log–log) ——
        with col1:
            from matplotlib.ticker import ScalarFormatter

            fig, ax = plt.subplots(figsize=(3.5, 3.5), constrained_layout=True)
            ax.set_box_aspect(1)

            for name in to_plot:
                df, _ = processed[name]
                label = re.sub(r'(?i)\.txt$|\.erp$|\.csv$', '', name)
                ax.plot(df["Time"], df["Torque_smooth"], color=color_map[name], linewidth=LINEWIDTH, label=label)

            ax.set_xscale("log"); ax.set_yscale("log")

            fmt = ScalarFormatter(); fmt.set_scientific(False); fmt.set_useOffset(False)
            ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)

            ax.set_title("Time vs Torque", fontsize=TITLE_FS)
            ax.set_xlabel("Time [min]", fontsize=LABEL_FS)
            ax.set_ylabel("Torque [dNm]", fontsize=LABEL_FS)
            ax.tick_params(axis="both", labelsize=TICK_FS)
            ax.grid(which="major", linestyle="-", linewidth=0.5)

            leg = ax.legend(title="Runs", fontsize=LEGEND_FS, title_fontsize=LEGEND_TITLE_FS, loc="best", frameon=True, edgecolor="black")
            leg.get_frame().set_linewidth(0.5)

            st.pyplot(fig, use_container_width=False)


        # —— RIGHT PANEL: Strain vs Modulus (linear) ——
        with col2:
            fig, ax = plt.subplots(figsize=(3.5, 3.5), constrained_layout=True)
            ax.set_box_aspect(1)
            for name in to_plot:
                df, _ = processed[name]
                label = re.sub(r'(?i)\.txt$|\.erp$|\.csv$', '', name)
                ax.plot(df["Time"], df["Modulus_smooth"], color=color_map[name], linewidth=LINEWIDTH, label=label)

            ax.set_xscale("log"); ax.set_yscale("linear")
            fmt = ScalarFormatter(); fmt.set_scientific(False); fmt.set_useOffset(False)
            ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)

            ax.set_title("Time vs Modulus", fontsize=TITLE_FS)
            ax.set_xlabel("Time [min]", fontsize=LABEL_FS)
            ax.set_ylabel("Modulus [kPa]", fontsize=LABEL_FS)
            ax.tick_params(axis="both", labelsize=TICK_FS)
            ax.grid(which="major", linestyle="-", linewidth=0.5)
            leg = ax.legend(title="Runs", fontsize=LEGEND_FS, title_fontsize=LEGEND_TITLE_FS, loc="best", frameon=True, edgecolor="black")
            leg.get_frame().set_linewidth(0.5)
            st.pyplot(fig, use_container_width=False)

     



# — Key Values —
with tab_key:
    st.subheader(f"{mode} — key values")

    if mode == "Cure Test":
        summary = {}
        thresholds = {
            'TS2 (min)':  0.02,
            'TC30 (min)': 0.30,
            'TC50 (min)': 0.50,
            'TC70 (min)': 0.70,
            'TC90 (min)': 0.90,
            'TC95 (min)': 0.95,
            'TC99 (min)': 0.99,
            'TC100 (min)':1.00
        }

        for name, (df, temp) in processed.items():
            # basic stats
            total_time = df['Time'].max()
            sp_max     = df['Sp'].max()
            sp_min     = df['Sp'].min()
            sp_range   = sp_max - sp_min

            # find times to reach thresholds
            times = {}
            for label, frac in thresholds.items():
                # first time where Sp ≥ frac * max
                hits = df.loc[df['Sp'] >= frac * sp_max, 'Time']
                times[label] = hits.iloc[0] if not hits.empty else float('nan')

            # collect row in dict-of-dicts
            summary[name] = {
                'Total Time  (min)': total_time,
                'Max Sp (dNm)':     sp_max,
                'Min Sp (dNm)':     sp_min,
                'Sp Range (dNm)':   sp_range,
                **times
            }


        # build the DataFrame, then reorder its columns (filenames) alphabetically
        summary_df = pd.DataFrame(summary).T
        summary_df = summary_df.sort_index(axis=0)
        summary_df.index = summary_df.index.str.replace(r'(?i)\.erp$', '', regex=True)
        st.dataframe(summary_df, use_container_width=True)

        ####################################################################
        st.markdown("---")
        st.subheader("Cure Law")

        # compute the overall max time once
        max_time = max(df['Time'].max() for (df, temp) in processed.values())
        t_select = st.number_input(
            "Evaluate at time (min):",
            min_value=0.0,
            max_value=float(max_time),
            value=0.0,
            step=0.1,
            key="key_vals_cure_law"
        )

        results = []
        for name, (df, temp) in processed.items():
            sp_at_t = float(np.interp(t_select, df['Time'], df['Sp']))
            pct     = (sp_at_t / df['Sp'].max() * 100) if df['Sp'].max() else 0.0
            results.append({
                "Mix":    name,
                "Sp at time 't' (dNm)": round(sp_at_t, 3),
                "%": round(pct, 1)
            })

        cure_law_df = pd.DataFrame(results).set_index("Mix")
        cure_law_df.index = cure_law_df.index.str.replace(r'(?i)\.erp$', '', regex=True)
        st.dataframe(cure_law_df, use_container_width=True)



    elif mode == "Scorch Test":
        summary = {}
        thresholds = {
            'T5 (min)':  1.05,
            'T35 (min)': 1.35
        }

        for name, (df, temp) in processed.items():
            # find the minimum Sp and the last time it occurs
            sp_min = df['Sp'].min()
            t0 = df.loc[df['Sp'] == sp_min, 'Time'].iloc[-1]

            # now restrict to times after T0
            df_after = df[df['Time'] > t0]

            # find each threshold time
            times = {}
            for label, factor in thresholds.items():
                target = sp_min * factor
                hits = df_after.loc[df_after['Sp'] >= target, 'Time']
                times[label] = hits.iloc[0] if not hits.empty else float('nan')

            summary[name] = {
                'Min Sp (dNm)':     sp_min,
                'T0 (min)':   t0,
                'T5 (min)':   times['T5 (min)'],
                'T35 (min)':  times['T35 (min)']
            }

        # build DataFrame with metrics as rows, filenames as columns
        summary_df = pd.DataFrame(summary).T
        summary_df = summary_df.sort_index(axis=0)
        summary_df.index = summary_df.index.str.replace(r'(?i)\.erp$', '', regex=True)
        st.dataframe(summary_df, use_container_width=True)



    elif mode == "Dynamic Test":
        # Phase selector
        phase = st.radio("Phase", ["Both", "Go", "Return"], horizontal=True, key="dyn_test")

        thresholds = [10, 20, 50]  # T10→5%, T20→10%, T50→25%
        summary = {}
        intersection = {}

        for name, (df, temp) in processed.items():
            # --- Go/Return split + TanDelta thresholds ---
            peak_idx = df['Strain'].idxmax()
            go_df  = df.loc[:peak_idx]
            ret_df = df.loc[peak_idx:]

            d = {
                'Max TanD (Go)':  go_df['TanDelta'].max(),
                'Max TanD (Ret)': ret_df['TanDelta'].max(),
            }
            for t in thresholds:
                cutoff = t / 2
                idx_go  = (go_df['Strain'] - cutoff).abs().idxmin()
                idx_ret = (ret_df['Strain'] - cutoff).abs().idxmin()
                d[f'T{t} (Go)']  = go_df.loc[idx_go,  'TanDelta']
                d[f'T{t} (Ret)'] = ret_df.loc[idx_ret, 'TanDelta']
            summary[name] = d

            # --- intersection of G' and G'' ---
            diffs = df['Gp'] - df['Gpp']
            if (diffs > 0).all() or (diffs < 0).all():
                # no crossing
                intersection[name] = None
            else:
                idx_int = diffs.abs().idxmin()
                intersection[name] = df.loc[idx_int, 'Strain']

        # --- build & display main summary table ---
        summary_df = pd.DataFrame(summary).T.sort_index()
        go_cols  = [c for c in summary_df.columns if "(Go)"  in c]
        ret_cols = [c for c in summary_df.columns if "(Ret)" in c]
        summary_df = summary_df[go_cols + ret_cols]

        if phase == "Go":
            summary_df = summary_df[go_cols]
        elif phase == "Return":
            summary_df = summary_df[ret_cols]

        # scrub .erp/.eRP from mix names
        summary_df.index = summary_df.index.str.replace(r'(?i)\.erp$', '', regex=True)
        st.dataframe(summary_df, use_container_width=True)

        # --- build & display intersection table ---
        inter_df = pd.DataFrame.from_dict(intersection, orient='index', columns=["Strain at G'=G''"])
        inter_df.index = inter_df.index.str.replace(r'(?i)\.erp$', '', regex=True)
        inter_df = inter_df.fillna("N/A")
        st.dataframe(inter_df, use_container_width=True)



    elif mode == "Temperature Sweep":
        intersection_temp = {}
        for name, (df, temp) in processed.items():
            # signed difference
            diff = df['Gp'] - df['Gpp']

            # if no sign change at all, mark N/A
            if (diff > 0).all() or (diff < 0).all():
                intersection_temp[name] = None
            else:
                # otherwise pick the row closest to the zero‐crossing
                idx_int = diff.abs().idxmin()
                intersection_temp[name] = df.loc[idx_int, 'Temp']

        # build the output DataFrame, replacing None with "N/A"
        inter_temp_df = pd.DataFrame.from_dict(intersection_temp, orient='index', columns=["Temperature at G'=G''"]).fillna("N/A")
        inter_temp_df.index = inter_temp_df.index.str.replace(r'(?i)\.erp$', '', regex=True)
        st.dataframe(inter_temp_df, use_container_width=True)



    elif mode == "Plastequiv Test":

        summary = []
        for name in sorted(processed):
            df, temp = processed[name]
            # grab last non‐NaN Np value
            last_np = pd.to_numeric(df["Np"], errors="coerce").dropna().iloc[-1]
            summary.append({
                "Mix": name,
                "Viscosity - Np (kPA)": last_np / 1000
            })
        summary_df = pd.DataFrame(summary).set_index("Mix")
        summary_df.index = summary_df.index.str.replace(r'(?i)\.erp$', '', regex=True)
        st.dataframe(summary_df, use_container_width=True)



    elif mode == "IVE Test":
        summary = []
        for name in sorted(processed):
            df, temp = processed[name]
            # drop rows missing the needed columns and sort by frequency
            df_clean = (df.dropna(subset=["Freq", "Gp", "Gpp"]).sort_values("Freq"))

            # 1) IVE: ratio Gp/Gpp at the lowest reported freq
            f_min   = df_clean["Freq"].iloc[0]
            gp_min  = df_clean["Gp"].iloc[0]
            gpp_min = df_clean["Gpp"].iloc[0]
            ive_val = gp_min / gpp_min if gpp_min else float("nan")

            # 2) crossover freq where Gp = Gpp
            diffs = df_clean["Gp"] - df_clean["Gpp"]
            # find the first index where the sign flips
            idx = np.where(diffs.values[:-1] * diffs.values[1:] <= 0)[0]
            if idx.size:
                i = idx[0]
                x0, x1 = df_clean["Freq"].iloc[i],   df_clean["Freq"].iloc[i+1]
                y0, y1 = diffs.iloc[i],             diffs.iloc[i+1]
                f_cross = x0 - y0 * (x1 - x0) / (y1 - y0)
            else:
                f_cross = float("nan")

            summary.append({
                "Mix": name,
                "IVE": ive_val,
                "Crossover Frequency (Hz)": f_cross
            })

        summary_df = pd.DataFrame(summary).set_index("Mix")
        summary_df.index = summary_df.index.str.replace(r'(?i)\.erp$', '', regex=True)
        st.dataframe(summary_df, use_container_width=True)



    elif mode == "Indus - Plastequiv Test":
        # for each mix, compute Ss & Sp max, end‐of‐run, and Sp/Spp crossover time
        summary = []
        for name, (df, temp) in processed.items():
            ss_max   = df['Ss'].max()
            ss_final = df['Ss'].iloc[-1]
            sp_max   = df['Sp'].max()
            sp_final = df['Sp'].iloc[-1]

            # detect zero‐crossing of Sp–Spp
            diff = df['Sp'] - df['Spp']
            if (diff > 0).all() or (diff < 0).all():
                t_cross = None
            else:
                idx_cross = diff.abs().idxmin()
                t_cross   = df.loc[idx_cross, 'Time']

            summary.append({
                "Mix":                  name,
                "Ss Maximum":           ss_max,
                "Ss Final Value":       ss_final,
                "Overshoot":            ss_max / ss_final,
                "Sp Maximum":           sp_max,
                "Sp Final Value":       sp_final,
                "Time at Sp=Spp (min)": t_cross
            })

        summary_df = pd.DataFrame(summary).set_index("Mix")
        # scrub the .erp extension
        summary_df.index = summary_df.index.str.replace(r'(?i)\.erp$', '', regex=True)
        # replace missing crossings with "N/A"
        summary_df = summary_df.fillna("N/A")

        st.dataframe(summary_df, use_container_width=True)





    else:
        st.info("Key values for this mode coming soon.")





# — Data Interface —
with tab_data:

    for name in sorted(processed):
        st.markdown(f"**{name}**")
        df, temp = processed[name]

        if mode == "Cure Test":
            df_display = df.iloc[:, :22].copy()
            st.dataframe(df_display, use_container_width=True)

        elif mode == "Scorch Test":
            df_display = df.iloc[:, :22].copy()
            st.dataframe(df_display, use_container_width=True)

        elif mode == "Dynamic Test":
            df_display = df.iloc[:, :27].copy()
            st.dataframe(df_display, use_container_width=True)
        
        elif mode == "Temperature Sweep":
            df_display = df.iloc[:, :22].copy()
            st.dataframe(df_display, use_container_width=True)

        elif mode == "IVE Test":
            df_display = df.iloc[:, :27].copy()
            freq_idx = df_display.columns.get_loc("Freq")
            df_display.insert(freq_idx + 1, "Angular (rad/s)", df_display["Freq"] * 2 * np.pi)
            st.dataframe(df_display, use_container_width=True)

        elif mode == "Plastequiv Test":
            df_display = df.iloc[:, :26].copy()
            st.dataframe(df_display, use_container_width=True)
        
        elif mode == "Indus - Plastequiv Test":
            df_display = df.iloc[:, :22].copy()
            st.dataframe(df_display, use_container_width=True)
        
        elif mode == "Indus - Stress Decay":
            df_display = df.iloc[:, :4].copy()
            st.dataframe(df_display, use_container_width=True)
