import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator, LogFormatter, FuncFormatter
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
    idx = next((i for i, L in enumerate(lines) if L.strip() == search_header), None)
    if idx is None or idx < 7:
        return None
    temp_line = lines[idx - 10].strip()
    parts = [p.strip() for p in temp_line.split(',') if p.strip() != ""]
    try:
        return float(parts[-1])
    except:
        return None


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
    raw = buffer.readlines() if hasattr(buffer, "readlines") else open(buffer, 'rb').readlines()
    lines = [
        L.decode('utf-8', errors='replace') if isinstance(L, (bytes, bytearray)) else L
        for L in raw
    ]
    search_header = (
        "GenericA,GenericA,Time,Temp,Temp,Strain,Freq,Strain,Temp,,Torque,Torque,Torque,"
        "Modulus,Modulus,Modulus,Compl,Compl,Compl,Visc,Visc,Visc,"
        "GenericB,Shear,Reserve1,Reserve2,Pressure"
    )
    idx = next((i for i,L in enumerate(lines) if L.strip() == search_header), None)
    if idx is None:
        raise ValueError("Dynamic header not found in file.")
    data_str = "".join(lines[idx+1:])

    temp = _read_test_temp(lines, search_header)
    return pd.read_csv(StringIO(data_str), names=col_names), temp


# ——— IVE low-level loader ———
def _load_raw_df_ive(buffer, col_names):
    raw = buffer.readlines() if hasattr(buffer, "readlines") else open(buffer, 'rb').readlines()
    lines = [
        L.decode('utf-8', errors='replace') if isinstance(L, (bytes, bytearray)) else L
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
    data_str = "".join(lines[idx + 1:])
    temp = _read_test_temp(lines, "Time,Strain,Torque,Torque,Torque,Modulus,Modulus,Modulus,Compl,"
                "Compl,Compl,Visc,Visc,Visc,GenericB,Temp,Temp,Temp,Pressure,Force,Reserve1,Reserve2")
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
    df['Strain']      = pd.to_numeric(df['Strain'], errors='coerce')
    df['Gp']          = pd.to_numeric(df['Gp'],     errors='coerce')
    df['TanDelta']    = pd.to_numeric(df['TDelt'],    errors='coerce') / df['Gp']
    df['Gp_smooth']       = df['Gp'].rolling(window=3, center=True).mean()
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
    df['Gp'] = pd.to_numeric(df['Gp'], errors='coerce')
    df['Gpp'] = pd.to_numeric(df['Gpp'], errors='coerce')
    df['Gp_smooth'] = df['Gp'].rolling(window=3, center=True).mean()
    df['Gpp_smooth'] = df['Gpp'].rolling(window=3, center=True).mean()
    # ensure time is numeric
    df['Freq'] = pd.to_numeric(df['Freq'], errors='coerce')
    return df, temp



# ——— UI ———
st.title("RPA Post-Processing Tool")

mode = st.selectbox(
    "Choose mode:",
    ["Cure Test", "Scorch Test", "Dynamic Test", "IVE Test"],
)

# per-mode uploader
key_map = {
    "Cure Test":   "uploader_cure",
    "Scorch Test": "uploader_scorch",
    "Dynamic Test": "uploader_dynamic",
    "IVE Test": "uploader_ive"
}
labels = {
    "Cure Test":      "Upload Cure-test .erp files",
    "Scorch Test":    "Upload Scorch-test .erp files",
    "Dynamic Test":   "Upload dynamic-test .erp files",
    "IVE Test": "Upload IVE-test .erp files"
}

uploaded = st.file_uploader(
    labels[mode],
    type=['erp','txt','csv'],
    accept_multiple_files=True,
    key=key_map[mode]
)

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
        elif mode == "Dynamic Test":
            df, temp = clean_dynamic_file(f)
        else:
            df, temp = clean_ive_file(f)
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

with tab_graph:
    st.subheader(f"{mode} — pick curves to plot")

    # choose metrics and axes per mode
    if mode in ("Cure Test", "Scorch Test"):
        opts    = ["Sp", "Gp", "Alpha"]
        x_axis  = "Time"
        x_label = "Time [min]"
    elif mode == "Dynamic Test":
        opts    = ["TanDelta", "Gp"]
        x_axis  = "Strain"
        x_label = "Strain"
    elif mode == "IVE Test":
        opts    = ["Gp & Gpp", "Gp", "Gpp"]
        x_axis  = "Freq"
        x_label = "Frequency [Hz]"

    # controls layout
    if mode == "Dynamic Test":
        col1, col2, col3 = st.columns([1, 1, 1], gap="large")
        with col1:
            metric = st.radio("Metric", opts, horizontal=True)
        with col2:
            phase = st.radio("Phase", ["Both", "Go", "Return"], horizontal=True)
        with col3:
            legend_choice = st.radio("Legend label:", ["Filename", "Nickname"], horizontal=True)
    else:
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            metric = st.radio("Metric", opts, horizontal=True)
        with col2:
            legend_choice = st.radio("Legend label:", ["Filename", "Nickname"], horizontal=True)


    select_all = st.checkbox("Select All", value=True)
    to_plot    = [
        name for name in sorted(processed)
        if st.checkbox(name, value=select_all, key=f"cb_{mode}_{name}")
    ]

    if not to_plot:
        st.info("Select at least one file to plot.")
    else:
        # prep colors & nicknames
        palette   = plt.get_cmap('tab20').colors
        names     = sorted(processed.keys())
        color_map = {n: palette[i % len(palette)] for i, n in enumerate(names)}
        nicknames = {n: f"Mix{i+1}" for i, n in enumerate(names)}

        # compute labels
        if mode in ("Cure Test", "Scorch Test"):
            temp    = next(iter(processed.values()))[1]
            max_t   = max(df['Time'].max() for df, _ in processed.values())
            time_lb = f"{max_t:.0f}"
            temp_lb = f"{temp:.0f}"
        else:
            temp     = next(iter(processed.values()))[1]
            temp_lb  = f"{temp:.0f}"
            lo_str   = min(df['Strain'].min() for df, _ in processed.values())
            hi_str   = max(df['Strain'].max() for df, _ in processed.values())
            range_lb = f"{lo_str:.0f}-{hi_str:.0f}"

        # — Square figure & aspect —
        fig, ax = plt.subplots(figsize=(3.5, 3.5), constrained_layout=True)
        ax.set_box_aspect(1)

        # log-scale for Dynamic
        if mode == "Dynamic Test":
            ax.set_xscale("log")
            ticks = [1, 10, 100, 1000]
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks], fontsize=TICK_FS)

        # log–log & decade ticks for IVE Test
        if mode == "IVE Test":
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(0.01, 100)
            ax.set_ylim(0.001, 1)

            xticks = [0.01, 0.1, 1, 10, 100]
            yticks = [0.001, 0.01, 0.1, 1]
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(t) for t in xticks], fontsize=TICK_FS)
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(t) for t in yticks], fontsize=TICK_FS)

            ax.grid(which="major", linestyle="-", linewidth=0.5)


        # plot each mix
        for name in to_plot:
            df, _ = processed[name]

            # split Go/Return
            if mode == "Dynamic Test" and phase != "Both":
                peak = df[x_axis].idxmax()
                df = df.iloc[:peak+1] if phase == "Go" else df.iloc[peak:]
            


            # select y
            if metric in ("Gp", "Sp", "Gpp"):
                y = df[f"{metric}_smooth"]
            elif metric == "Alpha":
                y = df["Sp_smooth"] / df["Sp"].max()
            elif metric == "TanDelta":
                y = df["TDelt"]
            elif metric in ("Np", "Ns"):
                y = df[f"{metric}_smooth"]


            lbl = name if legend_choice == "Filename" else nicknames[name]

            if mode == "IVE Test":
                if metric == "Gp & Gpp":
                    ax.plot(df[x_axis], df["Gp_smooth"],  linewidth=LINEWIDTH, color=color_map[name], label=f"{lbl} Gp")
                    ax.plot(df[x_axis], df["Gpp_smooth"], linewidth=LINEWIDTH, color=color_map[name], linestyle="--", label=f"{lbl} Gpp")
                else:
                    ax.plot(df[x_axis], df[f"{metric}_smooth"], linewidth=LINEWIDTH, label=lbl)
                continue


            ax.plot(df[x_axis], y, color=color_map[name], linewidth=LINEWIDTH, label=lbl)



        # titles & labels
        if mode in ("Cure Test", "Scorch Test"):
            title = f"RPA - {mode} {temp_lb}°C/{time_lb}min - {metric} vs {x_axis}"
        elif mode == "Dynamic Test":
            title = f"RPA - Strain Sweep {range_lb} at {temp_lb}°C - {metric} vs {x_axis}"
        elif mode == "IVE Test":
            title = f"RPA - IVE Test {temp_lb}°C - {metric} vs Frequency"

        ax.set_title(title, fontsize=TITLE_FS)
        ax.set_xlabel(x_label, fontsize=LABEL_FS)
        
        if metric in ("Gp", "Sp", "Gpp"):
            y_label = f"Torque ({metric}) [dNm]"
        elif metric == "Gp & Gpp":
            y_label = "Torque (Gp & Gpp) [dNm]"
        else:
            y_label = metric
        ax.set_ylabel(y_label, fontsize=LABEL_FS)


        ax.tick_params(axis="both", labelsize=TICK_FS)


        # pin bottom & left to data‐origin
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.spines["right"].set_visible(True)
        ax.spines["top"].set_visible(True)

    

        # legend _inside_ plot
        handles, labels = ax.get_legend_handles_labels()
        if legend_choice == "Filename":
            labels = [re.sub(r"(?i)\.erp", "", lbl) for lbl in labels]

        sorted_pairs    = sorted(zip(labels, handles), key=lambda x: x[0])
        lbls, hnds      = zip(*sorted_pairs)
        if mode == "IVE Test":
            leg = ax.legend(hnds, lbls, title="Mixes", fontsize=LEGEND_FS, title_fontsize=LEGEND_TITLE_FS, loc="lower right", frameon=True, edgecolor='black')
        else:
            leg = ax.legend(hnds, lbls, title="Mixes", fontsize=LEGEND_FS, title_fontsize=LEGEND_TITLE_FS, loc="lower right", frameon=True, edgecolor='black')
        leg.get_frame().set_linewidth(0.5)

        # render and download
        st.pyplot(fig, use_container_width=False)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        st.download_button("Download plot as PNG", data=buf, file_name="rpa_plot.png", mime="image/png")







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
        st.dataframe(summary_df, use_container_width=True)

    elif mode == "Dynamic Test":
        phase = st.radio("Phase", ["Both", "Go", "Return"], horizontal=True, key="dyn_test")

        summary = {}
        for name, (df, temp) in processed.items():
            peak = df['Strain'].idxmax()
            go_df  = df.loc[:peak]
            ret_df = df.loc[peak:]
            summary[name] = {
                'Max TanD (Go)':     go_df['TanDelta'].max(),
                'T10 (Go)':          go_df.loc[go_df['Strain']>=10,  'TanDelta'].iloc[0],
                'T20 (Go)':          go_df.loc[go_df['Strain']>=20,  'TanDelta'].iloc[0],
                'T50 (Go)':          go_df.loc[go_df['Strain']>=50,  'TanDelta'].iloc[0],
                'Max TanD (Ret)': ret_df['TanDelta'].max(),
                'T10 (Ret)':         ret_df.loc[ret_df['Strain']>=10, 'TanDelta'].iloc[0],
                'T20 (Ret)':         ret_df.loc[ret_df['Strain']>=20, 'TanDelta'].iloc[0],
                'T50 (Ret)':         ret_df.loc[ret_df['Strain']>=50, 'TanDelta'].iloc[0]
            }

        summary_df = pd.DataFrame(summary).T.sort_index()

        if phase == "Go":
            cols = [c for c in summary_df.columns if "(Go)" in c]
            summary_df = summary_df[cols]
        elif phase == "Return":
            cols = [c for c in summary_df.columns if "(Ret)" in c]
            summary_df = summary_df[cols]

        st.dataframe(summary_df, use_container_width=True)


    else:
        st.info("Key values for this mode coming soon.")





# — Data Interface —
with tab_data:
    st.subheader("Inspect the Cleaned DataFrames")

    for name in sorted(processed):
        st.markdown(f"**{name}**")
        df, temp = processed[name]

        if mode == "Cure Test":
            # first 22 columns → rename to cure_names
            df_display = df.iloc[:, :22].copy()
            alpha = df_display["Sp"] / df_display["Sp"].max()
            df_display.insert(loc=2, column="Alpha", value=alpha)
            st.dataframe(df_display, use_container_width=True)

        elif mode == "Scorch Test":
            # first 22 columns → rename to scorch_names
            df_display = df.iloc[:, :22].copy()
            alpha = df_display["Sp"] / df_display["Sp"].max()
            df_display.insert(loc=2, column="Alpha", value=alpha)
            st.dataframe(df_display, use_container_width=True)

        elif mode == "Dynamic Test":
            # first 27 columns → rename to sweep_names
            df_display = df.iloc[:, :27].copy()
            st.dataframe(df_display, use_container_width=True)

        elif mode == "IVE Test":
            df_display = df.iloc[:, :27].copy()
            st.dataframe(df_display, use_container_width=True)
