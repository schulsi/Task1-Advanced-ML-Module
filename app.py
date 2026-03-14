import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

st.set_page_config(
    page_title="DoS Detection",
    page_icon="🛡️",
    layout="centered",
)

TOP_FEATURES = [
    "Avg Bwd Segment Size", "Max Packet Length", "Bwd Packet Length Max",
    "Packet Length Std", "Packet Length Mean", "Bwd Packet Length Mean",
    "Average Packet Size", "Packet Length Variance", "Subflow Bwd Bytes",
    "Total Length of Bwd Packets", "Fwd Packet Length Max", "Bwd Packet Length Std",
    "Init_Win_bytes_forward", "Flow Duration", "Flow Bytes/s", "Flow IAT Mean",
    "Fwd IAT Total", "Flow IAT Std", "Idle Mean", "Active Mean",
]

# Real median values confirmed to predict correctly with this model
EXAMPLES = {
    "BENIGN": {
        "Avg Bwd Segment Size":        71.0,
        "Max Packet Length":           72.0,
        "Bwd Packet Length Max":       71.0,
        "Packet Length Std":           17.5,
        "Packet Length Mean":          51.0,
        "Bwd Packet Length Mean":      71.0,
        "Average Packet Size":         63.5,
        "Packet Length Variance":      307.0,
        "Subflow Bwd Bytes":           140.0,
        "Total Length of Bwd Packets": 140.0,
        "Fwd Packet Length Max":       33.0,
        "Bwd Packet Length Std":       0.0,
        "Init_Win_bytes_forward":      -1.0,
        "Flow Duration":               230.0,
        "Flow Bytes/s":                20389.0,
        "Flow IAT Mean":               85.0,
        "Fwd IAT Total":               3.0,
        "Flow IAT Std":                92.7,
        "Idle Mean":                   0.0,
        "Active Mean":                 0.0,
    },
    "DoS GoldenEye": {
        "Avg Bwd Segment Size":        1661.7,
        "Max Packet Length":           4392.0,
        "Bwd Packet Length Max":       4392.0,
        "Packet Length Std":           1580.9,
        "Packet Length Mean":          746.9,
        "Bwd Packet Length Mean":      1661.7,
        "Average Packet Size":         795.4,
        "Packet Length Variance":      2499208.0,
        "Subflow Bwd Bytes":           11632.0,
        "Total Length of Bwd Packets": 11632.0,
        "Fwd Packet Length Max":       382.0,
        "Bwd Packet Length Std":       2184.4,
        "Init_Win_bytes_forward":      29200.0,
        "Flow Duration":               11485031.0,
        "Flow Bytes/s":                1010.9,
        "Flow IAT Mean":               1035509.0,
        "Fwd IAT Total":               6542860.0,
        "Flow IAT Std":                2180951.5,
        "Idle Mean":                   6487531.5,
        "Active Mean":                 766.0,
    },
}

DEFAULTS = {f: 0.0 for f in TOP_FEATURES}
DEFAULTS["Init_Win_bytes_forward"] = -1.0
DEFAULTS["Flow Duration"]          = 230.0
DEFAULTS["Flow Bytes/s"]           = 20389.0
DEFAULTS["Flow IAT Mean"]          = 85.0
DEFAULTS["Flow IAT Std"]           = 92.7
DEFAULTS["Fwd IAT Total"]          = 3.0
DEFAULTS["Avg Bwd Segment Size"]   = 71.0
DEFAULTS["Max Packet Length"]      = 72.0
DEFAULTS["Bwd Packet Length Max"]  = 71.0
DEFAULTS["Packet Length Std"]      = 17.5
DEFAULTS["Packet Length Mean"]     = 51.0
DEFAULTS["Bwd Packet Length Mean"] = 71.0
DEFAULTS["Average Packet Size"]    = 63.5
DEFAULTS["Packet Length Variance"] = 307.0
DEFAULTS["Subflow Bwd Bytes"]      = 140.0
DEFAULTS["Total Length of Bwd Packets"] = 140.0
DEFAULTS["Fwd Packet Length Max"]  = 33.0


@st.cache_resource
def load_models():
    try:
        return joblib.load("binary_model.pkl"), joblib.load("attack_model.pkl")
    except FileNotFoundError:
        return None, None

binary_model, attack_model = load_models()


def predict(values: dict):
    df    = pd.DataFrame([values])[TOP_FEATURES]
    s1    = binary_model.predict(df)[0]
    conf1 = float(max(binary_model.predict_proba(df)[0]))
    if s1 == "BENIGN":
        return "BENIGN", conf1, None, None
    s2    = attack_model.predict(df)[0]
    conf2 = float(max(attack_model.predict_proba(df)[0]))
    return "ATTACK", conf1, s2, conf2


# ── Init session state ─────────────────────────────────────────────────────
if "vals" not in st.session_state:
    st.session_state.vals = dict(DEFAULTS)


# ═══════════════════════════════════════════════════════════════════════════
st.title("🛡️ DoS Detection")
st.caption("Two-stage DoS attack detection · Random Forest · CIC-IDS2017")
st.divider()

if binary_model is None:
    st.error("Models not found. Place `binary_model.pkl` and `attack_model.pkl` next to `app.py`.")
    st.stop()

# ── Quick Fill ─────────────────────────────────────────────────────────────
st.subheader("Quick Fill")
st.caption("Pre-fill all sliders with a real sample from the dataset:")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("✅ BENIGN example", use_container_width=True):
        st.session_state.vals = dict(EXAMPLES["BENIGN"])
        st.rerun()
with col2:
    if st.button("🔴 GoldenEye example", use_container_width=True):
        st.session_state.vals = dict(EXAMPLES["DoS GoldenEye"])
        st.rerun()
with col3:
    if st.button("🔄 Reset to defaults", use_container_width=True):
        st.session_state.vals = dict(DEFAULTS)
        st.rerun()

v = st.session_state.vals

st.divider()
st.subheader("Flow Features")

# ── Group 1: Backward packet stats (top 3 features = 67% of decision) ─────
with st.expander("Backward Packet Stats  –  drives 67% of the model decision", expanded=True):
    st.caption("These three features are by far the most important. Set all three high to trigger ATTACK.")
    c1, c2 = st.columns(2)
    with c1:
        v["Bwd Packet Length Std"] = st.slider(
            "Bwd Packet Length Std Dev",
            0, 10000, int(v["Bwd Packet Length Std"]), step=10,
            help="BENIGN: 0 · GoldenEye: ~2184  ← key trigger",
        )
        v["Avg Bwd Segment Size"] = st.slider(
            "Avg Bwd Segment Size (Bytes)",
            0, 10000, int(v["Avg Bwd Segment Size"]), step=10,
            help="BENIGN: ~71 B · GoldenEye: ~1662 B  ← key trigger",
        )
    with c2:
        v["Bwd Packet Length Mean"] = st.slider(
            "Bwd Packet Length Mean (Bytes)",
            0, 10000, int(v["Bwd Packet Length Mean"]), step=10,
            help="BENIGN: ~71 B · GoldenEye: ~1662 B  ← key trigger",
        )
        v["Bwd Packet Length Max"] = st.slider(
            "Bwd Packet Length Max (Bytes)",
            0, 65535, int(v["Bwd Packet Length Max"]), step=10,
            help="BENIGN: ~71 B · GoldenEye: ~4392 B",
        )

# ── Group 2: Packet size spread ────────────────────────────────────────────
with st.expander("Packet Size Spread", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        v["Packet Length Variance"] = st.slider(
            "Packet Length Variance",
            0, 5_000_000, int(v["Packet Length Variance"]), step=1000,
            help="BENIGN: ~307 · GoldenEye: ~2,499,208",
        )
        v["Packet Length Std"] = st.slider(
            "Packet Length Std Dev",
            0, 10000, int(v["Packet Length Std"]), step=10,
            help="BENIGN: ~18 · GoldenEye: ~1581",
        )
        v["Max Packet Length"] = st.slider(
            "Max Packet Length (Bytes)",
            0, 65535, int(v["Max Packet Length"]), step=10,
            help="BENIGN: ~72 B · GoldenEye: ~4392 B",
        )
    with c2:
        v["Packet Length Mean"] = st.slider(
            "Packet Length Mean (Bytes)",
            0, 10000, int(v["Packet Length Mean"]), step=10,
            help="BENIGN: ~51 B · GoldenEye: ~747 B",
        )
        v["Average Packet Size"] = st.slider(
            "Average Packet Size (Bytes)",
            0, 10000, int(v["Average Packet Size"]), step=10,
            help="BENIGN: ~64 B · GoldenEye: ~795 B",
        )
        v["Fwd Packet Length Max"] = st.slider(
            "Fwd Packet Length Max (Bytes)",
            0, 65535, int(v["Fwd Packet Length Max"]), step=10,
            help="BENIGN: ~33 B · GoldenEye: ~382 B",
        )

# ── Group 3: Volume ────────────────────────────────────────────────────────
with st.expander("Volume", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        v["Subflow Bwd Bytes"] = st.slider(
            "Subflow Bwd Bytes",
            0, 100_000, int(v["Subflow Bwd Bytes"]), step=100,
            help="BENIGN: ~140 B · GoldenEye: ~11,632 B",
        )
    with c2:
        v["Total Length of Bwd Packets"] = st.slider(
            "Total Length of Bwd Packets (Bytes)",
            0, 100_000, int(v["Total Length of Bwd Packets"]), step=100,
            help="BENIGN: ~140 B · GoldenEye: ~11,632 B",
        )

# ── Group 4: Timing ────────────────────────────────────────────────────────
with st.expander("Timing", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        v["Flow Duration"] = st.slider(
            "Flow Duration (µs)",
            0, 120_000_000, int(v["Flow Duration"]), step=1000,
            help="BENIGN: ~230 µs · GoldenEye: ~11,485,031 µs",
        )
        v["Flow IAT Mean"] = st.slider(
            "Flow IAT Mean (µs)",
            0, 120_000_000, int(v["Flow IAT Mean"]), step=1000,
            help="BENIGN: ~85 µs · GoldenEye: ~1,035,509 µs",
        )
        v["Flow IAT Std"] = st.slider(
            "Flow IAT Std Dev (µs)",
            0, 120_000_000, int(v["Flow IAT Std"]), step=1000,
            help="BENIGN: ~93 µs · GoldenEye: ~2,180,952 µs",
        )
    with c2:
        v["Fwd IAT Total"] = st.slider(
            "Fwd IAT Total (µs)",
            0, 120_000_000, int(v["Fwd IAT Total"]), step=1000,
            help="BENIGN: ~3 µs · GoldenEye: ~6,542,860 µs",
        )
        v["Idle Mean"] = st.slider(
            "Idle Mean (µs)",
            0, 120_000_000, int(v["Idle Mean"]), step=1000,
            help="BENIGN: 0 · GoldenEye: ~6,487,532 µs",
        )
        v["Active Mean"] = st.slider(
            "Active Mean (µs)",
            0, 10_000, int(v["Active Mean"]), step=10,
            help="BENIGN: 0 · GoldenEye: ~766 µs",
        )

# ── Group 5: TCP & Rate ────────────────────────────────────────────────────
with st.expander("TCP & Flow Rate", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        v["Flow Bytes/s"] = st.slider(
            "Flow Bytes/s",
            0, 500_000, int(v["Flow Bytes/s"]), step=100,
            help="BENIGN: ~20,389 · GoldenEye: ~1,011",
        )
    with c2:
        valid_wins = [-1, 239, 1024, 8192, 14600, 29200, 65535]
        cur_win = int(v["Init_Win_bytes_forward"])
        if cur_win not in valid_wins:
            cur_win = -1
        v["Init_Win_bytes_forward"] = float(st.select_slider(
            "Initial TCP Window (Bytes)",
            options=valid_wins,
            value=cur_win,
            format_func=lambda x: f"{x} B" if x >= 0 else "-1 (auto / BENIGN default)",
            help="BENIGN: -1 · GoldenEye: 29200 · Hulk/Slowloris: 14600",
        ))

st.divider()

# ── Analyze ────────────────────────────────────────────────────────────────
if st.button("Analyze Flow", use_container_width=True):
    with st.spinner("Running model…"):
        time.sleep(0.3)
        verdict, conf1, attack_type, conf2 = predict(v)

    st.subheader("Result")
    r1, r2 = st.columns(2)

    with r1:
        st.markdown("**Stage 1 – Attack or normal traffic?**")
        if verdict == "BENIGN":
            st.success("BENIGN – No attack detected")
        else:
            st.error("ATTACK – Attack detected!")
        st.caption(f"Confidence: {conf1:.1%}")
        st.progress(conf1)

    with r2:
        st.markdown("**Stage 2 – Attack type**")
        if verdict == "BENIGN":
            st.info("No attack → Stage 2 not executed")
        else:
            st.error(f"{attack_type}")
            st.caption(f"Confidence: {conf2:.1%}")
            st.progress(conf2)

    with st.expander("Show all submitted values"):
        st.dataframe(
            pd.DataFrame(v.items(), columns=["Feature", "Value"]),
            use_container_width=True, hide_index=True,
        )