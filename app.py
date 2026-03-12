import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

st.set_page_config(
    page_title="DoS Shield",
    layout="wide",
)

FEATURES = {
    "Avg Bwd Segment Size": ("Average Backward Segment Size (Bytes)", 0.0, 0.0, 65535.0, 1.0),
    "Max Packet Length": ("Max Packet Length (Bytes)", 0.0, 0.0, 65535.0, 1.0),
    "Bwd Packet Length Max": ("Backward Max Packet Length (Bytes)", 0.0, 0.0, 65535.0, 1.0),
    "Packet Length Std": ("Packet Length Standard Deviation", 0.0, 0.0, 65535.0, 0.1),
    "Packet Length Mean": ("Average Packet Length (Bytes)", 0.0, 0.0, 65535.0, 0.1),
    "Bwd Packet Length Mean": ("Backward Average Packet Length (Bytes)", 0.0, 0.0, 65535.0, 0.1),
    "Average Packet Size": ("Average Packet Size (Bytes)", 0.0, 0.0, 65535.0, 0.1),
    "Packet Length Variance": ("Packet Length Variance", 0.0, 0.0, 1e8, 1.0),
    "Subflow Bwd Bytes": ("Subflow Backward Bytes", 0.0, 0.0, 1e7, 1.0),
    "Total Length of Bwd Packets": ("Total Length of Backward Packets (Bytes)", 0.0, 0.0, 1e7, 1.0),
    "Fwd Packet Length Max": ("Forward Max Packet Length (Bytes)", 0.0, 0.0, 65535.0, 1.0),
    "Bwd Packet Length Std": ("Backward Packet Length Standard Deviation", 0.0, 0.0, 65535.0, 0.1),
    "Init_Win_bytes_forward": ("Initial Forward TCP Window (Bytes)", 0.0, -1.0, 65535.0, 1.0),
    "Flow Duration": ("Flow Duration (Microseconds)", 0.0, 0.0, 1.2e8, 100.0),
    "Flow Bytes/s": ("Flow Bytes per Second", 0.0, 0.0, 1e8, 1.0),
    "Flow IAT Mean": ("Average Flow IAT (Microseconds)", 0.0, 0.0, 5e6, 1.0),
    "Fwd IAT Total": ("Total Forward IAT (Microseconds)", 0.0, 0.0, 1.2e8, 1.0),
    "Flow IAT Std": ("Flow IAT Standard Deviation (Microseconds)", 0.0, 0.0, 5e6, 1.0),
    "Idle Mean": ("Average Idle Time (Microseconds)", 0.0, 0.0, 1.2e8, 1.0),
    "Active Mean": ("Average Active Time (Microseconds)", 0.0, 0.0, 1.2e8, 1.0),
}

FEATURE_GROUPS = {
    "Packet Sizes": [
        "Max Packet Length",
        "Fwd Packet Length Max",
        "Packet Length Mean",
        "Packet Length Std",
        "Packet Length Variance",
        "Average Packet Size",
    ],
    "Backward Direction": [
        "Avg Bwd Segment Size",
        "Bwd Packet Length Max",
        "Bwd Packet Length Mean",
        "Bwd Packet Length Std",
        "Subflow Bwd Bytes",
        "Total Length of Bwd Packets",
    ],
    "Timing and Flow": [
        "Flow Duration",
        "Flow Bytes/s",
        "Flow IAT Mean",
        "Flow IAT Std",
        "Fwd IAT Total",
        "Idle Mean",
        "Active Mean",
    ],
    "TCP": [
        "Init_Win_bytes_forward",
    ],
}


@st.cache_resource
def load_models():
    try:
        binary_model = joblib.load("binary_model.pkl")
        attack_model = joblib.load("attack_model.pkl")
        return binary_model, attack_model
    except FileNotFoundError:
        return None, None


binary_model, attack_model = load_models()


def predict(values: dict):
    df = pd.DataFrame([values])

    stage1 = binary_model.predict(df)[0]
    stage1_prob = binary_model.predict_proba(df)[0]
    conf1 = float(max(stage1_prob))

    if stage1 == "BENIGN":
        return "BENIGN", conf1, None, None

    stage2 = attack_model.predict(df)[0]
    stage2_prob = attack_model.predict_proba(df)[0]
    conf2 = float(max(stage2_prob))

    return "ATTACK", conf1, stage2, conf2


st.title("DoS Shield - Network Analysis")
st.caption("Two-stage DoS detection using Random Forest")
st.divider()

if binary_model is None:
    st.error(
        "Model not found. Place binary_model.pkl and attack_model.pkl in the same folder as app.py."
    )
    st.stop()

tab_manual, tab_csv = st.tabs(["Manual Input", "CSV Upload"])

with tab_manual:
    st.subheader("Enter Flow Features")

    values = {}
    for group_name, feat_list in FEATURE_GROUPS.items():
        with st.expander(group_name, expanded=True):
            cols = st.columns(2)
            for idx, feat in enumerate(feat_list):
                desc, default, fmin, fmax, step = FEATURES[feat]
                with cols[idx % 2]:
                    values[feat] = st.number_input(
                        label=desc,
                        min_value=float(fmin),
                        max_value=float(fmax),
                        value=float(default),
                        step=float(step),
                        key=f"manual_{feat}",
                    )

    st.divider()

    if st.button("Analyze Flow", use_container_width=True):
        with st.spinner("Analyzing flow..."):
            time.sleep(0.3)
            verdict, conf1, attack_type, conf2 = predict(values)

        st.subheader("Result")

        col_stage1, col_stage2 = st.columns(2)

        with col_stage1:
            st.markdown("**Stage 1 - BENIGN / ATTACK**")
            if verdict == "BENIGN":
                st.success(f"BENIGN | Confidence: {conf1:.1%}")
            else:
                st.error(f"ATTACK detected | Confidence: {conf1:.1%}")
            st.progress(conf1)

        with col_stage2:
            st.markdown("**Stage 2 - Attack Type**")
            if verdict == "BENIGN":
                st.info("No attack. Stage 2 not used.")
            else:
                st.error(f"{attack_type} | Confidence: {conf2:.1%}")
                st.progress(conf2)

        with st.expander("Show Input Values"):
            st.dataframe(
                pd.DataFrame([values]).T.rename(columns={0: "Value"}),
                use_container_width=True,
            )

with tab_csv:
    st.subheader("Upload CSV File")
    st.info(
        "CSV must contain the 20 feature columns. Optional column 'Label' can be used for accuracy comparison."
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        df_upload = pd.read_csv(uploaded)
        df_upload.columns = df_upload.columns.str.strip()

        feature_cols = list(FEATURES.keys())
        missing = [c for c in feature_cols if c not in df_upload.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            st.success(f"Loaded {len(df_upload):,} rows.")

            X_upload = df_upload[feature_cols].copy()
            X_upload = X_upload.replace([np.inf, -np.inf], np.nan).fillna(0)

            with st.spinner("Classifying flows..."):
                stage1_preds = binary_model.predict(X_upload)
                final_preds = list(stage1_preds)

                attack_mask = stage1_preds == "ATTACK"
                if attack_mask.any():
                    stage2_preds = attack_model.predict(X_upload[attack_mask])
                    for i, pred in zip(np.where(attack_mask)[0], stage2_preds):
                        final_preds[i] = pred

            df_upload["Prediction"] = final_preds

            st.divider()
            st.subheader("Result Summary")

            counts = pd.Series(final_preds).value_counts()
            total = len(final_preds)
            benign_n = counts.get("BENIGN", 0)
            attack_n = total - benign_n

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Total Flows", f"{total:,}")
            col_m2.metric("BENIGN", f"{benign_n:,}", f"{benign_n/total:.1%}")
            col_m3.metric("ATTACKS", f"{attack_n:,}", f"{attack_n/total:.1%}")

            st.bar_chart(counts)

            if "Label" in df_upload.columns:
                st.divider()
                st.subheader("Comparison with True Labels")

                y_true = df_upload["Label"].astype(str).str.strip()
                y_pred = pd.Series(final_preds)
                acc = (y_true.values == y_pred.values).mean()

                st.metric("Accuracy", f"{acc:.2%}")

                from sklearn.metrics import classification_report

                report_df = pd.DataFrame(
                    classification_report(y_true, y_pred, output_dict=True)
                ).T.round(3)

                st.dataframe(report_df, use_container_width=True)

            st.divider()
            csv_out = df_upload.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Results as CSV",
                data=csv_out,
                file_name="dos_shield_results.csv",
                mime="text/csv",
                use_container_width=True,
            )