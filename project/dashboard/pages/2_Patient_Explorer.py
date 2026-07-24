import streamlit as st
import pandas as pd
from pathlib import Path

st.header("Patient Metadata Explorer")

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
metrics_path = PROJECT_DIR / "stage2" / "metrics" / "stage2_quality_metrics.csv"

if not metrics_path.exists():
    st.warning("Please run the Stage 2 pipeline first to populate patient data.")
    st.stop()

df = pd.read_csv(metrics_path)

c1, c2, c3 = st.columns(3)
with c1:
    dataset_choice = st.selectbox("Select Dataset", options=["Brain", "Spine"])

sub_df = df[df["Dataset"] == dataset_choice]
patients = sorted(sub_df["Patient_ID"].unique())

with c2:
    patient_choice = st.selectbox("Select Patient", options=patients)

p_sub = sub_df[sub_df["Patient_ID"] == patient_choice]
modalities = sorted(p_sub["Modality"].unique())

with c3:
    modality_choice = st.selectbox("Select MRI Modality", options=modalities)

selected_row = p_sub[p_sub["Modality"] == modality_choice].iloc[0]

st.markdown("---")
st.subheader(f"Metadata for {patient_choice} ({modality_choice})")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Dataset", selected_row["Dataset"])
m2.metric("Patient ID", selected_row["Patient_ID"])
m3.metric("Modality", selected_row["Modality"])
m4.metric("Processing Time", f"{selected_row['Processing_Time_Sec']} s")

st.markdown("### Quality Metrics Summary")
st.json(selected_row.to_dict())
