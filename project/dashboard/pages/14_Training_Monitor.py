import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.header("Stage 3 Training Monitor and Learning Curves")

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
logs_dir = PROJECT_DIR / "stage3" / "logs"

dncnn_log = logs_dir / "dncnn_history.json"
swinir_log = logs_dir / "swinir_history.json"

if not dncnn_log.exists() and not swinir_log.exists():
    st.warning("No training log files found. Please run `run_stage3.py` to initiate model training.")
    st.stop()

model_choice = st.radio("Select Model Training Logs", options=["SwinIR", "DnCNN"])
target_file = logs_dir / f"{model_choice.lower()}_history.json"

if not target_file.exists():
    st.warning(f"Log file for {model_choice} not found.")
    st.stop()

with open(target_file, "r") as f:
    data = json.load(f)

history = data.get("history", [])
if not history:
    st.warning("Training history is empty.")
    st.stop()

df_hist = pd.DataFrame(history)

st.markdown(f"### Training Progression: {model_choice} (Total Time: {data.get('total_train_time_sec', 0)}s)")

c1, c2 = st.columns(2)

with c1:
    st.subheader("Loss Curves (Train vs Val)")
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=df_hist["epoch"], y=df_hist["train_loss"], name="Train Loss", line=dict(color="#0ea5e9", width=2)))
    fig_loss.add_trace(go.Scatter(x=df_hist["epoch"], y=df_hist["val_loss"], name="Val Loss", line=dict(color="#ef4444", width=2)))
    fig_loss.update_layout(template="plotly_dark", title="Epoch Loss Curve", xaxis_title="Epoch", yaxis_title="Loss")
    st.plotly_chart(fig_loss, use_container_width=True)

with c2:
    st.subheader("Validation PSNR Curve (dB)")
    fig_psnr = px.line(df_hist, x="epoch", y="val_psnr", title="Validation PSNR per Epoch", template="plotly_dark")
    fig_psnr.update_traces(line_color="#10b981", line_width=2)
    st.plotly_chart(fig_psnr, use_container_width=True)

c3, c4 = st.columns(2)

with c3:
    st.subheader("Learning Rate Schedule")
    fig_lr = px.line(df_hist, x="epoch", y="learning_rate", title="Cosine Annealing Learning Rate", template="plotly_dark")
    fig_lr.update_traces(line_color="#f59e0b", line_width=2)
    st.plotly_chart(fig_lr, use_container_width=True)

with c4:
    st.subheader("GPU Memory Allocation (MB)")
    fig_gpu = px.line(df_hist, x="epoch", y="gpu_mem_mb", title="Peak GPU Memory per Epoch", template="plotly_dark")
    fig_gpu.update_traces(line_color="#8b5cf6", line_width=2)
    st.plotly_chart(fig_gpu, use_container_width=True)
