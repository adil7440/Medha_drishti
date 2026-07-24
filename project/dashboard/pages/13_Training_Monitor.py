import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

st.markdown("""
<div class="main-header">
    <h1 style="color:#f59e0b; margin:0 0 4px 0;">Training Monitor</h1>
    <p>Real-time training metrics, loss curves, and learning rate schedules for all models.</p>
</div>
""", unsafe_allow_html=True)

logs_dir = PROJECT_DIR / "stage3" / "logs"
history_files = list(logs_dir.glob("*_history.json"))

if not history_files:
    st.warning("No training logs found. Run `run_stage3.py` first.")
    st.stop()

model_names = [f.stem.replace("_history", "").title() for f in history_files]
model_choice = st.radio("Select Model", model_names, horizontal=True)

target_file = logs_dir / f"{model_choice.lower()}_history.json"
if not target_file.exists():
    # Try matching by case-insensitive
    for f in history_files:
        if f.stem.startswith(model_choice.lower()):
            target_file = f
            break

with open(target_file, "r") as f:
    data = json.load(f)

history = data.get("history", [])
if not history:
    st.warning("Empty training history.")
    st.stop()

df = pd.DataFrame(history)
total_time = data.get("total_train_time_sec", 0)
param_count = data.get("param_count", 0)
model_size = data.get("model_size_mb", 0)
best_psnr = data.get("best_val_psnr", 0)
epochs_trained = data.get("epochs_trained", len(history))

# Summary metrics
st.markdown(f"### Training Summary: {model_choice}")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Epochs", epochs_trained)
with c2:
    st.metric("Total Time", f"{total_time:.0f}s")
with c3:
    st.metric("Best PSNR", f"{best_psnr:.2f} dB")
with c4:
    st.metric("Parameters", f"{param_count:,}")
with c5:
    st.metric("Model Size", f"{model_size:.2f} MB")

st.markdown("---")

# Loss Curves
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["epoch"], y=df["train_loss"], name="Train Loss",
                             line=dict(color="#0ea5e9", width=2)))
    fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_loss"], name="Val Loss",
                             line=dict(color="#ef4444", width=2)))
    fig.update_layout(template="plotly_dark", title="Loss Curves", xaxis_title="Epoch", yaxis_title="Loss")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.line(df, x="epoch", y="val_psnr", title="Validation PSNR (dB)",
                  template="plotly_dark")
    fig.update_traces(line_color="#10b981", line_width=2.5)
    fig.update_layout(xaxis_title="Epoch", yaxis_title="PSNR (dB)")
    st.plotly_chart(fig, use_container_width=True)

# LR & GPU
c1, c2 = st.columns(2)
with c1:
    fig = px.line(df, x="epoch", y="learning_rate", title="Learning Rate Schedule",
                  template="plotly_dark")
    fig.update_traces(line_color="#f59e0b", line_width=2)
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Learning Rate")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.line(df, x="epoch", y="gpu_mem_mb", title="GPU Memory (MB)",
                  template="plotly_dark")
    fig.update_traces(line_color="#8b5cf6", line_width=2)
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Memory (MB)")
    st.plotly_chart(fig, use_container_width=True)

# Epoch Time
fig = px.bar(df, x="epoch", y="epoch_time_sec", title="Per-Epoch Training Time (s)",
             template="plotly_dark")
fig.update_traces(marker_color="#06b6d4")
fig.update_layout(xaxis_title="Epoch", yaxis_title="Time (s)")
st.plotly_chart(fig, use_container_width=True)

# Full history table
st.markdown("### Full Epoch History")
st.dataframe(df, use_container_width=True)
