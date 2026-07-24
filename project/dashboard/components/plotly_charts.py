import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


DARK_TEMPLATE = "plotly_dark"
COLOR_PRIMARY = "#0ea5e9"
COLOR_SECONDARY = "#10b981"
COLOR_ACCENT = "#f59e0b"


def create_histogram_comparison(img_before: np.ndarray, img_after: np.ndarray) -> go.Figure:
    """
    Renders an interactive voxel intensity distribution histogram (Before vs After).
    """
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=img_before.ravel(),
        name="Before Preprocessing",
        opacity=0.6,
        marker_color="#ef4444",
        nbinsx=100
    ))

    fig.add_trace(go.Histogram(
        x=img_after.ravel(),
        name="After Preprocessing",
        opacity=0.6,
        marker_color="#10b981",
        nbinsx=100
    ))

    fig.update_layout(
        title="Voxel Intensity Distribution (Before vs After)",
        xaxis_title="Voxel Intensity",
        yaxis_title="Voxel Count",
        barmode="overlay",
        template=DARK_TEMPLATE,
        paper_bgcolor="rgba(11, 15, 25, 0)",
        plot_bgcolor="rgba(17, 24, 39, 0.6)",
        font=dict(color="#e5e7eb")
    )
    return fig


def create_radar_chart(metrics_dict: dict) -> go.Figure:
    """
    Renders a multi-metric radar chart for normalized quality scores.
    """
    categories = ['PSNR (norm)', 'SSIM', 'UQI', 'FSIM', 'Entropy (norm)', 'Contrast (norm)']
    
    psnr_norm = min(1.0, metrics_dict.get('PSNR', 0.0) / 40.0)
    ssim_val = metrics_dict.get('SSIM', 0.0)
    uqi_val = metrics_dict.get('UQI', 0.0)
    fsim_val = metrics_dict.get('FSIM', 0.0)
    ent_norm = min(1.0, metrics_dict.get('Entropy_After', 0.0) / 8.0)
    cont_norm = min(1.0, metrics_dict.get('Contrast_After', 0.0) / 100.0)

    values = [psnr_norm, ssim_val, uqi_val, fsim_val, ent_norm, cont_norm]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(14, 165, 233, 0.25)',
        line=dict(color=COLOR_PRIMARY, width=2),
        name="Processed Volume Quality"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.0], color="#9ca3af"),
            bgcolor="rgba(17, 24, 39, 0.6)"
        ),
        title="Multi-Metric Quality Radar Profile",
        template=DARK_TEMPLATE,
        paper_bgcolor="rgba(11, 15, 25, 0)",
        font=dict(color="#e5e7eb")
    )
    return fig


def create_metrics_boxplot(df: pd.DataFrame, metric_col: str = "PSNR") -> go.Figure:
    """
    Renders boxplots comparing metric distributions across Brain vs Spine datasets.
    """
    fig = px.box(
        df,
        x="Dataset",
        y=metric_col,
        color="Dataset",
        points="all",
        color_discrete_map={"Brain": "#0ea5e9", "Spine": "#10b981"},
        title=f"{metric_col} Distribution Across Datasets"
    )
    fig.update_layout(
        template=DARK_TEMPLATE,
        paper_bgcolor="rgba(11, 15, 25, 0)",
        plot_bgcolor="rgba(17, 24, 39, 0.6)",
        font=dict(color="#e5e7eb")
    )
    return fig


def create_scatter_plot(df: pd.DataFrame, x_col: str = "PSNR", y_col: str = "SSIM") -> go.Figure:
    """
    Renders a scatter plot comparing two quality metrics.
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color="Dataset",
        hover_data=["Patient_ID", "Modality"],
        color_discrete_map={"Brain": "#0ea5e9", "Spine": "#10b981"},
        title=f"{y_col} vs {x_col} Scatter Comparison"
    )
    fig.update_layout(
        template=DARK_TEMPLATE,
        paper_bgcolor="rgba(11, 15, 25, 0)",
        plot_bgcolor="rgba(17, 24, 39, 0.6)",
        font=dict(color="#e5e7eb")
    )
    return fig
