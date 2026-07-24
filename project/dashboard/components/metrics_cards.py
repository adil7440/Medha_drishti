import streamlit as st


def render_metric_card(title: str, value: str, before_val: str = None, delta: str = None, is_positive: bool = True):
    """
    Renders a clinical metric card with value, before/after comparison, and delta tag.
    """
    delta_html = ""
    if delta:
        css_cls = "metric-delta-positive" if is_positive else "metric-delta-negative"
        sign = "+" if is_positive else ""
        delta_html = f'<span class="{css_cls}">{sign}{delta}</span>'

    before_html = f'<div style="font-size:0.75rem; color:#6b7280; margin-top:4px;">Before: {before_val}</div>' if before_val else ""

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
            {before_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_summary_row(metrics_dict: dict):
    """
    Renders a row of key metric cards.
    """
    cols = st.columns(4)
    with cols[0]:
        psnr = metrics_dict.get("PSNR", 0.0)
        render_metric_card("PSNR", f"{psnr:.2f} dB", is_positive=psnr > 25.0)
    with cols[1]:
        ssim = metrics_dict.get("SSIM", 0.0)
        render_metric_card("SSIM", f"{ssim:.4f}", is_positive=ssim > 0.8)
    with cols[2]:
        entropy = metrics_dict.get("Entropy_After", 0.0)
        bef = metrics_dict.get("Entropy_Before", 0.0)
        render_metric_card("Entropy", f"{entropy:.3f}", before_val=f"{bef:.3f}", is_positive=True)
    with cols[3]:
        contrast = metrics_dict.get("Contrast_After", 0.0)
        bef_c = metrics_dict.get("Contrast_Before", 0.0)
        render_metric_card("Contrast", f"{contrast:.3f}", before_val=f"{bef_c:.3f}", is_positive=True)
