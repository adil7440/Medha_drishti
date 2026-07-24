import streamlit as st
import pandas as pd
from pathlib import Path

st.header("Technical Reports & Export Center")

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
stage2_dir = PROJECT_DIR / "stage2"
report_path = stage2_dir / "reports" / "stage2_preprocessing_report.md"
metrics_path = stage2_dir / "metrics" / "stage2_quality_metrics.csv"

st.markdown(
    """
    <div class="main-header">
        <h3 style="margin-top:0; color:#f9fafb;">Export Technical Reports and Metrics</h3>
        <p>Download executive technical reports and full CSV quality metric datasets for documentation.</p>
    </div>
    """,
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

with c1:
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            rep_text = f.read()
        st.download_button(
            label="Download Stage 2 Report (.md)",
            data=rep_text,
            file_name="stage2_preprocessing_report.md",
            mime="text/markdown"
        )
    else:
        st.button("Download Stage 2 Report (Not Ready)", disabled=True)

with c2:
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            csv_text = f.read()
        st.download_button(
            label="Download CSV Metrics (.csv)",
            data=csv_text,
            file_name="stage2_quality_metrics.csv",
            mime="text/csv"
        )
    else:
        st.button("Download CSV Metrics (Not Ready)", disabled=True)

with c3:
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        summary_csv = df.describe().to_csv()
        st.download_button(
            label="Download Summary Stats (.csv)",
            data=summary_csv,
            file_name="stage2_processed_summary.csv",
            mime="text/csv"
        )
    else:
        st.button("Download Summary Stats (Not Ready)", disabled=True)

st.markdown("---")
if report_path.exists():
    st.subheader("Report Preview")
    st.markdown(rep_text)
