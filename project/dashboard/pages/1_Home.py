import streamlit as st
import pandas as pd
from pathlib import Path

st.header("Project and Dataset Overview")

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
metrics_path = PROJECT_DIR / "stage2" / "metrics" / "stage2_quality_metrics.csv"

st.markdown(
    """
    <div class="main-header">
        <h3 style="margin-top:0; color:#f9fafb;">Stage 2: MRI Preprocessing Pipeline</h3>
        <p>
            Standardized preprocessing workflow for Brain and Spine MRI volumes. 
            Implements validation, isotropic resampling, intensity normalization, non-local and bilateral denoising, 
            N4 bias field correction, and CLAHE contrast optimization.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

c1, c2 = st.columns(2)

with c1:
    st.markdown("### Brain Dataset (BraTS 2020)")
    st.markdown("- **Total Patients**: 369 Patients")
    st.markdown("- **Modalities**: T1, T1CE, T2, FLAIR")
    st.markdown("- **Format**: NIfTI (.nii)")
    st.markdown("- **Skull Stripping**: Otsu + Morphological Connected Component Extraction")

with c2:
    st.markdown("### Spine Dataset")
    st.markdown("- **Sub-datasets**: Normal & Pathological Spine MRI")
    st.markdown("- **Modalities**: T1W, T2W, STIR, SURVEY")
    st.markdown("- **Format**: NIfTI (.nii.gz)")
    st.markdown("- **Skull Stripping**: Bypassed for Spine")

st.markdown("---")
st.subheader("Preprocessing Status")

if metrics_path.exists():
    df = pd.read_csv(metrics_path)
    st.success(f"Preprocessing pipeline execution complete. Successfully processed {len(df)} MRI volumes.")
    st.dataframe(df[["Dataset", "Patient_ID", "Modality", "PSNR", "SSIM", "Processing_Time_Sec"]].head(10), use_container_width=True)
else:
    st.warning("Stage 2 pipeline metrics file not found. Run stage2_pipeline.py to execute preprocessing.")
