import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from scripts.stage4_inference import MedicalOrchestrator

DARK_BG = '#0b0f19'
st.set_page_config(page_title="19 - Upload MRI", layout="wide", page_icon="📁")
st.markdown(f"<style>.stApp {{ background-color: {DARK_BG}; color: #f9fafb; }}</style>", unsafe_allow_html=True)
st.title("📁 19 - Upload MRI Workstation")

uploads_dir = Path("project/stage4/uploads")
uploads_dir.mkdir(parents=True, exist_ok=True)

uploaded_file = st.file_uploader("Upload 3D MRI Volume (.nii.gz, .nii, .dcm)", type=['nii', 'nii.gz', 'npz', 'dcm'])
if uploaded_file:
    file_path = uploads_dir / uploaded_file.name
    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
    
    modality_override = st.selectbox("Select MRI Modality (Overrides Auto-Detection)", ["Auto-Detect", "Brain", "Spine"])
    
    st.success("File uploaded successfully. Initializing AI Pipeline...")
    
    if st.button("▶ Run Full AI Diagnosis Pipeline (Stages 2, 3, 4)", type="primary"):
        progress_bar = st.progress(0, text="Initializing Medical Pipeline...")
        
        def update_progress(val, text):
            progress_bar.progress(val, text=text)
            
        with st.spinner("Executing Full Medical Pipeline..."):
            pipeline = MedicalOrchestrator("project")
            
            # Pass modality override if user selected it
            selected_modality = None if modality_override == "Auto-Detect" else modality_override
            
            result, vol, vol2, vol3, mask = pipeline.analyze_mri(str(file_path), modality=selected_modality, progress_callback=update_progress)
            progress_bar.progress(1.0, text="Pipeline Execution Complete!")
            
            st.session_state['stage4_result'] = result
            st.session_state['vol_original'] = vol
            st.session_state['vol_preprocessed'] = vol2
            st.session_state['vol_enhanced'] = vol3
            st.session_state['mask'] = mask
            
        st.success("Pipeline executed successfully! Navigate through the sidebar pages to view the clinical results.")
