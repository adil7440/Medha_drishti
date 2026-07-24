import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

DARK_BG = '#0b0f19'
st.set_page_config(page_title="23 - Segmentation Viewer", layout="wide", page_icon="🧠")
st.markdown(f"""
    <style>
    .stApp {{ background-color: {DARK_BG}; color: #f9fafb; }}
    .metric-card {{ background: #111827; border-radius: 12px; padding: 20px; border: 1px solid #1f2937; text-align: center; }}
    .metric-value {{ font-size: 24px; font-weight: 700; color: #0ea5e9; }}
    </style>
""", unsafe_allow_html=True)
st.title("🎯 23 - Segmentation Viewer")

if 'stage4_result' not in st.session_state:
    st.warning("Please upload and analyze an MRI volume in '19_Upload_MRI.py' first.")
    st.stop()

result = st.session_state['stage4_result']

vol = st.session_state['vol_enhanced']
mask = st.session_state['mask']
z = st.slider("Z-Slice", 0, vol.shape[0]-1, result['measurements']['Most_Affected_Slice'])
fig = go.Figure(data=go.Heatmap(z=vol[z], colorscale='gray', showscale=False))
import cv2

if np.any(mask[z]):
    # Extract contours for "proper lines or outlines" aesthetic
    mask_slice = (mask[z] > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if len(contour) > 2:
            x = contour[:, 0, 0]
            y = contour[:, 0, 1]
            # Close the loop
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(color='#ef4444', width=2.5), # Vibrant Red Outline
                hoverinfo='skip',
                showlegend=False
            ))
            
            # Calculate centroid to point the arrow at the affected zone
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                # Add clinical arrow pointing to the lesion/herniation
                fig.add_annotation(
                    x=cx, y=cy,
                    ax=cx + 50, ay=cy - 50, # Offset the tail of the arrow
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=2,
                    arrowwidth=3,
                    arrowcolor="#ef4444", # Red arrow matching reference image
                    text="" # No text, just the arrow
                )
fig.update_layout(width=600, height=600, margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG)
fig.update_yaxes(autorange="reversed")
st.plotly_chart(fig, use_container_width=True)
