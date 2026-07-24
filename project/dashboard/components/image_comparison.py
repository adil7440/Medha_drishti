import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def display_side_by_side(
    img1: np.ndarray,
    img2: np.ndarray,
    label1: str = "Original MRI",
    label2: str = "Preprocessed MRI",
    cmap: str = "gray"
):
    """
    Renders two 2D MRI slices side-by-side along with an absolute intensity difference heatmap.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#0b0f19')
    
    # 1. Original
    axes[0].imshow(img1, cmap=cmap)
    axes[0].set_title(label1, color='#f9fafb', fontsize=11, fontweight='600')
    axes[0].axis('off')
    
    # 2. Processed
    axes[1].imshow(img2, cmap=cmap)
    axes[1].set_title(label2, color='#f9fafb', fontsize=11, fontweight='600')
    axes[1].axis('off')

    # 3. Difference Heatmap
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    axes[2].imshow(diff, cmap='inferno')
    axes[2].set_title("Absolute Difference", color='#f9fafb', fontsize=11, fontweight='600')
    axes[2].axis('off')

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def display_stage_grid(stages_dict: dict, cmap: str = "gray"):
    """
    Renders a multi-stage grid pipeline view.
    """
    keys = list(stages_dict.keys())
    n = len(keys)
    if n == 0:
        return

    cols = st.columns(n)
    for idx, (title, img) in enumerate(stages_dict.items()):
        with cols[idx]:
            fig, ax = plt.subplots(figsize=(4, 4), facecolor='#0b0f19')
            ax.imshow(img, cmap=cmap)
            ax.set_title(title, color='#f9fafb', fontsize=9, fontweight='600')
            ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
