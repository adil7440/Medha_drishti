import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

# ==============================================================================
# Diagram Generation Functions
# ==============================================================================
def draw_box(ax, x, y, width, height, text, bg_color, text_color='white'):
    rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='black', facecolor=bg_color, alpha=0.9, zorder=2)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, color=text_color, fontsize=12, fontweight='bold', ha='center', va='center', zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, text=""):
    ax.annotate(text, xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                fontsize=10, ha='center', va='bottom', color='#333333', zorder=1)

def generate_pipeline_diagram(output_path):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis('off')

    draw_box(ax, 5, 3, 15, 4, "Raw NIfTI\n(Brain/Spine)", "#64748b")
    draw_arrow(ax, 20, 5, 27, 5)
    
    draw_box(ax, 27, 3, 16, 4, "Stage 2:\nPreprocessing\n(N4ITK, Z-Score)", "#0284c7")
    draw_arrow(ax, 43, 5, 50, 5)
    
    draw_box(ax, 50, 3, 16, 4, "Stage 3:\nEnhancement\n(SE-DnCNN)", "#16a34a")
    draw_arrow(ax, 66, 5, 73, 5)
    
    draw_box(ax, 73, 3, 22, 4, "Stage 4:\nSegmentation\n(3D Attention U-Net)", "#9333ea")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_dncnn_diagram(output_path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 10)
    ax.axis('off')

    draw_box(ax, 2, 3, 12, 4, "Input\nSlice", "#64748b")
    draw_arrow(ax, 14, 5, 20, 5)
    
    draw_box(ax, 20, 2, 60, 6, "", "#f1f5f9", text_color="black") # Main container
    ax.text(50, 8.5, "Deep Convolutional Neural Network (DnCNN)", fontsize=14, fontweight='bold', ha='center')
    
    draw_box(ax, 22, 3.5, 12, 3, "Conv2D +\nReLU", "#3b82f6")
    draw_arrow(ax, 34, 5, 40, 5)
    
    draw_box(ax, 40, 3.5, 16, 3, "15x (Conv2D +\nBN + ReLU)", "#eab308", "black")
    draw_arrow(ax, 56, 5, 62, 5)
    
    draw_box(ax, 62, 3.5, 15, 3, "Squeeze &\nExcitation", "#ec4899")
    
    draw_arrow(ax, 80, 5, 88, 5)
    
    draw_box(ax, 88, 3, 12, 4, "Conv2D", "#3b82f6")
    draw_arrow(ax, 100, 5, 106, 5)
    
    draw_box(ax, 106, 3, 12, 4, "Residual\nOutput", "#22c55e")
    
    # Global skip connection
    ax.annotate("", xy=(108, 7.5), xytext=(8, 7.5), arrowprops=dict(facecolor='black', shrink=0.01, width=1, headwidth=6))
    ax.plot([8, 8], [7, 7.5], color='black', lw=2)
    ax.plot([108, 108], [7, 7.5], color='black', lw=2)
    ax.text(50, 7.7, "+ (Global Residual Learning)", fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_unet_diagram(output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    ax.text(50, 95, "3D Attention U-Net Architecture", fontsize=16, fontweight='bold', ha='center')

    # Encoder
    draw_box(ax, 10, 70, 15, 10, "Conv3D\n(64)", "#3b82f6")
    draw_arrow(ax, 17.5, 70, 17.5, 60, "MaxPool3D")
    
    draw_box(ax, 10, 50, 15, 10, "Conv3D\n(128)", "#2563eb")
    draw_arrow(ax, 17.5, 50, 17.5, 40, "MaxPool3D")
    
    draw_box(ax, 10, 30, 15, 10, "Conv3D\n(256)", "#1d4ed8")
    
    # Bottleneck
    draw_arrow(ax, 17.5, 30, 35, 15)
    draw_box(ax, 35, 10, 30, 10, "Bottleneck Conv3D (512)", "#1e40af")
    draw_arrow(ax, 65, 15, 82.5, 30)
    
    # Decoder
    draw_box(ax, 75, 30, 15, 10, "Conv3D\n(256)", "#1d4ed8")
    draw_arrow(ax, 82.5, 40, 82.5, 50, "UpConv3D")
    
    draw_box(ax, 75, 50, 15, 10, "Conv3D\n(128)", "#2563eb")
    draw_arrow(ax, 82.5, 60, 82.5, 70, "UpConv3D")
    
    draw_box(ax, 75, 70, 15, 10, "Conv3D\n(64)", "#3b82f6")
    
    # Skip connections with Attention
    draw_arrow(ax, 25, 75, 50, 75)
    draw_box(ax, 50, 71, 15, 8, "Attention\nGate", "#f59e0b")
    draw_arrow(ax, 65, 75, 75, 75)
    
    draw_arrow(ax, 25, 55, 50, 55)
    draw_box(ax, 50, 51, 15, 8, "Attention\nGate", "#f59e0b")
    draw_arrow(ax, 65, 55, 75, 55)
    
    # Output
    draw_arrow(ax, 90, 75, 96, 75)
    draw_box(ax, 96, 71, 4, 8, "1x1", "#10b981")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==============================================================================
# PDF Generation
# ==============================================================================
def build_pdf(output_path, fig_paths):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=50, leftMargin=50,
        topMargin=50, bottomMargin=50
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=20, spaceAfter=20, alignment=TA_CENTER, textColor=colors.HexColor("#0f172a"))
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=16, spaceBefore=20, spaceAfter=10, textColor=colors.HexColor("#1d4ed8"))
    body_style = ParagraphStyle('Body', parent=styles['BodyText'], fontSize=11, spaceAfter=10, alignment=TA_JUSTIFY, leading=14)

    Story = []

    # PAGE 1
    Story.append(Paragraph("MedhaDrishti AI: Architecture & Pipeline Report", title_style))
    
    Story.append(Paragraph("1. End-to-End System Pipeline", heading_style))
    Story.append(Paragraph("The flowchart below illustrates the data lifecycle from raw medical tensor input to clinical dashboard output.", body_style))
    Story.append(Spacer(1, 10))
    Story.append(RLImage(fig_paths['pipeline'], width=6*inch, height=1.8*inch))
    Story.append(Spacer(1, 10))
    Story.append(Paragraph("• <b>Stage 1 & 2</b>: Raw volumes undergo metadata extraction, N4ITK bias correction, and Z-score standardization.", body_style))
    Story.append(Paragraph("• <b>Stage 3</b>: AI Enhancement restores structural fidelity.", body_style))
    Story.append(Paragraph("• <b>Stage 4</b>: Sub-regional pixel-wise segmentation is executed.", body_style))

    Story.append(Paragraph("2. Stage 3 Architecture: SE-DnCNN", heading_style))
    Story.append(Paragraph("The chosen enhancement architecture is a Deep Convolutional Neural Network integrated with Squeeze-and-Excitation (SE) mechanisms.", body_style))
    Story.append(Spacer(1, 10))
    Story.append(RLImage(fig_paths['dncnn'], width=7*inch, height=2.3*inch))
    Story.append(Spacer(1, 10))
    Story.append(Paragraph("<b>Global Residual Learning</b>: Instead of predicting the clean image directly, the network predicts the latent noise map, which is subtracted from the input. This severely reduces convergence time.", body_style))
    Story.append(Paragraph("<b>Squeeze-and-Excitation</b>: An attention mechanism that assigns adaptive weights to feature channels, effectively suppressing chaotic MRI artifacts while enhancing sharp anatomical boundaries.", body_style))

    Story.append(PageBreak())
    
    # PAGE 2
    Story.append(Paragraph("3. Stage 4 Architecture: 3D Attention U-Net", heading_style))
    Story.append(Paragraph("The segmentation pipeline relies on a fully volumetric 3D Attention U-Net.", body_style))
    Story.append(Spacer(1, 10))
    Story.append(RLImage(fig_paths['unet'], width=6.5*inch, height=4*inch))
    Story.append(Spacer(1, 10))
    Story.append(Paragraph("<b>Encoder-Decoder</b>: 3D Convolutions map spatial hierarchies, contracting to a 512-channel bottleneck before symmetrically expanding.", body_style))
    Story.append(Paragraph("<b>Attention Gates</b>: Placed at the skip connections, these gates actively filter out irrelevant background activations, directing the network's focus strictly onto the Region of Interest (Tumor Core, Edema, Disc Herniation).", body_style))

    Story.append(Paragraph("4. Training Pipeline & Loss Functions", heading_style))
    Story.append(Paragraph("• <b>Optimizer</b>: AdamW optimizer utilized to prevent severe weight decay.", body_style))
    Story.append(Paragraph("• <b>Loss Function (Stage 3)</b>: Charbonnier Loss combined with structural SSIM loss to preserve edge gradients.", body_style))
    Story.append(Paragraph("• <b>Loss Function (Stage 4)</b>: Hybrid Dice-Focal Loss applied to counter extreme class imbalances (e.g., small metastatic tumors vs large healthy tissue volumes).", body_style))

    doc.build(Story)
    print(f"[Success] Architecture Report generated at: {output_path}")

if __name__ == "__main__":
    out_dir = Path("project/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig_dir = Path("project/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {
        'pipeline': str(fig_dir / "arch_pipeline.png"),
        'dncnn': str(fig_dir / "arch_dncnn.png"),
        'unet': str(fig_dir / "arch_unet.png")
    }
    
    print("Generating diagrams...")
    generate_pipeline_diagram(paths['pipeline'])
    generate_dncnn_diagram(paths['dncnn'])
    generate_unet_diagram(paths['unet'])
    
    print("Building PDF...")
    build_pdf(out_dir / "Architecture_Pipeline_Report.pdf", paths)
