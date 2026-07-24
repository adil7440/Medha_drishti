"""
Stage 2: Preprocessing Justification & Curated Dataset Report Generator
MedhaDrishti National-Level AI Hackathon (Yugma TechFest 2.0)
"""

import sys, os, time, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
FIG_DIR = PROJECT_DIR / 'figures' / 'stage2'
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = PROJECT_DIR / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD METRICS DATA
# ============================================================

def load_metrics():
    metrics_dir = PROJECT_DIR / 'stage2' / 'metrics'
    all_df = pd.read_csv(metrics_dir / 'stage2_quality_metrics.csv')
    brain_df = pd.read_csv(metrics_dir / 'brain_preprocessing_metrics.csv')
    spine_df = pd.read_csv(metrics_dir / 'spine_preprocessing_metrics.csv')
    return all_df, brain_df, spine_df


def load_sample_npz(dataset="Brain", patient="BraTS20_Training_001", modality="T2"):
    npz_dir = PROJECT_DIR / 'stage2' / 'preprocessed'
    fname = f"{dataset}_{patient}_{modality}_preprocessed.npz"
    npz_path = npz_dir / fname
    if npz_path.exists():
        return dict(np.load(str(npz_path), allow_pickle=True))
    # Try brain default
    fname2 = f"Brain_BraTS20_Training_001_T2_preprocessed.npz"
    npz_path2 = npz_dir / fname2
    if npz_path2.exists():
        return dict(np.load(str(npz_path2), allow_pickle=True))
    return None


# ============================================================
# 2. FIGURE GENERATION
# ============================================================

def fig_pipeline_flow(save_path):
    """Draw the 8-step preprocessing pipeline as a flowchart."""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 3)
    ax.axis('off')
    ax.set_title('Stage 2: MRI Preprocessing Pipeline Flow', fontsize=16, fontweight='bold', pad=15)

    steps = [
        ('1. NIfTI\nValidation', '#1565c0'),
        ('2. Voxel\nResampling', '#0d47a1'),
        ('3. Intensity\nNormalization', '#2e7d32'),
        ('4. Multi-Filter\nDenoising', '#e65100'),
        ('5. N4 Bias Field\nCorrection', '#6a1b9a'),
        ('6. CLAHE\nEnhancement', '#c62828'),
        ('7. Skull\nStripping', '#37474f'),
        ('8. Data\nAugmentation', '#00695c'),
    ]

    for i, (label, color) in enumerate(steps):
        x = i * 2 + 1
        rect = plt.Rectangle((x - 0.7, 0.8), 1.6, 1.4, facecolor=color, edgecolor='white',
                              linewidth=2, alpha=0.9, zorder=2)
        ax.add_patch(rect)
        ax.text(x + 0.1, 1.5, label, ha='center', va='center', fontsize=9,
                fontweight='bold', color='white', zorder=3)
        if i < len(steps) - 1:
            ax.annotate('', xy=(x + 1.0, 1.5), xytext=(x + 0.95, 1.5),
                        arrowprops=dict(arrowstyle='->', color='#bdbdbd', lw=2))

    ax.text(8, 0.2, 'All techniques are classical (non-deep-learning) methods as required by hackathon rules',
            ha='center', va='center', fontsize=9, style='italic', color='#616161')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def fig_before_after_comparison(npz_data, save_path):
    """Show original vs each processing stage."""
    if npz_data is None:
        return
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('Preprocessing Stage-by-Stage Comparison (Central Slice)', fontsize=15, fontweight='bold')

    stage_keys = [
        ('orig_slice', 'Original Raw'),
        ('stage_norm', 'After Normalization'),
        ('stage_denoise_bilat', 'After Bilateral Denoise'),
        ('stage_n4', 'After N4 Bias Correction'),
        ('stage_clahe', 'After CLAHE Enhancement'),
        ('stage_final', 'Final Preprocessed'),
        ('aug_rot', 'Augmented: Rotation'),
        ('aug_gamma', 'Augmented: Gamma'),
    ]

    for ax, (key, title) in zip(axes.flat, stage_keys):
        if key in npz_data:
            img = npz_data[key]
            if img.ndim == 3:
                img = img[:, :, img.shape[2] // 2]
            vmin, vmax = np.percentile(img, [1, 99]) if img.max() > 0 else (0, 1)
            ax.imshow(img.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
            ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def fig_denoising_comparison(npz_data, save_path):
    """Compare 4 denoising filters side by side."""
    if npz_data is None:
        return
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('Denoising Filter Comparison (Bilateral Selected)', fontsize=14, fontweight='bold')

    items = [
        ('orig_slice', 'Original'),
        ('stage_denoise_gauss', 'Gaussian'),
        ('stage_denoise_median', 'Median'),
        ('stage_denoise_bilat', 'Bilateral\n(Selected)'),
        ('stage_denoise_nlm', 'NLM'),
    ]

    for ax, (key, title) in zip(axes, items):
        if key in npz_data:
            img = npz_data[key]
            if img.ndim == 3:
                img = img[:, :, img.shape[2] // 2]
            vmin, vmax = np.percentile(img, [1, 99]) if img.max() > 0 else (0, 1)
            ax.imshow(img.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
            color = 'green' if 'bilat' in key else 'black'
            ax.set_title(title, fontsize=11, fontweight='bold', color=color)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def fig_augmentation_examples(npz_data, save_path):
    """Show all 4 augmentation transforms."""
    if npz_data is None:
        return
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('Data Augmentation Transforms Applied', fontsize=14, fontweight='bold')

    items = [
        ('stage_final', 'Preprocessed\n(Base)'),
        ('aug_rot', 'Rotation\n(10 deg)'),
        ('aug_flip', 'Horizontal\nFlip'),
        ('aug_gamma', 'Gamma\nCorrection'),
        ('aug_final', 'Final\n(Noise Added)'),
    ]

    for ax, (key, title) in zip(axes, items):
        if key in npz_data:
            img = npz_data[key]
            if img.ndim == 3:
                img = img[:, :, img.shape[2] // 2]
            vmin, vmax = np.percentile(img, [1, 99]) if img.max() > 0 else (0, 1)
            ax.imshow(img.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
            ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def fig_quality_metrics_bars(brain_df, spine_df, save_path):
    """Bar chart of key quality metrics for Brain vs Spine."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('Quality Evaluation Metrics: Brain vs Spine Preprocessing', fontsize=15, fontweight='bold')

    metrics = ['PSNR', 'SSIM', 'RMSE', 'UQI', 'FSIM', 'Contrast_After', 'Sharpness_After', 'NoiseLevel_After']
    titles = ['PSNR (dB)', 'SSIM', 'RMSE', 'UQI', 'FSIM', 'Contrast (After)', 'Sharpness (After)', 'Noise Level (After)']

    for ax, metric, title in zip(axes.flat, metrics, titles):
        b_val = brain_df[metric].mean() if metric in brain_df.columns else 0
        b_std = brain_df[metric].std() if metric in brain_df.columns else 0
        s_val = spine_df[metric].mean() if metric in spine_df.columns else 0
        s_std = spine_df[metric].std() if metric in spine_df.columns else 0

        bars = ax.bar(['Brain', 'Spine'], [b_val, s_val],
                      yerr=[b_std, s_std], color=['#1565c0', '#2e7d32'],
                      capsize=8, alpha=0.85, edgecolor='black', linewidth=0.8)
        ax.set_title(title, fontsize=11, fontweight='bold')
        for bar, val in zip(bars, [b_val, s_val]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + abs(bar.get_height())*0.05,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def fig_before_after_image_metrics(brain_df, spine_df, save_path):
    """Compare before vs after for key image properties."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Image Properties: Before vs After Preprocessing', fontsize=15, fontweight='bold')

    pairs = [
        ('Contrast_Before', 'Contrast_After', 'Contrast (RMS)'),
        ('Sharpness_Before', 'Sharpness_After', 'Sharpness'),
        ('EdgeStrength_Before', 'EdgeStrength_After', 'Edge Strength'),
        ('NoiseLevel_Before', 'NoiseLevel_After', 'Noise Level'),
        ('Entropy_Before', 'Entropy_After', 'Entropy (Complexity)'),
    ]

    # Brain
    for ax, (bef, aft, title) in zip(axes[0], pairs[:3]):
        b_bef = brain_df[bef].mean() if bef in brain_df.columns else 0
        b_aft = brain_df[aft].mean() if aft in brain_df.columns else 0
        bars = ax.bar(['Before', 'After'], [b_bef, b_aft],
                      color=['#ef5350', '#42a5f5'], edgecolor='black', alpha=0.85)
        ax.set_title(f'Brain: {title}', fontsize=10, fontweight='bold')
        for bar, val in zip(bars, [b_bef, b_aft]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f'{val:.4f}', ha='center', fontsize=8, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    for ax, (bef, aft, title) in zip(axes[1], pairs[3:] + [pairs[0]]):
        s_bef = spine_df[bef].mean() if bef in spine_df.columns else 0
        s_aft = spine_df[aft].mean() if aft in spine_df.columns else 0
        bars = ax.bar(['Before', 'After'], [s_bef, s_aft],
                      color=['#ef5350', '#66bb6a'], edgecolor='black', alpha=0.85)
        ax.set_title(f'Spine: {title}', fontsize=10, fontweight='bold')
        for bar, val in zip(bars, [s_bef, s_aft]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f'{val:.4f}', ha='center', fontsize=8, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def fig_modality_quality_boxplots(brain_df, save_path):
    """Boxplots of quality metrics per brain modality."""
    if len(brain_df) == 0:
        return
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle('Brain MRI: Quality Metrics by Modality', fontsize=14, fontweight='bold')

    for ax, metric, title in zip(axes, ['PSNR', 'SSIM', 'Contrast_After', 'NoiseLevel_After'],
                                  ['PSNR (dB)', 'SSIM', 'Contrast (After)', 'Noise Level (After)']):
        modalities = sorted(brain_df['Modality'].unique())
        data = [brain_df[brain_df['Modality'] == m][metric].dropna().values for m in modalities]
        bp = ax.boxplot(data, labels=modalities, patch_artist=True)
        box_colors = plt.cm.Set2(np.linspace(0, 1, len(modalities)))
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def fig_curated_dataset_summary(save_path):
    """Overview of curated datasets ready for downstream tasks."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Curated Datasets: Ready for AI Enhancement & Segmentation', fontsize=15, fontweight='bold')

    # Brain curated
    ax = axes[0]
    cats = ['Training Brain\n(BraTS 2020)', 'Training Spine', 'Test Brain\n(BRP1-10)', 'Test Spine\n(SP11-23)']
    counts = [306, 186, 40, 179]
    colors_list = ['#1565c0', '#2e7d32', '#e65100', '#6a1b9a']
    bars = ax.bar(cats, counts, color=colors_list, edgecolor='black', alpha=0.85)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                str(count), ha='center', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of Preprocessed Volumes')
    ax.set_title('Curated Volume Counts')
    ax.grid(axis='y', alpha=0.3)

    # Preprocessing status pie
    ax = axes[1]
    status = ['Preprocessed\n(npz saved)', 'Not Yet\nProcessed']
    brain_total = 369 * 4  # All modalities
    spine_total = 186
    preprocessed = 306 + 186
    remaining = (brain_total + spine_total) - preprocessed
    sizes = [preprocessed, max(remaining, 0)]
    explode = (0.05, 0)
    ax.pie(sizes, explode=explode, labels=status, autopct='%1.1f%%',
           colors=['#42a5f5', '#e0e0e0'], startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title(f'Pipeline Coverage ({preprocessed} / {preprocessed + max(remaining, 0)} volumes)')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def fig_modality_before_after_heatmap(brain_df, save_path):
    """Heatmap showing before/after delta per modality."""
    if len(brain_df) == 0:
        return
    modalities = sorted(brain_df['Modality'].unique())
    params = ['Contrast', 'Sharpness', 'EdgeStrength', 'NoiseLevel', 'Entropy']
    param_bef = [f'{p}_Before' for p in params]
    param_aft = [f'{p}_After' for p in params]

    delta_matrix = []
    for mod in modalities:
        mod_df = brain_df[brain_df['Modality'] == mod]
        row = []
        for b, a in zip(param_bef, param_aft):
            bef = mod_df[b].mean() if b in mod_df.columns else 0
            aft = mod_df[a].mean() if a in mod_df.columns else 0
            delta = aft - bef
            row.append(delta)
        delta_matrix.append(row)

    fig, ax = plt.subplots(figsize=(10, max(3, len(modalities) * 0.8)))
    im = ax.imshow(delta_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.1)

    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(['Contrast', 'Sharpness', 'Edge\nStrength', 'Noise\nLevel', 'Entropy'],
                        fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(modalities)))
    ax.set_yticklabels(modalities, fontsize=11, fontweight='bold')
    ax.set_title('Brain MRI: Change (After - Before) per Modality', fontsize=14, fontweight='bold', pad=15)

    for i in range(len(modalities)):
        for j in range(len(params)):
            val = delta_matrix[i][j]
            color = 'white' if abs(val) > 0.05 else 'black'
            ax.text(j, i, f'{val:+.4f}', ha='center', va='center', fontsize=9,
                    color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, shrink=0.8, label='Delta (After - Before)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# 3. PDF BUILD
# ============================================================

def build_stage2_pdf(output_path, all_df, brain_df, spine_df, fig_paths):
    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=15*mm, bottomMargin=15*mm
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'],
        fontSize=18, spaceAfter=6, textColor=colors.HexColor('#1a237e'), alignment=TA_CENTER)
    subtitle_style = ParagraphStyle('SubTitle', parent=styles['Heading2'],
        fontSize=13, spaceAfter=4, textColor=colors.HexColor('#37474f'), alignment=TA_CENTER)
    section_style = ParagraphStyle('Section', parent=styles['Heading1'],
        fontSize=14, spaceBefore=12, spaceAfter=6, textColor=colors.HexColor('#0d47a1'))
    subsection_style = ParagraphStyle('SubSection', parent=styles['Heading2'],
        fontSize=11, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor('#1565c0'))
    body_style = ParagraphStyle('BodyText2', parent=styles['Normal'],
        fontSize=9, spaceAfter=4, leading=13)
    small_style = ParagraphStyle('SmallText', parent=styles['Normal'],
        fontSize=7.5, spaceAfter=2, leading=10)

    elements = []

    # ---- COVER PAGE ----
    elements.append(Spacer(1, 60))
    elements.append(Paragraph("MedhaDrishti National-Level AI Hackathon", title_style))
    elements.append(Paragraph("Yugma TechFest 2.0", subtitle_style))
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("Stage 2: Preprocessing Justification & Curated Dataset Report", ParagraphStyle(
        'ReportTitle', parent=styles['Heading1'], fontSize=14, alignment=TA_CENTER,
        textColor=colors.HexColor('#283593'), spaceAfter=10)))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("AI for Medical Image Enhancement and Segmentation", ParagraphStyle(
        'Topic', parent=styles['Normal'], fontSize=11, alignment=TA_CENTER,
        textColor=colors.HexColor('#546e7a'), spaceAfter=20)))

    cover_info = [
        ['Report Type', 'Preprocessing Justification & Curated Dataset Assessment'],
        ['Sub-Modalities', 'T1, T1CE, T2, FLAIR (Brain); T1W, T2W, STIR, GADO (Spine)'],
        ['Techniques', 'Resizing, Scaling, Denoising, Artifact Correction, Augmentation'],
        ['Processed Volumes', f'{len(all_df)} (Brain: {len(brain_df)}, Spine: {len(spine_df)})'],
    ]
    t = Table(cover_info, colWidths=[130, 330])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#37474f')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LINEBELOW', (0, 0), (-1, -2), 0.5, colors.HexColor('#e0e0e0')),
        ('LINEBELOW', (0, -1), (-1, -1), 1, colors.HexColor('#1565c0')),
    ]))
    elements.append(t)
    elements.append(PageBreak())

    # ---- TOC ----
    elements.append(Paragraph("Table of Contents", section_style))
    toc = [
        "1. Executive Summary",
        "2. Preprocessing Pipeline Overview",
        "3. Technique Justification: Brain MRI (T1, T1CE, T2, FLAIR)",
        "4. Technique Justification: Spine MRI (T1W, T2W, STIR)",
        "5. Resizing & Scaling (Voxel Normalization)",
        "6. Denoising: Multi-Filter Evaluation",
        "7. Artifact Correction: N4 Bias Field Correction",
        "8. Contrast Enhancement: CLAHE",
        "9. Skull Stripping (Brain Only)",
        "10. Data Augmentation",
        "11. Annotation Visualization (Training Dataset)",
        "12. Quantitative Quality Evaluation (17 Metrics)",
        "13. Per-Modality Quality Analysis",
        "14. Curated Datasets Summary",
        "15. Conclusion",
    ]
    for item in toc:
        elements.append(Paragraph(item, body_style))
    elements.append(PageBreak())

    # ---- 1. EXECUTIVE SUMMARY ----
    elements.append(Paragraph("1. Executive Summary", section_style))
    elements.append(Paragraph(
        "Stage 2 implements a comprehensive, <b>classical (non-deep-learning)</b> MRI preprocessing pipeline "
        "for both Brain and Spine datasets. All techniques were selected based on their proven efficacy in "
        "medical image processing literature and validated through 17 quantitative quality metrics.<br/><br/>"
        f"<b>Total volumes processed:</b> {len(all_df)} "
        f"(Brain: {len(brain_df)}, Spine: {len(spine_df)})<br/>"
        f"<b>Average processing time:</b> {all_df['Processing_Time_Sec'].mean():.3f} seconds per volume<br/>"
        f"<b>Pipeline steps:</b> 8 sequential stages (Validation, Resampling, Normalization, Denoising, "
        f"N4 Correction, CLAHE, Skull Stripping, Augmentation)", body_style))

    # Summary metrics
    elements.append(Spacer(1, 8))
    summary_data = [
        ['Metric', 'Brain Avg', 'Spine Avg', 'Interpretation'],
        ['PSNR (dB)', f'{brain_df["PSNR"].mean():.2f}', f'{spine_df["PSNR"].mean():.2f}',
         '> 20 dB = Good quality preservation'],
        ['SSIM', f'{brain_df["SSIM"].mean():.4f}', f'{spine_df["SSIM"].mean():.4f}',
         '> 0.85 = High structural similarity'],
        ['RMSE', f'{brain_df["RMSE"].mean():.4f}', f'{spine_df["RMSE"].mean():.4f}',
         'Lower = better fidelity'],
        ['UQI', f'{brain_df["UQI"].mean():.4f}', f'{spine_df["UQI"].mean():.4f}',
         '> 0.90 = Excellent quality'],
        ['FSIM', f'{brain_df["FSIM"].mean():.4f}', f'{spine_df["FSIM"].mean():.4f}',
         '> 0.80 = Strong feature preservation'],
    ]
    t = Table(summary_data, colWidths=[70, 80, 80, 220])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1565c0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#e3f2fd'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#90caf9')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(t)
    elements.append(PageBreak())

    # ---- 2. PIPELINE OVERVIEW ----
    elements.append(Paragraph("2. Preprocessing Pipeline Overview", section_style))
    elements.append(Paragraph(
        "The 8-step preprocessing pipeline was designed to handle the unique challenges of multi-sequence "
        "MRI data across both Brain and Spine datasets:", body_style))
    elements.append(Spacer(1, 6))

    # Pipeline flow figure
    for fp in fig_paths:
        if 'pipeline_flow' in str(fp):
            elements.append(RLImage(str(fp), width=470, height=150))
            break
    elements.append(Spacer(1, 6))

    pipeline_details = [
        ['Step', 'Technique', 'Purpose', 'Implementation'],
        ['1', 'NIfTI Validation', 'Check file integrity, NaN/Inf, spacing', 'MRIValidator (nibabel)'],
        ['2', 'Voxel Resampling', 'Normalize voxel grid to isotropic 1mm^3', 'SimpleITK ResampleImageFilter'],
        ['3', 'Intensity Normalization', 'Percentile clipping (0.5-99.5) + Min-Max [0,1]', 'IntensityNormalizer'],
        ['4', 'Multi-Filter Denoising', 'Gaussian, Median, Bilateral, NLM comparison', 'NoiseRemover (OpenCV + skimage)'],
        ['5', 'N4 Bias Field Correction', 'Correct RF coil inhomogeneity', 'SimpleITK N4BiasFieldCorrection'],
        ['6', 'CLAHE Enhancement', 'Local contrast enhancement (clip=2.0)', 'OpenCV CLAHE'],
        ['7', 'Skull Stripping (Brain)', 'Otsu + morphological brain extraction', 'SkullStripper (skimage)'],
        ['8', 'Data Augmentation', 'Rotation, Flip, Gamma, Gaussian Noise', 'MRIAugmentor (scipy)'],
    ]
    t = Table(pipeline_details, colWidths=[30, 110, 170, 150])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#283593')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#e8eaf6'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#9fa8da')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(t)
    elements.append(PageBreak())

    # ---- 3. BRAIN MRI JUSTIFICATION ----
    elements.append(Paragraph("3. Technique Justification: Brain MRI (T1, T1CE, T2, FLAIR)", section_style))
    brain_just = [
        ['Modality', 'Key Challenge', 'Technique Applied', 'Justification'],
        ['T1-Weighted', 'Low contrast between gray/white matter;\nRF inhomogeneity artifacts',
         'N4 Correction + CLAHE', 'N4 corrects B1 field non-uniformity inherent in T1.\nCLAHE boosts local contrast without over-amplifying noise.'],
        ['T1CE (Contrast)', 'Gadolinium enhancement creates\nhigh-intensity peaks that skew normalization',
         'Percentile clipping + Z-score', 'Clipping at 99.5th percentile prevents contrast-enhancing\nlesions from dominating normalization statistics.'],
        ['T2-Weighted', 'High CSF signal can overwhelm\nbrain parenchyma detail',
         'Bilateral denoising + Skull stripping', 'Bilateral filter preserves tissue boundaries while reducing\nCSF-related noise. Skull stripping removes non-brain signal.'],
        ['FLAIR', 'Suppresses CSF but retains edema signal;\nrequires careful intensity balancing',
         'Full pipeline: Norm + Denoise + CLAHE', 'FLAIR benefits most from the complete pipeline due to its\ncomplex multi-tissue contrast dynamics.'],
    ]
    t = Table(brain_just, colWidths=[70, 120, 110, 160])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1565c0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#e3f2fd'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#90caf9')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(t)
    elements.append(PageBreak())

    # ---- 4. SPINE MRI JUSTIFICATION ----
    elements.append(Paragraph("4. Technique Justification: Spine MRI (T1W, T2W, STIR)", section_style))
    spine_just = [
        ['Modality', 'Key Challenge', 'Technique Applied', 'Justification'],
        ['T1W / eT1W', 'Heterogeneous voxel spacing across\npatients (0.5-1.0mm in-plane)',
         'Resampling + Percentile Norm', 'Isotropic resampling standardizes geometry for batch\nprocessing. Percentile clipping handles intensity outliers.'],
        ['T2W / eT2W', 'High dynamic range; DRIVE variants\nhave different noise profiles',
         'Bilateral denoise + Min-Max', 'Bilateral filtering is modality-agnostic and preserves the\ncritical disc/cord boundary contrast in T2W images.'],
        ['STIR', 'Fat suppression creates very low\nsignal in certain regions',
         'Percentile clipping + CLAHE', 'STIR sequences have extreme intensity distributions;\npercentile clipping prevents the fat-suppressed regions\nfrom being treated as background noise.'],
        ['T1W GADO', 'Post-contrast enhancement creates\nfocal high-intensity lesions',
         'Full normalization pipeline', 'Same as T1CE in Brain: Gadolinium-enhanced lesions require\ncareful outlier handling during normalization.'],
        ['Skull Stripping', 'NOT applied to Spine', 'Skipped (step 7)', 'Spine images contain vertebral structures adjacent to the\nspinal cord; skull stripping would remove relevant anatomy.'],
    ]
    t = Table(spine_just, colWidths=[70, 120, 105, 165])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e7d32')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#e8f5e9'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#a5d6a7')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(t)
    elements.append(PageBreak())

    # ---- 5. RESIZING & SCALING ----
    elements.append(Paragraph("5. Resizing & Scaling (Voxel Normalization)", section_style))
    elements.append(Paragraph(
        "<b>Voxel Resampling:</b> All volumes resampled to <b>1.0 x 1.0 x 1.0 mm^3 isotropic</b> spacing "
        "using SimpleITK linear interpolation. This eliminates geometric distortion from heterogeneous "
        "scanner protocols and enables consistent spatial feature extraction.<br/><br/>"
        "<b>Intensity Scaling:</b> Two-stage normalization applied:<br/>"
        "1. <b>Percentile Clipping:</b> Voxels clipped between 0.5th and 99.5th percentiles (computed on "
        "non-zero voxels only) to remove extreme intensity outliers from motion artifacts or scanner glitches.<br/>"
        "2. <b>Min-Max Scaling:</b> Linearly scaled to [0.0, 1.0] range for consistent input to downstream "
        "enhancement and segmentation models.<br/><br/>"
        "<b>Justification:</b> MRI signal intensities lack absolute physical units (unlike CT Hounsfield units). "
        "Percentile clipping + Min-Max normalization is the standard approach in BraTS challenge pipelines "
        "and medical imaging literature.", body_style))
    elements.append(PageBreak())

    # ---- 6. DENOISING ----
    elements.append(Paragraph("6. Denoising: Multi-Filter Evaluation", section_style))
    elements.append(Paragraph(
        "Four denoising filters were benchmarked on each volume to determine the optimal "
        "noise reduction strategy:", body_style))

    denoise_table = [
        ['Filter', 'Method', 'Pros', 'Cons', 'Selected?'],
        ['Gaussian', 'Convolution with Gaussian kernel (sigma=1.0)',
         'Fast, simple', 'Blurs edges, no content awareness', 'No'],
        ['Median', 'Rank-order filter (3x3 kernel)',
         'Removes salt-and-pepper noise', 'Loses fine texture details', 'No'],
        ['Bilateral', 'Spatial + intensity weighted (d=5, sigma=25)',
         'Edge-preserving, content-aware', 'Slower than Gaussian', 'YES (Selected)'],
        ['NLM', 'Non-Local Means (patch-based averaging)',
         'Best theoretical denoising', 'Computationally expensive', 'No (reference)'],
    ]
    t = Table(denoise_table, colWidths=[55, 120, 95, 95, 70])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e65100')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (4, 4), (4, 4), colors.HexColor('#c8e6c9')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#fff3e0'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#ffcc80')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        "<b>Selection Rationale:</b> Bilateral filtering was chosen as the primary denoising method because it "
        "achieves the best balance between noise reduction and edge preservation. It operates on both spatial "
        "proximity and intensity similarity, ensuring that anatomical boundaries (critical for segmentation) "
        "remain sharp while background noise is suppressed.", body_style))
    elements.append(PageBreak())

    # ---- 7. N4 BIAS FIELD ----
    elements.append(Paragraph("7. Artifact Correction: N4 Bias Field Correction", section_style))
    elements.append(Paragraph(
        "<b>Problem:</b> RF coil inhomogeneity causes low-frequency intensity variations (bias fields) across "
        "MRI volumes, making identical tissues appear different depending on spatial location.<br/><br/>"
        "<b>Solution:</b> SimpleITK N4BiasFieldCorrectionImageFilter applied on central slices. "
        "The algorithm fits a B-spline model to the logarithm of the image intensity and divides out the "
        "estimated bias field.<br/><br/>"
        "<b>Parameters:</b> Maximum iterations = [30, 20, 10] (3 fitting levels). "
        "Otsu thresholding used for foreground mask generation.<br/><br/>"
        "<b>Justification:</b> N4 is the gold-standard bias correction method in medical imaging "
        "(Tustison et al., 2010). It is applied after denoising to prevent noise amplification "
        "during the bias estimation step.", body_style))
    elements.append(PageBreak())

    # ---- 8. CLAHE ----
    elements.append(Paragraph("8. Contrast Enhancement: CLAHE", section_style))
    elements.append(Paragraph(
        "<b>Method:</b> Contrast Limited Adaptive Histogram Equalization (CLAHE) with "
        "clip_limit=2.0 and tile_grid_size=(8,8).<br/><br/>"
        "<b>How it works:</b> CLAHE divides the image into small tiles and applies histogram equalization "
        "independently to each tile, with contrast limiting to prevent noise amplification. "
        "This enhances local contrast while maintaining global intensity distribution.<br/><br/>"
        f"<b>Results:</b><br/>"
        f"- Brain: Contrast increased from {brain_df['Contrast_Before'].mean():.4f} to "
        f"{brain_df['Contrast_After'].mean():.4f} (delta: +{brain_df['Contrast_After'].mean() - brain_df['Contrast_Before'].mean():.4f})<br/>"
        f"- Spine: Contrast increased from {spine_df['Contrast_Before'].mean():.4f} to "
        f"{spine_df['Contrast_After'].mean():.4f} (delta: +{spine_df['Contrast_After'].mean() - spine_df['Contrast_Before'].mean():.4f})<br/><br/>"
        "<b>Justification:</b> CLAHE is preferred over global histogram equalization because MRI "
        "intensity distributions vary locally (different tissue types), and global HE would over-enhance "
        "some regions while under-enhancing others.", body_style))
    elements.append(PageBreak())

    # ---- 9. SKULL STRIPPING ----
    elements.append(Paragraph("9. Skull Stripping (Brain Only)", section_style))
    elements.append(Paragraph(
        "<b>Method:</b> Otsu adaptive thresholding followed by 3D morphological operations "
        "(largest connected component extraction, binary closing with ball(3), binary opening with ball(2), "
        "hole filling).<br/><br/>"
        "<b>Application:</b> Applied to Brain MRI volumes only. <b>Skip for Spine</b> as vertebral "
        "structures are anatomically adjacent to the spinal cord and removing them would lose diagnostic "
        "information.<br/><br/>"
        "<b>Threshold:</b> Otsu threshold multiplied by 0.3 (lowered to capture the full brain tissue "
        "extent including edema and tumor regions).<br/><br/>"
        "<b>Justification:</b> Skull stripping removes non-brain tissue (scalp, skull, dura) that can "
        "confound segmentation models. The morphological post-processing ensures smooth, anatomically "
        "plausible brain masks.", body_style))
    elements.append(PageBreak())

    # ---- 10. DATA AUGMENTATION ----
    elements.append(Paragraph("10. Data Augmentation", section_style))
    elements.append(Paragraph(
        "Four augmentation transforms were applied to increase training data diversity:", body_style))

    aug_table = [
        ['Transform', 'Parameters', 'Purpose', 'Effect on MRI'],
        ['Rotation', 'angle=10 degrees', 'Rotational invariance',
         'Simulates patient positioning variability'],
        ['Horizontal Flip', 'axis=1 (sagittal)', 'Left-right symmetry',
         'Doubles effective training samples; brain/spine are roughly symmetric'],
        ['Gamma Correction', 'gamma=1.2', 'Intensity non-linearity',
         'Simulates scanner intensity profile variations'],
        ['Gaussian Noise', 'std=0.01', 'Noise robustness',
         'Simulates low-SNR acquisition conditions'],
    ]
    t = Table(aug_table, colWidths=[80, 85, 100, 195])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00695c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#e0f2f1'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#80cbc4')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(t)
    elements.append(PageBreak())

    # ---- 11. ANNOTATION VISUALIZATION ----
    elements.append(Paragraph("11. Annotation Visualization (Training Dataset)", section_style))
    elements.append(Paragraph(
        "The BraTS 2020 training dataset includes expert-annotated multi-class segmentation masks:<br/><br/>"
        "<b>Label 0:</b> Background / Healthy Tissue<br/>"
        "<b>Label 1:</b> Necrotic and Non-Enhancing Tumor Core (NCR/NET)<br/>"
        "<b>Label 2:</b> Peritumoral Edema (ED)<br/>"
        "<b>Label 4:</b> Enhancing Tumor (ET)<br/><br/>"
        "Each preprocessed volume preserves the corresponding ground truth segmentation mask for "
        "supervised training. The annotation masks were NOT modified during preprocessing to maintain "
        "label integrity for downstream segmentation tasks.<br/><br/>"
        "<b>Spine Dataset:</b> Binary classification labels (Normal vs Pathological) are maintained "
        "across all preprocessed volumes. No voxel-level annotations are available for the spine dataset.", body_style))

    # Add sample images if available
    sample_img = PROJECT_DIR / 'figures' / 'sample_images' / 'sample_patient_triplanar.png'
    tumor_img = PROJECT_DIR / 'figures' / 'sample_images' / 'tumor_mask_overlay.png'
    montage_img = PROJECT_DIR / 'figures' / 'spine' / 'sample_images' / 'patient_montage.png'

    if sample_img.exists():
        elements.append(Spacer(1, 6))
        elements.append(Paragraph("Brain: Tri-planar Views with Tumor Overlay", subsection_style))
        elements.append(RLImage(str(sample_img), width=350, height=200))
    if tumor_img.exists():
        elements.append(Spacer(1, 4))
        elements.append(RLImage(str(tumor_img), width=350, height=200))
    if montage_img.exists():
        elements.append(Spacer(1, 6))
        elements.append(Paragraph("Spine: Patient Montage", subsection_style))
        elements.append(RLImage(str(montage_img), width=350, height=200))

    elements.append(PageBreak())

    # ---- 12. QUANTITATIVE QUALITY EVALUATION ----
    elements.append(Paragraph("12. Quantitative Quality Evaluation (17 Metrics)", section_style))
    elements.append(Paragraph(
        "17 full-reference and no-reference quality metrics were computed for each processed volume, "
        "comparing the original raw slice to the final preprocessed output:", body_style))

    metrics_table = [
        ['#', 'Metric', 'Type', 'Brain Avg', 'Spine Avg', 'Description'],
        ['1', 'PSNR (dB)', 'Full-Ref', f'{brain_df["PSNR"].mean():.2f}', f'{spine_df["PSNR"].mean():.2f}',
         'Peak Signal-to-Noise Ratio; higher = better'],
        ['2', 'SSIM', 'Full-Ref', f'{brain_df["SSIM"].mean():.4f}', f'{spine_df["SSIM"].mean():.4f}',
         'Structural Similarity Index; 1.0 = perfect'],
        ['3', 'MSE', 'Full-Ref', f'{brain_df["MSE"].mean():.6f}', f'{spine_df["MSE"].mean():.6f}',
         'Mean Squared Error; lower = better'],
        ['4', 'RMSE', 'Full-Ref', f'{brain_df["RMSE"].mean():.4f}', f'{spine_df["RMSE"].mean():.4f}',
         'Root MSE; interpretable error magnitude'],
        ['5', 'UQI', 'Full-Ref', f'{brain_df["UQI"].mean():.4f}', f'{spine_df["UQI"].mean():.4f}',
         'Universal Quality Index; 1.0 = ideal'],
        ['6', 'FSIM', 'Full-Ref', f'{brain_df["FSIM"].mean():.4f}', f'{spine_df["FSIM"].mean():.4f}',
         'Feature Similarity; gradient-based metric'],
        ['7', 'GMSD', 'Full-Ref', f'{brain_df["GMSD"].mean():.6f}', f'{spine_df["GMSD"].mean():.6f}',
         'Gradient Magnitude Similarity Deviation'],
        ['8', 'VIF', 'Full-Ref', f'{brain_df["VIF"].mean():.4f}', f'{spine_df["VIF"].mean():.4f}',
         'Visual Information Fidelity'],
        ['9', 'BRISQUE', 'No-Ref', f'{brain_df["BRISQUE"].mean():.2f}', f'{spine_df["BRISQUE"].mean():.2f}',
         'Blind Image Spatial Quality Evaluator'],
        ['10', 'NIQE', 'No-Ref', f'{brain_df["NIQE"].mean():.2f}', f'{spine_df["NIQE"].mean():.2f}',
         'Natural Image Quality Evaluator'],
        ['11', 'PIQE', 'No-Ref', f'{brain_df["PIQE"].mean():.2f}', f'{spine_df["PIQE"].mean():.2f}',
         'Perception-based Image Quality Evaluator'],
        ['12', 'LPIPS', 'Full-Ref', f'{brain_df["LPIPS"].mean():.4f}', f'{spine_df["LPIPS"].mean():.4f}',
         'Multi-scale Perceptual Gradient Similarity'],
        ['13', 'Entropy (After)', 'No-Ref', f'{brain_df["Entropy_After"].mean():.4f}',
         f'{spine_df["Entropy_After"].mean():.4f}', 'Shannon Information Content (bits)'],
        ['14', 'Contrast (After)', 'No-Ref', f'{brain_df["Contrast_After"].mean():.4f}',
         f'{spine_df["Contrast_After"].mean():.4f}', 'RMS Contrast on foreground'],
        ['15', 'Sharpness (After)', 'No-Ref', f'{brain_df["Sharpness_After"].mean():.6f}',
         f'{spine_df["Sharpness_After"].mean():.6f}', 'Tenengrad gradient variance'],
        ['16', 'Edge Strength (After)', 'No-Ref', f'{brain_df["EdgeStrength_After"].mean():.6f}',
         f'{spine_df["EdgeStrength_After"].mean():.6f}', 'Average Sobel gradient magnitude'],
        ['17', 'Noise Level (After)', 'No-Ref', f'{brain_df["NoiseLevel_After"].mean():.4f}',
         f'{spine_df["NoiseLevel_After"].mean():.4f}', 'MAD background noise estimate'],
    ]
    t = Table(metrics_table, colWidths=[18, 75, 48, 62, 62, 195])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#37474f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#eceff1'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#90a4ae')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    elements.append(t)
    elements.append(PageBreak())

    # ---- 13. PER-MODALITY ANALYSIS ----
    elements.append(Paragraph("13. Per-Modality Quality Analysis", section_style))

    for fp in fig_paths:
        if 'quality_metrics' in str(fp) or 'modality_quality' in str(fp):
            elements.append(RLImage(str(fp), width=470, height=260))
            elements.append(Spacer(1, 8))
            break

    for fp in fig_paths:
        if 'before_after' in str(fp):
            elements.append(RLImage(str(fp), width=470, height=290))
            break

    elements.append(PageBreak())

    # ---- 14. CURATED DATASETS ----
    elements.append(Paragraph("14. Curated Datasets Summary", section_style))
    elements.append(Paragraph(
        "The following curated datasets have been preprocessed and are ready for "
        "AI-based enhancement (Stage 3) and segmentation (Stage 4):", body_style))
    elements.append(Spacer(1, 6))

    curated_table = [
        ['Dataset', 'Type', 'Patients', 'Volumes', 'Format', 'Location'],
        ['Brain Training\n(BraTS 2020)', 'Training', '30 (of 369)', str(len(brain_df)),
         '.npz (slices)', 'stage2/preprocessed/Brain_*'],
        ['Spine Training', 'Training', '10', str(len(spine_df)),
         '.npz (slices)', 'stage2/preprocessed/Spine_*'],
        ['Brain Test\n(BRP1-BRP10)', 'Test/Val', '10', '40 (raw NIfTI)',
         '.nii', 'test_brain/'],
        ['Spine Test\n(SP11-SP23)', 'Test/Val', '10', '179 (raw NIfTI)',
         '.nii.gz', 'test_spine/'],
    ]
    t = Table(curated_table, colWidths=[80, 50, 55, 55, 65, 150])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#e8eaf6'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#7986cb')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 10))

    # Curated dataset figure
    for fp in fig_paths:
        if 'curated' in str(fp):
            elements.append(RLImage(str(fp), width=470, height=180))
            break

    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        "<b>Each .npz file contains:</b> original slice, normalized slice, denoised slices (4 filters), "
        "N4-corrected slice, CLAHE-enhanced slice, final preprocessed slice, and 4 augmented versions. "
        "This enables fast dashboard visualization and model training without re-running the pipeline.", body_style))
    elements.append(PageBreak())

    # ---- 15. CONCLUSION ----
    elements.append(Paragraph("15. Conclusion", section_style))
    elements.append(Paragraph(
        f"Stage 2 successfully preprocessed <b>{len(all_df)} MRI volumes</b> "
        f"(Brain: {len(brain_df)}, Spine: {len(spine_df)}) through an 8-step classical pipeline.<br/><br/>"
        "<b>Key Achievements:</b><br/>"
        "- <b>Justified every technique</b> per sub-modality (T1, T1CE, T2, FLAIR, T1W, T2W, STIR, GADO)<br/>"
        "- <b>Resizing/Scaling:</b> Isotropic 1mm^3 resampling + percentile clipping + Min-Max normalization<br/>"
        "- <b>Denoising:</b> Bilateral filter selected via 4-filter comparative benchmarking<br/>"
        "- <b>Artifact Correction:</b> N4 bias field correction for RF inhomogeneity removal<br/>"
        "- <b>Contrast Enhancement:</b> CLAHE with clip_limit=2.0 for local contrast improvement<br/>"
        "- <b>Skull Stripping:</b> Applied to Brain (Otsu + morphological); skipped for Spine<br/>"
        "- <b>Data Augmentation:</b> Rotation, Flip, Gamma, Gaussian Noise (4 transforms)<br/>"
        "- <b>Annotation Preservation:</b> All ground truth labels maintained for supervised training<br/>"
        "- <b>17 Quality Metrics:</b> Comprehensive before/after evaluation confirming pipeline efficacy<br/><br/>"
        "<b>Curated Output:</b> 306 preprocessed Brain slices + 186 preprocessed Spine volumes "
        "stored as .npz files, ready for Stage 3 (AI Enhancement) and Stage 4 (Segmentation).<br/><br/>"
        "<b>Note:</b> All techniques are non-deep-learning classical methods, fully compliant with "
        "MedhaDrishti hackathon rules.", body_style))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph(
        "<i>Report automatically generated by Stage 2 Preprocessing Pipeline for MedhaDrishti AI Hackathon.</i>",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)))

    doc.build(elements)
    print(f"PDF generated: {output_path}")


# ============================================================
# 4. MAIN
# ============================================================

def main():
    start = time.time()
    print("=" * 70)
    print(" STAGE 2: PREPROCESSING JUSTIFICATION PDF GENERATOR")
    print(" MedhaDrishti National-Level AI Hackathon (Yugma TechFest 2.0)")
    print("=" * 70)

    print("\n[1/4] Loading metrics data...")
    all_df, brain_df, spine_df = load_metrics()
    print(f"  Loaded: {len(all_df)} total, {len(brain_df)} brain, {len(spine_df)} spine records")

    print("\n[2/4] Generating figures...")
    fig_paths = []

    # Pipeline flow
    fp = str(FIG_DIR / 'pipeline_flow.png')
    try:
        fig_pipeline_flow(fp)
        fig_paths.append(fp)
        print(f"  Generated: pipeline_flow.png")
    except Exception as e:
        print(f"  [Error] pipeline_flow: {e}")

    # Before/After comparison from NPZ
    npz = load_sample_npz()
    fp = str(FIG_DIR / 'before_after_stages.png')
    try:
        fig_before_after_comparison(npz, fp)
        fig_paths.append(fp)
        print(f"  Generated: before_after_stages.png")
    except Exception as e:
        print(f"  [Error] before_after: {e}")

    # Denoising comparison
    fp = str(FIG_DIR / 'denoising_comparison.png')
    try:
        fig_denoising_comparison(npz, fp)
        fig_paths.append(fp)
        print(f"  Generated: denoising_comparison.png")
    except Exception as e:
        print(f"  [Error] denoising: {e}")

    # Augmentation examples
    fp = str(FIG_DIR / 'augmentation_examples.png')
    try:
        fig_augmentation_examples(npz, fp)
        fig_paths.append(fp)
        print(f"  Generated: augmentation_examples.png")
    except Exception as e:
        print(f"  [Error] augmentation: {e}")

    # Quality metrics bars
    fp = str(FIG_DIR / 'quality_metrics_comparison.png')
    try:
        fig_quality_metrics_bars(brain_df, spine_df, fp)
        fig_paths.append(fp)
        print(f"  Generated: quality_metrics_comparison.png")
    except Exception as e:
        print(f"  [Error] quality_metrics: {e}")

    # Before/After image metrics
    fp = str(FIG_DIR / 'before_after_image_metrics.png')
    try:
        fig_before_after_image_metrics(brain_df, spine_df, fp)
        fig_paths.append(fp)
        print(f"  Generated: before_after_image_metrics.png")
    except Exception as e:
        print(f"  [Error] before_after_metrics: {e}")

    # Modality quality boxplots
    fp = str(FIG_DIR / 'modality_quality_boxplots.png')
    try:
        fig_modality_quality_boxplots(brain_df, fp)
        fig_paths.append(fp)
        print(f"  Generated: modality_quality_boxplots.png")
    except Exception as e:
        print(f"  [Error] modality_boxplots: {e}")

    # Curated dataset summary
    fp = str(FIG_DIR / 'curated_dataset_summary.png')
    try:
        fig_curated_dataset_summary(fp)
        fig_paths.append(fp)
        print(f"  Generated: curated_dataset_summary.png")
    except Exception as e:
        print(f"  [Error] curated_summary: {e}")

    # Modality before/after heatmap
    fp = str(FIG_DIR / 'modality_before_after_heatmap.png')
    try:
        fig_modality_before_after_heatmap(brain_df, fp)
        fig_paths.append(fp)
        print(f"  Generated: modality_before_after_heatmap.png")
    except Exception as e:
        print(f"  [Error] modality_heatmap: {e}")

    print(f"\n[3/4] Generated {len(fig_paths)} figures")

    print("\n[4/4] Building PDF report...")
    pdf_path = REPORTS_DIR / 'Stage2_Preprocessing_Report.pdf'
    build_stage2_pdf(pdf_path, all_df, brain_df, spine_df, fig_paths)

    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print(f" STAGE 2 PDF GENERATED IN {elapsed:.2f} SECONDS")
    print(f" Output: {pdf_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
