import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib

# Set clean aesthetic styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

class DatasetVisualizer:
    """
    Generates publication-quality visualizations for Stage 1 Brain MRI Dataset Analysis.
    Saves all figures as PNG files into specified subdirectories under project/figures/.
    """
    def __init__(self, properties_df, patient_df, figures_dir):
        self.properties_df = properties_df
        self.patient_df = patient_df
        self.figures_dir = Path(figures_dir)
        
        # Subdirectories
        self.dirs = {
            'sample_images': self.figures_dir / 'sample_images',
            'histograms': self.figures_dir / 'histograms',
            'boxplots': self.figures_dir / 'boxplots',
            'modality_comparison': self.figures_dir / 'modality_comparison',
            'resolution_analysis': self.figures_dir / 'resolution_analysis',
            'quality_analysis': self.figures_dir / 'quality_analysis'
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        self.colors = {
            'T1': '#1f77b4',
            'T1CE': '#ff7f0e',
            'T2': '#2ca02c',
            'FLAIR': '#d62728',
            'SEG': '#9467bd'
        }

    def generate_all_figures(self, sample_patient_dir=None):
        """Generates the full suite of Stage 1 analysis figures."""
        print("[Visualizer] Generating sample image figures...")
        self._plot_sample_patient_and_montages(sample_patient_dir)

        print("[Visualizer] Generating intensity histograms...")
        self._plot_histograms()

        print("[Visualizer] Generating boxplots...")
        self._plot_boxplots()

        print("[Visualizer] Generating modality comparison figures...")
        self._plot_modality_comparisons()

        print("[Visualizer] Generating resolution & spatial analysis figures...")
        self._plot_resolution_analysis()

        print("[Visualizer] Generating quality analysis figures...")
        self._plot_quality_analysis()

        print("[Visualizer] All publication-quality figures successfully generated!")

    def _plot_sample_patient_and_montages(self, sample_patient_dir):
        """Plots sample slices, multi-patient montages, and tumor label overlays."""
        if sample_patient_dir is None:
            # Pick first complete patient
            comp = self.patient_df[self.patient_df['Status'] == 'Complete']
            if not comp.empty:
                sample_patient_dir = Path(comp.iloc[0]['Patient_Dir'])
            else:
                return

        p_dir = Path(sample_patient_dir)
        p_id = p_dir.name

        # Load modalities
        mods = {}
        for m in ['t1', 't1ce', 't2', 'flair', 'seg']:
            match = list(p_dir.glob(f"*{m}.nii*"))
            if match:
                mods[m.upper()] = nib.load(str(match[0])).get_fdata(dtype=np.float32)

        if not mods:
            return

        # 1. Four-Panel Modality Comparison Figure for representative patient (Middle Axial Slice)
        fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=300)
        mid_z = mods['FLAIR'].shape[2] // 2 if 'FLAIR' in mods else 77

        for idx, m_name in enumerate(['T1', 'T1CE', 'T2', 'FLAIR']):
            ax = axes[idx]
            if m_name in mods:
                slc = mods[m_name][:, :, mid_z]
                # Rotate for standard neurological display orientation
                slc = np.rot90(slc)
                ax.imshow(slc, cmap='gray')
                ax.set_title(f"{m_name} Modality\n(Axial Slice {mid_z})", fontsize=12, fontweight='bold', pad=10)
            ax.axis('off')

        plt.suptitle(f"Multi-Sequence Brain MRI Scan Comparison — Patient {p_id}", fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.dirs['modality_comparison'] / 'modality_4panel_comparison.png', bbox_inches='tight')
        plt.close()

        # 2. Multi-View & Mask Overlay (Axial, Coronal, Sagittal)
        fig, axes = plt.subplots(3, 5, figsize=(18, 11), dpi=300)
        shape = mods['FLAIR'].shape
        cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2

        mod_keys = ['T1', 'T1CE', 'T2', 'FLAIR', 'SEG']
        for col, m_name in enumerate(mod_keys):
            if m_name not in mods:
                continue
            data = mods[m_name]
            cmap = 'gray' if m_name != 'SEG' else 'tab10'

            # Axial (Z)
            axes[0, col].imshow(np.rot90(data[:, :, cz]), cmap=cmap)
            axes[0, col].set_title(f"{m_name}\nAxial (Z={cz})", fontsize=11, fontweight='bold')
            axes[0, col].axis('off')

            # Coronal (Y)
            axes[1, col].imshow(np.rot90(data[:, cy, :]), cmap=cmap)
            axes[1, col].set_title(f"Coronal (Y={cy})", fontsize=10)
            axes[1, col].axis('off')

            # Sagittal (X)
            axes[2, col].imshow(np.rot90(data[cx, :, :]), cmap=cmap)
            axes[2, col].set_title(f"Sagittal (X={cx})", fontsize=10)
            axes[2, col].axis('off')

        plt.suptitle(f"Tri-Planar MRI Views & Ground Truth Segmentation — Patient {p_id}", fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.dirs['sample_images'] / 'sample_patient_triplanar.png', bbox_inches='tight')
        plt.close()

        # 3. Tumor Segmentation Overlay Visual (BraTS Color Coding)
        if 'FLAIR' in mods and 'SEG' in mods:
            flair_slc = np.rot90(mods['FLAIR'][:, :, cz])
            seg_slc = np.rot90(mods['SEG'][:, :, cz])

            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
            ax.imshow(flair_slc, cmap='gray')

            # Overlay mask with alpha: Red=Label 1 (NCR/NET), Green=Label 2 (ED), Yellow=Label 4 (ET)
            mask_rgb = np.zeros((*seg_slc.shape, 4), dtype=np.float32)
            mask_rgb[seg_slc == 1] = [1.0, 0.0, 0.0, 0.6]  # Red
            mask_rgb[seg_slc == 2] = [0.0, 0.8, 0.0, 0.5]  # Green
            mask_rgb[seg_slc == 4] = [1.0, 1.0, 0.0, 0.7]  # Yellow

            ax.imshow(mask_rgb)
            ax.set_title(f"FLAIR Volume with BraTS Pathological Tumor Labels Overlay\n(Patient: {p_id})", fontsize=13, fontweight='bold')
            
            # Custom Legend
            red_patch = mpatches.Patch(color='red', label='Label 1: Necrotic/Non-Enhancing Core (NCR/NET)')
            green_patch = mpatches.Patch(color='green', label='Label 2: Peritumoral Edema (ED)')
            yellow_patch = mpatches.Patch(color='yellow', label='Label 4: Enhancing Tumor Core (ET)')
            ax.legend(handles=[red_patch, green_patch, yellow_patch], loc='lower right', facecolor='#111111', edgecolor='white', labelcolor='white')
            
            ax.axis('off')
            plt.savefig(self.dirs['sample_images'] / 'tumor_mask_overlay.png', bbox_inches='tight')
            plt.close()

        # 4. Multi-Patient 4x4 Montage Image
        montage_patients = self.patient_df.head(16)
        fig, axes = plt.subplots(4, 4, figsize=(12, 12), dpi=300)
        axes = axes.flatten()

        for idx, (_, row) in enumerate(montage_patients.iterrows()):
            p_dir_idx = Path(row['Patient_Dir'])
            flair_files = list(p_dir_idx.glob("*flair.nii*"))
            if flair_files:
                f_data = nib.load(str(flair_files[0])).get_fdata(dtype=np.float32)
                mid_slice = np.rot90(f_data[:, :, f_data.shape[2] // 2])
                axes[idx].imshow(mid_slice, cmap='gray')
                axes[idx].set_title(row['Patient_ID'], fontsize=9, pad=4)
            axes[idx].axis('off')

        plt.suptitle("BraTS 2020 Dataset Representative FLAIR Slice Montage (16 Patients)", fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.dirs['sample_images'] / 'patient_montage.png', bbox_inches='tight')
        plt.close()

    def _plot_histograms(self):
        """Generates intensity distribution histograms and tumor label distributions."""
        img_df = self.properties_df[~self.properties_df['Modality'].str.lower().isin(['seg', 'seg_mask', 'segmentation'])]

        # 1. Modal Intensity Histograms (Individual Panels)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
        axes = axes.flatten()

        for idx, mod in enumerate(['T1', 'T1CE', 'T2', 'FLAIR']):
            mod_data = img_df[img_df['Modality'] == mod]
            ax = axes[idx]
            color = self.colors.get(mod, '#333333')

            ax.hist(mod_data['Mean_Intensity'], bins=25, color=color, alpha=0.7, edgecolor='black')
            ax.set_title(f"{mod} Mean Foreground Intensity Distribution", fontsize=11, fontweight='bold')
            ax.set_xlabel("Mean Intensity", fontsize=10)
            ax.set_ylabel("Patient Count", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)

        plt.suptitle("Intensity Distribution Across BraTS MRI Modalities", fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.dirs['histograms'] / 'intensity_histograms.png', bbox_inches='tight')
        plt.close()

        # 2. Combined Intensity KDE / Overlay Plot
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        for mod in ['T1', 'T1CE', 'T2', 'FLAIR']:
            mod_data = img_df[img_df['Modality'] == mod]['Mean_Intensity']
            ax.hist(mod_data, bins=30, alpha=0.4, label=mod, color=self.colors[mod], density=True)

        ax.set_title("Comparative Intensity Probability Density Across Modalities", fontsize=13, fontweight='bold')
        ax.set_xlabel("Mean Intensity Value", fontsize=11)
        ax.set_ylabel("Probability Density", fontsize=11)
        ax.legend(title="Modality", frameon=True)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(self.dirs['histograms'] / 'intensity_distribution_overlay.png', bbox_inches='tight')
        plt.close()

        # 3. Tumor Label Distribution Histogram
        seg_df = self.properties_df[self.properties_df['Modality'].str.lower().isin(['seg', 'seg_mask', 'segmentation'])]
        if not seg_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            label_names = ['NCR/NET (Label 1)', 'Edema (Label 2)', 'Enhancing Tumor (Label 4)']
            avg_vols = [
                seg_df['NCR_Volume_cm3'].mean(),
                seg_df['ED_Volume_cm3'].mean(),
                seg_df['ET_Volume_cm3'].mean()
            ]
            colors_bar = ['#d9534f', '#5cb85c', '#f0ad4e']

            bars = ax.bar(label_names, avg_vols, color=colors_bar, width=0.5, edgecolor='black')
            ax.set_title("Average Tumor Sub-Region Volume Distribution Across Dataset", fontsize=13, fontweight='bold')
            ax.set_ylabel("Mean Volume (cm³)", fontsize=11)
            ax.grid(axis='y', linestyle='--', alpha=0.5)

            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f} cm³',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold')

            plt.savefig(self.dirs['histograms'] / 'tumor_label_histogram.png', bbox_inches='tight')
            plt.close()

    def _plot_boxplots(self):
        """Generates comparative boxplots for Mean Intensity, Contrast, Entropy, Sharpness, Noise."""
        img_df = self.properties_df[~self.properties_df['Modality'].str.lower().isin(['seg', 'seg_mask', 'segmentation'])]
        modalities = ['T1', 'T1CE', 'T2', 'FLAIR']

        metrics = [
            ('Mean_Intensity', 'Mean Intensity', 'intensity_boxplot.png'),
            ('Contrast', 'RMS Contrast', 'contrast_boxplot.png'),
            ('Entropy', 'Shannon Entropy (bits)', 'entropy_boxplot.png'),
            ('Sharpness', 'Sharpness (Laplacian Variance)', 'sharpness_boxplot.png'),
            ('Noise_Estimate', 'Noise Estimation (MAD)', 'noise_boxplot.png'),
            ('SNR', 'Signal-to-Noise Ratio (SNR)', 'snr_boxplot.png')
        ]

        # Individual Boxplots
        for col_name, title, filename in metrics:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
            data_to_plot = [img_df[img_df['Modality'] == m][col_name].dropna() for m in modalities]

            bp = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=modalities)
            for patch, m in zip(bp['boxes'], modalities):
                patch.set_facecolor(self.colors[m])
                patch.set_alpha(0.7)

            ax.set_title(f"Comparative {title} Across MRI Modalities", fontsize=13, fontweight='bold')
            ax.set_ylabel(title, fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.savefig(self.dirs['boxplots'] / filename, bbox_inches='tight')
            plt.close()

        # Grid of All Metrics
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
        axes = axes.flatten()

        for idx, (col_name, title, _) in enumerate(metrics):
            ax = axes[idx]
            data_to_plot = [img_df[img_df['Modality'] == m][col_name].dropna() for m in modalities]
            bp = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=modalities)
            for patch, m in zip(bp['boxes'], modalities):
                patch.set_facecolor(self.colors[m])
                patch.set_alpha(0.7)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)

        plt.suptitle("Statistical & Quality Metric Distributions Across MRI Modalities", fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.dirs['boxplots'] / 'property_comparison_grid.png', bbox_inches='tight')
        plt.close()

    def _plot_modality_comparisons(self):
        """Generates property comparison bar charts and radar charts comparing T1, T1CE, T2, FLAIR."""
        img_df = self.properties_df[~self.properties_df['Modality'].str.lower().isin(['seg', 'seg_mask', 'segmentation'])]

        metrics = ['Mean_Intensity', 'Contrast', 'Entropy', 'Sharpness', 'Noise_Estimate', 'SNR', 'Edge_Strength']
        mean_metrics = img_df.groupby('Modality')[metrics].mean()

        # Bar chart comparison
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        # Normalize each metric to [0, 1] for relative visual comparison
        norm_metrics = (mean_metrics - mean_metrics.min()) / (mean_metrics.max() - mean_metrics.min() + 1e-8)

        x = np.arange(len(metrics))
        width = 0.18

        for idx, mod in enumerate(['T1', 'T1CE', 'T2', 'FLAIR']):
            vals = norm_metrics.loc[mod].values if mod in norm_metrics.index else np.zeros(len(metrics))
            ax.bar(x + idx * width, vals, width, label=mod, color=self.colors[mod], alpha=0.85)

        ax.set_title("Normalized Image Quality & Statistical Profile Comparison Across Modalities", fontsize=13, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([m.replace('_', ' ') for m in metrics], rotation=15, fontsize=10)
        ax.set_ylabel("Normalized Scale (0 to 1)", fontsize=11)
        ax.legend(title="Modality", frameon=True)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        plt.savefig(self.dirs['modality_comparison'] / 'modality_bar_comparison.png', bbox_inches='tight')
        plt.close()

    def _plot_resolution_analysis(self):
        """Generates charts for spatial resolution (image dimensions and voxel spacing)."""
        df = self.properties_df

        # 1. Dimensions Plot
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        dim_counts = df.groupby(['Width', 'Height', 'Depth']).size().reset_index(name='Count')
        dim_labels = [f"{r.Width}x{r.Height}x{r.Depth}" for _, r in dim_counts.iterrows()]

        bars = ax.bar(dim_labels, dim_counts['Count'], color='#4a90e2', width=0.4, edgecolor='black')
        ax.set_title("BraTS 2020 MRI Volume Dimension Distribution", fontsize=13, fontweight='bold')
        ax.set_xlabel("Volume Shape (W x H x D)", fontsize=11)
        ax.set_ylabel("Number of Volumes", fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

        plt.savefig(self.dirs['resolution_analysis'] / 'image_dimensions_distribution.png', bbox_inches='tight')
        plt.close()

        # 2. Voxel Spacing Plot
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        sp_counts = df.groupby(['Spacing_X', 'Spacing_Y', 'Spacing_Z']).size().reset_index(name='Count')
        sp_labels = [f"{r.Spacing_X}x{r.Spacing_Y}x{r.Spacing_Z} mm³" for _, r in sp_counts.iterrows()]

        bars = ax.bar(sp_labels, sp_counts['Count'], color='#50e3c2', width=0.4, edgecolor='black')
        ax.set_title("BraTS 2020 Isotropic Voxel Resolution Distribution", fontsize=13, fontweight='bold')
        ax.set_xlabel("Voxel Spacing (dx x dy x dz)", fontsize=11)
        ax.set_ylabel("Number of Volumes", fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

        plt.savefig(self.dirs['resolution_analysis'] / 'voxel_spacing_distribution.png', bbox_inches='tight')
        plt.close()

    def _plot_quality_analysis(self):
        """Generates charts for quality analysis (SNR distribution and data completeness)."""
        df = self.properties_df[~self.properties_df['Modality'].str.lower().isin(['seg', 'seg_mask', 'segmentation'])]

        # 1. SNR Distribution
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        for mod in ['T1', 'T1CE', 'T2', 'FLAIR']:
            snr_data = df[df['Modality'] == mod]['SNR']
            ax.hist(snr_data, bins=25, alpha=0.5, label=mod, color=self.colors[mod])

        ax.set_title("Signal-to-Noise Ratio (SNR) Distribution Across Modalities", fontsize=13, fontweight='bold')
        ax.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=11)
        ax.set_ylabel("Volume Count", fontsize=11)
        ax.legend(title="Modality")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(self.dirs['quality_analysis'] / 'snr_distribution.png', bbox_inches='tight')
        plt.close()

        # 2. Data Completeness Pie Chart
        fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
        status_counts = self.patient_df['Status'].value_counts()

        colors_pie = ['#2ecc71', '#e74c3c'] if 'Complete' in status_counts.index else ['#3498db']
        wedges, texts, autotexts = ax.pie(
            status_counts.values,
            labels=status_counts.index,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors_pie,
            explode=[0.05] + [0]*(len(status_counts)-1),
            textprops=dict(color="black", fontweight="bold")
        )
        ax.set_title("BraTS 2020 Dataset Completeness Status (369 Patients)", fontsize=13, fontweight='bold')
        plt.savefig(self.dirs['quality_analysis'] / 'quality_checks_summary.png', bbox_inches='tight')
        plt.close()
