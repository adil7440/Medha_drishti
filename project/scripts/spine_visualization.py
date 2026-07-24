from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


class SpineDatasetVisualizer:
    """
    Generates publication-quality visualizations for Stage 1 Spine MRI
    Dataset Analysis. Saves all figures as PNG files.
    """

    def __init__(self, properties_df, patient_df, figures_dir):
        self.properties_df = properties_df
        self.patient_df = patient_df
        self.figures_dir = Path(figures_dir)

        self.dirs = {
            'sample_images': self.figures_dir / 'sample_images',
            'histograms': self.figures_dir / 'histograms',
            'boxplots': self.figures_dir / 'boxplots',
            'modality_comparison': self.figures_dir / 'modality_comparison',
            'resolution_analysis': self.figures_dir / 'resolution_analysis',
            'quality_analysis': self.figures_dir / 'quality_analysis',
            'class_comparison': self.figures_dir / 'class_comparison',
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        self.colors = {
            'T1W': '#1f77b4',
            'T1W_GADO': '#17becf',
            'T2W': '#2ca02c',
            'STIR': '#d62728',
            'Survey': '#9467bd',
            'SPAIR': '#e377c2',
            'Other': '#7f7f7f',
        }

    def generate_all_figures(self):
        """Generates the full suite of analysis figures."""
        print("[SpineVisualizer] Generating sample image figures...")
        self._plot_sample_patient_and_montages()

        print("[SpineVisualizer] Generating intensity histograms...")
        self._plot_histograms()

        print("[SpineVisualizer] Generating boxplots...")
        self._plot_boxplots()

        print("[SpineVisualizer] Generating modality comparison figures...")
        self._plot_modality_comparisons()

        print("[SpineVisualizer] Generating resolution & spatial analysis...")
        self._plot_resolution_analysis()

        print("[SpineVisualizer] Generating quality analysis figures...")
        self._plot_quality_analysis()

        print("[SpineVisualizer] Generating class comparison figures...")
        self._plot_class_comparison()

        print("[SpineVisualizer] All figures generated successfully!")

    def _plot_sample_patient_and_montages(self):
        """Plots sample slices and multi-patient montages."""
        # Pick one Normal and one Pathological patient with T1W
        for cls_label, cmap_color in [('Normal', 'gray'), ('Pathological', 'gray')]:
            cls_patients = self.patient_df[self.patient_df['Class'] == cls_label]
            if cls_patients.empty:
                continue

            sample_row = cls_patients.iloc[0]
            p_dir = Path(sample_row['Patient_Dir'])
            p_id = sample_row['Patient_ID']

            # Find a T1W or T2W file
            t1_files = list(p_dir.glob("*T1W*.nii.gz"))
            t2_files = list(p_dir.glob("*T2W*.nii.gz"))

            mod_files = {}
            if t1_files:
                mod_files['T1W'] = t1_files[0]
            if t2_files:
                mod_files['T2W'] = t2_files[0]

            if not mod_files:
                continue

            n_mods = len(mod_files)
            fig, axes = plt.subplots(1, n_mods, figsize=(7 * n_mods, 6), dpi=300)
            if n_mods == 1:
                axes = [axes]

            for idx, (m_name, m_file) in enumerate(mod_files.items()):
                data = nib.load(str(m_file)).get_fdata(dtype=np.float32)
                mid_z = data.shape[2] // 2
                slc = np.rot90(data[:, :, mid_z])
                axes[idx].imshow(slc, cmap=cmap_color)
                axes[idx].set_title(
                    f"{m_name} Modality\n(Axial Slice {mid_z})",
                    fontsize=12, fontweight='bold', pad=10)
                axes[idx].axis('off')

            plt.suptitle(
                f"Spine MRI Scan — Patient {p_id} ({cls_label})",
                fontsize=15, fontweight='bold', y=1.02)
            plt.tight_layout()
            safe_label = cls_label.lower()
            plt.savefig(
                self.dirs['modality_comparison'] /
                f'{safe_label}_modality_comparison.png',
                bbox_inches='tight')
            plt.close()

        # Multi-Patient Montage (all patients, one representative slice each)
        n_patients = len(self.patient_df)
        n_cols = 5
        n_rows = (n_patients + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), dpi=300)
        axes = axes.flatten() if n_patients > 1 else [axes]

        for idx, (_, row) in enumerate(self.patient_df.iterrows()):
            p_dir = Path(row['Patient_Dir'])
            t2_files = list(p_dir.glob("*T2W*.nii.gz"))
            if t2_files:
                data = nib.load(str(t2_files[0])).get_fdata(dtype=np.float32)
                mid_slice = np.rot90(data[:, :, data.shape[2] // 2])
                axes[idx].imshow(mid_slice, cmap='gray')
            cls_tag = 'N' if row['Class'] == 'Normal' else 'P'
            axes[idx].set_title(f"{row['Patient_ID']}\n({cls_tag})",
                                fontsize=9, pad=4)
            axes[idx].axis('off')

        for idx in range(n_patients, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(
            "Spine MRI Dataset — Representative T2W Slice Montage (10 Patients)",
            fontsize=14, fontweight='bold', y=1.0)
        plt.tight_layout()
        plt.savefig(
            self.dirs['sample_images'] / 'patient_montage.png',
            bbox_inches='tight')
        plt.close()

    def _plot_histograms(self):
        """Generates intensity distribution histograms."""
        img_df = self.properties_df[
            ~self.properties_df['Modality_Category'].isin(['Survey', 'Other'])]

        modalities = ['T1W', 'T1W_GADO', 'T2W', 'STIR']
        avail = [m for m in modalities if m in img_df['Modality_Category'].values]

        if not avail:
            return

        n_mods = len(avail)
        ncols = min(n_mods, 2)
        nrows = (n_mods + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), dpi=300)
        if n_mods == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, mod in enumerate(avail):
            mod_data = img_df[img_df['Modality_Category'] == mod]
            ax = axes[idx]
            color = self.colors.get(mod, '#333333')
            ax.hist(mod_data['Mean_Intensity'], bins=25, color=color,
                    alpha=0.7, edgecolor='black')
            ax.set_title(f"{mod} Mean Foreground Intensity Distribution",
                         fontsize=11, fontweight='bold')
            ax.set_xlabel("Mean Intensity", fontsize=10)
            ax.set_ylabel("Volume Count", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)

        for idx in range(n_mods, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(
            "Intensity Distribution Across Spine MRI Modalities",
            fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(
            self.dirs['histograms'] / 'intensity_histograms.png',
            bbox_inches='tight')
        plt.close()

        # Combined overlay
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        for mod in avail:
            mod_data = img_df[img_df['Modality_Category'] == mod]['Mean_Intensity']
            ax.hist(mod_data, bins=30, alpha=0.4, label=mod,
                    color=self.colors.get(mod, '#333'), density=True)
        ax.set_title(
            "Comparative Intensity Probability Density — Spine MRI Modalities",
            fontsize=13, fontweight='bold')
        ax.set_xlabel("Mean Intensity Value", fontsize=11)
        ax.set_ylabel("Probability Density", fontsize=11)
        ax.legend(title="Modality", frameon=True)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(
            self.dirs['histograms'] / 'intensity_distribution_overlay.png',
            bbox_inches='tight')
        plt.close()

        # Per-class intensity distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
        for cls_idx, cls_label in enumerate(['Normal', 'Pathological']):
            ax = axes[cls_idx]
            cls_pids = self.patient_df[
                self.patient_df['Class'] == cls_label]['Patient_ID'].tolist()
            cls_data = img_df[img_df['Patient_ID'].isin(cls_pids)]
            for mod in avail:
                mod_sub = cls_data[cls_data['Modality_Category'] == mod]
                if not mod_sub.empty:
                    ax.hist(mod_sub['Mean_Intensity'], bins=25, alpha=0.5,
                            label=mod, color=self.colors.get(mod, '#333'),
                            density=True)
            ax.set_title(f"{cls_label} Patients — Intensity Distribution",
                         fontsize=12, fontweight='bold')
            ax.set_xlabel("Mean Intensity", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.legend(title="Modality", fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.5)

        plt.suptitle(
            "Intensity Distribution by Patient Class",
            fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(
            self.dirs['histograms'] / 'intensity_by_class.png',
            bbox_inches='tight')
        plt.close()

    def _plot_boxplots(self):
        """Generates comparative boxplots for image quality metrics."""
        img_df = self.properties_df[
            ~self.properties_df['Modality_Category'].isin(['Survey', 'Other'])]
        modalities = [m for m in ['T1W', 'T1W_GADO', 'T2W', 'STIR']
                      if m in img_df['Modality_Category'].values]

        if not modalities:
            return

        metrics = [
            ('Mean_Intensity', 'Mean Intensity', 'intensity_boxplot.png'),
            ('Contrast', 'RMS Contrast', 'contrast_boxplot.png'),
            ('Entropy', 'Shannon Entropy (bits)', 'entropy_boxplot.png'),
            ('Sharpness', 'Sharpness (Laplacian Variance)', 'sharpness_boxplot.png'),
            ('Noise_Estimate', 'Noise Estimation (MAD)', 'noise_boxplot.png'),
            ('SNR', 'Signal-to-Noise Ratio (SNR)', 'snr_boxplot.png'),
        ]

        for col_name, title, filename in metrics:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
            data_to_plot = [img_df[img_df['Modality_Category'] == m][col_name].dropna()
                            for m in modalities]
            bp = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=modalities)
            for patch, m in zip(bp['boxes'], modalities):
                patch.set_facecolor(self.colors.get(m, '#999'))
                patch.set_alpha(0.7)
            ax.set_title(f"Comparative {title} Across Spine MRI Modalities",
                         fontsize=13, fontweight='bold')
            ax.set_ylabel(title, fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.savefig(self.dirs['boxplots'] / filename, bbox_inches='tight')
            plt.close()

        # Grid of all metrics
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
        axes = axes.flatten()
        for idx, (col_name, title, _) in enumerate(metrics):
            ax = axes[idx]
            data_to_plot = [img_df[img_df['Modality_Category'] == m][col_name].dropna()
                            for m in modalities]
            bp = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=modalities)
            for patch, m in zip(bp['boxes'], modalities):
                patch.set_facecolor(self.colors.get(m, '#999'))
                patch.set_alpha(0.7)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)

        plt.suptitle(
            "Statistical & Quality Metrics Across Spine MRI Modalities",
            fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(
            self.dirs['boxplots'] / 'property_comparison_grid.png',
            bbox_inches='tight')
        plt.close()

        # Boxplots: Normal vs Pathological for key metrics
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
        axes_flat = axes.flatten()
        class_colors = {'Normal': '#3498db', 'Pathological': '#e74c3c'}

        for idx, (col_name, title, _) in enumerate(metrics):
            ax = axes_flat[idx]
            data_list = []
            labels = []
            for cls in ['Normal', 'Pathological']:
                pids = self.patient_df[
                    self.patient_df['Class'] == cls]['Patient_ID'].tolist()
                subset = img_df[img_df['Patient_ID'].isin(pids)][col_name].dropna()
                data_list.append(subset)
                labels.append(cls)
            bp = ax.boxplot(data_list, patch_artist=True, tick_labels=labels)
            for patch, lbl in zip(bp['boxes'], labels):
                patch.set_facecolor(class_colors[lbl])
                patch.set_alpha(0.7)
            ax.set_title(f"{title} — Normal vs Pathological",
                         fontsize=11, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)

        plt.suptitle(
            "Normal vs Pathological: MRI Quality Metrics Comparison",
            fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(
            self.dirs['class_comparison'] / 'normal_vs_pathological_boxplots.png',
            bbox_inches='tight')
        plt.close()

    def _plot_modality_comparisons(self):
        """Generates bar chart and radar-style comparison of modalities."""
        img_df = self.properties_df[
            ~self.properties_df['Modality_Category'].isin(['Survey', 'Other'])]

        metrics = ['Mean_Intensity', 'Contrast', 'Entropy', 'Sharpness',
                    'Noise_Estimate', 'SNR', 'Edge_Strength']
        mean_metrics = img_df.groupby('Modality_Category')[metrics].mean()
        modalities = [m for m in mean_metrics.index if m != 'Other']

        if len(modalities) < 2:
            return

        # Bar chart
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        norm_metrics = (mean_metrics - mean_metrics.min()) / (
                mean_metrics.max() - mean_metrics.min() + 1e-8)
        x = np.arange(len(metrics))
        width = 0.18

        for idx, mod in enumerate(modalities):
            vals = norm_metrics.loc[mod].values if mod in norm_metrics.index else np.zeros(len(metrics))
            ax.bar(x + idx * width, vals, width, label=mod,
                   color=self.colors.get(mod, '#999'), alpha=0.85)

        ax.set_title(
            "Normalized Image Quality Profile — Spine MRI Modalities",
            fontsize=13, fontweight='bold')
        ax.set_xticks(x + width * (len(modalities) - 1) / 2)
        ax.set_xticklabels([m.replace('_', ' ') for m in metrics],
                           rotation=15, fontsize=10)
        ax.set_ylabel("Normalized Scale (0 to 1)", fontsize=11)
        ax.legend(title="Modality", frameon=True)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.savefig(
            self.dirs['modality_comparison'] / 'modality_bar_comparison.png',
            bbox_inches='tight')
        plt.close()

    def _plot_resolution_analysis(self):
        """Generates charts for spatial resolution."""
        df = self.properties_df

        # Dimensions
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        dim_counts = df.groupby(['Width', 'Height', 'Depth']).size().reset_index(name='Count')
        dim_labels = [f"{r.Width}x{r.Height}x{r.Depth}" for _, r in dim_counts.iterrows()]
        bars = ax.bar(dim_labels, dim_counts['Count'], color='#4a90e2',
                      width=0.4, edgecolor='black')
        ax.set_title("Spine MRI Volume Dimension Distribution",
                     fontsize=13, fontweight='bold')
        ax.set_xlabel("Volume Shape (W x H x D)", fontsize=11)
        ax.set_ylabel("Number of Volumes", fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        plt.xticks(rotation=30, ha='right')
        plt.savefig(
            self.dirs['resolution_analysis'] / 'image_dimensions_distribution.png',
            bbox_inches='tight')
        plt.close()

        # Voxel Spacing
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        sp_counts = df.groupby(
            ['Spacing_X', 'Spacing_Y', 'Spacing_Z']).size().reset_index(name='Count')
        sp_labels = [f"{r.Spacing_X}x{r.Spacing_Y}x{r.Spacing_Z} mm"
                     for _, r in sp_counts.iterrows()]
        bars = ax.bar(sp_labels, sp_counts['Count'], color='#50e3c2',
                      width=0.4, edgecolor='black')
        ax.set_title("Spine MRI Voxel Resolution Distribution",
                     fontsize=13, fontweight='bold')
        ax.set_xlabel("Voxel Spacing (dx x dy x dz)", fontsize=11)
        ax.set_ylabel("Number of Volumes", fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        plt.xticks(rotation=30, ha='right')
        plt.savefig(
            self.dirs['resolution_analysis'] / 'voxel_spacing_distribution.png',
            bbox_inches='tight')
        plt.close()

        # File size distribution per patient
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        patient_sizes = df.groupby('Patient_ID')['File_Size_MB'].sum().sort_values()
        cls_map = dict(zip(self.patient_df['Patient_ID'],
                           self.patient_df['Class']))
        bar_colors = ['#3498db' if cls_map.get(p) == 'Normal' else '#e74c3c'
                      for p in patient_sizes.index]
        bars = ax.barh(patient_sizes.index, patient_sizes.values, color=bar_colors,
                       edgecolor='black')
        ax.set_title("Total Dataset Size per Patient",
                     fontsize=13, fontweight='bold')
        ax.set_xlabel("Total Size (MB)", fontsize=11)
        blue_patch = mpatches.Patch(color='#3498db', label='Normal')
        red_patch = mpatches.Patch(color='#e74c3c', label='Pathological')
        ax.legend(handles=[blue_patch, red_patch], loc='lower right')
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        plt.savefig(
            self.dirs['resolution_analysis'] / 'patient_size_distribution.png',
            bbox_inches='tight')
        plt.close()

    def _plot_quality_analysis(self):
        """Generates quality analysis charts."""
        df = self.properties_df[
            ~self.properties_df['Modality_Category'].isin(['Survey', 'Other'])]

        # SNR Distribution
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        avail_mods = [m for m in ['T1W', 'T2W', 'STIR']
                      if m in df['Modality_Category'].values]
        for mod in avail_mods:
            snr_data = df[df['Modality_Category'] == mod]['SNR']
            ax.hist(snr_data, bins=25, alpha=0.5, label=mod,
                    color=self.colors.get(mod, '#333'))
        ax.set_title("SNR Distribution Across Spine MRI Modalities",
                     fontsize=13, fontweight='bold')
        ax.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=11)
        ax.set_ylabel("Volume Count", fontsize=11)
        ax.legend(title="Modality")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(
            self.dirs['quality_analysis'] / 'snr_distribution.png',
            bbox_inches='tight')
        plt.close()

        # Data Completeness Pie
        fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
        status_counts = self.patient_df['Status'].value_counts()
        colors_pie = ['#2ecc71', '#e74c3c'] if 'Complete' in status_counts.index else ['#3498db']
        wedges, texts, autotexts = ax.pie(
            status_counts.values, labels=status_counts.index,
            autopct='%1.1f%%', startangle=140, colors=colors_pie,
            explode=[0.05] + [0] * (len(status_counts) - 1),
            textprops=dict(color="black", fontweight="bold"))
        ax.set_title("Spine MRI Dataset Completeness Status",
                     fontsize=13, fontweight='bold')
        plt.savefig(
            self.dirs['quality_analysis'] / 'quality_checks_summary.png',
            bbox_inches='tight')
        plt.close()

        # Noise vs SNR scatter
        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
        for cls_label, marker, color in [
            ('Normal', 'o', '#3498db'), ('Pathological', 's', '#e74c3c')
        ]:
            pids = self.patient_df[
                self.patient_df['Class'] == cls_label]['Patient_ID'].tolist()
            subset = df[df['Patient_ID'].isin(pids)]
            ax.scatter(subset['Noise_Estimate'], subset['SNR'],
                       marker=marker, c=color, alpha=0.6, label=cls_label,
                       edgecolors='black', linewidths=0.5)
        ax.set_title("Noise Estimate vs SNR — Normal vs Pathological",
                     fontsize=13, fontweight='bold')
        ax.set_xlabel("Noise Estimate (MAD)", fontsize=11)
        ax.set_ylabel("SNR", fontsize=11)
        ax.legend(title="Class")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(
            self.dirs['quality_analysis'] / 'noise_vs_snr_scatter.png',
            bbox_inches='tight')
        plt.close()

    def _plot_class_comparison(self):
        """Generates bar charts comparing Normal vs Pathological statistics."""
        img_df = self.properties_df[
            ~self.properties_df['Modality_Category'].isin(['Survey', 'Other'])]

        metrics = ['Mean_Intensity', 'Contrast', 'Entropy', 'Sharpness',
                    'SNR', 'Noise_Estimate']

        normal_pids = self.patient_df[
            self.patient_df['Class'] == 'Normal']['Patient_ID'].tolist()
        path_pids = self.patient_df[
            self.patient_df['Class'] == 'Pathological']['Patient_ID'].tolist()

        normal_data = img_df[img_df['Patient_ID'].isin(normal_pids)]
        path_data = img_df[img_df['Patient_ID'].isin(path_pids)]

        normal_means = [normal_data[m].mean() for m in metrics]
        path_means = [path_data[m].mean() for m in metrics]

        # Normalize for visual comparison
        max_vals = [max(n, p, 1e-8) for n, p in zip(normal_means, path_means)]
        norm_vals = [n / mx for n, mx in zip(normal_means, max_vals)]
        path_vals = [p / mx for p, mx in zip(path_means, max_vals)]

        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        x = np.arange(len(metrics))
        width = 0.35
        ax.bar(x - width / 2, norm_vals, width, label='Normal',
               color='#3498db', alpha=0.85)
        ax.bar(x + width / 2, path_vals, width, label='Pathological',
               color='#e74c3c', alpha=0.85)
        ax.set_title(
            "Normalized MRI Quality Metrics — Normal vs Pathological",
            fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ') for m in metrics],
                           rotation=15, fontsize=10)
        ax.set_ylabel("Normalized Scale (0 to 1)", fontsize=11)
        ax.legend(title="Class", frameon=True)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.savefig(
            self.dirs['class_comparison'] / 'class_bar_comparison.png',
            bbox_inches='tight')
        plt.close()

        # File count comparison
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        cls_file_counts = img_df.groupby(
            ['Patient_ID', 'Modality_Category']).size().unstack(fill_value=0)
        cls_labels_map = dict(zip(self.patient_df['Patient_ID'],
                                  self.patient_df['Class']))
        cls_file_counts['Class'] = cls_file_counts.index.map(cls_labels_map)

        for cls_label, color in [('Normal', '#3498db'), ('Pathological', '#e74c3c')]:
            cls_sub = cls_file_counts[cls_file_counts['Class'] == cls_label]
            if not cls_sub.empty:
                cls_sub.drop(columns='Class').sum().plot(
                    kind='bar', ax=ax, color=[self.colors.get(c, '#999')
                                              for c in cls_sub.drop(columns='Class').columns],
                    alpha=0.7, label=cls_label, position=0 if cls_label == 'Normal' else 1)

        ax.set_title("Modality File Count by Patient Class",
                     fontsize=13, fontweight='bold')
        ax.set_ylabel("File Count", fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.xticks(rotation=30, ha='right')
        plt.savefig(
            self.dirs['class_comparison'] / 'modality_count_by_class.png',
            bbox_inches='tight')
        plt.close()
