import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
import datetime

class Stage4ReportGenerator:
    """Generates a professional clinical PDF report for Stage 4."""
    
    @staticmethod
    def generate_report(result_dict: dict, image_paths: dict, output_pdf_path: str):
        """
        result_dict: {patient_id, modality, diagnosis, disease_class, confidence, measurements, shape, voxel_spacing}
        image_paths: {"original": str, "enhanced": str, "segmented": str, "heatmap": str}
        """
        Path(output_pdf_path).parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(str(output_pdf_path), pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = styles['Heading1']
        title_style.alignment = 1 # Center
        h2 = styles['Heading2']
        normal = styles['Normal']
        
        elements = []
        
        # Header
        elements.append(Paragraph("AI-Assisted Medical MRI Analysis Report", title_style))
        elements.append(Paragraph(f"Date Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", normal))
        elements.append(Spacer(1, 12))
        
        # Patient Details Table
        data = [
            ["Patient ID:", result_dict.get('patient_id', 'Unknown'), "Modality:", result_dict.get('modality', 'Unknown')],
            ["Volume Shape:", str(result_dict.get('shape', 'N/A')), "Voxel Spacing:", str(result_dict.get('voxel_spacing', 'N/A'))],
            ["Diagnosis:", result_dict.get('diagnosis', 'N/A'), "Confidence:", f"{result_dict.get('confidence', 0) * 100:.1f}%"]
        ]
        
        t = Table(data, colWidths=[100, 150, 100, 150])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('BACKGROUND', (0,2), (3,2), colors.lightblue),
            ('FONTNAME', (0,2), (3,2), 'Helvetica-Bold')
        ]))
        elements.append(t)
        elements.append(Spacer(1, 20))
        
        # Measurements
        elements.append(Paragraph("Clinical Measurements", h2))
        meas = result_dict.get('measurements', {})
        meas_data = [
            ["Measurement", "Value"],
            ["Total Volume", f"{meas.get('Volume_mm3', 0)} mm³"],
            ["Max Area (Cross Section)", f"{meas.get('Area_mm2_max_slice', 0)} mm²"],
            ["Approx. Max Diameter", f"{meas.get('Max_Diameter_mm', 0)} mm"],
            ["Number of Lesions", str(meas.get('Num_Lesions', 0))],
            ["Lesion Centroid (Z,Y,X)", str(meas.get('Centroid', (0,0,0)))],
            ["Most Affected Slice (Z)", str(meas.get('Most_Affected_Slice', 0))]
        ]
        
        mt = Table(meas_data, colWidths=[200, 200])
        mt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (1,0), colors.HexColor('#0ea5e9')),
            ('TEXTCOLOR', (0,0), (1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        elements.append(mt)
        elements.append(Spacer(1, 20))
        
        # Images Section
        elements.append(Paragraph("Visualizations", h2))
        
        # Row 1: Original vs Enhanced
        img_row_1 = []
        if 'original' in image_paths and os.path.exists(image_paths['original']):
            img_row_1.append(Image(image_paths['original'], width=200, height=200))
        if 'enhanced' in image_paths and os.path.exists(image_paths['enhanced']):
            img_row_1.append(Image(image_paths['enhanced'], width=200, height=200))
            
        if img_row_1:
            img_t1 = Table([img_row_1])
            elements.append(img_t1)
            
        elements.append(Spacer(1, 10))
        
        # Row 2: Segmented vs Heatmap
        img_row_2 = []
        if 'segmented' in image_paths and os.path.exists(image_paths['segmented']):
            img_row_2.append(Image(image_paths['segmented'], width=200, height=200))
        if 'heatmap' in image_paths and os.path.exists(image_paths['heatmap']):
            img_row_2.append(Image(image_paths['heatmap'], width=200, height=200))
            
        if img_row_2:
            img_t2 = Table([img_row_2])
            elements.append(img_t2)
            
        # Clinical Summary
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Clinical Summary & Recommendations", h2))
        summary_text = (f"The AI model analyzed the {result_dict.get('modality', '')} MRI "
                        f"and predicts '{result_dict.get('diagnosis', 'Unknown')}' with "
                        f"{(result_dict.get('confidence', 0)*100):.1f}% confidence. "
                        f"The primary affected region is centered at slice {meas.get('Most_Affected_Slice', 0)} "
                        f"with an approximate volume of {meas.get('Volume_mm3', 0)} mm³.")
        elements.append(Paragraph(summary_text, normal))
        
        # Build PDF
        doc.build(elements)
        print(f"[Done] Clinical Report generated at: {output_pdf_path}")
        
if __name__ == "__main__":
    # Test
    pass
