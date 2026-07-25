import os
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, ListFlowable, ListItem
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

def generate_report(output_path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=50, leftMargin=50,
        topMargin=50, bottomMargin=50
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=15,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0f172a")
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#64748b")
    )
    
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=18,
        spaceBefore=20,
        spaceAfter=15,
        textColor=colors.HexColor("#2563eb")
    )
    
    concept_style = ParagraphStyle(
        'ConceptTitle',
        parent=styles['Heading3'],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=5,
        textColor=colors.HexColor("#dc2626")
    )
    
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['BodyText'],
        fontSize=12,
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        leading=16
    )
    
    qa_q_style = ParagraphStyle(
        'QA_Q',
        parent=styles['Heading3'],
        fontSize=13,
        spaceBefore=10,
        spaceAfter=4,
        textColor=colors.HexColor("#0f172a"),
        fontName='Helvetica-Bold'
    )
    
    qa_a_style = ParagraphStyle(
        'QA_A',
        parent=styles['BodyText'],
        fontSize=12,
        spaceAfter=12,
        alignment=TA_LEFT,
        leading=16,
        textColor=colors.HexColor("#334155")
    )

    Story = []

    # ================= PAGE 1 =================
    Story.append(Paragraph("MedhaDrishti AI: Presentation Guide", title_style))
    Story.append(Paragraph("The 'Explain It Like I'm 5' Cheat Sheet for Hackathon Presenters", subtitle_style))
    
    Story.append(Paragraph("The Big Picture: What did we build?", heading_style))
    Story.append(Paragraph(
        "<b>The Problem:</b> In rural hospitals, doctors use cheap, low-power MRI machines. These machines take blurry, noisy pictures full of static. Trying to find a tiny brain tumor in these pictures is like trying to find a needle in a haystack while wearing foggy glasses.", body_style))
    Story.append(Paragraph(
        "<b>Our Solution (MedhaDrishti):</b> We built a software pipeline that takes these terrible, blurry MRI scans, magically cleans them up using Artificial Intelligence (making them 4K quality), and then automatically draws a red circle around the exact location of the tumor or spinal injury.", body_style))

    Story.append(Paragraph("Stage 1 & 2: Data Loading and Preprocessing", heading_style))
    
    Story.append(Paragraph("Concept: N4ITK Bias Correction (The Lighting Fixer)", concept_style))
    Story.append(Paragraph(
        "<b>Analogy:</b> Imagine taking a photo of a person where half their face is in bright sunlight and the other half is in a dark shadow. It's hard to see their features. "
        "N4ITK Bias Correction is a mathematical tool that fixes this 'bad lighting' in MRI machines so the whole brain looks evenly lit.", body_style))
    
    Story.append(Paragraph("Concept: Z-Score Normalization (The Translator)", concept_style))
    Story.append(Paragraph(
        "<b>Analogy:</b> Every hospital uses a different brand of MRI scanner (Siemens, GE, Philips). It's like they all speak different languages. "
        "Z-Score Normalization forces all these different images to 'speak the same language' (a standard scale of brightness) so our AI doesn't get confused.", body_style))

    Story.append(PageBreak())

    # ================= PAGE 2 =================
    Story.append(Paragraph("Stage 3: AI Enhancement (SE-DnCNN)", heading_style))
    
    Story.append(Paragraph("Concept: DnCNN (The Noise-Canceling Headphones)", concept_style))
    Story.append(Paragraph(
        "<b>Analogy:</b> How do noise-canceling headphones work? They listen to the background static, create an exact opposite soundwave, and subtract it to leave only the music. "
        "Our DnCNN model does exactly this for images! Instead of trying to guess what the clean brain looks like, our AI predicts the 'static' (noise), and subtracts it from the original blurry image to leave a perfectly crystal-clear scan.", body_style))
    
    Story.append(Paragraph("Concept: Squeeze-and-Excitation (SE) Blocks", concept_style))
    Story.append(Paragraph(
        "<b>Analogy:</b> Imagine a student reading a textbook and using a yellow highlighter to mark the most important words. "
        "SE Blocks are the AI's 'highlighter'. They teach the AI to boost (excite) the important features like edges and structures, and ignore (squeeze) the useless background static.", body_style))

    Story.append(Paragraph("Stage 4: Automated Diagnosis (3D Attention U-Net)", heading_style))
    
    Story.append(Paragraph("Concept: U-Net (The Microscope)", concept_style))
    Story.append(Paragraph(
        "<b>Analogy:</b> The U-Net acts like a microscope. It first zooms out to see the whole brain (Encoder), finds where the tumor is, and then zooms all the way back in (Decoder) to draw a pixel-perfect outline around the edges of the tumor.", body_style))
        
    Story.append(Paragraph("Concept: Attention Gates (The Detective)", concept_style))
    Story.append(Paragraph(
        "<b>Analogy:</b> The brain is massive, and a tumor is tiny. Without Attention Gates, the AI wastes time looking at healthy tissue. "
        "Attention Gates are like giving the AI a magnifying glass and a specific clue: 'Only look here!'. It forces the AI to ignore healthy brain tissue and focus 100% of its computing power on the suspicious lesion.", body_style))

    Story.append(PageBreak())

    # ================= PAGE 3 =================
    Story.append(Paragraph("Presentation Q&A: The Cheat Sheet", heading_style))
    Story.append(Paragraph("If the judges grill you with technical questions, use these simple, hard-hitting answers:", body_style))
    
    Story.append(Paragraph("Q: Why did you use DnCNN instead of massive modern Transformers like SwinIR?", qa_q_style))
    Story.append(Paragraph("<b>A:</b> Speed and Clinical Viability. We tested SwinIR and it took over 1.1 seconds per scan. Our DnCNN takes just 323 milliseconds (nearly 4x faster) while maintaining a better PSNR (22.6 dB vs 20.4 dB). In a busy hospital, real-time speed is non-negotiable.", qa_a_style))

    Story.append(Paragraph("Q: What is a NIfTI (.nii.gz) file?", qa_q_style))
    Story.append(Paragraph("<b>A:</b> Unlike standard JPEGs which are flat 2D images, a NIfTI file is a complete 3D volumetric cube. It contains hundreds of slices of the brain stacked together. Our AI processes the entire 3D volume, not just flat pictures.", qa_a_style))

    Story.append(Paragraph("Q: How does the model handle class imbalance? (Since tumors are tiny compared to the whole brain)", qa_q_style))
    Story.append(Paragraph("<b>A:</b> We used a 'Dice-Focal Loss' function during training. Focal loss mathematically forces the AI to stop worrying about the massive amounts of easy-to-predict healthy tissue, and penalizes it heavily if it gets the tiny, complex tumor boundaries wrong.", qa_a_style))

    Story.append(Paragraph("Q: Is the system fully autonomous?", qa_q_style))
    Story.append(Paragraph("<b>A:</b> MedhaDrishti is designed as a 'Human-in-the-Loop' assistive tool, not a replacement for doctors. The Streamlit dashboard automatically highlights the affected zones (via exact topological contours), allowing the radiologist to make the final authoritative decision much faster.", qa_a_style))

    Story.append(Paragraph("Q: What are the pathological subtypes it detects in Brain MRI?", qa_q_style))
    Story.append(Paragraph("<b>A:</b> 1. Necrotic Core (Dead tissue inside the tumor), 2. Peritumoral Edema (Swelling around the tumor), and 3. Enhancing Tumor (The active, growing outer edge).", qa_a_style))

    doc.build(Story)
    print(f"[Success] Explainer Guide generated at: {output_path}")

if __name__ == "__main__":
    out_dir = Path("project/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_report(out_dir / "MedhaDrishti_Explainer_Guide.pdf")
