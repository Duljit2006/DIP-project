import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

# Output presentation path
PPTX_PATH = "presentation.pptx"

# Initialize 16:9 Widescreen Presentation
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]  # Blank slide layout

# ==============================================================================
# Theme Palette: Light Theme with Blue Accents
# ==============================================================================
# Backgrounds
C_BG_WHITE       = RGBColor(255, 255, 255)  # #FFFFFF
C_BG_SLATE_TINT  = RGBColor(248, 250, 252)  # #F8FAFC
C_BG_BLUE_TINT   = RGBColor(240, 247, 255)  # #F0F7FF
C_BG_CARD_BLUE   = RGBColor(235, 245, 255)  # #EBF5FF

# Primary & Accents
C_NAVY_TITLE     = RGBColor(15, 41, 66)     # #0F2942
C_ROYAL_BLUE     = RGBColor(29, 78, 216)    # #1D4ED8
C_ACCENT_BLUE    = RGBColor(59, 130, 246)   # #3B82F6
C_BORDER_LIGHT   = RGBColor(219, 234, 254)  # #DBEAFE
C_BORDER_BLUE    = RGBColor(147, 197, 253)  # #93C5FD

# Text
C_TEXT_DARK      = RGBColor(30, 41, 59)     # #1E293B
C_TEXT_MUTED     = RGBColor(100, 116, 139)  # #64748B
C_TEXT_WHITE     = RGBColor(255, 255, 255)

# Status Highlights
C_SUCCESS_GREEN  = RGBColor(16, 185, 129)   # #10B981
C_SUCCESS_DARK   = RGBColor(22, 101, 52)    # #166534
C_SUCCESS_BG     = RGBColor(240, 253, 244)  # #F0FDF4
C_SUCCESS_BORDER = RGBColor(134, 239, 172)  # #86EFAC

C_ALERT_RED      = RGBColor(239, 68, 68)    # #EF4444
C_ALERT_BG       = RGBColor(254, 242, 242)  # #FEF2F2
C_ALERT_BORDER   = RGBColor(254, 202, 202)  # #FECACA

C_WARN_AMBER     = RGBColor(245, 158, 11)   # #F59E0B
C_WARN_BG        = RGBColor(255, 251, 235)  # #FFFBEB
C_WARN_BORDER    = RGBColor(253, 230, 138)  # #FDE68A

# Strict 2 Fonts
FONT_TITLE = "Segoe UI"
FONT_BODY  = "Calibri"

# ==============================================================================
# Helper UI Functions
# ==============================================================================

def set_slide_background(slide, color):
    bg_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg_shape.fill.solid()
    bg_shape.fill.fore_color.rgb = color
    bg_shape.line.fill.background() # no border

def add_header(slide, title_text, category_tag="DIGITAL IMAGE PROCESSING", subtitle_text=None):
    # Category Pill
    pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.4), Inches(2.8), Inches(0.32))
    pill.fill.solid()
    pill.fill.fore_color.rgb = C_BG_CARD_BLUE
    pill.line.color.rgb = C_BORDER_BLUE
    pill.line.width = Pt(1)
    
    tf = pill.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = category_tag.upper()
    p.font.name = FONT_TITLE
    p.font.size = Pt(9.5)
    p.font.bold = True
    p.font.color.rgb = C_ROYAL_BLUE
    p.alignment = PP_ALIGN.CENTER
    
    # Title Text
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.65))
    tf2 = tb.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = title_text
    p2.font.name = FONT_TITLE
    p2.font.size = Pt(24)
    p2.font.bold = True
    p2.font.color.rgb = C_NAVY_TITLE

    # Optional Subtitle
    if subtitle_text:
        p3 = tf2.add_paragraph()
        p3.text = subtitle_text
        p3.font.name = FONT_BODY
        p3.font.size = Pt(12)
        p3.font.color.rgb = C_TEXT_MUTED
        p3.space_before = Pt(3)

def add_card(slide, left, top, width, height, bg_color=C_BG_WHITE, border_color=C_BORDER_LIGHT, corner_radius=True):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if corner_radius else MSO_SHAPE.RECTANGLE
    card = slide.shapes.add_shape(shape_type, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1.2)
    return card

# ==============================================================================
# SLIDE 1: Title Slide (Split Hero Layout)
# ==============================================================================
slide1 = prs.slides.add_slide(blank_layout)
set_slide_background(slide1, C_BG_WHITE)

# Left Side: Project Identity & Titles
title_card = add_card(slide1, Inches(0.8), Inches(1.0), Inches(6.8), Inches(5.5), C_BG_BLUE_TINT, C_BORDER_BLUE)

tb = slide1.shapes.add_textbox(Inches(1.2), Inches(1.3), Inches(6.0), Inches(4.8))
tf = tb.text_frame
tf.word_wrap = True

# Tag
p_tag = tf.paragraphs[0]
p_tag.text = "RESEARCH PROJECT DEFENSE"
p_tag.font.name = FONT_TITLE
p_tag.font.size = Pt(11)
p_tag.font.bold = True
p_tag.font.color.rgb = C_ROYAL_BLUE

# Main Title
p_title = tf.add_paragraph()
p_title.text = "Dual-Domain JPEG\nArtifact Restoration"
p_title.font.name = FONT_TITLE
p_title.font.size = Pt(34)
p_title.font.bold = True
p_title.font.color.rgb = C_NAVY_TITLE
p_title.space_before = Pt(10)

# Subtitle
p_sub = tf.add_paragraph()
p_sub.text = "Harmonizing Spatial Convolutional Networks with Spectral Graph Message Passing"
p_sub.font.name = FONT_BODY
p_sub.font.size = Pt(14)
p_sub.font.color.rgb = C_TEXT_DARK
p_sub.space_before = Pt(10)

# Highlights Pill Box
p_hl = tf.add_paragraph()
p_hl.text = "★ Best Benchmark Gain: +4.13 dB (Set5 Butterfly) | Zero Spectral Drift"
p_hl.font.name = FONT_BODY
p_hl.font.size = Pt(11.5)
p_hl.font.bold = True
p_hl.font.color.rgb = C_SUCCESS_DARK
p_hl.space_before = Pt(20)

# Team / Guide
p_team = tf.add_paragraph()
p_team.text = "Presented by: DIP Project Team  |  Guide: Research Committee"
p_team.font.name = FONT_BODY
p_team.font.size = Pt(11)
p_team.font.color.rgb = C_TEXT_MUTED
p_team.space_before = Pt(24)

# Right Side: AI Generated 3D Hero Visual
if os.path.exists("assets/title_hero.jpg"):
    img_card = add_card(slide1, Inches(7.8), Inches(1.0), Inches(4.7), Inches(5.5), C_BG_WHITE, C_BORDER_BLUE)
    slide1.shapes.add_picture("assets/title_hero.jpg", Inches(7.9), Inches(1.1), Inches(4.5), Inches(5.3))


# ==============================================================================
# SLIDE 2: The Core Dilemma (3-Column Card Layout)
# ==============================================================================
slide2 = prs.slides.add_slide(blank_layout)
set_slide_background(slide2, C_BG_SLATE_TINT)
add_header(slide2, "The Real-World Dilemma: If Lossless Formats Exist, Why Does JPEG Still Dominate?",
           "PROBLEM MOTIVATION", "PNG and WebP exist, yet over 75% of internet images remain compressed in JPEG.")

cards_data_s2 = [
    ("1. Global Edge Bandwidth",
     "• Mobile networks, social media (WhatsApp, Instagram), and satellites must move billions of images daily.\n\n• Lossless formats (PNG, RAW) produce 5× to 10× larger files, resulting in severe latency and bandwidth bottlenecks.",
     C_BG_CARD_BLUE, C_ROYAL_BLUE),
    ("2. The Unavoidable Trade-Off",
     "• To achieve high compression ratios, JPEG discards high-frequency mathematical data.\n\n• This aggressive loss causes severe visual degradation: muddy blurred edges and ugly 8×8 block checkerboard seams.",
     C_WARN_BG, C_WARN_AMBER),
    ("3. Our Research Mission",
     "• Can we transmit tiny, highly compressed JPEG files over the network...\n\n• ...and use Dual-Domain AI to mathematically reconstruct the missing high-frequency spectrum at the receiver?",
     C_SUCCESS_BG, C_SUCCESS_GREEN)
]

for idx, (ctitle, cdesc, cbg, cborder) in enumerate(cards_data_s2):
    x = Inches(0.8 + idx * 3.95)
    c = add_card(slide2, x, Inches(1.6), Inches(3.8), Inches(4.9), cbg, cborder)
    tb = slide2.shapes.add_textbox(x + Inches(0.25), Inches(1.85), Inches(3.3), Inches(4.4))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = ctitle
    p.font.name = FONT_TITLE
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = C_NAVY_TITLE
    
    p2 = tf.add_paragraph()
    p2.text = cdesc
    p2.font.name = FONT_BODY
    p2.font.size = Pt(13)
    p2.font.color.rgb = C_TEXT_DARK
    p2.space_before = Pt(14)


# ==============================================================================
# SLIDE 3: How JPEG Works (Top-Bottom Split Layout)
# ==============================================================================
slide3 = prs.slides.add_slide(blank_layout)
set_slide_background(slide3, C_BG_WHITE)
add_header(slide3, "The Mechanics of JPEG Compression: From Spatial Pixels to Frequency Domain",
           "JPEG ARCHITECTURE", "Understanding how 2D Discrete Cosine Transform separates low and high frequencies.")

# Top Step Indicators (4 steps in a row)
steps_s3 = [
    ("Step 1: 8×8 Block Slicing", "Image is divided into independent 8×8 pixel blocks."),
    ("Step 2: 2D-DCT Transform", "Pixels are converted into 64 orthogonal frequency coefficients."),
    ("Step 3: Zigzag Truncation", "Top-left 15 low frequencies kept; 49 high AC frequencies zeroed out."),
    ("Step 4: Quantization & Save", "Remaining coefficients entropy encoded for minimal storage."),
]

for i, (st, sd) in enumerate(steps_s3):
    sx = Inches(0.8 + i * 2.95)
    add_card(slide3, sx, Inches(1.6), Inches(2.8), Inches(1.3), C_BG_BLUE_TINT, C_BORDER_LIGHT)
    tb = slide3.shapes.add_textbox(sx + Inches(0.15), Inches(1.7), Inches(2.5), Inches(1.1))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = st
    p.font.name = FONT_TITLE
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_ROYAL_BLUE
    
    p2 = tf.add_paragraph()
    p2.text = sd
    p2.font.name = FONT_BODY
    p2.font.size = Pt(9.5)
    p2.font.color.rgb = C_TEXT_DARK
    p2.space_before = Pt(4)

# Bottom: Diagram of JPEG Flaw & Truncation
if os.path.exists("assets/diagram_jpeg_flaw.png"):
    add_card(slide3, Inches(0.8), Inches(3.1), Inches(11.733), Inches(3.9), C_BG_WHITE, C_BORDER_LIGHT)
    slide3.shapes.add_picture("assets/diagram_jpeg_flaw.png", Inches(1.0), Inches(3.2), Inches(11.3), Inches(3.7))


# ==============================================================================
# SLIDE 4: The Flaws in JPEG (Side-by-Side Dual Flaw Breakdown)
# ==============================================================================
slide4 = prs.slides.add_slide(blank_layout)
set_slide_background(slide4, C_BG_SLATE_TINT)
add_header(slide4, "The Two Inherent Mathematical Flaws in JPEG Compression",
           "ROOT CAUSE ANALYSIS", "Aggressive quantization triggers both spectral and spatial artifacts.")

# Left Box: Spectral Loss
c_left = add_card(slide4, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.1), C_ALERT_BG, C_ALERT_BORDER)
tb_l = slide4.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(5.1), Inches(4.5))
tf_l = tb_l.text_frame
tf_l.word_wrap = True

p_l1 = tf_l.paragraphs[0]
p_l1.text = "FLAW 1: Spectral Information Loss"
p_l1.font.name = FONT_TITLE
p_l1.font.size = Pt(18)
p_l1.font.bold = True
p_l1.font.color.rgb = RGBColor(153, 27, 27)

p_l2 = tf_l.add_paragraph()
p_l2.text = "• The 49 High-Frequency AC coefficients contain the fine mathematical variation of edges, textures, hair, and sharp boundaries.\n\n" \
            "• Zeroing them causes severe smoothing and blur.\n\n" \
            "• Traditional spatial filters cannot restore this lost mathematical frequency information."
p_l2.font.name = FONT_BODY
p_l2.font.size = Pt(13)
p_l2.font.color.rgb = C_TEXT_DARK
p_l2.space_before = Pt(14)

# Right Box: Spatial Seams
c_right = add_card(slide4, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1), C_WARN_BG, C_WARN_BORDER)
tb_r = slide4.shapes.add_textbox(Inches(7.1), Inches(1.9), Inches(5.1), Inches(4.5))
tf_r = tb_r.text_frame
tf_r.word_wrap = True

p_r1 = tf_r.paragraphs[0]
p_r1.text = "FLAW 2: Dirichlet Boundary Grid Seams"
p_r1.font.name = FONT_TITLE
p_r1.font.size = Pt(18)
p_r1.font.bold = True
p_r1.font.color.rgb = RGBColor(146, 64, 14)

p_r2 = tf_r.add_paragraph()
p_r2.text = "• Because each 8×8 block is computed independently, there is NO mathematical continuity constraint across borders.\n\n" \
            "• At the borders, neighboring pixels experience unnatural step jumps.\n\n" \
            "• Result: The infamous visible checkerboard grid lines (blocking artifacts) seen across compressed photos."
p_r2.font.name = FONT_BODY
p_r2.font.size = Pt(13)
p_r2.font.color.rgb = C_TEXT_DARK
p_r2.space_before = Pt(14)


# ==============================================================================
# SLIDE 5: Existing Literature & Limitations (Timeline Layout)
# ==============================================================================
slide5 = prs.slides.add_slide(blank_layout)
set_slide_background(slide5, C_BG_WHITE)
add_header(slide5, "Prior Literature and Their Inherent Architectural Limitations",
           "LITERATURE REVIEW", "Why classical filters and deep pure-spatial CNNs leave the problem unsolved.")

lit_items = [
    ("Classical Deblocking Filters (Bilateral, Wavelet)",
     "• Applied post-decompression in spatial domain.\n• Simple threshold smoothing that washes out real edges along with artifacts.\n• Cannot infer true missing frequency harmonics.",
     "Classical Era"),
    ("Pure Spatial CNNs (ARCNN, DnCNN, SRCNN)",
     "• Slide standard 2D convolutions over pixel grid.\n• Great at smoothing seam lines across 8×8 boundaries.\n• FATAL FLAW: Blind to DCT physics. Alters the true 15 low frequencies (Spectral Drift)!",
     "Deep CNN Era"),
    ("Pure Frequency Domain Models (FSRCNN-variants)",
     "• Predict DCT coefficients in frequency tables.\n• Operates on isolated blocks without inter-block spatial context.\n• Fails completely to eliminate visible boundary grid seams.",
     "Spectral Era"),
]

for idx, (ltitle, ldesc, lera) in enumerate(lit_items):
    lx = Inches(0.8 + idx * 3.95)
    c = add_card(slide5, lx, Inches(1.6), Inches(3.8), Inches(5.0), C_BG_SLATE_TINT, C_BORDER_LIGHT)
    tb = slide5.shapes.add_textbox(lx + Inches(0.2), Inches(1.8), Inches(3.4), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True
    
    # Era Tag
    p_era = tf.paragraphs[0]
    p_era.text = lera.upper()
    p_era.font.name = FONT_TITLE
    p_era.font.size = Pt(9.5)
    p_era.font.bold = True
    p_era.font.color.rgb = C_ROYAL_BLUE
    
    p = tf.add_paragraph()
    p.text = ltitle
    p.font.name = FONT_TITLE
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_NAVY_TITLE
    p.space_before = Pt(6)
    
    p2 = tf.add_paragraph()
    p2.text = ldesc
    p2.font.name = FONT_BODY
    p2.font.size = Pt(12.5)
    p2.font.color.rgb = C_TEXT_DARK
    p2.space_before = Pt(12)


# ==============================================================================
# SLIDE 6: Our 3-Stage Investigation Roadmap (Horizontal Ribbon)
# ==============================================================================
slide6 = prs.slides.add_slide(blank_layout)
set_slide_background(slide6, C_BG_BLUE_TINT)
add_header(slide6, "Our 3-Stage Scientific Investigation: From Failure to Breakthrough",
           "RESEARCH ROADMAP", "We evaluated three distinct architectural paradigms to discover the true hybrid synergy.")

phases = [
    ("Phase 1: Pure Frequency GNN",
     "Can harmonic graph message passing infer missing ACs directly from low-frequency anchors?",
     "Result: Failed (+0.01 dB gain)\nDiagnosis: Block border blindness",
     C_ALERT_BG, C_ALERT_BORDER, RGBColor(153, 27, 27)),
    ("Phase 2: Hybrid GNN ➔ CNN",
     "Predict high frequencies in GNN first, then pass through spatial CNN to deblock seams.",
     "Result: Partial Success (+1.58 dB)\nLimitation: Spectral Drift (low freqs distorted)",
     C_WARN_BG, C_WARN_BORDER, RGBColor(146, 64, 14)),
    ("Phase 3: Hybrid CNN ➔ GNN ★",
     "Deblock boundaries in spatial CNN first, then lock true low freqs & refine harmonics via GNN.",
     "Result: Breakthrough (+1.91 dB to +4.13 dB!)\nVictory: Zero drift & clean harmonic physics",
     C_SUCCESS_BG, C_SUCCESS_BORDER, C_SUCCESS_DARK),
]

for idx, (pname, phypo, presult, pbg, pborder, ptitle_col) in enumerate(phases):
    px = Inches(0.8 + idx * 3.95)
    c = add_card(slide6, px, Inches(1.6), Inches(3.8), Inches(5.1), pbg, pborder)
    tb = slide6.shapes.add_textbox(px + Inches(0.25), Inches(1.9), Inches(3.3), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = pname
    p.font.name = FONT_TITLE
    p.font.size = Pt(16.5)
    p.font.bold = True
    p.font.color.rgb = ptitle_col
    
    p2 = tf.add_paragraph()
    p2.text = "HYPOTHESIS:"
    p2.font.name = FONT_TITLE
    p2.font.size = Pt(10)
    p2.font.bold = True
    p2.font.color.rgb = C_TEXT_MUTED
    p2.space_before = Pt(12)
    
    p3 = tf.add_paragraph()
    p3.text = phypo
    p3.font.name = FONT_BODY
    p3.font.size = Pt(12.5)
    p3.font.color.rgb = C_TEXT_DARK
    p3.space_before = Pt(4)

    p4 = tf.add_paragraph()
    p4.text = "OUTCOME:"
    p4.font.name = FONT_TITLE
    p4.font.size = Pt(10)
    p4.font.bold = True
    p4.font.color.rgb = C_TEXT_MUTED
    p4.space_before = Pt(14)
    
    p5 = tf.add_paragraph()
    p5.text = presult
    p5.font.name = FONT_BODY
    p5.font.size = Pt(12)
    p5.font.bold = True
    p5.font.color.rgb = ptitle_col
    p5.space_before = Pt(4)


# ==============================================================================
# SLIDE 7: Methodology 1 Architecture (Diagram + Topology Split)
# ==============================================================================
slide7 = prs.slides.add_slide(blank_layout)
set_slide_background(slide7, C_BG_WHITE)
add_header(slide7, "Methodology 1: Pure Frequency Graph Neural Network Architecture",
           "METHODOLOGY 01: ARCHITECTURE", "Constructing 64 frequency nodes connected by harmonic octave relations.")

if os.path.exists("assets/diagram_gnn_pipeline.png"):
    add_card(slide7, Inches(0.8), Inches(1.6), Inches(11.733), Inches(2.6), C_BG_WHITE, C_BORDER_LIGHT)
    slide7.shapes.add_picture("assets/diagram_gnn_pipeline.png", Inches(1.0), Inches(1.7), Inches(11.3), Inches(2.4))

if os.path.exists("assets/diagram_graph_topology.png"):
    add_card(slide7, Inches(0.8), Inches(4.4), Inches(11.733), Inches(2.6), C_BG_SLATE_TINT, C_BORDER_LIGHT)
    slide7.shapes.add_picture("assets/diagram_graph_topology.png", Inches(1.0), Inches(4.5), Inches(11.3), Inches(2.4))


# ==============================================================================
# SLIDE 8: Methodology 1 Results & Failure Diagnosis (Stat + Diagnosis)
# ==============================================================================
slide8 = prs.slides.add_slide(blank_layout)
set_slide_background(slide8, C_BG_SLATE_TINT)
add_header(slide8, "Methodology 1 Results: Why Pure GNN Flatlined at +0.01 dB Gain",
           "METHODOLOGY 01: SCIENTIFIC DIAGNOSIS", "Empirical proof that isolated spectral graphs cannot solve boundary artifacts.")

# Left: Stat Callout
c_stat = add_card(slide8, Inches(0.8), Inches(1.6), Inches(4.0), Inches(5.1), C_ALERT_BG, C_ALERT_BORDER)
tb_s = slide8.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(3.6), Inches(4.5))
tf_s = tb_s.text_frame
tf_s.word_wrap = True

p_s1 = tf_s.paragraphs[0]
p_s1.text = "GNN PSNR GAIN:"
p_s1.font.name = FONT_TITLE
p_s1.font.size = Pt(12)
p_s1.font.bold = True
p_s1.font.color.rgb = RGBColor(153, 27, 27)

p_s2 = tf_s.add_paragraph()
p_s2.text = "+0.01 dB"
p_s2.font.name = FONT_TITLE
p_s2.font.size = Pt(44)
p_s2.font.bold = True
p_s2.font.color.rgb = RGBColor(185, 28, 28)
p_s2.space_before = Pt(6)

p_s3 = tf_s.add_paragraph()
p_s3.text = "Baseline: 32.75 dB\nGNN Output: 32.76 dB\nImprovement: Statistically Negligible"
p_s3.font.name = FONT_BODY
p_s3.font.size = Pt(13)
p_s3.font.color.rgb = C_TEXT_DARK
p_s3.space_before = Pt(10)

# Right: The Diagnosis
c_diag = add_card(slide8, Inches(5.1), Inches(1.6), Inches(7.433), Inches(5.1), C_BG_WHITE, C_BORDER_LIGHT)
tb_d = slide8.shapes.add_textbox(Inches(5.4), Inches(1.9), Inches(6.8), Inches(4.5))
tf_d = tb_d.text_frame
tf_d.word_wrap = True

p_d1 = tf_d.paragraphs[0]
p_d1.text = "The Diagnosis: Inter-Block Border Blindness"
p_d1.font.name = FONT_TITLE
p_d1.font.size = Pt(18)
p_d1.font.bold = True
p_d1.font.color.rgb = C_NAVY_TITLE

p_d2 = tf_d.add_paragraph()
p_d2.text = "1. Frequency graphs operate strictly inside the 8×8 transform window. Each graph has ZERO topological connections to pixels in adjacent blocks.\n\n" \
            "2. Human perception of JPEG degradation is heavily dominated by the step discontinuities along the borders (the grid checkerboard).\n\n" \
            "3. Because the GNN could not observe or bridge across borders, it could not eliminate the grid seams.\n\n" \
            "4. Even when we added 8 multi-block spatial context nodes (72-node graph), GNN alone lacked the continuous translational convolutional filters needed to smooth pixel seams."
p_d2.font.name = FONT_BODY
p_d2.font.size = Pt(13)
p_d2.font.color.rgb = C_TEXT_DARK
p_d2.space_before = Pt(14)


# ==============================================================================
# SLIDE 9: Methodology 2 Architecture (GNN -> CNN Workflow)
# ==============================================================================
slide9 = prs.slides.add_slide(blank_layout)
set_slide_background(slide9, C_BG_WHITE)
add_header(slide9, "Methodology 2 Architecture: Hybrid GNN ➔ Spatial CNN",
           "METHODOLOGY 02: ARCHITECTURE", "Synthesizing frequency harmonics first, followed by spatial convolutional deblocking.")

if os.path.exists("assets/diagram_gnn_cnn_pipeline.png"):
    add_card(slide9, Inches(0.8), Inches(1.6), Inches(11.733), Inches(2.9), C_BG_WHITE, C_BORDER_LIGHT)
    slide9.shapes.add_picture("assets/diagram_gnn_cnn_pipeline.png", Inches(1.0), Inches(1.7), Inches(11.3), Inches(2.7))

# Bottom Two Insight Cards
c_b1 = add_card(slide9, Inches(0.8), Inches(4.7), Inches(5.7), Inches(2.1), C_BG_BLUE_TINT, C_BORDER_BLUE)
tb_b1 = slide9.shapes.add_textbox(Inches(1.0), Inches(4.85), Inches(5.3), Inches(1.8))
tf_b1 = tb_b1.text_frame
tf_b1.word_wrap = True
p = tf_b1.paragraphs[0]
p.text = "Role of Stage 1: Frequency GNN"
p.font.name = FONT_TITLE
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = C_ROYAL_BLUE
p2 = tf_b1.add_paragraph()
p2.text = "Receives truncated 15 low frequencies and uses 652 harmonic graph edges to predict initial high-frequency amplitudes in the spectral domain."
p2.font.name = FONT_BODY
p2.font.size = Pt(12)
p2.font.color.rgb = C_TEXT_DARK
p2.space_before = Pt(4)

c_b2 = add_card(slide9, Inches(6.8), Inches(4.7), Inches(5.7), Inches(2.1), C_BG_BLUE_TINT, C_BORDER_BLUE)
tb_b2 = slide9.shapes.add_textbox(Inches(7.0), Inches(4.85), Inches(5.3), Inches(1.8))
tf_b2 = tb_b2.text_frame
tf_b2.word_wrap = True
p = tf_b2.paragraphs[0]
p.text = "Role of Stage 2: Spatial CNN"
p.font.name = FONT_TITLE
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = C_ROYAL_BLUE
p2 = tf_b2.add_paragraph()
p2.text = "Converts GNN output to pixels via 1st IDCT, then applies 6 residual convolutional blocks across block borders to smooth out the visible grid lines."
p2.font.name = FONT_BODY
p2.font.size = Pt(12)
p2.font.color.rgb = C_TEXT_DARK
p2.space_before = Pt(4)


# ==============================================================================
# SLIDE 10: Methodology 2 Results & Spectral Drift (The Dilemma)
# ==============================================================================
slide10 = prs.slides.add_slide(blank_layout)
set_slide_background(slide10, C_BG_SLATE_TINT)
add_header(slide10, "Methodology 2 Results: Gaining +1.58 dB but Hitting 'Spectral Drift'",
           "METHODOLOGY 02: ANALYSIS", "PSNR jumps to 34.34 dB, but placing CNN at the end creates an unforeseen side effect.")

# Stat Card
c_stat2 = add_card(slide10, Inches(0.8), Inches(1.6), Inches(4.0), Inches(5.1), C_WARN_BG, C_WARN_BORDER)
tb_s2 = slide10.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(3.6), Inches(4.5))
tf_s2 = tb_s2.text_frame
tf_s2.word_wrap = True

p_s1 = tf_s2.paragraphs[0]
p_s1.text = "GNN ➔ CNN GAIN:"
p_s1.font.name = FONT_TITLE
p_s1.font.size = Pt(12)
p_s1.font.bold = True
p_s1.font.color.rgb = RGBColor(146, 64, 14)

p_s2 = tf_s2.add_paragraph()
p_s2.text = "+1.58 dB"
p_s2.font.name = FONT_TITLE
p_s2.font.size = Pt(44)
p_s2.font.bold = True
p_s2.font.color.rgb = RGBColor(180, 83, 9)
p_s2.space_before = Pt(6)

p_s3 = tf_s2.add_paragraph()
p_s3.text = "Baseline: 32.75 dB\nGNN➔CNN: 34.34 dB\nClear visual deblocking of 8x8 seams!"
p_s3.font.name = FONT_BODY
p_s3.font.size = Pt(13)
p_s3.font.color.rgb = C_TEXT_DARK
p_s3.space_before = Pt(10)

# The Dilemma Explanation
c_drift = add_card(slide10, Inches(5.1), Inches(1.6), Inches(7.433), Inches(5.1), C_BG_WHITE, C_BORDER_LIGHT)
tb_dr = slide10.shapes.add_textbox(Inches(5.4), Inches(1.9), Inches(6.8), Inches(4.5))
tf_dr = tb_dr.text_frame
tf_dr.word_wrap = True

p_dr1 = tf_dr.paragraphs[0]
p_dr1.text = "The Dilemma: Why Putting CNN Last Causes 'Spectral Drift'"
p_dr1.font.name = FONT_TITLE
p_dr1.font.size = Pt(18)
p_dr1.font.bold = True
p_dr1.font.color.rgb = C_NAVY_TITLE

p_dr2 = tf_dr.add_paragraph()
p_dr2.text = "1. In the original JPEG file, the first 15 low frequencies are 100% mathematically correct and known with certainty.\n\n" \
             "2. When the Spatial CNN operates at the end of the pipeline, its 3×3 convolutions slightly modify ALL pixel intensities to smooth borders.\n\n" \
             "3. As an accidental side effect, the CNN corrupts the 15 low frequencies that were originally perfect!\n\n" \
             "4. This causes slight color bleeding and ringing around high-contrast edges, preventing the model from reaching higher PSNR."
p_dr2.font.name = FONT_BODY
p_dr2.font.size = Pt(13)
p_dr2.font.color.rgb = C_TEXT_DARK
p_dr2.space_before = Pt(14)


# ==============================================================================
# SLIDE 11: Methodology 3 Architecture (The Champion: CNN -> GNN)
# ==============================================================================
slide11 = prs.slides.add_slide(blank_layout)
set_slide_background(slide11, C_BG_WHITE)
add_header(slide11, "Methodology 3 Architecture: Spatial First, Spectral Second (The Breakthrough)",
           "METHODOLOGY 03: THE WINNING ARCHITECTURE", "Why the 2-DCT and 2-IDCT handshake solves both spatial seams and spectral drift.")

if os.path.exists("assets/diagram_cnn_gnn_pipeline.png"):
    add_card(slide11, Inches(0.8), Inches(1.6), Inches(11.733), Inches(3.2), C_BG_WHITE, C_BORDER_LIGHT)
    slide11.shapes.add_picture("assets/diagram_cnn_gnn_pipeline.png", Inches(1.0), Inches(1.7), Inches(11.3), Inches(3.0))

# Bottom Two Steps Explaining the 2 DCTs & 2 IDCTs
c_p1 = add_card(slide11, Inches(0.8), Inches(5.0), Inches(5.7), Inches(1.8), C_BG_BLUE_TINT, C_BORDER_BLUE)
tb_p1 = slide11.shapes.add_textbox(Inches(1.0), Inches(5.15), Inches(5.3), Inches(1.5))
tf_p1 = tb_p1.text_frame
tf_p1.word_wrap = True
p = tf_p1.paragraphs[0]
p.text = "Stage 1: 1st IDCT ➔ Spatial Deblocking CNN"
p.font.name = FONT_TITLE
p.font.size = Pt(13.5)
p.font.bold = True
p.font.color.rgb = C_ROYAL_BLUE
p2 = tf_p1.add_paragraph()
p2.text = "1st IDCT turns compressed numbers into pixels. The CNN immediately eliminates 8×8 block seams and establishes smooth continuous macro edges."
p2.font.name = FONT_BODY
p2.font.size = Pt(11.5)
p2.font.color.rgb = C_TEXT_DARK
p2.space_before = Pt(3)

c_p2 = add_card(slide11, Inches(6.8), Inches(5.0), Inches(5.7), Inches(1.8), C_SUCCESS_BG, C_SUCCESS_BORDER)
tb_p2 = slide11.shapes.add_textbox(Inches(7.0), Inches(5.15), Inches(5.3), Inches(1.5))
tf_p2 = tb_p2.text_frame
tf_p2.word_wrap = True
p = tf_p2.paragraphs[0]
p.text = "Stage 2: 2nd DCT ➔ GNN & Hard Spectral Lock"
p.font.name = FONT_TITLE
p.font.size = Pt(13.5)
p.font.bold = True
p.font.color.rgb = C_SUCCESS_DARK
p2 = tf_p2.add_paragraph()
p2.text = "2nd DCT translates CNN pixels back to frequencies. GNN enforces harmonic physics AND locks the 15 low frequencies back to exact truth!"
p2.font.name = FONT_BODY
p2.font.size = Pt(11.5)
p2.font.color.rgb = C_TEXT_DARK
p2.space_before = Pt(3)


# ==============================================================================
# SLIDE 12: The Secret Weapon: Hard Spectral Consistency
# ==============================================================================
slide12 = prs.slides.add_slide(blank_layout)
set_slide_background(slide12, C_SUCCESS_BG)
add_header(slide12, "The Secret Weapon: Hard Spectral Consistency & Harmonic Refinement",
           "MATHEMATICAL INVARIANT", "How our model guarantees zero spectral drift and eliminates generative hallucination.")

c_card12_l = add_card(slide12, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.1), C_BG_WHITE, C_SUCCESS_BORDER)
tb_12l = slide12.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(5.1), Inches(4.5))
tf_12l = tb_12l.text_frame
tf_12l.word_wrap = True

p = tf_12l.paragraphs[0]
p.text = "1. Hard Spectral Consistency (The Anchor)"
p.font.name = FONT_TITLE
p.font.size = Pt(17)
p.font.bold = True
p.font.color.rgb = C_SUCCESS_DARK

p2 = tf_12l.add_paragraph()
p2.text = "• Mathematically, the 15 low frequencies in the compressed JPEG file were never corrupted — they are ground truth.\n\n" \
          "• Before the final 2nd IDCT, we explicitly clamp the first 15 coefficients back to their original values:\n" \
          "  Y_final[0:15] = Y_true[0:15]\n" \
          "  Y_final[15:64] = GNN_refined[15:64]\n\n" \
          "• This guarantees ZERO spectral drift, zero color shift, and 100% preservation of true scene illumination."
p2.font.name = FONT_BODY
p2.font.size = Pt(12.5)
p2.font.color.rgb = C_TEXT_DARK
p2.space_before = Pt(12)

c_card12_r = add_card(slide12, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1), C_BG_WHITE, C_SUCCESS_BORDER)
tb_12r = slide12.shapes.add_textbox(Inches(7.1), Inches(1.9), Inches(5.1), Inches(4.5))
tf_12r = tb_12r.text_frame
tf_12r.word_wrap = True

p = tf_12r.paragraphs[0]
p.text = "2. Harmonic Octave Refinement"
p.font.name = FONT_TITLE
p.font.size = Pt(17)
p.font.bold = True
p.font.color.rgb = C_NAVY_TITLE

p2 = tf_12r.add_paragraph()
p2.text = "• CNNs often produce Gibbs ringing (artificial rippling waves near sharp borders).\n\n" \
          "• Our GNN passes messages along octave chains (u <-> 2u, v <-> 2v).\n\n" \
          "• If a high-frequency coefficient predicted by the CNN violates physical harmonic proportions with its fundamental tone, the GNN dampens it.\n\n" \
          "• If it aligns with true edge physics, the GNN sharpens it."
p2.font.name = FONT_BODY
p2.font.size = Pt(12.5)
p2.font.color.rgb = C_TEXT_DARK
p2.space_before = Pt(12)


# ==============================================================================
# SLIDE 13: Quantitative Benchmark Results (Full Scoreboard + Strip)
# ==============================================================================
slide13 = prs.slides.add_slide(blank_layout)
set_slide_background(slide13, C_BG_WHITE)
add_header(slide13, "Quantitative Benchmarks: Proving Dual-Domain Dominance across Set5",
           "EMPIRICAL RESULTS", "Average PSNR jumps +1.91 dB, with massive texture reconstruction (+4.13 dB on Butterfly).")

# Top: Scoreboard Cards (4 cards)
scores = [
    ("Baseline JPEG", "32.75 dB", "0.00 dB (Cutoff=15)", C_BG_SLATE_TINT, C_BORDER_LIGHT, C_TEXT_MUTED),
    ("1. Frequency GNN", "32.76 dB", "+0.01 dB (Border blind)", C_ALERT_BG, C_ALERT_BORDER, RGBColor(153, 27, 27)),
    ("2. Hybrid (GNN➔CNN)", "34.34 dB", "+1.58 dB (Spectral drift)", C_WARN_BG, C_WARN_BORDER, RGBColor(146, 64, 14)),
    ("3. Hybrid (CNN➔GNN)", "34.66 dB", "+1.91 dB AVG (Up to +4.13 dB!)", C_SUCCESS_BG, C_SUCCESS_BORDER, C_SUCCESS_DARK),
]

for i, (name, psnr, sub, bg, bcol, tcol) in enumerate(scores):
    sx = Inches(0.8 + i * 2.95)
    add_card(slide13, sx, Inches(1.5), Inches(2.8), Inches(1.3), bg, bcol)
    tb = slide13.shapes.add_textbox(sx + Inches(0.15), Inches(1.55), Inches(2.5), Inches(1.2))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = name
    p.font.name = FONT_TITLE
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_NAVY_TITLE
    
    p2 = tf.add_paragraph()
    p2.text = psnr
    p2.font.name = FONT_TITLE
    p2.font.size = Pt(22)
    p2.font.bold = True
    p2.font.color.rgb = tcol
    
    p3 = tf.add_paragraph()
    p3.text = sub
    p3.font.name = FONT_BODY
    p3.font.size = Pt(9.5)
    p3.font.color.rgb = C_TEXT_MUTED

# Bottom: 5-Panel Visual Comparison Strip from Results
if os.path.exists("results/Set14_img_001_SRF_2_comparison.png"):
    add_card(slide13, Inches(0.8), Inches(3.0), Inches(11.733), Inches(4.0), C_BG_WHITE, C_BORDER_LIGHT)
    slide13.shapes.add_picture("results/Set14_img_001_SRF_2_comparison.png", Inches(0.9), Inches(3.1), Inches(11.533), Inches(3.8))


# ==============================================================================
# SLIDE 14: Visual Proof: 8x8 Block Seam Elimination & Error Heatmaps
# ==============================================================================
slide14 = prs.slides.add_slide(blank_layout)
set_slide_background(slide14, C_BG_SLATE_TINT)
add_header(slide14, "Visual Verification: Zoomed Seam Elimination & 4-Panel Error Heatmaps",
           "VISUAL VERIFICATION", "Close-up evidence showing Dirichlet boundary seams vanishing under CNN ➔ GNN.")

# Top Half: Zoomed Seam Elimination Patches
if os.path.exists("assets/zoomed_seam_patch.png"):
    add_card(slide14, Inches(0.8), Inches(1.5), Inches(11.733), Inches(2.6), C_BG_WHITE, C_BORDER_LIGHT)
    slide14.shapes.add_picture("assets/zoomed_seam_patch.png", Inches(1.0), Inches(1.55), Inches(11.3), Inches(2.5))

# Bottom Half: 4-Panel Error Heatmap Gallery
if os.path.exists("results/Set14_img_001_SRF_2_error_heatmap.png"):
    add_card(slide14, Inches(0.8), Inches(4.3), Inches(11.733), Inches(2.7), C_BG_WHITE, C_BORDER_LIGHT)
    slide14.shapes.add_picture("results/Set14_img_001_SRF_2_error_heatmap.png", Inches(1.0), Inches(4.35), Inches(11.3), Inches(2.6))


# ==============================================================================
# SLIDE 15: Comparative Synthesis (Why Neither Model Works Alone)
# ==============================================================================
slide15 = prs.slides.add_slide(blank_layout)
set_slide_background(slide15, C_BG_WHITE)
add_header(slide15, "Synthesis: Why Neither Model Works Alone (The Dual-Domain Principle)",
           "THEORETICAL SYNTHESIS", "Spatial convolution and spectral graph passing are fundamentally complementary.")

c_syn_l = add_card(slide15, Inches(0.8), Inches(1.6), Inches(3.8), Inches(5.1), C_BG_CARD_BLUE, C_BORDER_BLUE)
tb_sl = slide15.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(3.4), Inches(4.5))
tf_sl = tb_sl.text_frame
tf_sl.word_wrap = True
p = tf_sl.paragraphs[0]
p.text = "Spatial CNN Alone"
p.font.name = FONT_TITLE
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = C_ROYAL_BLUE
p2 = tf_sl.add_paragraph()
p2.text = "STRENGTH:\n• Slides continuously across 8×8 block borders.\n• Excels at identifying and smoothing visual step jumps (seams).\n\nFATAL WEAKNESS:\n• Blind to DCT orthogonal basis physics.\n• Hallucinates and alters known true low frequencies (Spectral Drift)."
p2.font.name = FONT_BODY
p2.font.size = Pt(12.5)
p2.font.color.rgb = C_TEXT_DARK
p2.space_before = Pt(12)

c_syn_m = add_card(slide15, Inches(4.75), Inches(1.6), Inches(3.8), Inches(5.1), C_ALERT_BG, C_ALERT_BORDER)
tb_sm = slide15.shapes.add_textbox(Inches(4.95), Inches(1.85), Inches(3.4), Inches(4.5))
tf_sm = tb_sm.text_frame
tf_sm.word_wrap = True
p = tf_sm.paragraphs[0]
p.text = "Frequency GNN Alone"
p.font.name = FONT_TITLE
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = RGBColor(185, 28, 28)
p2 = tf_sm.add_paragraph()
p2.text = "STRENGTH:\n• Respects harmonic octave physics (u <-> 2u, v <-> 2v).\n• Guarantees mathematical frequency constraints.\n\nFATAL WEAKNESS:\n• Isolated within the 8×8 block window.\n• Completely blind to inter-block boundary discontinuities."
p2.font.name = FONT_BODY
p2.font.size = Pt(12.5)
p2.font.color.rgb = C_TEXT_DARK
p2.space_before = Pt(12)

c_syn_r = add_card(slide15, Inches(8.7), Inches(1.6), Inches(3.8), Inches(5.1), C_SUCCESS_BG, C_SUCCESS_BORDER)
tb_sr = slide15.shapes.add_textbox(Inches(8.9), Inches(1.85), Inches(3.4), Inches(4.5))
tf_sr = tb_sr.text_frame
tf_sr.word_wrap = True
p = tf_sr.paragraphs[0]
p.text = "Our Dual-Domain Hybrid ★"
p.font.name = FONT_TITLE
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = C_SUCCESS_DARK
p2 = tf_sr.add_paragraph()
p2.text = "THE SYNERGY:\n• CNN acts as the Spatial Seam Smoother.\n• GNN acts as the Spectral Physics Regulator.\n• Hard Spectral Lock guarantees 0% spectral drift.\n\nOUTCOME:\n• The highest PSNR (+4.13 dB gain).\n• Crisp, artifact-free restoration."
p2.font.name = FONT_BODY
p2.font.size = Pt(12.5)
p2.font.color.rgb = C_TEXT_DARK
p2.space_before = Pt(12)


# ==============================================================================
# SLIDE 16: Interactive Demo & Q&A (Grand Finale)
# ==============================================================================
slide16 = prs.slides.add_slide(blank_layout)
set_slide_background(slide16, C_BG_BLUE_TINT)
add_header(slide16, "Live Demonstration & Research Defense Q&A",
           "CONCLUSION & INVITATION", "Experience the model live on any uploaded image via our web interface.")

# Left Card: Live Demo Invitation & QR
c_demo = add_card(slide16, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.1), C_BG_WHITE, C_BORDER_BLUE)
tb_dm = slide16.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(5.1), Inches(4.5))
tf_dm = tb_dm.text_frame
tf_dm.word_wrap = True

p = tf_dm.paragraphs[0]
p.text = "📱 Try the Model Live on Your Phone"
p.font.name = FONT_TITLE
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = C_ROYAL_BLUE

p2 = tf_dm.add_paragraph()
p2.text = "• Web Interface: Built with FastAPI / Gradio & PyTorch.\n\n" \
          "• Upload any PNG or camera photo to see the 8×8 JPEG block seams disappear in real-time.\n\n" \
          "• Dual Deployment: Running offline locally on presenter laptop for zero-lag presentation, and hosted on cloud for judges' mobile devices."
p2.font.name = FONT_BODY
p2.font.size = Pt(13)
p2.font.color.rgb = C_TEXT_DARK
p2.space_before = Pt(14)

# Right Card: Summary of Contributions
c_sum = add_card(slide16, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1), C_BG_WHITE, C_SUCCESS_BORDER)
tb_sm = slide16.shapes.add_textbox(Inches(7.1), Inches(1.9), Inches(5.1), Inches(4.5))
tf_sm = tb_sm.text_frame
tf_sm.word_wrap = True

p = tf_sm.paragraphs[0]
p.text = "Summary of Contributions"
p.font.name = FONT_TITLE
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = C_SUCCESS_DARK

p2 = tf_sm.add_paragraph()
p2.text = "1. Identified the root cause of pure GNN failure (+0.01 dB) as inter-block boundary blindness.\n\n" \
          "2. Uncovered the 'Spectral Drift' dilemma when placing CNN after GNN (+1.58 dB).\n\n" \
          "3. Engineered the CNN ➔ GNN Dual-Domain Architecture with Hard Spectral Consistency, achieving up to +4.13 dB gain on benchmark images.\n\n" \
          "Thank you! We welcome your questions."
p2.font.name = FONT_BODY
p2.font.size = Pt(12.5)
p2.font.color.rgb = C_TEXT_DARK
p2.space_before = Pt(14)


# Save presentation
prs.save(PPTX_PATH)
print(f"[SUCCESS] 16-Slide Presentation successfully generated and saved to: {PPTX_PATH}")
