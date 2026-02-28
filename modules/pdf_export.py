"""MODULE: PDF Lookbook Generator using FPDF2"""
from fpdf import FPDF
from datetime import datetime
import os
import re


def sanitize(text: str) -> str:
    """Strip all non-latin1 characters so Helvetica never throws a Unicode error."""
    if not text:
        return ""
    # Replace common Unicode punctuation with ASCII equivalents
    replacements = {
        "\u2022": "-",   # bullet •
        "\u2013": "-",   # en dash
        "\u2014": "-",   # em dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u00d7": "x",   # multiplication sign
        "\u2019": "'",
        "\u20b9": "Rs.", # rupee sign
        "\u00e9": "e",
        "\u00e8": "e",
        "\u00ea": "e",
        "\u00e0": "a",
        "\u00e2": "a",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # Final sweep: encode to latin-1, replacing anything still invalid
    return text.encode("latin-1", errors="replace").decode("latin-1")


class FashionLookbook(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 18)
        self.set_text_color(30, 30, 30)
        self.cell(0, 12, 'My AI Fashion Lookbook', align='C', new_x="LMARGIN", new_y="NEXT")
        self.set_font('Helvetica', '', 9)
        self.set_text_color(150, 150, 150)
        self.cell(0, 7, sanitize(f'Generated {datetime.now().strftime("%B %d, %Y")} | Powered by Gemini 2.5 Flash + ML'),
                  align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(180, 180, 180)
        self.cell(0, 10, f'AI Fashion Design Generator | Page {self.page_no()}', align='C')


def create_lookbook_pdf(designs: list, filename: str = "my_lookbook.pdf") -> str:
    pdf = FashionLookbook()
    pdf.set_auto_page_break(auto=True, margin=15)
    for i, design in enumerate(designs):
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 10, sanitize(f"Design {i+1}: {design.get('name', 'Untitled')}"),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(70, 70, 70)
        pdf.multi_cell(0, 6, sanitize(design.get('description', '')[:600]))
        pdf.ln(3)
        if design.get('colors'):
            pdf.set_font('Helvetica', 'B', 11)
            pdf.set_text_color(40, 40, 40)
            pdf.cell(0, 8, 'Color Palette:', new_x="LMARGIN", new_y="NEXT")
            pdf.set_font('Helvetica', '', 10)
            for color in design.get('colors', [])[:4]:
                # Use ASCII hyphen-dash instead of bullet
                pdf.cell(0, 6, sanitize(f'  - {color}'), new_x="LMARGIN", new_y="NEXT")
        if design.get('style_tags'):
            pdf.ln(2)
            pdf.set_font('Helvetica', 'B', 11)
            pdf.cell(0, 8, 'Style DNA:', new_x="LMARGIN", new_y="NEXT")
            pdf.set_font('Helvetica', '', 10)
            pdf.cell(0, 6, sanitize(' | '.join(design.get('style_tags', []))),
                     new_x="LMARGIN", new_y="NEXT")
        if design.get('ml_analysis'):
            pdf.ln(2)
            pdf.set_font('Helvetica', 'B', 11)
            pdf.cell(0, 8, 'ML Analysis:', new_x="LMARGIN", new_y="NEXT")
            pdf.set_font('Helvetica', '', 9)
            ml = design['ml_analysis']
            pdf.cell(0, 6, sanitize(f"CLIP Style: {ml.get('primary_style', 'N/A')} ({ml.get('confidence', 0):.0%} confidence)"),
                     new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_text_color(160, 160, 160)
        pdf.cell(0, 6, sanitize(f"Created: {design.get('created_at', '')}"),
                 new_x="LMARGIN", new_y="NEXT")
    import tempfile
    out = os.path.join(tempfile.gettempdir(), filename)
    pdf.output(out)
    return out