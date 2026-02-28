"""
MODULE: Gemini 2.5 Flash Integration
Model: gemini-2.5-flash (multimodal - text + vision)
Role: Fashion design generation with ML-enriched context
"""

import google.generativeai as genai
import os, base64, io
from PIL import Image
import streamlit as st


def configure_gemini():
    try:
        api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key:
        genai.configure(api_key=api_key)
    return bool(api_key)


def generate_design_description(user_prompt: str, style_tags: list = None,
                                  color_palette: list = None, occasion: str = None,
                                  budget: str = None, trend_tags: list = None) -> dict:
    """Gemini 2.5 Flash design generation with ML pipeline context injection"""
    if not configure_gemini():
        return {"description": "Set GEMINI_API_KEY in .env", "keywords": ""}

    context_parts = []
    if style_tags:   context_parts.append(f"CLIP style classification: {', '.join(style_tags)}")
    if color_palette: context_parts.append(f"KMeans color palette: {', '.join(color_palette[:3])}")
    if occasion:     context_parts.append(f"Occasion: {occasion}")
    if budget:       context_parts.append(f"Budget: {budget}")
    if trend_tags:   context_parts.append(f"Trending styles: {', '.join(trend_tags)}")

    prompt = f"""You are a senior fashion designer. 
USER REQUEST: {user_prompt}
ML PIPELINE CONTEXT:\n{chr(10).join(context_parts)}

Return structured design:

**DESIGN CONCEPT**
[2-3 vivid sentences]

**KEY ELEMENTS**
- Silhouette: [type]
- Neckline: [style]
- Sleeves: [style]
- Fabric: [fabric + texture]
- Pattern: [print or embellishment]
- Length: [length]

**COLOR PALETTE**
- Primary: [name] — [#hex]
- Secondary: [name] — [#hex]
- Accent: [name] — [#hex]

**STYLING TIPS**
1. [tip one]
2. [tip two]

**OCCASION FIT**
[occasions]

SEARCH_KEYWORDS: [5-7 comma-separated shopping keywords]"""

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        text = response.text
        keywords = ""
        for line in text.split('\n'):
            if 'SEARCH_KEYWORDS:' in line:
                keywords = line.replace('SEARCH_KEYWORDS:', '').strip()
                break
        return {"description": text, "keywords": keywords}
    except Exception as e:
        return {"description": f"Error: {str(e)}", "keywords": ""}


def analyze_fashion_image_gemini(image: Image.Image) -> dict:
    """Gemini Vision analysis complementing CLIP classification"""
    if not configure_gemini():
        return {"analysis": "API key not configured", "tags": []}

    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    prompt = """Analyze this fashion image as an expert stylist:
1. GARMENT IDENTIFICATION: What items are visible?
2. STYLE AESTHETIC: What fashion aesthetic?
3. COLOR STORY: Palette in fashion terms
4. FABRIC ASSESSMENT: Fabrics used?
5. OCCASION: Suitable occasions?
6. TREND ALIGNMENT: Current trend alignment?

STYLE_TAGS: [comma-separated: aesthetic, occasion, season, formality]"""

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": img_b64}])
        text = response.text
        tags = []
        for line in text.split('\n'):
            if 'STYLE_TAGS:' in line:
                tags = [t.strip() for t in line.replace('STYLE_TAGS:', '').split(',')]
                break
        return {"analysis": text, "tags": tags}
    except Exception as e:
        return {"analysis": f"Error: {str(e)}", "tags": []}


def generate_trend_fusion_design(user_prompt: str, selected_trends: list, clip_style: str = None) -> dict:
    """Fuse user vision with trending aesthetics"""
    if not configure_gemini():
        return {"description": "API key not configured", "keywords": ""}
    clip_ctx = f"\nCLIP-detected base style: {clip_style}" if clip_style else ""
    prompt = f"""Blend these into a cohesive fashion design:
USER VISION: {user_prompt}
TRENDING STYLES: {', '.join(selected_trends)}{clip_ctx}

Use: **DESIGN CONCEPT**, **KEY ELEMENTS**, **COLOR PALETTE**, **STYLING TIPS**
End with: SEARCH_KEYWORDS: [keywords]"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        text = response.text
        keywords = ""
        for line in text.split('\n'):
            if 'SEARCH_KEYWORDS:' in line:
                keywords = line.replace('SEARCH_KEYWORDS:', '').strip()
                break
        return {"description": text, "keywords": keywords}
    except Exception as e:
        return {"description": f"Error: {str(e)}", "keywords": ""}


def generate_occasion_outfit(occasion: str, preferences: str, budget: str = None, season: str = None) -> dict:
    """Complete occasion-specific outfit generator"""
    if not configure_gemini():
        return {"description": "API key not configured", "keywords": ""}
    budget_note = f"Budget: {budget}" if budget else "Open budget"
    prompt = f"""Design complete outfit for:
OCCASION: {occasion}
PREFERENCES: {preferences}
{budget_note}  SEASON: {season or 'Any'}

Include: **COMPLETE LOOK**, **ACCESSORIES**, **FOOTWEAR**, **BUDGET BREAKDOWN**
End with: SEARCH_KEYWORDS: [main piece keywords]"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        text = response.text
        keywords = ""
        for line in text.split('\n'):
            if 'SEARCH_KEYWORDS:' in line:
                keywords = line.replace('SEARCH_KEYWORDS:', '').strip()
                break
        return {"description": text, "keywords": keywords}
    except Exception as e:
        return {"description": f"Error: {str(e)}", "keywords": ""}
