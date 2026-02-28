"""
Helper stub for text-based color DNA — called from app.py
The full implementation is in color_ml.py
"""

# This function is imported in app.py but defined here as a stub
# Full color analysis from text prompts uses style_classifier keyword mapping

def get_style_dna_from_text_for_prompt(text: str) -> dict:
    """
    Extract color and style DNA from text description.
    Used when no image is uploaded (text-only mode).
    """
    from modules.color_ml import COLOR_PALETTES
    from modules.style_classifier import STYLE_KNOWLEDGE_BASE
    
    text_lower = text.lower()
    
    # Detect color mood from keywords
    warm_kws = ["warm", "orange", "red", "yellow", "earthy", "rust", "terracotta", "amber"]
    cool_kws = ["cool", "blue", "purple", "teal", "lavender", "mint", "sage", "ocean"]
    pastel_kws = ["pastel", "soft", "baby", "light pink", "blush", "powder", "dusty"]
    earth_kws = ["earth", "brown", "khaki", "camel", "tan", "sand", "beige", "nude"]
    
    if any(kw in text_lower for kw in pastel_kws):
        mood = "pastel"
    elif any(kw in text_lower for kw in earth_kws):
        mood = "earth"
    elif any(kw in text_lower for kw in warm_kws):
        mood = "warm"
    elif any(kw in text_lower for kw in cool_kws):
        mood = "cool"
    else:
        mood = "neutral"
    
    return {
        "color_mood": mood,
        "suggested_palette": COLOR_PALETTES.get(mood, COLOR_PALETTES["neutral"])
    }
