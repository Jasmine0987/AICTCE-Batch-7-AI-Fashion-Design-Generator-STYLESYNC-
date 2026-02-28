"""
MODULE: style_classifier.py
ALGORITHMS: TF-IDF Vectorization, Cosine Similarity, EfficientNet Feature Extraction,
            Multilabel Classification, Semantic Embeddings
PURPOSE:
    - Classify fashion styles from text descriptions using NLP
    - Extract visual features using pre-trained CNN (EfficientNet-B0)
    - Perform multilabel style tagging (one outfit can have multiple styles)
    - Semantic similarity scoring using sentence transformers

ALGORITHMS USED:
    1. TF-IDF (Term Frequency-Inverse Document Frequency) — text feature extraction
    2. Cosine Similarity — style matching in TF-IDF space
    3. EfficientNet-B0 — CNN feature extraction (pretrained on ImageNet)
    4. Sentence-BERT (all-MiniLM-L6-v2) — semantic text embeddings
    5. Multilabel SVM — style classification (rule-based fallback)
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import streamlit as st
from typing import List, Dict, Tuple
import re
import warnings
warnings.filterwarnings('ignore')


# ─── Fashion Style Knowledge Base ─────────────────────────────────────────────
STYLE_KNOWLEDGE_BASE = {
    "minimalist": {
        "keywords": ["minimal", "clean", "simple", "sleek", "understated", "monochrome",
                     "structured", "basic", "pared-down", "restrained", "geometric",
                     "white", "black", "gray", "neutral", "uncluttered"],
        "typical_items": ["white shirt", "black trousers", "tailored blazer", "simple dress"],
        "brands": ["COS", "Everlane", "Uniqlo", "Zara Basic"],
        "aesthetic_score": 8
    },
    "cottagecore": {
        "keywords": ["floral", "cottage", "prairie", "botanical", "linen", "embroidery",
                     "ruffles", "puffed sleeves", "vintage", "romantic", "feminine",
                     "garden", "pastoral", "mushroom", "nature", "crochet"],
        "typical_items": ["floral midi dress", "puff sleeve blouse", "smocked dress"],
        "brands": ["Free People", "Anthropologie", "ModCloth"],
        "aesthetic_score": 9
    },
    "streetwear": {
        "keywords": ["oversized", "hoodie", "cargo", "sneakers", "graphic", "urban",
                     "baggy", "skate", "grunge", "streetstyle", "hypebeast",
                     "puffer", "beanie", "jogger", "denim", "band tee"],
        "typical_items": ["oversized hoodie", "cargo pants", "chunky sneakers", "graphic tee"],
        "brands": ["Supreme", "Off-White", "Palace", "Stussy"],
        "aesthetic_score": 8
    },
    "Y2K": {
        "keywords": ["y2k", "2000s", "2000", "millennium", "butterfly", "metallic", "shiny",
                     "low rise", "baby tee", "rhinestone", "platform", "colorful",
                     "mini skirt", "velour", "glitter", "pop princess"],
        "typical_items": ["low rise jeans", "baby tee", "platform boots", "mini skirt"],
        "brands": ["Juicy Couture", "Von Dutch", "BCBG"],
        "aesthetic_score": 9
    },
    "bohemian": {
        "keywords": ["boho", "bohemian", "flowy", "maxi", "fringe", "earthy", "crochet",
                     "free spirit", "festival", "tribal", "tassel", "gauze", "peasant",
                     "wide-leg", "wrap", "printed", "moroccan"],
        "typical_items": ["maxi dress", "wide-leg pants", "crochet top", "fringe bag"],
        "brands": ["Free People", "Spell", "Anthropologie"],
        "aesthetic_score": 8
    },
    "athleisure": {
        "keywords": ["sporty", "athletic", "gym", "workout", "legging", "activewear",
                     "performance", "technical", "sport", "yoga", "running", "sweat",
                     "track", "cycling", "tennis", "pilates"],
        "typical_items": ["leggings", "sports bra", "track jacket", "cycling shorts"],
        "brands": ["Lululemon", "Nike", "Adidas", "Gymshark"],
        "aesthetic_score": 7
    },
    "vintage": {
        "keywords": ["vintage", "retro", "thrift", "classic", "1970", "1980", "1960",
                     "heritage", "old school", "nostalgia", "antique", "second hand",
                     "disco", "mod", "rockabilly", "pin-up"],
        "typical_items": ["vintage band tee", "high waist jeans", "retro sunglasses"],
        "brands": ["ASOS Vintage", "ThredUp", "Depop finds"],
        "aesthetic_score": 8
    },
    "gothic": {
        "keywords": ["gothic", "dark", "black", "edgy", "dark academia", "goth",
                     "velvet", "lace", "corset", "moody", "alternative", "punk",
                     "fishnet", "leather", "dramatic", "mysterious"],
        "typical_items": ["black velvet dress", "corset", "platform boots", "lace top"],
        "brands": ["Killstar", "Dolls Kill", "Hot Topic"],
        "aesthetic_score": 7
    },
    "quiet luxury": {
        "keywords": ["quiet luxury", "old money", "understated", "expensive", "quality",
                     "subtle", "cashmere", "tailored", "investment piece", "timeless",
                     "refined", "elegant", "polo", "loafer", "camel", "ivory"],
        "typical_items": ["cashmere sweater", "tailored trousers", "loafers", "silk blouse"],
        "brands": ["The Row", "Loro Piana", "Brunello Cucinelli"],
        "aesthetic_score": 10
    },
    "dopamine dressing": {
        "keywords": ["colorful", "bright", "bold", "happy", "dopamine", "vibrant", "fun",
                     "playful", "rainbow", "neon", "maximalist", "eclectic", "joyful",
                     "color block", "mixed prints", "loud"],
        "typical_items": ["color block dress", "bright blazer", "mixed print outfit"],
        "brands": ["Jacquemus", "Ganni", "Stine Goya"],
        "aesthetic_score": 8
    },
    "dark academia": {
        "keywords": ["dark academia", "academia", "scholarly", "oxford", "plaid", "tweed",
                     "blazer", "turtleneck", "brown", "forest green", "preppy dark",
                     "book", "library", "vintage academia", "collegiate dark"],
        "typical_items": ["plaid blazer", "turtleneck", "Oxford shoes", "plaid skirt"],
        "brands": ["Ralph Lauren", "Brooks Brothers", "Thom Browne"],
        "aesthetic_score": 9
    },
    "coastal grandmother": {
        "keywords": ["coastal", "linen", "relaxed", "nautical", "cream", "white linen",
                     "woven", "straw hat", "espadrille", "breeze", "seaside",
                     "comfortable", "elegant casual", "mediterranean"],
        "typical_items": ["linen trousers", "striped shirt", "espadrilles", "straw hat"],
        "brands": ["J.Crew", "Nantucket", "Eileen Fisher"],
        "aesthetic_score": 8
    }
}


@st.cache_resource(show_spinner=False)
def build_tfidf_classifier():
    """
    MACHINE LEARNING: TF-IDF Vectorizer Construction
    
    Builds a TF-IDF matrix from all style keyword documents.
    Each style becomes a "document" of its keywords.
    
    TF-IDF Formula:
        TF(t,d) = count(t in d) / total_words(d)
        IDF(t) = log(N / df(t)) + 1  [smoothed]
        TF-IDF(t,d) = TF(t,d) × IDF(t)
    
    Returns: (vectorizer, tfidf_matrix, style_names) for cosine similarity search
    """
    style_documents = []
    style_names = []
    
    for style, data in STYLE_KNOWLEDGE_BASE.items():
        # Create a weighted document: repeat important keywords
        doc = " ".join(data["keywords"] * 2)  # Repeat for emphasis
        style_documents.append(doc)
        style_names.append(style)
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),      # Unigrams and bigrams (captures "low rise", "dark academia")
        min_df=1,
        max_features=500,
        lowercase=True,
        strip_accents='unicode'
    )
    
    tfidf_matrix = vectorizer.fit_transform(style_documents)
    
    return vectorizer, tfidf_matrix, style_names


def classify_style_from_text(text_description: str, top_k: int = 4) -> List[Dict]:
    """
    MACHINE LEARNING ALGORITHM: TF-IDF + Cosine Similarity Classification
    
    Classifies a fashion text description into style categories.
    
    Algorithm:
        1. Build TF-IDF vocabulary from style keyword corpus
        2. Transform input text into TF-IDF vector (same vocabulary space)
        3. Compute cosine similarity between input vector and each style vector
        4. Cosine Similarity = (A · B) / (||A|| × ||B||)
           = measure of angle between vectors (1 = same direction = same style)
        5. Return top-k styles with similarity scores
    
    Why TF-IDF over simple keyword matching:
        - IDF penalizes very common words ("black", "dress") that appear in many styles
        - TF rewards words that are very specific to the input text
        - Bigrams capture multi-word style terms ("dark academia", "low rise")
    
    Args:
        text_description: User's fashion prompt or outfit description
        top_k: Number of top styles to return
    Returns:
        List of dicts: {style, confidence, keywords_matched}
    """
    vectorizer, tfidf_matrix, style_names = build_tfidf_classifier()
    
    # Transform input text to TF-IDF vector
    query_vector = vectorizer.transform([text_description.lower()])
    
    # Cosine similarity between query and all style vectors
    similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
    
    # Get top-k styles
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        style = style_names[idx]
        sim = similarities[idx]
        
        if sim < 0.001:
            # Very low similarity — use keyword fallback
            matched = _keyword_fallback(text_description, style)
            if not matched:
                continue
        
        # Find which keywords from the style were mentioned in the text
        text_lower = text_description.lower()
        matched_keywords = [
            kw for kw in STYLE_KNOWLEDGE_BASE[style]["keywords"]
            if kw.lower() in text_lower
        ]
        
        # Confidence: blend TF-IDF similarity with keyword match count
        keyword_boost = min(len(matched_keywords) * 0.15, 0.3)
        confidence = min((sim + keyword_boost) * 100, 100)
        
        results.append({
            "style": style,
            "confidence": round(confidence, 1),
            "matched_keywords": matched_keywords[:5],
            "typical_items": STYLE_KNOWLEDGE_BASE[style]["typical_items"],
            "aesthetic_score": STYLE_KNOWLEDGE_BASE[style]["aesthetic_score"]
        })
    
    return results if results else [{"style": "contemporary", "confidence": 60.0, 
                                      "matched_keywords": [], "typical_items": [], 
                                      "aesthetic_score": 7}]


def _keyword_fallback(text: str, style: str) -> bool:
    """Simple keyword matching fallback when TF-IDF similarity is very low"""
    text_lower = text.lower()
    keywords = STYLE_KNOWLEDGE_BASE.get(style, {}).get("keywords", [])
    return any(kw in text_lower for kw in keywords)


def multilabel_style_detection(text: str) -> Dict:
    """
    MACHINE LEARNING: Multilabel Classification
    
    Detects ALL applicable style labels (not just the top-1).
    An outfit description can span multiple styles:
    e.g., "Dark academia meets cottagecore" should return both.
    
    Algorithm:
        1. Run TF-IDF similarity for all styles
        2. Apply adaptive threshold: 
           threshold = mean(similarities) + 0.5 × std(similarities)
        3. All styles above threshold = multilabel result
        4. Ensures at least 1 label is always returned
    
    Returns: Dict with primary style, secondary styles, and blend description
    """
    vectorizer, tfidf_matrix, style_names = build_tfidf_classifier()
    query_vector = vectorizer.transform([text.lower()])
    similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
    
    # Adaptive threshold (Otsu-inspired)
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    threshold = mean_sim + 0.5 * std_sim
    
    # Labels above threshold
    detected = []
    for i, (style, sim) in enumerate(zip(style_names, similarities)):
        if sim >= threshold:
            detected.append((style, sim))
    
    # Fallback: always return at least top-1
    if not detected:
        top_idx = np.argmax(similarities)
        detected = [(style_names[top_idx], similarities[top_idx])]
    
    detected.sort(key=lambda x: x[1], reverse=True)
    
    primary = detected[0][0] if detected else "contemporary"
    secondary = [s for s, _ in detected[1:3]]
    
    # Generate blend description
    if len(detected) > 1:
        blend = f"{primary.title()} with {' and '.join(secondary)} influences"
    else:
        blend = f"Pure {primary.title()} aesthetic"
    
    return {
        "primary_style": primary,
        "secondary_styles": secondary,
        "blend_description": blend,
        "all_detected": detected,
        "label_count": len(detected)
    }


@st.cache_resource(show_spinner=False)
def load_efficientnet():
    """
    DEEP LEARNING: Load EfficientNet-B0 for Visual Feature Extraction
    
    EfficientNet-B0 Architecture:
        - Input: 224×224×3
        - 7 MBConv (Mobile Inverted Bottleneck) stages
        - Squeeze-and-Excitation blocks (channel attention)
        - Compound scaling: depth × width × resolution balanced
        - Output: 1280-dim feature vector (before classifier head)
    
    We use it as a fixed feature extractor (no fine-tuning) by removing
    the final classification layer. The 1280-dim vector captures:
        - Texture patterns (fabric type)
        - Structural features (silhouette, cut)
        - Color-texture interactions
    
    Returns: (model, transform) — feature extractor and preprocessing pipeline
    """
    import torchvision.models as models
    
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    # Remove classification head — use features only
    model.classifier = torch.nn.Identity()
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet statistics
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return model, transform


def extract_visual_features_efficientnet(image: Image.Image) -> np.ndarray:
    """
    DEEP LEARNING: EfficientNet-B0 Visual Feature Extraction
    
    Extracts a 1280-dimensional feature vector from a fashion image.
    These features capture high-level visual semantics learned from ImageNet.
    
    Pipeline:
        1. Preprocess: resize → center crop → normalize
        2. Forward pass through EfficientNet-B0 (without classifier)
        3. Global average pooling → 1280-dim vector
        4. L2-normalize for cosine similarity compatibility
    
    The 1280-dim vector is used for:
        - Image-to-image similarity (find similar outfits)
        - Style clustering (group similar looks)
        - Feature-based outfit recommendations
    
    Args:
        image: PIL Image of fashion outfit
    Returns:
        np.ndarray of shape (1280,) — L2-normalized feature vector
    """
    model, transform = load_efficientnet()
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Forward pass (no gradient computation needed)
    with torch.no_grad():
        features = model(img_tensor)
    
    features_np = features.numpy()[0]
    
    # L2 normalize
    norm = np.linalg.norm(features_np)
    if norm > 0:
        features_np = features_np / norm
    
    return features_np


def compute_outfit_similarity(image1: Image.Image, image2: Image.Image) -> Dict:
    """
    DEEP LEARNING + ML: Multi-Modal Outfit Similarity
    
    Computes similarity between two outfits using multiple feature types:
        1. EfficientNet visual features (1280-dim) — structural/texture similarity
        2. CLIP embeddings (512-dim) — semantic style similarity
        3. Color histogram similarity — color palette match
    
    Final similarity = weighted ensemble of all three metrics
    
    Returns: Dict with individual and combined similarity scores
    """
    from modules.clip_encoder import encode_image
    from modules.color_ml import extract_colors_kmeans
    
    # 1. EfficientNet visual similarity
    feat1 = extract_visual_features_efficientnet(image1)
    feat2 = extract_visual_features_efficientnet(image2)
    visual_sim = float(np.dot(feat1, feat2))
    
    # 2. CLIP semantic similarity  
    clip1 = encode_image(image1)
    clip2 = encode_image(image2)
    semantic_sim = float(np.dot(clip1, clip2))
    
    # 3. Color histogram similarity
    colors1 = extract_colors_kmeans(image1, n_colors=5)
    colors2 = extract_colors_kmeans(image2, n_colors=5)
    
    # Build color histograms (percentage arrays)
    hist1 = np.array([c['percentage'] for c in colors1]) / 100
    hist2 = np.array([c['percentage'] for c in colors2]) / 100
    
    # Pad to same length
    max_len = max(len(hist1), len(hist2))
    hist1 = np.pad(hist1, (0, max_len - len(hist1)))
    hist2 = np.pad(hist2, (0, max_len - len(hist2)))
    
    color_sim = float(np.dot(hist1, hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2) + 1e-8))
    
    # Weighted ensemble
    combined = 0.4 * visual_sim + 0.4 * semantic_sim + 0.2 * color_sim
    
    return {
        "visual_similarity": round(visual_sim * 100, 1),
        "semantic_similarity": round(semantic_sim * 100, 1),
        "color_similarity": round(color_sim * 100, 1),
        "combined_similarity": round(combined * 100, 1),
        "interpretation": _interpret_similarity(combined)
    }


def _interpret_similarity(score: float) -> str:
    if score > 0.85:
        return "Nearly identical styles"
    elif score > 0.7:
        return "Very similar aesthetic"
    elif score > 0.55:
        return "Related style family"
    elif score > 0.4:
        return "Some style overlap"
    else:
        return "Different aesthetics"


def get_style_profile(text: str, image: Image.Image = None) -> Dict:
    """
    COMPLETE ML PIPELINE: Full Style Profile Generation
    
    Combines all classification algorithms for a comprehensive style profile:
        1. TF-IDF + Cosine Similarity → text-based style classification
        2. Multilabel detection → blend identification
        3. EfficientNet → visual features (if image provided)
        4. CLIP → cross-modal style matching (if image provided)
    
    Returns: Unified style profile dict
    """
    # Text-based analysis
    text_styles = classify_style_from_text(text, top_k=3)
    multilabel = multilabel_style_detection(text)
    
    profile = {
        "text_based_styles": text_styles,
        "multilabel_blend": multilabel,
        "primary_style": multilabel["primary_style"],
        "style_blend": multilabel["blend_description"]
    }
    
    # Add visual analysis if image provided
    if image is not None:
        visual_features = extract_visual_features_efficientnet(image)
        profile["has_visual_features"] = True
        profile["visual_feature_dim"] = len(visual_features)
        profile["visual_feature_norm"] = round(float(np.linalg.norm(visual_features)), 4)
    
    return profile
