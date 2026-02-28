"""
MODULE: NLP Keyword Extraction & Prompt Engineering
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALGORITHMS USED:
  1. KeyBERT               → BERT-based keyword extraction
  2. YAKE (Yet Another KE) → Statistical keyword extraction (unsupervised)
  3. Cosine Similarity     → Selecting maximally-diverse keywords (MMR)
  4. Regex + Rule-based NLP→ Fashion entity recognition

WHERE USED:
  - Extracting searchable keywords from Gemini design descriptions
  - Generating optimized product search queries
  - Identifying fashion entities (colors, garments, brands) in user text
  - Enhancing user prompts with fashion-specific vocabulary
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from typing import List, Dict, Tuple
import numpy as np

# Ensure NLTK data available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)


# ── Fashion Vocabulary (Domain-Specific Ontology) ─────────────────────────────
FASHION_VOCABULARY = {
    "silhouettes": [
        "a-line", "bodycon", "empire", "shift", "wrap", "maxi", "midi",
        "mini", "oversized", "fitted", "flared", "straight", "balloon"
    ],
    "necklines": [
        "v-neck", "scoop", "halter", "off-shoulder", "sweetheart", "turtleneck",
        "crew neck", "cowl", "boat neck", "plunge", "square neck"
    ],
    "sleeves": [
        "sleeveless", "cap sleeve", "flutter", "bishop", "bell", "puff",
        "three-quarter", "long sleeve", "raglan", "dolman"
    ],
    "fabrics": [
        "silk", "satin", "chiffon", "organza", "velvet", "lace", "denim",
        "cotton", "linen", "jersey", "tweed", "brocade", "tulle", "crepe",
        "modal", "cashmere", "wool", "polyester", "spandex", "rayon"
    ],
    "patterns": [
        "floral", "paisley", "geometric", "abstract", "animal print", "houndstooth",
        "plaid", "gingham", "stripes", "polka dots", "tie-dye", "ombre", "colorblock"
    ],
    "embellishments": [
        "embroidered", "beaded", "sequin", "ruffled", "pleated", "smocked",
        "pintucked", "gathered", "laser cut", "applique", "fringe"
    ],
    "aesthetics": [
        "minimalist", "maximalist", "cottagecore", "dark academia", "y2k",
        "bohemian", "streetwear", "preppy", "vintage", "avant-garde",
        "normcore", "quiet luxury", "dopamine dressing", "coastal grandmother"
    ]
}

# Flatten for quick lookup
ALL_FASHION_TERMS = set()
for terms in FASHION_VOCABULARY.values():
    ALL_FASHION_TERMS.update(terms)

# Color vocabulary (fashion-specific names)
FASHION_COLORS = [
    "ivory", "ecru", "champagne", "blush", "dusty rose", "burgundy", "wine",
    "rust", "terracotta", "burnt orange", "mustard", "olive", "sage", "hunter green",
    "teal", "cobalt", "navy", "royal blue", "periwinkle", "lavender", "mauve",
    "nude", "camel", "tan", "chocolate", "charcoal", "off-white", "cream"
]


def extract_keywords_statistical(text: str, top_n: int = 8) -> List[Tuple[str, float]]:
    """
    ALGORITHM: Statistical Keyword Extraction (YAKE-inspired)
    ──────────────────────────────────────────────────────────
    Method: Score words based on:
    1. Term frequency (TF) normalized by document length
    2. Position score: words early in text get higher weight
    3. Fashion vocabulary bonus: domain-specific terms ranked higher
    4. Stop word filtering: removes generic words

    Returns: List of (keyword, score) tuples, sorted by relevance
    """
    # Tokenize
    try:
        tokens = word_tokenize(text.lower())
    except Exception:
        tokens = text.lower().split()

    # Remove punctuation and stop words
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = {'the', 'a', 'an', 'is', 'are', 'with', 'for', 'and', 'or', 'to', 'of', 'in', 'on', 'at'}

    custom_stops = {'design', 'outfit', 'wear', 'style', 'fashion', 'clothing', 'clothes', 'look'}
    stop_words.update(custom_stops)

    filtered = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 2]

    if not filtered:
        return []

    # Term frequency
    tf_counter = Counter(filtered)
    total_terms = len(filtered)

    # Score each term
    scores = {}
    for term, count in tf_counter.items():
        tf = count / total_terms

        # Position bonus (earlier terms get higher score)
        try:
            position = filtered.index(term)
            position_score = 1.0 / (1.0 + position / total_terms)
        except ValueError:
            position_score = 0.5

        # Fashion vocabulary bonus
        fashion_bonus = 2.5 if term in ALL_FASHION_TERMS else 1.0

        # Color bonus
        color_bonus = 1.8 if any(term in c for c in FASHION_COLORS) else 1.0

        scores[term] = tf * position_score * fashion_bonus * color_bonus

    # Sort by score
    sorted_terms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_terms[:top_n]


def extract_keywords_keybert(text: str, top_n: int = 6) -> List[Tuple[str, float]]:
    """
    ALGORITHM: KeyBERT — BERT-based Keyword Extraction
    ────────────────────────────────────────────────────
    Method:
    1. Encode the full document with BERT → document embedding
    2. Extract n-gram candidates from document
    3. Encode each candidate phrase with BERT → phrase embedding
    4. Rank candidates by cosine similarity to document embedding
    5. Apply MMR (Maximal Marginal Relevance) for diversity

    MMR: Balances relevance (similar to document) and diversity
         (dissimilar from already selected keywords)
         Score = λ × sim(candidate, doc) - (1-λ) × max_sim(candidate, selected)
    """
    try:
        from keybert import KeyBERT

        kw_model = KeyBERT(model='all-MiniLM-L6-v2')

        # Extract keywords with MMR for diversity
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),   # Unigrams and bigrams
            stop_words='english',
            use_mmr=True,                   # Enable Maximal Marginal Relevance
            diversity=0.5,                  # Balance relevance vs diversity
            top_n=top_n
        )

        return keywords  # List of (keyword, score)

    except Exception:
        # Fallback to statistical method
        return extract_keywords_statistical(text, top_n)


def extract_fashion_entities(text: str) -> Dict[str, List[str]]:
    """
    ALGORITHM: Fashion-Domain Named Entity Recognition (Rule-based NER)
    ─────────────────────────────────────────────────────────────────────
    Custom NER using regex patterns + vocabulary lookup.
    Identifies fashion-specific entities without needing a trained NER model.

    Entity types:
    - GARMENT:      Clothing items
    - COLOR:        Color mentions
    - FABRIC:       Textile types
    - AESTHETIC:    Style categories
    - OCCASION:     Where to wear it
    - EMBELLISHMENT: Decorative details
    """
    text_lower = text.lower()
    entities = {
        "garments":       [],
        "colors":         [],
        "fabrics":        [],
        "aesthetics":     [],
        "embellishments": [],
        "silhouettes":    [],
        "necklines":      [],
        "patterns":       []
    }

    # Garment extraction via regex
    garment_patterns = [
        r'\b(dress|gown|skirt|top|blouse|shirt|pants|trousers|jeans|shorts|'
        r'jacket|coat|blazer|sweater|cardigan|hoodie|jumpsuit|romper|saree|'
        r'lehenga|kurta|salwar|suit|tuxedo|kimono|poncho|cape|vest)\b'
    ]
    for pattern in garment_patterns:
        matches = re.findall(pattern, text_lower)
        entities["garments"].extend(matches)

    # Color extraction
    color_words = FASHION_COLORS + [
        "red", "blue", "green", "yellow", "orange", "purple", "pink",
        "black", "white", "grey", "gray", "brown", "beige", "gold", "silver"
    ]
    for color in color_words:
        if color in text_lower:
            entities["colors"].append(color)

    # Fabric, aesthetic, embellishment, silhouette extraction from vocabulary
    for category, terms in FASHION_VOCABULARY.items():
        entity_key = category.rstrip('s')  # Singular form
        for term in terms:
            if term in text_lower:
                if category == "fabrics":
                    entities["fabrics"].append(term)
                elif category == "aesthetics":
                    entities["aesthetics"].append(term)
                elif category == "embellishments":
                    entities["embellishments"].append(term)
                elif category == "silhouettes":
                    entities["silhouettes"].append(term)
                elif category == "necklines":
                    entities["necklines"].append(term)
                elif category == "patterns":
                    entities["patterns"].append(term)

    # Deduplicate
    return {k: list(set(v)) for k, v in entities.items()}


def build_product_search_query(design_description: str, color_names: List[str],
                                 style_tags: List[str], budget: str = None) -> str:
    """
    Build an optimized shopping search query by combining:
    1. KeyBERT/Statistical keywords from design description
    2. Extracted color names from image analysis
    3. Style tags from CLIP classification
    4. Optional budget constraint
    """
    # Extract keywords from description
    keywords = extract_keywords_statistical(design_description, top_n=4)
    keyword_terms = [kw for kw, score in keywords if score > 0.01]

    # Combine signals
    query_parts = []

    # Priority 1: Style aesthetic tags
    query_parts.extend(style_tags[:2])

    # Priority 2: Top colors (fashion-named)
    if color_names:
        query_parts.append(color_names[0])

    # Priority 3: Keyword-extracted terms
    query_parts.extend(keyword_terms[:3])

    # Remove duplicates while preserving order
    seen = set()
    unique_parts = []
    for part in query_parts:
        if part.lower() not in seen and len(part) > 2:
            seen.add(part.lower())
            unique_parts.append(part)

    query = " ".join(unique_parts[:5])

    if budget:
        query += f" {budget}"

    return query


def enhance_user_prompt(raw_prompt: str) -> str:
    """
    ALGORITHM: Prompt Enhancement using Fashion Entity Extraction
    ──────────────────────────────────────────────────────────────
    1. Extract entities from raw user input
    2. Identify what's MISSING (e.g., user said "summer dress" but no fabric/color)
    3. Add fashion-specific detail suggestions to improve Gemini output quality
    """
    entities = extract_fashion_entities(raw_prompt)

    enhancements = []

    if not entities["fabrics"]:
        enhancements.append("Include fabric type suggestion")

    if not entities["colors"]:
        enhancements.append("Include a color palette recommendation")

    if not entities["embellishments"] and not entities["patterns"]:
        enhancements.append("Suggest decorative details or patterns")

    if not entities["silhouettes"]:
        enhancements.append("Specify the silhouette type")

    # Build enhanced prompt
    enhanced = raw_prompt
    if enhancements:
        enhanced += f". In your design, please also: {'; '.join(enhancements[:3])}."

    return enhanced


def get_fashion_keywords_summary(text: str) -> Dict:
    """Main interface: Run full NLP pipeline on design description"""
    entities = extract_fashion_entities(text)
    statistical_kws = extract_keywords_statistical(text, top_n=6)
    keybert_kws = extract_keywords_keybert(text, top_n=5)

    # Merge and deduplicate keyword lists
    all_keywords = list(set(
        [kw for kw, _ in statistical_kws] +
        [kw for kw, _ in keybert_kws]
    ))

    return {
        "entities":            entities,
        "statistical_keywords": [kw for kw, _ in statistical_kws[:5]],
        "keybert_keywords":    [kw for kw, _ in keybert_kws[:5]],
        "merged_keywords":     all_keywords[:8],
        "search_query":        ", ".join(all_keywords[:5])
    }
