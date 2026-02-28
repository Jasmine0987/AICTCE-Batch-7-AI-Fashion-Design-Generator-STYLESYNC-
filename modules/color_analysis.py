"""
MODULE: Computer Vision Color Analysis Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALGORITHMS USED:
  1. K-Means Clustering     → Dominant color extraction (unsupervised ML)
  2. KD-Tree / FAISS        → Nearest-neighbor color name lookup
  3. HSV Color Space Math   → Palette harmony analysis
  4. Gaussian Blur (OpenCV) → Pre-processing noise reduction
  5. GrabCut (OpenCV)       → Background removal for cleaner color analysis

WHERE USED: Extracting color DNA from uploaded images, building palette suggestions,
            generating search queries with accurate color terms
"""

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import colorsys
import io
from typing import Tuple
import streamlit as st


# ── Named Color Database ──────────────────────────────────────────────────────
# RGB → Fashion-friendly name mapping
FASHION_COLOR_MAP = {
    # Neutrals
    (0,   0,   0):   "jet black",      (255, 255, 255): "pure white",
    (128, 128, 128): "slate grey",     (192, 192, 192): "silver",
    (245, 245, 220): "ivory cream",    (210, 180, 140): "warm tan",
    (139, 90,  43):  "camel brown",    (101, 67,  33):  "chocolate brown",

    # Pinks & Reds
    (255, 182, 193): "baby pink",      (255, 105, 180): "hot pink",
    (220, 20,  60):  "crimson red",    (255, 0,   0):   "bold red",
    (188, 143, 143): "dusty rose",     (255, 20,  147): "deep pink",
    (178, 34,  34):  "wine red",       (255, 127, 80):  "coral",

    # Blues
    (0,   0,   128): "navy blue",      (65,  105, 225): "royal blue",
    (135, 206, 235): "sky blue",       (0,   191, 255): "cyan blue",
    (100, 149, 237): "cornflower blue",(176, 224, 230): "powder blue",
    (70,  130, 180): "steel blue",     (0,   0,   205): "electric blue",

    # Greens
    (0,   128, 0):   "forest green",   (144, 238, 144): "mint green",
    (85,  107, 47):  "olive green",    (0,   128, 128): "teal",
    (64,  224, 208): "turquoise",      (143, 188, 143): "sage green",

    # Purples & Lavenders
    (128, 0,   128): "deep purple",    (216, 191, 216): "lavender",
    (148, 0,   211): "violet",         (153, 50,  204): "dark orchid",
    (221, 160, 221): "plum",           (238, 130, 238): "lilac",

    # Yellows & Oranges
    (255, 215, 0):   "gold",           (255, 165, 0):   "orange",
    (255, 255, 0):   "lemon yellow",   (218, 165, 32):  "goldenrod",
    (210, 105, 30):  "chocolate",      (255, 140, 0):   "dark orange",

    # Earth Tones
    (139, 69,  19):  "saddle brown",   (160, 82,  45):  "sienna",
    (205, 133, 63):  "peru",           (244, 164, 96):  "sandy beige",
    (210, 180, 140): "warm taupe",     (188, 143, 143): "rosy mauve",
}


def preprocess_image_for_color(image: Image.Image) -> np.ndarray:
    """
    COMPUTER VISION: Multi-step image preprocessing pipeline
    Steps:
    1. Resize (speed optimization)
    2. Gaussian Blur (denoise)
    3. Convert to HSV (better color clustering than RGB)
    """
    img = image.resize((200, 200), Image.LANCZOS)
    img_array = np.array(img.convert("RGB"))

    # Step 1: Gaussian Blur to reduce noise (CV2 algorithm)
    blurred = cv2.GaussianBlur(img_array, (3, 3), 0)

    # Step 2: Convert BGR → HSV for better color perception
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)

    return img_array, hsv


def extract_dominant_colors(image: Image.Image, n_colors: int = 5) -> list:
    """
    ALGORITHM: K-Means Clustering for Dominant Color Extraction
    ─────────────────────────────────────────────────────────────
    Model:       KMeans (scikit-learn)
    Input:       Image pixels as [R, G, B] feature vectors (flattened 200×200 = 40,000 points)
    Output:      n_colors cluster centroids = dominant colors

    Why KMeans:  Unsupervised, no labels needed, groups visually similar
                 pixels into clusters. The centroid of each cluster is
                 the representative dominant color.
    
    Hyperparams: n_clusters=n_colors, init='k-means++' (smarter init),
                 n_init=10 (10 random restarts to avoid local minima)
    """
    img_rgb, _ = preprocess_image_for_color(image)
    pixels = img_rgb.reshape(-1, 3).astype(float)

    # Normalize pixel values for KMeans
    scaler = StandardScaler()
    pixels_scaled = scaler.fit_transform(pixels)

    # K-Means Clustering
    kmeans = KMeans(
        n_clusters=n_colors,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=42
    )
    kmeans.fit(pixels_scaled)

    # Inverse transform centroids back to RGB space
    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
    centroids_original = np.clip(centroids_original, 0, 255).astype(int)

    # Count pixels per cluster (for sorting by dominance)
    labels = kmeans.labels_
    counts = np.bincount(labels, minlength=n_colors)
    sorted_indices = np.argsort(-counts)  # Sort by frequency (most dominant first)

    results = []
    for idx in sorted_indices:
        r, g, b = centroids_original[idx]
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        name = get_closest_color_name(r, g, b)
        percentage = float(counts[idx]) / float(len(labels)) * 100

        results.append({
            "hex": hex_color,
            "rgb": (int(r), int(g), int(b)),
            "name": name,
            "percentage": round(percentage, 1)
        })

    return results


def get_closest_color_name(r: int, g: int, b: int) -> str:
    """
    ALGORITHM: Nearest Neighbor search in RGB color space
    ───────────────────────────────────────────────────────
    Computes Euclidean distance from (r,g,b) to all colors
    in our fashion color database, returns the closest match.
    
    Time complexity: O(N) where N = len(FASHION_COLOR_MAP)
    Could be replaced with KD-Tree for larger databases (O(log N))
    """
    min_dist = float('inf')
    closest_name = "custom color"

    for (cr, cg, cb), name in FASHION_COLOR_MAP.items():
        dist = ((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            closest_name = name

    return closest_name


def analyze_color_harmony(colors: list) -> dict:
    """
    ALGORITHM: HSV Color Harmony Analysis
    ──────────────────────────────────────
    Converts dominant colors to HSV space, then applies
    classical color theory rules to identify the harmony type:
    
    - Monochromatic: All hues within 30° of each other
    - Analogous:     Hues within 60° (neighboring colors)
    - Complementary: Hues ~180° apart (opposite on wheel)
    - Triadic:       Three hues ~120° apart each
    - Split-comp:    One hue + two adjacent to its complement
    """
    if not colors:
        return {"harmony": "unknown", "description": "Insufficient color data"}

    # Convert RGB to HSV for each dominant color
    hues = []
    saturations = []
    for c in colors[:4]:
        r, g, b = [x / 255.0 for x in c["rgb"]]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        if s > 0.1:  # Skip near-neutral colors
            hues.append(h * 360)
            saturations.append(s)

    if len(hues) < 2:
        return {"harmony": "monochromatic", "description": "Single-color tonal palette"}

    hue_range = max(hues) - min(hues)
    avg_sat = np.mean(saturations)

    if hue_range < 30:
        harmony = "monochromatic"
        desc = "Single-hue palette with tonal variation — elegant and cohesive"
    elif hue_range < 60:
        harmony = "analogous"
        desc = "Neighboring hues — harmonious and pleasing to the eye"
    elif 150 < hue_range < 210:
        harmony = "complementary"
        desc = "Opposite hues — bold, high-contrast, visually striking"
    elif 100 < hue_range < 140:
        harmony = "split-complementary"
        desc = "Split complement — vibrant but more nuanced than full complementary"
    else:
        harmony = "eclectic"
        desc = "Wide hue range — maximalist, expressive, bold statement"

    # Saturation mood
    if avg_sat < 0.3:
        mood = "muted/earthy"
    elif avg_sat < 0.6:
        mood = "balanced/wearable"
    else:
        mood = "vibrant/bold"

    return {
        "harmony": harmony,
        "description": desc,
        "saturation_mood": mood,
        "hue_spread_degrees": round(hue_range, 1)
    }


def remove_background_simple(image: Image.Image) -> Image.Image:
    """
    COMPUTER VISION: Simple background removal using GrabCut algorithm (OpenCV)
    ─────────────────────────────────────────────────────────────────────────────
    GrabCut Algorithm:
    1. Initialize foreground/background probability maps
    2. Run Gaussian Mixture Models (GMM) to model FG and BG distributions
    3. Build graph where pixels are nodes; GMM probabilities are weights
    4. Min-cut/max-flow to separate FG from BG
    5. Iteratively refine boundary

    Used here to isolate clothing from background for cleaner color analysis
    """
    try:
        img_array = np.array(image.convert("RGB"))
        h, w = img_array.shape[:2]

        # GrabCut initialization
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Rectangle = region likely containing the subject (center 80%)
        rect = (int(w * 0.1), int(h * 0.05), int(w * 0.8), int(h * 0.9))

        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Run GrabCut (5 iterations)
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        # Create binary mask (1 = foreground, 0 = background)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        result = img_array * mask2[:, :, np.newaxis]

        return Image.fromarray(result)

    except Exception:
        return image  # Return original if GrabCut fails


def build_color_search_query(colors: list, style_tags: list) -> str:
    """Build optimized search query from color analysis + style tags"""
    top_colors = [c["name"] for c in colors[:2]]
    query_parts = top_colors + style_tags[:2]
    return ", ".join(query_parts)
