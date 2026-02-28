"""
MODULE: color_ml.py
ALGORITHMS: K-Means Clustering, Gaussian Mixture Models (GMM), Color Space Transformations
PURPOSE:
    - Extract dominant colors from fashion images using unsupervised ML
    - Classify color palettes into fashion-meaningful categories
    - Generate complementary color recommendations using color theory
    - Measure color harmony scores

ALGORITHMS USED:
    1. K-Means Clustering — dominant color extraction from pixel data
    2. Gaussian Mixture Model — soft color clustering (handles color gradients better)
    3. LAB Color Space Conversion — perceptually uniform color distance (Delta-E)
    4. Cosine Similarity — palette matching to predefined fashion palettes
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import colorsys
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


# ─── Fashion Color Palettes (Ground Truth) ────────────────────────────────────
FASHION_PALETTES = {
    "Monochrome": {
        "colors": [(0, 0, 0), (255, 255, 255), (128, 128, 128), (64, 64, 64)],
        "description": "Classic black, white and grays"
    },
    "Earth Tones": {
        "colors": [(139, 90, 43), (188, 143, 91), (205, 170, 125), (101, 67, 33)],
        "description": "Warm browns, tans, terracottas"
    },
    "Pastels": {
        "colors": [(255, 182, 193), (176, 224, 230), (152, 251, 152), (221, 160, 221)],
        "description": "Soft pinks, baby blues, mint greens"
    },
    "Jewel Tones": {
        "colors": [(65, 105, 225), (128, 0, 128), (0, 128, 128), (220, 20, 60)],
        "description": "Rich sapphires, amethysts, emeralds"
    },
    "Warm Sunset": {
        "colors": [(255, 127, 80), (255, 165, 0), (220, 20, 60), (255, 215, 0)],
        "description": "Oranges, reds, warm yellows"
    },
    "Cool Ocean": {
        "colors": [(70, 130, 180), (0, 191, 255), (72, 209, 204), (100, 149, 237)],
        "description": "Blues, teals, aquamarines"
    },
    "Neutral Luxe": {
        "colors": [(245, 245, 220), (210, 180, 140), (188, 143, 143), (112, 128, 144)],
        "description": "Beige, taupe, mauve, slate"
    },
    "Bold Primary": {
        "colors": [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 128, 0)],
        "description": "Vibrant reds, blues, yellows"
    }
}


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB tuple to hex color string"""
    return '#{:02x}{:02x}{:02x}'.format(int(r), int(g), int(b))


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    ALGORITHM: CIE LAB Color Space Conversion
    
    Converts RGB to LAB (L*a*b*) — a perceptually uniform color space where
    Euclidean distance between colors matches human visual perception of difference.
    
    L* = Lightness (0=black, 100=white)
    a* = Red-Green axis (negative=green, positive=red)
    b* = Blue-Yellow axis (negative=blue, positive=yellow)
    
    Used for: accurate color distance measurement (Delta-E metric)
    """
    # Normalize RGB to 0-1
    rgb_norm = rgb / 255.0
    
    # Apply sRGB gamma correction (linearize)
    mask = rgb_norm > 0.04045
    rgb_linear = np.where(mask, ((rgb_norm + 0.055) / 1.055) ** 2.4, rgb_norm / 12.92)
    
    # RGB to XYZ using sRGB matrix (D65 illuminant)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    if rgb_linear.ndim == 1:
        xyz = M @ rgb_linear
    else:
        xyz = rgb_linear @ M.T
    
    # Normalize by D65 white point
    xyz_norm = xyz / np.array([0.95047, 1.00000, 1.08883])
    
    # XYZ to LAB
    epsilon = 0.008856
    kappa = 903.3
    
    f = np.where(xyz_norm > epsilon, 
                 np.cbrt(xyz_norm), 
                 (kappa * xyz_norm + 16) / 116)
    
    if f.ndim == 1:
        L = 116 * f[1] - 16
        a = 500 * (f[0] - f[1])
        b = 200 * (f[1] - f[2])
        return np.array([L, a, b])
    else:
        L = 116 * f[:, 1] - 16
        a_ch = 500 * (f[:, 0] - f[:, 1])
        b_ch = 200 * (f[:, 1] - f[:, 2])
        return np.column_stack([L, a_ch, b_ch])


def delta_e(color1_rgb: np.ndarray, color2_rgb: np.ndarray) -> float:
    """
    ALGORITHM: Delta-E Color Difference Metric (CIE76)
    
    Computes perceptual color difference between two colors.
    Delta-E < 1: imperceptible difference
    Delta-E 1-2: slight difference (trained eye)
    Delta-E 2-10: clear difference
    Delta-E > 10: different colors
    
    Used for: Color palette similarity matching
    """
    lab1 = rgb_to_lab(np.array(color1_rgb, dtype=float))
    lab2 = rgb_to_lab(np.array(color2_rgb, dtype=float))
    return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))


def extract_colors_kmeans(image: Image.Image, n_colors: int = 5) -> List[Dict]:
    """
    MACHINE LEARNING ALGORITHM: K-Means Clustering
    
    Extracts dominant colors from a fashion image by clustering pixels.
    
    Algorithm:
        1. Resize image to 150x150 (speed optimization, color info preserved)
        2. Flatten to array of N pixels × 3 channels (RGB)
        3. Initialize k cluster centroids randomly (k=n_colors)
        4. Assign each pixel to nearest centroid (Euclidean distance in RGB space)
        5. Update centroids as mean of assigned pixels
        6. Repeat steps 4-5 until convergence (centroid shift < threshold)
        7. Centroids = dominant colors; cluster sizes = color proportions
    
    Complexity: O(n × k × iterations) where n = pixels (~22,500), k = colors
    
    Why K-Means for color: Pixel clusters in RGB space naturally correspond to
    perceptually distinct colors. Centroid = "average" color in each cluster.
    
    Returns: List of color dicts with hex, rgb, percentage
    """
    # Preprocessing: resize for speed
    img_resized = image.resize((150, 150), Image.LANCZOS)
    
    # Convert to RGB if needed
    if img_resized.mode != 'RGB':
        img_resized = img_resized.convert('RGB')
    
    # Flatten to pixel array: (22500, 3)
    pixels = np.array(img_resized).reshape(-1, 3).astype(float)
    
    # K-Means clustering
    kmeans = KMeans(
        n_clusters=n_colors,
        random_state=42,
        n_init=10,          # 10 random initializations, best result kept
        max_iter=300,        # Maximum iterations
        algorithm='lloyd'    # Standard Lloyd's algorithm
    )
    kmeans.fit(pixels)
    
    # Get cluster centers (dominant colors) and their sizes
    centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    
    # Calculate proportion of each color
    total_pixels = len(labels)
    color_results = []
    
    for i, center in enumerate(centers):
        count = np.sum(labels == i)
        percentage = round((count / total_pixels) * 100, 1)
        
        r, g, b = center
        color_results.append({
            "hex": rgb_to_hex(r, g, b),
            "rgb": (int(r), int(g), int(b)),
            "percentage": percentage,
            "cluster_size": int(count)
        })
    
    # Sort by percentage (most dominant first)
    return sorted(color_results, key=lambda x: x['percentage'], reverse=True)


def extract_colors_gmm(image: Image.Image, n_components: int = 5) -> List[Dict]:
    """
    MACHINE LEARNING ALGORITHM: Gaussian Mixture Model (GMM)
    
    Alternative to K-Means — models pixel colors as mixture of Gaussian distributions.
    Better than K-Means for fashion images with gradients and subtle color variations.
    
    Algorithm (Expectation-Maximization):
        E-step: Compute probability of each pixel belonging to each Gaussian component
        M-step: Update Gaussian parameters (mean, covariance, weight) to maximize likelihood
        Repeat until log-likelihood converges
    
    Advantages over K-Means:
        - Soft assignments (pixel can partially belong to multiple clusters)
        - Models color spread via covariance matrices (handles color gradients)
        - Better for fashion fabrics with texture variations
    
    Returns: List of color dicts with hex, rgb, weight (probability)
    """
    img_resized = image.resize((100, 100), Image.LANCZOS)
    if img_resized.mode != 'RGB':
        img_resized = img_resized.convert('RGB')
    
    pixels = np.array(img_resized).reshape(-1, 3).astype(float) / 255.0  # Normalize to 0-1
    
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',    # Full covariance matrix (captures color correlations)
        random_state=42,
        max_iter=100,
        n_init=3
    )
    gmm.fit(pixels)
    
    # Get component means (dominant colors) and weights (proportions)
    means = (gmm.means_ * 255).astype(int)
    weights = gmm.weights_
    
    color_results = []
    for i, (mean, weight) in enumerate(zip(means, weights)):
        r, g, b = np.clip(mean, 0, 255)
        color_results.append({
            "hex": rgb_to_hex(r, g, b),
            "rgb": (int(r), int(g), int(b)),
            "percentage": round(weight * 100, 1),
            "method": "GMM"
        })
    
    return sorted(color_results, key=lambda x: x['percentage'], reverse=True)


def match_palette_to_fashion_category(extracted_colors: List[Dict]) -> Dict:
    """
    MACHINE LEARNING ALGORITHM: Nearest Neighbor Matching in LAB Color Space
    
    Matches extracted image colors to predefined fashion palette categories
    using Delta-E (perceptual color distance) as the distance metric.
    
    Algorithm:
        1. For each extracted color, compute Delta-E against all palette colors
        2. Sum minimum Delta-E scores for each palette category
        3. Category with lowest total distance = best match
        4. Score confidence as inverse of distance
    
    Returns: Best matching palette with confidence score
    """
    extracted_rgbs = [c['rgb'] for c in extracted_colors[:5]]
    
    palette_scores = {}
    
    for palette_name, palette_data in FASHION_PALETTES.items():
        palette_colors = palette_data['colors']
        total_distance = 0
        
        # For each extracted color, find the closest palette color
        for ext_rgb in extracted_rgbs:
            min_dist = min(
                delta_e(np.array(ext_rgb), np.array(pal_rgb))
                for pal_rgb in palette_colors
            )
            total_distance += min_dist
        
        avg_distance = total_distance / len(extracted_rgbs)
        # Convert distance to confidence (lower distance = higher confidence)
        confidence = max(0, 100 - avg_distance * 2)
        palette_scores[palette_name] = {
            "score": round(confidence, 1),
            "avg_distance": round(avg_distance, 2),
            "description": palette_data['description']
        }
    
    # Sort by confidence
    sorted_palettes = sorted(palette_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    best_match = sorted_palettes[0]
    return {
        "best_palette": best_match[0],
        "confidence": best_match[1]['score'],
        "description": best_match[1]['description'],
        "all_scores": dict(sorted_palettes[:4])
    }


def compute_color_harmony_score(colors: List[str]) -> Dict:
    """
    ALGORITHM: Color Theory Rule-Based Harmony Scoring
    
    Evaluates how harmonious a set of colors is using HSV color wheel rules:
        - Complementary: 180° apart on color wheel → high contrast, bold
        - Analogous: within 30° → cohesive, calming
        - Triadic: 120° apart → balanced, vibrant
        - Monochromatic: same hue, different saturation/value → elegant
    
    Returns: Harmony type and score (0-100)
    """
    if not colors:
        return {"score": 50, "harmony_type": "Unknown", "advice": ""}
    
    hues = []
    saturations = []
    values = []
    
    for hex_color in colors[:5]:
        try:
            r, g, b = hex_to_rgb(hex_color)
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            hues.append(h * 360)  # Convert to degrees
            saturations.append(s)
            values.append(v)
        except:
            continue
    
    if not hues:
        return {"score": 50, "harmony_type": "Neutral", "advice": "Balanced neutral palette"}
    
    hue_range = max(hues) - min(hues) if len(hues) > 1 else 0
    avg_saturation = np.mean(saturations)
    value_range = max(values) - min(values) if len(values) > 1 else 0
    
    # Harmony classification
    if avg_saturation < 0.15:
        harmony_type = "Monochromatic/Achromatic"
        score = 85
        advice = "Elegant neutral palette — perfect for minimalist and quiet luxury looks"
    elif hue_range < 30:
        harmony_type = "Analogous"
        score = 90
        advice = "Cohesive analogous colors — creates a harmonious, calming aesthetic"
    elif 150 <= hue_range <= 210:
        harmony_type = "Complementary"
        score = 78
        advice = "High contrast complementary palette — bold and eye-catching"
    elif 100 <= hue_range <= 140:
        harmony_type = "Split-Complementary"
        score = 82
        advice = "Balanced split-complementary — vibrant but not overwhelming"
    elif hue_range > 200:
        harmony_type = "Triadic / Tetradic"
        score = 70
        advice = "Multi-hue palette — great for maximalist and dopamine dressing styles"
    else:
        harmony_type = "Mixed"
        score = 65
        advice = "Eclectic palette — works for experimental and creative fashion"
    
    return {
        "score": score,
        "harmony_type": harmony_type,
        "advice": advice,
        "hue_range_deg": round(hue_range, 1),
        "avg_saturation_pct": round(avg_saturation * 100, 1)
    }


def generate_complementary_palette(base_hex: str, scheme: str = "analogous") -> List[str]:
    """
    ALGORITHM: HSV Color Wheel Mathematics for Palette Generation
    
    Generates a fashion-appropriate color palette from a base color using
    color theory relationships on the HSV color wheel.
    
    Schemes:
        - analogous: base ± 30° (harmonious)
        - complementary: base + 180° (high contrast)
        - triadic: base + 120° and + 240° (balanced vibrancy)
        - tetradic: base + 90°, 180°, 270° (complex richness)
    
    Returns: List of hex color strings forming the palette
    """
    r, g, b = hex_to_rgb(base_hex)
    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    h_deg = h * 360
    
    if scheme == "analogous":
        angles = [h_deg, h_deg + 30, h_deg - 30, h_deg + 15]
    elif scheme == "complementary":
        angles = [h_deg, h_deg + 180, h_deg + 30, h_deg + 210]
    elif scheme == "triadic":
        angles = [h_deg, h_deg + 120, h_deg + 240, h_deg + 60]
    elif scheme == "tetradic":
        angles = [h_deg, h_deg + 90, h_deg + 180, h_deg + 270]
    else:
        angles = [h_deg, h_deg + 30, h_deg - 30, h_deg + 180]
    
    palette = []
    for angle in angles:
        angle_norm = (angle % 360) / 360
        r_new, g_new, b_new = colorsys.hsv_to_rgb(angle_norm, s * 0.9, min(v * 1.05, 1.0))
        palette.append(rgb_to_hex(int(r_new * 255), int(g_new * 255), int(b_new * 255)))
    
    return palette


def full_color_analysis(image: Image.Image) -> Dict:
    """
    COMPLETE ML PIPELINE for Color Analysis
    
    Combines multiple algorithms:
        1. K-Means (n=6) → dominant colors with percentages
        2. GMM (n=4) → soft-clustered palette (handles gradients)
        3. LAB Delta-E → match to fashion palette categories
        4. Color Harmony Scoring → aesthetic quality score
        5. Complementary Palette Generation → styling suggestions
    
    Returns: Complete color intelligence report
    """
    # Step 1: K-Means dominant colors
    kmeans_colors = extract_colors_kmeans(image, n_colors=6)
    
    # Step 2: GMM soft colors
    gmm_colors = extract_colors_gmm(image, n_components=4)
    
    # Step 3: Fashion palette matching
    palette_match = match_palette_to_fashion_category(kmeans_colors)
    
    # Step 4: Harmony scoring
    hex_colors = [c['hex'] for c in kmeans_colors]
    harmony = compute_color_harmony_score(hex_colors)
    
    # Step 5: Generate complementary palettes (from most dominant color)
    if kmeans_colors:
        dominant = kmeans_colors[0]['hex']
        complementary = generate_complementary_palette(dominant, "analogous")
    else:
        complementary = []
    
    return {
        "dominant_colors_kmeans": kmeans_colors,
        "soft_colors_gmm": gmm_colors,
        "palette_match": palette_match,
        "harmony_score": harmony,
        "recommended_palette": complementary,
        "top_3_colors": [c['hex'] for c in kmeans_colors[:3]]
    }
