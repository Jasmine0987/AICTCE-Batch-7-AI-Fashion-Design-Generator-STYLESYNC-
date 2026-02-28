"""
MODULE: clip_encoder.py
ALGORITHM: CLIP (Contrastive Language-Image Pretraining) — Deep Learning
PURPOSE:
    - Encodes fashion images into 512-dim embedding vectors using OpenAI's CLIP ViT-B/32
    - Encodes text style descriptions into the same embedding space
    - Enables cross-modal similarity: find images matching text prompts and vice versa
    - Used for: style similarity search, outfit matching, text-to-style retrieval

ARCHITECTURE:
    - Vision Encoder: ViT-B/32 (Vision Transformer, 12 layers, 12 heads, 768 hidden dim)
    - Text Encoder: 12-layer Transformer with masked self-attention
    - Both projected to 512-dim joint embedding space
    - Trained with contrastive loss on 400M image-text pairs
"""

import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import streamlit as st
from typing import Union, List
import os

# ─── Model Configuration ──────────────────────────────────────────────────────
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner=False)
def load_clip_model():
    """
    Load CLIP model and processor.
    Cached with st.cache_resource so it loads only once per session.
    
    Returns: (CLIPModel, CLIPProcessor) tuple
    """
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return model, processor


def encode_image(image: Image.Image) -> np.ndarray:
    """
    DEEP LEARNING - CLIP Vision Encoder (ViT-B/32)
    
    Converts a PIL image into a 512-dimensional embedding vector.
    This embedding captures semantic visual features (style, color, silhouette,
    texture) in a way that's comparable to text embeddings in the same space.
    
    Process:
        1. Resize & normalize image to 224x224 (CLIP standard)
        2. Patch the image into 7x7 grid of 32x32 patches
        3. Add [CLS] token and positional embeddings
        4. Pass through 12-layer ViT transformer
        5. Project [CLS] output to 512-dim via linear projection
        6. L2-normalize the final embedding
    
    Args:
        image: PIL Image of any fashion item
    Returns:
        np.ndarray of shape (512,) — normalized embedding vector
    """
    model, processor = load_clip_model()
    
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        # L2 normalize (unit sphere projection — standard for CLIP)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return image_features.cpu().numpy()[0]


def encode_text(text: str) -> np.ndarray:
    """
    DEEP LEARNING - CLIP Text Encoder (Transformer)
    
    Converts a text description into a 512-dimensional embedding vector.
    The embedding lives in the same geometric space as image embeddings,
    enabling direct text-to-image similarity measurement.
    
    Process:
        1. Tokenize text with BPE tokenizer (max 77 tokens)
        2. Add [SOS] and [EOS] tokens
        3. Pass through 12-layer causal attention transformer
        4. Use [EOS] token representation
        5. Project to 512-dim via linear layer
        6. L2-normalize
    
    Args:
        text: Fashion style description or design prompt
    Returns:
        np.ndarray of shape (512,) — normalized embedding vector
    """
    model, processor = load_clip_model()
    
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features.cpu().numpy()[0]


def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Cosine similarity between two CLIP embeddings.
    Since both are L2-normalized, cosine similarity = dot product.
    
    Range: -1 (opposite) to 1 (identical)
    Typical same-style range: 0.6 - 0.9
    
    ALGORITHM: Cosine Similarity
        similarity = (A · B) / (||A|| × ||B||)
        = dot(A, B)  [since embeddings are already unit vectors]
    """
    return float(np.dot(embedding1, embedding2))


def rank_styles_by_similarity(query_embedding: np.ndarray, 
                               style_embeddings: dict) -> list:
    """
    MACHINE LEARNING - Nearest Neighbor Retrieval in Embedding Space
    
    Ranks predefined fashion styles by their cosine similarity to a query
    embedding (from text or image). This is a k-NN search in embedding space.
    
    Returns: List of (style_name, similarity_score) sorted descending
    """
    scores = []
    for style_name, style_emb in style_embeddings.items():
        sim = compute_similarity(query_embedding, style_emb)
        scores.append((style_name, sim))
    
    return sorted(scores, key=lambda x: x[1], reverse=True)


def get_style_embeddings_cache() -> dict:
    """
    Pre-compute and cache text embeddings for all fashion style categories.
    
    ALGORITHM: Embedding Space Construction
    Each style is encoded using descriptive sentences that capture its essence.
    These become "prototype vectors" for style classification via nearest-neighbor.
    
    Returns: Dict mapping style names to 512-dim CLIP embeddings
    """
    style_descriptions = {
        "minimalist": "clean simple minimalist fashion white neutral tones structured silhouette",
        "cottagecore": "floral cottagecore vintage linen prairie dress botanical feminine",
        "streetwear": "urban streetwear oversized hoodie cargo pants sneakers graphic tee",
        "Y2K": "Y2K 2000s retro metallic low rise butterfly print colorful pop",
        "bohemian": "boho bohemian flowy maxi dress fringe earthy tones crochet free-spirited",
        "athleisure": "athletic sporty gym leggings activewear performance technical fabric",
        "vintage": "vintage retro classic 1970s 1980s heritage old-school fashion",
        "gothic": "gothic dark black velvet edgy dark academia mysterious fashion",
        "preppy": "preppy collegiate polo shirt khaki blazer clean cut classic American",
        "avant-garde": "avant garde experimental artistic unconventional deconstructed fashion art",
        "quiet luxury": "quiet luxury old money subtle expensive quality understated elegance",
        "dopamine dressing": "colorful bright dopamine dressing vibrant bold cheerful maximalist color pop"
    }
    
    embeddings = {}
    for style, description in style_descriptions.items():
        embeddings[style] = encode_text(description)
    
    return embeddings


def classify_image_style(image: Image.Image, top_k: int = 3) -> list:
    """
    DEEP LEARNING PIPELINE - Zero-Shot Image Style Classification
    
    Algorithm:
        1. Encode image with CLIP Vision Transformer → 512-dim vector
        2. Encode all style category descriptions with CLIP Text Encoder
        3. Compute cosine similarity between image and each style
        4. Return top-k styles with confidence scores
    
    This is ZERO-SHOT classification — no fine-tuning on fashion data needed.
    CLIP's multimodal pretraining enables direct image-text comparison.
    
    Args:
        image: PIL Image of fashion outfit
        top_k: Number of top styles to return
    Returns:
        List of (style_name, confidence_score) tuples
    """
    image_emb = encode_image(image)
    style_embeddings = get_style_embeddings_cache()
    rankings = rank_styles_by_similarity(image_emb, style_embeddings)
    
    # Convert to percentage confidence using softmax-like normalization
    top_results = rankings[:top_k]
    scores = [s for _, s in top_results]
    
    # Normalize scores to 0-100% range
    min_s, max_s = min(scores), max(scores)
    if max_s > min_s:
        normalized = [(name, round((s - min_s) / (max_s - min_s) * 100, 1)) 
                      for name, s in top_results]
    else:
        normalized = [(name, 100.0) for name, _ in top_results]
    
    return normalized


def find_similar_text_prompts(image: Image.Image, candidate_prompts: list) -> list:
    """
    DEEP LEARNING - Image-to-Text Retrieval using CLIP
    
    Given an uploaded outfit image, finds which text prompts/descriptions
    best match it using cross-modal embedding similarity.
    
    Useful for: "What style is this?" and generating good search keywords.
    
    Returns: Ranked list of (prompt, similarity) pairs
    """
    image_emb = encode_image(image)
    
    results = []
    for prompt in candidate_prompts:
        text_emb = encode_text(prompt)
        sim = compute_similarity(image_emb, text_emb)
        results.append((prompt, round(sim * 100, 1)))
    
    return sorted(results, key=lambda x: x[1], reverse=True)
