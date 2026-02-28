"""
MODULE: FAISS-Based Vector Similarity Recommendation Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALGORITHMS:
  1. FAISS IndexFlatL2        - Exact L2 nearest-neighbor search
  2. Cosine Similarity        - Semantic similarity between embeddings
  3. Weighted Score Fusion    - Combine CLIP + CNN + rating scores
  4. Sentence Transformers    - BERT-based text similarity (MiniLM)
  5. TF-IDF + Cosine          - Keyword similarity for product matching
"""

import numpy as np
import faiss
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

EMBEDDING_DIM = 512
CNN_FEATURE_DIM = 1280
SENTENCE_DIM = 384


@st.cache_resource(show_spinner=False)
def load_sentence_transformer():
    """
    DEEP LEARNING: Load BERT-based Sentence Transformer (MiniLM-L6-v2)
    Architecture: 6-layer BERT -> Mean pooling -> 384-dim embedding
    Training: Contrastive sentence-pair learning (SNLI, MultiNLI)
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception:
        return None


class FashionRecommendationEngine:
    """
    ALGORITHM: FAISS-Based Approximate Nearest Neighbor Recommendation
    How FAISS works:
    1. All design embeddings stored in an index (vector database)
    2. Query embedding compared against all stored embeddings
    3. L2 distance finds the closest designs
    4. Results ranked by distance (lower = more similar)
    """

    def __init__(self):
        self.clip_index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.cnn_index  = faiss.IndexFlatIP(CNN_FEATURE_DIM)
        self.design_metadata: List[Dict] = []
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500, stop_words='english')
        self.tfidf_matrix = None
        self.corpus: List[str] = []
        self._tfidf_fitted = False

    def add_design(self, design_id: str, clip_embedding: np.ndarray,
                   cnn_features: np.ndarray, metadata: Dict) -> None:
        clip_norm = clip_embedding / (np.linalg.norm(clip_embedding) + 1e-8)
        clip_norm = clip_norm.astype(np.float32).reshape(1, -1)
        cnn_norm  = cnn_features  / (np.linalg.norm(cnn_features)  + 1e-8)
        cnn_norm  = cnn_norm.astype(np.float32).reshape(1, -1)
        self.clip_index.add(clip_norm)
        self.cnn_index.add(cnn_norm)
        self.design_metadata.append({"id": design_id, "index_position": len(self.design_metadata), **metadata})
        text = f"{metadata.get('name', '')} {metadata.get('description', '')} {' '.join(metadata.get('tags', []))}"
        self.corpus.append(text)
        self._refit_tfidf()

    def _refit_tfidf(self):
        """TF-IDF = Term Frequency x Inverse Document Frequency for keyword matching"""
        if len(self.corpus) >= 2:
            try:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus)
                self._tfidf_fitted = True
            except Exception:
                self._tfidf_fitted = False

    def find_similar_designs(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """FAISS L2 nearest neighbor search - converts distance to similarity score"""
        if len(self.design_metadata) == 0:
            return []
        top_k = min(top_k, len(self.design_metadata))
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        query_norm = query_norm.astype(np.float32).reshape(1, -1)
        distances, indices = self.clip_index.search(query_norm, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.design_metadata):
                similarity = float(1.0 / (1.0 + dist))
                result = dict(self.design_metadata[idx])
                result["similarity_score"] = round(similarity, 3)
                results.append(result)
        return sorted(results, key=lambda x: x["similarity_score"], reverse=True)

    def keyword_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """TF-IDF Cosine Similarity Search for keyword-based matching"""
        if not self._tfidf_fitted or len(self.corpus) < 2:
            return []
        try:
            query_vec = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_indices = np.argsort(-similarities)[:top_k]
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.01 and idx < len(self.design_metadata):
                    result = dict(self.design_metadata[idx])
                    result["tfidf_similarity"] = round(float(similarities[idx]), 3)
                    results.append(result)
            return results
        except Exception:
            return []


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    DEEP LEARNING: BERT-based semantic text similarity
    Better than TF-IDF: captures synonyms and semantic meaning
    Example: 'cottagecore dress' <-> 'pastoral feminine outfit' -> high similarity
    """
    model = load_sentence_transformer()
    if model is None:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        union  = words1 | words2
        return float(len(words1 & words2) / len(union)) if union else 0.0
    try:
        embeddings = model.encode([text1, text2], convert_to_numpy=True)
        sim = cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0]
        return float(max(0.0, min(1.0, sim)))
    except Exception:
        return 0.0


def rank_products_by_relevance(products: List[Dict], design_description: str, style_tags: List[str]) -> List[Dict]:
    """
    ALGORITHM: Multi-signal product relevance ranking
    Combines 3 signals:
      1. Semantic similarity (BERT): description <-> product title
      2. Tag overlap: style tags <-> product keywords
      3. Rating normalization: product rating as quality signal
    Final score = 0.5 * semantic + 0.3 * tag_overlap + 0.2 * rating
    """
    if not products:
        return products

    query_text    = f"{design_description} {' '.join(style_tags)}"
    product_texts = [p.get("title", "") + " " + p.get("source", "") for p in products]

    model = load_sentence_transformer()
    semantic_scores = [0.5] * len(products)
    if model:
        try:
            from sentence_transformers import SentenceTransformer
            query_emb = model.encode([query_text])[0]
            prod_embs = model.encode(product_texts)
            sims = cosine_similarity(query_emb.reshape(1, -1), prod_embs)[0]
            semantic_scores = sims.tolist()
        except Exception:
            pass

    tag_set     = set([t.lower() for t in style_tags])
    all_ratings = [float(p.get("rating", 3.0) or 3.0) for p in products]
    max_rating  = max(all_ratings) if all_ratings else 5.0

    ranked = []
    for i, product in enumerate(products):
        prod_words  = set(product.get("title", "").lower().split())
        tag_overlap = len(tag_set & prod_words) / max(len(tag_set), 1)
        rating_norm = all_ratings[i] / max_rating
        sem_score   = semantic_scores[i] if i < len(semantic_scores) else 0.5
        final_score = 0.5 * sem_score + 0.3 * tag_overlap + 0.2 * rating_norm
        ranked.append({**product, "relevance_score": round(final_score, 3),
                       "semantic_similarity": round(float(sem_score), 3)})

    return sorted(ranked, key=lambda x: x["relevance_score"], reverse=True)


@st.cache_resource(show_spinner=False)
def get_recommendation_engine() -> FashionRecommendationEngine:
    return FashionRecommendationEngine()
