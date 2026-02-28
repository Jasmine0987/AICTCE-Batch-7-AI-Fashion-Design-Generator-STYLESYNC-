# AI Fashion Design Generator
### Gemini 2.5 Flash × CLIP × MobileNetV2 × FAISS × Sentence-BERT × K-Means

---

## ML/DL Algorithms Used

| Algorithm | Type | Library | Where Used |
|-----------|------|---------|-----------|
| CLIP ViT-B/32 | Deep Learning (Vision-Language) | open-clip-torch | Style classification, text/image embedding |
| K-Means Clustering | Classical ML | scikit-learn | Dominant color extraction from pixels |
| MobileNetV2 | Deep Learning (CNN) | torchvision | Fashion attribute extraction (1280-d features) |
| Sentence-BERT (MiniLM-L6) | Deep Learning (NLP/BERT) | sentence-transformers | Product ranking, semantic similarity |
| FAISS IndexFlatL2 | Approximate Nearest Neighbor | faiss-cpu | Design similarity search & recommendation |
| KeyBERT | DL + Statistical | keybert | Keyword extraction from design descriptions |
| TF-IDF + Cosine | Classical ML/NLP | scikit-learn | Portfolio keyword search |
| GrabCut | Computer Vision | opencv | Background removal before color analysis |
| Gaussian Blur | Signal Processing | opencv | Image preprocessing/denoising |
| Gemini 2.5 Flash | Generative AI | google-generativeai | Design text generation |

---

## Full Pipeline Flow

```
User Input (Text or Image)
        │
        ▼
NLP Layer: KeyBERT + Fashion NER → keyword extraction, prompt enhancement
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
CLIP ViT-B/32                         K-Means Clustering
(512-d embedding, zero-shot classify)  (5 dominant colors from pixels)
        │                                      │
        └──────────────────────────────────────┘
                          │
                          ▼ (Enriched Context)
                   Gemini 2.5 Flash
                (Structured design generation)
                          │
              ┌───────────┼──────────────┐
              ▼           ▼              ▼
         FAISS Index   TF-IDF        Sentence-BERT
         (similarity   (keyword      (product relevance
          search)       search)       ranking)
              │
              ▼
        Streamlit UI
```

---

## Project Structure

```
fashion_ai/
├── app.py                              ← Main Streamlit application (all 5 modes)
├── modules/
│   ├── __init__.py
│   ├── clip_style_classifier.py       ← CLIP zero-shot classification + embeddings
│   ├── cnn_attribute_classifier.py    ← MobileNetV2 transfer learning features
│   ├── color_analysis.py              ← K-Means + GrabCut + color harmony
│   ├── nlp_keywords.py                ← KeyBERT + NER + prompt enhancement
│   ├── recommendation_engine.py       ← FAISS + Sentence-BERT + TF-IDF
│   ├── gemini_designer.py             ← Gemini 2.5 Flash integration
│   ├── product_finder.py              ← SerpAPI product search
│   └── pdf_export.py                  ← FPDF2 lookbook generator
├── requirements.txt
├── .env.example
├── setup.sh                           ← One-command setup script
└── README.md
```

---

## Quick Setup

### Option A: Automated (Recommended)
```bash
git clone <your-repo>
cd fashion_ai
bash setup.sh
# Then edit .env and add GEMINI_API_KEY
streamlit run app.py
```

### Option B: Manual
```bash
cd fashion_ai
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
cp .env.example .env
# Edit .env → add GEMINI_API_KEY
streamlit run app.py
```

---

## API Keys

| Key | Required | Get It |
|-----|----------|--------|
| `GEMINI_API_KEY` | **Yes** | [aistudio.google.com](https://aistudio.google.com/app/apikey) — Free |
| `SERPAPI_KEY` | No | [serpapi.com](https://serpapi.com) — 100 free/month |

---

## App Modes

1. **Text → Design**: Type a description → full ML pipeline → design + products
2. **Image → Design**: Upload outfit → CLIP + MobileNetV2 + K-Means analysis + remix
3. **Occasion Planner**: Specify event → complete outfit with budget awareness
4. **Trend Explorer**: Explore current fashion trends → instant AI designs
5. **ML Analytics**: Transparency dashboard showing all algorithms and data flow

---

## Academic Report Mapping

| Section | What to Write |
|---------|--------------|
| Problem Statement | Fashion inaccessibility: expensive tools, artistic barriers, no AI-to-product bridge |
| Proposed Solution | End-to-end ML pipeline: CLIP + MobileNetV2 + Gemini + FAISS + BERT |
| Technology Used | As listed in algorithm table above |
| Algorithm & Deployment | The pipeline flow diagram above; deployed on Streamlit Cloud |
| Result | Generation time, CLIP accuracy, user satisfaction, product relevance scores |
| Conclusion | Democratizes fashion design via accessible AI |
| Future Scope | AR try-on, fine-tuned DeepFashion2 model, collaborative rooms, price tracking |

---

## System Requirements

- Python 3.10+
- RAM: 4GB minimum (8GB recommended for all models in memory)
- Storage: ~3GB (ML model weights)
- GPU: Optional (CPU works fine, ~2-5s inference)
