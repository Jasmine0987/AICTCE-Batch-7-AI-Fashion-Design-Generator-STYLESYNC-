# 🎨 AI Fashion Design Generator

> An end-to-end AI-powered fashion studio built with **Gemini 2.5 Flash**, **CLIP**, **MobileNetV2**, **Sentence-BERT**, **FAISS**, **K-Means**, **KeyBERT**, **GrabCut**, and **TF-IDF** — deployed as a Streamlit web application.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square&logo=streamlit)
![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-orange?style=flat-square&logo=google)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## What This Project Does

Most fashion apps show you what's popular. This one understands what you *mean*.

You can type something as vague as *"something flowy and romantic for a summer wedding under ₹2000"* and the system will run it through a pipeline of 9 ML algorithms — extracting style intent with KeyBERT, classifying aesthetics with CLIP, generating a complete outfit description with Gemini 2.5 Flash, and returning budget-filtered, Sentence-BERT-ranked product recommendations. All in under 3 seconds, in a browser, for free.

There are five distinct modes, each powered by a different combination of models working together behind the scenes.

---

## Five App Modes

### ✏️ Text → Design
Type any style description or mood. The system extracts keywords with KeyBERT + MMR, maps them to known fashion aesthetics using CLIP embeddings, analyzes color context with K-Means, and generates a structured outfit with design concept, key elements, color palette, and styling tips via Gemini 2.5 Flash. FAISS indexes each result for similarity search. Sentence-BERT ranks product recommendations by semantic relevance to the generated design.

### 📸 Image → Design
Upload any outfit photo. GrabCut isolates the clothing from the background, K-Means extracts the dominant color palette (5 clusters from up to 40,000 pixels), CLIP classifies the style aesthetic, and MobileNetV2 predicts garment type, pattern, formality, and season. Gemini Vision then analyzes the image and generates a remixed or evolved version of the look with new product recommendations.

### 🎯 Occasion Planner
Choose an occasion (wedding, office, casual, festive, etc.), set a budget, and pick a season. Gemini generates a complete curated look — including accessories and footwear — filtered to your constraints. Products are ranked by Sentence-BERT semantic similarity to the generated description, then hard-filtered against your budget.

### 🌊 Trend Explorer
Eight live micro-trends (Mob Wife, Ballet Core, Quiet Luxury, Dopamine Dressing, Old Money, Dark Academia, Coastal Grandmother, Gorpcore) each get a dedicated card. Click Generate on any trend and the full ML pipeline runs — sidebar occasion and season filters apply, and ranked product cards appear below the result.

### 📊 ML Analytics Dashboard
Interactive Plotly charts showing exactly how each algorithm contributes: model embedding dimensions, pipeline latency breakdown per stage, BERT ranking signal weights, and — if you've saved designs to your portfolio — live keyword frequency and color palette analysis from your actual session data.

---

## ML Pipeline Architecture

```
User Input (Text / Image)
         │
         ▼
┌─────────────────────────────────┐
│  NLP Layer                      │
│  KeyBERT + MMR keyword extract  │
│  Fashion NER entity tagging     │
└──────────────┬──────────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
CLIP        MobileNetV2  K-Means
512-d       1280-d feat  k=5 RGB
embeddings  prediction   centroids
    │          │          │
    └──────────┼──────────┘
               │  (Enriched context)
               ▼
      ┌─────────────────┐
      │  Gemini 2.5 Flash│
      │  Structured JSON │
      └────────┬─────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
FAISS      TF-IDF    Sentence-BERT
Index      Cosine    384-d ranking
similarity search    3-signal fusion
               │
               ▼
     Streamlit UI Output
```

---

## Algorithm Reference

| Algorithm | Type | What It Does In This Project |
|-----------|------|------------------------------|
| **CLIP ViT-B/32** | Deep Learning | Converts outfit images and text prompts into 512-d vectors in a shared semantic space. Measures cosine similarity to classify style aesthetics (cottagecore, dark academia, etc.) without any labelled training data |
| **MobileNetV2** | CNN — Deep Learning | Extracts 1280-d feature vectors from outfit photos. Predicts garment type, pattern (floral/solid/plaid), formality level, and season |
| **Sentence-BERT** | Transformer NLP | Converts design descriptions and product titles into 384-d semantic embeddings. Products whose vectors align with the generated design score higher — contributes 50% of the final ranking score |
| **K-Means (k=5)** | Classical ML | Clusters up to 40,000 RGB pixel samples from uploaded images into 5 dominant color groups. Centroids sorted by cluster size give you the color palette in order of prominence |
| **FAISS IndexFlatL2** | Vector Search | Stores CLIP embeddings of every generated design in memory. Performs exact L2 nearest-neighbor search to find visually similar past designs |
| **KeyBERT + MMR** | NLP — Hybrid | Extracts keywords from user prompts using BERT document embeddings. Maximal Marginal Relevance ensures diversity — no repetitive keywords. Enriches prompts before Gemini generation |
| **TF-IDF Cosine** | Classical ML | Weights words by frequency-in-document vs rarity-across-portfolio. Used for portfolio keyword search — finds saved designs that match a search query |
| **GrabCut** | Computer Vision | Graph-cut algorithm that iteratively separates foreground clothing from background in uploaded images. Preceded by Gaussian Blur to reduce noise before segmentation |
| **Gemini 2.5 Flash** | Generative AI | Receives the ML-enriched context (style tags, color palette, occasion, budget, season) and generates structured outfit descriptions. Also powers Gemini Vision for image analysis and remixing |

---

## Project Structure

```
fashion_ai/
│
├── app.py                          # Main Streamlit application (all 5 modes)
├── theme.py                        # Luxury CSS design system
├── requirements.txt                # All pinned dependencies
├── setup.sh                        # First-run setup script
├── .env.example                    # Template for API keys
│
└── modules/
    ├── __init__.py
    ├── gemini_designer.py          # Gemini 2.5 Flash — design generation + vision
    ├── clip_encoder.py             # CLIP ViT-B/32 — image/text embeddings
    ├── clip_style_classifier.py    # CLIP — zero-shot style classification
    ├── cnn_attribute_classifier.py # MobileNetV2 — garment attribute prediction
    ├── color_analysis.py           # GrabCut + Gaussian Blur — background removal
    ├── color_ml.py                 # K-Means — dominant color extraction
    ├── color_dna_text.py           # Color name mapping utilities
    ├── nlp_keywords.py             # KeyBERT + MMR — keyword extraction
    ├── style_classifier.py         # TF-IDF — portfolio search
    ├── recommendation_engine.py    # Sentence-BERT + FAISS — product ranking
    ├── product_finder.py           # Product search + budget filtering catalogue
    └── pdf_export.py               # FPDF2 — lookbook PDF export
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- A [Google Gemini API key](https://aistudio.google.com) (free tier: 1500 req/day on gemini-1.5-flash)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fashion-ai-studio.git
cd fashion-ai-studio
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ First install takes 5–10 minutes — PyTorch, CLIP, and transformers are large packages.

### 4. Set Up API Keys

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Open `.env` and fill in:

```env
GEMINI_API_KEY=your_gemini_api_key_here
SERPAPI_KEY=your_serpapi_key_here   # Optional — mock products used if absent
```

Get your Gemini API key for free at [aistudio.google.com](https://aistudio.google.com).

### 5. Run the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Deploying to Streamlit Cloud

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit — AI Fashion Studio"
git remote add origin https://github.com/YOUR_USERNAME/fashion-ai-studio.git
git branch -M main
git push -u origin main
```

### Step 2 — Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **New app**
3. Select your repository and set **Main file path** to `app.py`
4. Click **Advanced settings → Secrets** and add:

```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
SERPAPI_KEY = "your_serpapi_key_here"
```

5. Click **Deploy** — your app will be live at `https://your-app-name.streamlit.app` in about 2 minutes

> The app runs entirely on CPU. No GPU required. Streamlit Cloud free tier is sufficient.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | Google Gemini 2.5 Flash API key from [aistudio.google.com](https://aistudio.google.com) |
| `SERPAPI_KEY` | ❌ Optional | SerpAPI key for live Google Shopping results. If absent, a curated mock catalogue of 100+ products is used instead |

---

## API Rate Limits

| Model | Free Tier Limit | Recommendation |
|-------|----------------|----------------|
| `gemini-2.5-flash` | 20 requests/day | Switch to `gemini-1.5-flash` for development (1500 req/day free) |
| `gemini-1.5-flash` | 1500 requests/day | Use this for taking screenshots / demos |

To switch models, open `modules/gemini_designer.py` and replace every instance of `gemini-2.5-flash` with `gemini-1.5-flash`.

---

## Key Dependencies

```
streamlit==1.35.0               # Web framework
google-generativeai==0.7.2      # Gemini 2.5 Flash API
torch==2.3.0                    # PyTorch (CLIP + MobileNetV2)
torchvision==0.18.0             # MobileNetV2 pretrained weights
open-clip-torch==2.24.0         # CLIP ViT-B/32
transformers==4.41.2            # HuggingFace transformers
sentence-transformers==3.0.1    # Sentence-BERT MiniLM-L6-v2
faiss-cpu==1.8.0                # Vector similarity search
scikit-learn==1.5.0             # K-Means, TF-IDF
keybert==0.8.4                  # KeyBERT keyword extraction
opencv-python-headless==4.9.0.80 # GrabCut + Gaussian Blur
Pillow==10.3.0                  # Image processing
plotly==5.22.0                  # Interactive analytics charts
fpdf2==2.7.9                    # PDF lookbook export
colorthief==0.2.1               # Color extraction utilities
```

---

## Features At a Glance

- 🧠 **9 ML algorithms** working in a single coherent pipeline
- 🎨 **5 app modes** — text, image, occasion, trends, analytics
- 💰 **Budget filtering** — hard price cap across all product results
- 🍂 **Season + Occasion awareness** — sidebar filters propagate everywhere
- 📊 **ML transparency** — Plotly dashboard showing latency, embeddings, ranking weights
- 📄 **PDF Export** — save your designs as a downloadable lookbook
- 🛒 **Smart product ranking** — Sentence-BERT semantic similarity + style tag overlap + ratings
- 🔄 **Mock product fallback** — works fully without SerpAPI key
- ☁️ **Zero-cost deployment** — runs on Streamlit Cloud free tier, CPU only

---

## Screenshots

> Add your screenshots here after running the app locally.

| Mode | Screenshot |
|------|------------|
| Hero / Homepage | `screenshots/hero.png` |
| Text → Design output | `screenshots/text_design.png` |
| Image → Design (CLIP + colors) | `screenshots/image_design.png` |
| Occasion Planner result | `screenshots/occasion.png` |
| Trend Explorer | `screenshots/trends.png` |
| ML Analytics Dashboard | `screenshots/analytics.png` |
| Product cards with budget filter | `screenshots/products.png` |

---

## Future Scope

- **Meta LLaMA 3.2-Vision** — local inference to replace Gemini, eliminating API costs
- **Stable Diffusion / DALL-E 3** — generate actual photorealistic outfit images
- **Virtual Try-On** — DressCode / HR-VITON overlay on user photos
- **Collaborative Filtering** — LightFM-based style DNA profile that learns over time
- **Live Product APIs** — Myntra / Flipkart API for real-time pricing and stock
- **Mobile PWA** — React Native app with camera integration

---

## References

1. Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* OpenAI. ICML 2021.
2. Sandler et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* Google Brain. CVPR 2018.
3. Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019.
4. Johnson, Douze & Jégou (2019). *Billion-scale similarity search with GPUs.* IEEE Transactions on Big Data.
5. Grootendorst (2020). *KeyBERT: Minimal keyword extraction with BERT.* Zenodo.
6. Rother, Kolmogorov & Blake (2004). *GrabCut: Interactive foreground extraction using iterated graph cuts.* SIGGRAPH 2004.
7. MacQueen (1967). *Some methods for classification and analysis of multivariate observations.* 5th Berkeley Symposium on Mathematical Statistics.
8. Google DeepMind (2024). *Gemini: A Family of Highly Capable Multimodal Models.* Technical Report.



<div align="center">
  Built with Streamlit · Gemini 2.5 Flash · CLIP · PyTorch · HuggingFace
</div>
