#!/bin/bash
echo "🚀 Setting up AI Fashion Design Generator..."
echo ""

# 1. Python version check
python_version=$(python3 --version 2>&1)
echo "✅ Python: $python_version"

# 2. Virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "✅ Virtual environment activated"

# 3. Upgrade pip
pip install --upgrade pip --quiet

# 4. Install dependencies
echo ""
echo "📥 Installing dependencies (this takes 3-5 minutes)..."
pip install -r requirements.txt --quiet

# 5. Download NLTK data
echo ""
echo "📚 Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True); print('✅ NLTK data downloaded')"

# 6. Pre-download ML models (optional but speeds up first run)
echo ""
echo "🧠 Pre-caching ML models (CLIP, BERT, MobileNetV2)..."
python3 -c "
print('  Loading CLIP ViT-B/32...')
try:
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    print('  ✅ CLIP loaded')
except Exception as e:
    print(f'  ⚠️ CLIP: {e}')

print('  Loading Sentence-BERT...')
try:
    from sentence_transformers import SentenceTransformer
    SentenceTransformer('all-MiniLM-L6-v2')
    print('  ✅ Sentence-BERT loaded')
except Exception as e:
    print(f'  ⚠️ BERT: {e}')

print('  Loading MobileNetV2...')
try:
    import torchvision.models as models
    models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    print('  ✅ MobileNetV2 loaded')
except Exception as e:
    print(f'  ⚠️ MobileNetV2: {e}')
"

# 7. Copy .env
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "⚠️  Created .env file. Please add your GEMINI_API_KEY to .env before running!"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your GEMINI_API_KEY"
echo "  2. Run: source venv/bin/activate"
echo "  3. Run: streamlit run app.py"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
