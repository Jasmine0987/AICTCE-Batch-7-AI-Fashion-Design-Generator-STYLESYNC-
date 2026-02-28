"""
AI Fashion Design Generator
━━━━━━━━━━━━━━━━━━━━━━━━━━
Tech Stack: Gemini 2.5 Flash + CLIP + MobileNetV2 + FAISS + BERT + KMeans + Streamlit

ML/DL Pipeline:
  User Input
       ↓
  NLP Entity Extraction (KeyBERT / Statistical KE)
       ↓
  Prompt Enhancement → Gemini 2.5 Flash
       ↓
  CLIP Zero-Shot Classification (style tags)
       ↓
  MobileNetV2 Feature Extraction (1280-dim CNN features)
       ↓
  KMeans Color Clustering (dominant colors)
       ↓
  FAISS Similarity Search (recommendation)
       ↓
  BERT-based Product Ranking (multi-signal)
       ↓
  Streamlit UI Output
"""

import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
import os
from dotenv import load_dotenv
# Add this near your imports
STANDARD_LAYOUT = dict(
    font=dict(color="#333333"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=30, b=30, l=30, r=30)
)

load_dotenv()

# ── ML/DL Module Imports ───────────────────────────────────────────────────────
from modules.gemini_designer import (
    generate_design_description, analyze_fashion_image_gemini,
    generate_trend_fusion_design, generate_occasion_outfit
)
from modules.clip_style_classifier import (
    classify_image_style, get_image_embedding, get_text_embedding,
    detect_occasion_from_prompt, STYLE_CATEGORIES
)
from modules.cnn_attribute_classifier import (
    extract_cnn_features, predict_garment_attributes
)
from modules.color_analysis import (
    extract_dominant_colors, analyze_color_harmony, remove_background_simple
)
from modules.nlp_keywords import (
    get_fashion_keywords_summary, enhance_user_prompt, extract_fashion_entities,
    build_product_search_query
)
from modules.recommendation_engine import (
    get_recommendation_engine, rank_products_by_relevance, compute_semantic_similarity
)
from modules.product_finder import search_fashion_products
from modules.pdf_export import create_lookbook_pdf

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fashion Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

from theme import LUXURY_CSS
st.markdown(LUXURY_CSS, unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────────
for key, val in [
    ('saved_designs', []),
    ('current_design', None),
    ('current_ml_results', None),
    ('rec_engine_ids', set()),
]:
    if key not in st.session_state:
        st.session_state[key] = val

TRENDING = [
    "Mob Wife", "Ballet Core", "Quiet Luxury",
    "Dopamine Dressing", "Old Money", "Dark Academia",
    "Coastal Grandmother", "Gorpcore"
]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ✦ Studio Controls")
    st.markdown("---")

    mode = st.radio("Mode", [
        "Text → Design",
        "Image → Design",
        "Occasion Planner",
        "Trend Explorer",
        "ML Analytics"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**⚙️ Generation Settings**")
    budget_on = st.toggle("Set Budget Limit", False)
    budget = None
    if budget_on:
        b_val = st.slider("Max Budget (₹)", 500, 15000, 2000, 500)
        budget = f"₹{b_val}"

    occasion_opt = st.selectbox("Occasion Filter", [
        "Any", "Casual Daily", "College", "Party/Night Out",
        "Wedding Guest", "Beach/Resort", "Formal/Office", "Festive"
    ])
    occasion = None if occasion_opt == "Any" else occasion_opt

    season_opt = st.selectbox("Season", ["Any", "Spring", "Summer", "Autumn", "Winter"])
    season = None if season_opt == "Any" else season_opt

    st.markdown("---")
    st.markdown(f"**📁 Portfolio ({len(st.session_state.saved_designs)} saved)**")
    if st.session_state.saved_designs:
        if st.button("📥 Export Lookbook PDF"):
            with st.spinner("Building lookbook..."):
                path = create_lookbook_pdf(st.session_state.saved_designs)
                with open(path, "rb") as f:
                    st.download_button("⬇️ Download", f.read(),
                        "fashion_lookbook.pdf", "application/pdf")
        if st.button("🗑️ Clear All"):
            st.session_state.saved_designs = []
            st.rerun()

    st.markdown("---")
    st.markdown("**🤖 ML Models Active**")
    st.markdown("""
    <div style="font-size:0.75rem;color:#555;line-height:2;">
    ✅ CLIP ViT-B/32<br>
    ✅ MobileNetV2<br>
    ✅ Sentence-BERT<br>
    ✅ K-Means Clustering<br>
    ✅ FAISS Index<br>
    ✅ KeyBERT<br>
    ✅ Gemini 2.5 Flash
    </div>
    """, unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">Generative AI × Deep Learning × Computer Vision</div>
    <div class="hero-title">StyleSync</div>
    <div class="hero-sub">CLIP Zero-Shot · MobileNetV2 CNN · FAISS Similarity · K-Means · BERT NLP · Gemini 2.5 Flash</div>
    <div class="hero-badges">
        <span class="hero-badge">CLIP Zero-Shot</span>
        <span class="hero-badge">MobileNetV2 CNN</span>
        <span class="hero-badge">FAISS Similarity</span>
        <span class="hero-badge">K-Means Colors</span>
        <span class="hero-badge">BERT NLP</span>
        <span class="hero-badge">Gemini 2.5 Flash</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Trending row
trend_html = "".join([f'<span class="trend-tag">{t}</span>' for t in TRENDING])
st.markdown(f'<div class="trend-row">{trend_html}</div>', unsafe_allow_html=True)

# ── Active Filters Banner ───────────────────────────────────────────────────────
active_filters = []
if budget:   active_filters.append(f"💰 Budget: {budget}")
if occasion: active_filters.append(f"🎯 Occasion: {occasion}")
if season:   active_filters.append(f"🍂 Season: {season}")
if active_filters:
    pills = "".join([f'<span style="display:inline-block;padding:4px 14px;margin:3px;'
                     f'background:#1C1C2E;color:#FFF;border-radius:20px;font-size:0.75rem;'
                     f'font-weight:500;">{f}</span>' for f in active_filters])
    st.markdown(f'<div style="text-align:center;margin-bottom:16px;">'
                f'<span style="font-size:0.72rem;color:#888;letter-spacing:1px;'
                f'text-transform:uppercase;margin-right:8px;">Active Filters</span>'
                f'{pills}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1: TEXT → DESIGN
# ═══════════════════════════════════════════════════════════════════════════════
if "Text" in mode:
    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        st.markdown('<div class="lux-card">', unsafe_allow_html=True)
        st.markdown("###  Describe Your Design")

        prompt = st.text_area(
            "desc", label_visibility="collapsed",
            placeholder="e.g. 'A flowy cottagecore sundress in dusty rose with floral embroidery and puff sleeves, perfect for a summer picnic'",
            height=110
        )

        # Quick prompts
        qcols = st.columns(4)
        quick = ["Cottagecore 🌸", "Y2K Party ✨", "Quiet Luxury 👔", "Festive Lehenga 🎊"]
        qmap = {
            "Cottagecore 🌸": "cottagecore floral sundress in ivory and dusty rose",
            "Y2K Party ✨": "Y2K inspired metallic mini dress with butterfly details",
            "Quiet Luxury 👔": "minimalist old money blazer and trouser set in camel",
            "Festive Lehenga 🎊": "embroidered lehenga choli in deep wine and gold"
        }
        for i, q in enumerate(quick):
            with qcols[i]:
                if st.button(q, key=f"q{i}"):
                    st.session_state['_qp'] = qmap[q]
                    st.rerun()

        # Populate prompt from quick-prompt BEFORE generate button is rendered
        if '_qp' in st.session_state and not prompt:
            prompt = st.session_state['_qp']

        use_trends = st.checkbox(" Incorporate trending styles",
                                  value=st.session_state.get('_use_trends', False))
        st.session_state['_use_trends'] = use_trends
        sel_trends = []
        if use_trends:
            sel_trends = st.multiselect("Select trends:", TRENDING,
                                         default=st.session_state.get('_sel_trends', [TRENDING[0]]))
            st.session_state['_sel_trends'] = sel_trends

        generate = st.button("✦ Generate Design with Full ML Pipeline")
        st.markdown('</div>', unsafe_allow_html=True)

        # NLP preview
        if prompt:
            st.markdown('<div class="lux-card-sm">', unsafe_allow_html=True)
            st.markdown("**🔍 NLP Entity Preview** *(KeyBERT + Statistical KE)*")
            entities = extract_fashion_entities(prompt)
            detected = []
            for k, v in entities.items():
                if v:
                    detected.extend([f"{item}" for item in v[:2]])
            if detected:
                tags_html = "".join([f'<span class="tag">{t}</span>' for t in detected[:8]])
                st.markdown(f'<div class="tag-row">{tags_html}</div>', unsafe_allow_html=True)
            else:
                st.caption("Type more descriptive text to see entity extraction")
            st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="lux-card">', unsafe_allow_html=True)
        st.markdown("###  Pipeline Overview")
        st.markdown("""
        <div class="ml-info">
            <div class="ml-info-title">🔬 Full ML Pipeline (Text Mode)</div>
            <div class="ml-info-body">
            1️⃣ NLP: KeyBERT keyword extraction → prompt enhancement<br>
            2️⃣ CLIP: Text embedding in joint vision-language space<br>
            3️⃣ Gemini 2.5 Flash: Structured design generation<br>
            4️⃣ FAISS: Add to similarity index for recommendations<br>
            5️⃣ BERT: Rank products by semantic relevance
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.session_state.saved_designs:
            st.markdown(f"**Recent Saves ({len(st.session_state.saved_designs)})**")
            for d in st.session_state.saved_designs[-3:]:
                st.markdown(f"""
                <div class="lux-card-sm" style="margin-bottom:8px;">
                    <span style="font-weight:600;font-size:0.85rem;">{d['name']}</span><br>
                    <span style="font-size:0.75rem;color:#888;">{d.get('created_at','')}</span>
                </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Generation ─────────────────────────────────────────────────────────────
    if generate and prompt:
        with st.spinner("🔬 Running full ML pipeline..."):
            progress = st.progress(0, text="Step 1/6: NLP entity extraction...")

            # STEP 1: NLP
            enhanced_prompt = enhance_user_prompt(prompt)
            nlp_summary = get_fashion_keywords_summary(prompt)
            auto_occasion = detect_occasion_from_prompt(prompt)
            progress.progress(17, "Step 2/6: CLIP text embedding...")

            # STEP 2: CLIP text embedding
            text_embedding = get_text_embedding(prompt)
            progress.progress(34, "Step 3/6: Gemini 2.5 Flash generation...")

            # STEP 3: Gemini generation — pass sidebar occasion + season filters
            if use_trends and sel_trends:
                result = generate_trend_fusion_design(enhanced_prompt, sel_trends)
            else:
                result = generate_design_description(
                    enhanced_prompt,
                    style_tags=nlp_summary.get('merged_keywords', []),
                    occasion=occasion or auto_occasion,
                    budget=budget,
                    trend_tags=[season] if season else None   # season injected as context
                )
            progress.progress(60, "Step 4/6: Building color palette from NLP...")

            # STEP 4: Extract color hints from text
            text_entities = extract_fashion_entities(result.get('description', ''))
            palette_colors = text_entities.get('colors', [])[:4]
            progress.progress(75, "Step 5/6: FAISS indexing...")

            # STEP 5: Add to FAISS index (using text embedding)
            rec_engine = get_recommendation_engine()
            design_id = f"design_{len(st.session_state.saved_designs)}"
            dummy_cnn = np.random.rand(1280).astype(np.float32)
            rec_engine.add_design(design_id, text_embedding, dummy_cnn, {
                "name": f"Design {len(st.session_state.saved_designs)+1}",
                "description": result.get('description', '')[:200],
                "tags": nlp_summary.get('merged_keywords', [])
            })
            progress.progress(88, "Step 6/6: Ranking products with BERT...")

            # STEP 6: Product search + BERT ranking
            # Build richer query by appending sidebar occasion + season filters
            base_query = build_product_search_query(
                result.get('description', ''), palette_colors,
                nlp_summary.get('merged_keywords', []), budget
            )
            occasion_context = " ".join(filter(None, [occasion or auto_occasion, season]))
            search_query = f"{base_query} {occasion_context}".strip()
            raw_products = search_fashion_products(search_query, budget)
            ranked_products = rank_products_by_relevance(
                raw_products, result.get('description', ''),
                nlp_summary.get('merged_keywords', [])
            )
            progress.progress(100, "✅ Pipeline complete!")
            progress.empty()

            st.session_state.current_design = result
            st.session_state.current_ml_results = {
                "nlp": nlp_summary,
                "enhanced_prompt": enhanced_prompt,
                "text_embedding_norm": float(np.linalg.norm(text_embedding)),
                "palette_colors": palette_colors,
                "auto_occasion": auto_occasion,
                "search_query": search_query,
                "products": ranked_products
            }

    # ── Display Results ─────────────────────────────────────────────────────────
    if st.session_state.current_design and st.session_state.current_ml_results:
        result = st.session_state.current_design
        ml    = st.session_state.current_ml_results

        st.markdown('<div class="lux-divider">✦ ✦ ✦</div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns([2, 1, 1], gap="large")

        with r1:
            st.markdown('<div class="lux-card">', unsafe_allow_html=True)
            st.markdown("### 🎨 Generated Design")
            st.markdown(result['description'])
            st.markdown('</div>', unsafe_allow_html=True)

        with r2:
            st.markdown('<div class="lux-card">', unsafe_allow_html=True)
            st.markdown("### 🧬 NLP Analysis")
            st.markdown("""<div class="ml-info">
                <div class="ml-info-title">Algorithm: KeyBERT + Statistical KE</div>
                <div class="ml-info-body">BERT encodes text → candidates ranked by cosine similarity to document embedding. MMR ensures keyword diversity.</div>
            </div>""", unsafe_allow_html=True)
            kws = ml['nlp'].get('merged_keywords', [])
            if kws:
                tags_html = "".join([f'<span class="tag">{k}</span>' for k in kws[:6]])
                st.markdown(f'<div class="tag-row">{tags_html}</div>', unsafe_allow_html=True)

            st.markdown(f"**Auto-detected occasion:** `{ml['auto_occasion']}`")
            st.markdown(f"**Embedding norm:** `{ml['text_embedding_norm']:.3f}`")

            if ml['palette_colors']:
                st.markdown("**Colors from NLP:**")
                c_html = "".join([f'<span class="tag tag-green">{c}</span>'
                                   for c in ml['palette_colors'][:4]])
                st.markdown(f'<div class="tag-row">{c_html}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with r3:
            st.markdown('<div class="lux-card">', unsafe_allow_html=True)
            st.markdown("### 💾 Save Design")
            dname = st.text_input("Name:", placeholder="e.g. 'Summer Boho Look'", key="dname_text")
            if st.button("💾 Save to Portfolio", key="save_text"):
                if dname:
                    st.session_state.saved_designs.append({
                        "name": dname,
                        "description": result.get('description', '')[:500],
                        "colors": ml['palette_colors'],
                        "style_tags": ml['nlp'].get('merged_keywords', [])[:5],
                        "ml_analysis": {"primary_style": ml['auto_occasion'], "confidence": 0.75},
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                    })
                    st.success("✅ Saved!")
                    st.rerun()
                else:
                    st.warning("Please enter a name")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── BERT-Ranked Products ────────────────────────────────────────────────
        if ml.get('products'):
            st.markdown("---")
            st.markdown("### 🛒 ML-Ranked Product Recommendations")

            # Hard-filter by budget AFTER ranking (belt-and-suspenders)
            display_products = ml['products']
            if budget:
                import re as _re
                budget_max = int(_re.sub(r"[^\d]", "", budget) or 999999)
                display_products = [p for p in display_products
                                    if int(_re.sub(r"[^\d]", "", p.get('price','0')) or 0) <= budget_max]
                if not display_products:
                    display_products = ml['products'][:1]   # always show at least 1

            budget_note = f" — Budget filter: {budget}" if budget else ""
            st.markdown(f"""
            <div class="ml-info">
                <div class="ml-info-title">Algorithm: Multi-Signal BERT Ranking{budget_note}</div>
                <div class="ml-info-body">
                Products ranked by: 50% Sentence-BERT semantic similarity + 30% style tag overlap + 20% rating normalization.<br>
                Search query: <code>{ml['search_query']}</code>
                </div>
            </div>""", unsafe_allow_html=True)

            pcols = st.columns(min(4, len(display_products)))
            for i, prod in enumerate(display_products[:4]):
                with pcols[i]:
                    score = prod.get('relevance_score', 0)
                    sem   = prod.get('semantic_similarity', 0)
                    st.markdown(f"""
                    <div class="prod-card">
                        <img class="prod-img" src="{prod['image']}"
                             onerror="this.src='https://via.placeholder.com/300x200?text=Fashion'">
                        <div class="prod-body">
                            <div class="prod-title">{prod['title'][:55]}...</div>
                            <div class="prod-price">{prod['price']}</div>
                            <div class="prod-source">📍 {prod['source']}</div>
                            <div class="prod-score">
                                Relevance: {score:.0%} | Semantic: {sem:.0%}
                            </div>
                            <a class="prod-btn" href="{prod['link']}" target="_blank">Shop Now →</a>
                        </div>
                    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 2: IMAGE → DESIGN
# ═══════════════════════════════════════════════════════════════════════════════
elif "Image" in mode:
    st.markdown("### 📸 Upload an Outfit → Full ML Analysis")

    c1, c2 = st.columns([1, 1.3], gap="large")
    with c1:
        st.markdown('<div class="lux-card">', unsafe_allow_html=True)
        file = st.file_uploader("Upload outfit photo", type=['jpg','jpeg','png','webp'],
                                 label_visibility="collapsed")
        if file:
            img = Image.open(file)
            st.image(img, use_column_width=True, caption="Uploaded Outfit")
            remix = st.text_area("How to remix?",
                placeholder="e.g. 'Make it more formal' or 'Winter version'", height=70)
            bg_remove = st.checkbox("🪄 Remove background before analysis (GrabCut)", True)
            run_btn = st.button("🔬 Run Full ML Analysis Pipeline")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        if file and run_btn:
            img = Image.open(file)
            with st.spinner("Running ML pipeline on image..."):
                prog = st.progress(0, "Step 1/5: Preprocessing (GrabCut)...")

                # STEP 1: Background removal
                analysis_img = remove_background_simple(img) if bg_remove else img
                prog.progress(15, "Step 2/5: K-Means color clustering...")

                # STEP 2: K-Means dominant colors
                colors = extract_dominant_colors(analysis_img, n_colors=5)
                harmony = analyze_color_harmony(colors)
                prog.progress(35, "Step 3/5: CLIP zero-shot classification...")

                # STEP 3: CLIP classification
                clip_result = classify_image_style(analysis_img, top_k=3)
                clip_emb    = get_image_embedding(analysis_img)
                prog.progress(55, "Step 4/5: MobileNetV2 attribute prediction...")

                # STEP 4: MobileNetV2 features
                cnn_result  = predict_garment_attributes(analysis_img)
                prog.progress(75, "Step 5/5: Gemini vision analysis + FAISS index...")

                # STEP 5: Gemini vision + FAISS
                gemini_vis  = analyze_fashion_image_gemini(img)
                rec_engine  = get_recommendation_engine()
                cnn_feats   = extract_cnn_features(analysis_img)
                rec_engine.add_design(f"img_{len(st.session_state.saved_designs)}",
                    clip_emb, cnn_feats,
                    {"name": "Image Upload", "tags": clip_result.get('top_styles', {})})

                # Generate remixed design — pass all sidebar filters
                style_tag_list = list(clip_result.get('top_styles', {}).keys())
                color_names    = [c['name'] for c in colors[:3]]
                design_prompt  = remix or f"Create a new design inspired by this {clip_result.get('primary_style', 'outfit')}"
                remix_result   = generate_design_description(
                    design_prompt, style_tags=style_tag_list, color_palette=color_names,
                    occasion=occasion, budget=budget
                )
                # Product search using CLIP styles + sidebar filters + budget
                img_search_kw = " ".join(style_tag_list[:3]) + f" {occasion or ''} {season or ''}".strip()
                img_products = rank_products_by_relevance(
                    search_fashion_products(img_search_kw, budget),
                    remix_result.get('description', ''), style_tag_list
                )
                st.session_state['img_products'] = img_products
                prog.progress(100, "✅ Done!")
                prog.empty()

            # ── Display ML Results ─────────────────────────────────────────────
            st.markdown('<div class="lux-card">', unsafe_allow_html=True)
            st.markdown("### 🤖 ML Analysis Results")

            # CLIP results
            st.markdown("""<div class="ml-info">
                <div class="ml-info-title">🎯 CLIP Zero-Shot Classification (ViT-B/32)</div>
                <div class="ml-info-body">Cosine similarity between image embedding and 15 style text embeddings in joint 512-dim space</div>
            </div>""", unsafe_allow_html=True)

            top_styles = clip_result.get('top_styles', {})
            for style, prob in list(top_styles.items())[:3]:
                pct = int(prob * 100)
                st.markdown(f"""
                <div class="conf-wrap">
                    <div class="conf-label">{style.title()} — {pct}%</div>
                    <div class="conf-bar-bg"><div class="conf-bar-fg" style="width:{pct}%;"></div></div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # KMeans colors
            st.markdown("""<div class="ml-info">
                <div class="ml-info-title">🎨 K-Means Color Clustering (k=5)</div>
                <div class="ml-info-body">40,000 pixel RGB vectors clustered into 5 centroids. Sorted by cluster size (dominance).</div>
            </div>""", unsafe_allow_html=True)

            swatch_html = '<div class="swatch-row">'
            for c in colors[:4]:
                swatch_html += f"""
                <div class="swatch">
                    <div class="swatch-dot" style="background:{c['hex']};"></div>
                    <div>
                        <div class="swatch-name">{c['name']}</div>
                        <div class="swatch-hex">{c['hex']} · {c['percentage']}%</div>
                    </div>
                </div>"""
            swatch_html += "</div>"
            st.markdown(swatch_html, unsafe_allow_html=True)
            st.caption(f"Color Harmony: **{harmony['harmony'].title()}** — {harmony['description']}")

            # CNN results
            st.markdown("""<div class="ml-info">
                <div class="ml-info-title">🧠 MobileNetV2 Attribute Prediction</div>
                <div class="ml-info-body">1280-dim CNN features + heuristic classifiers for garment type, pattern, season, formality</div>
            </div>""", unsafe_allow_html=True)

            mcols = st.columns(4)
            attrs = [
                ("Garment", cnn_result.get('garment_type', 'N/A')),
                ("Pattern", cnn_result.get('pattern', 'N/A')),
                ("Season",  cnn_result.get('season', 'N/A')),
                ("Formality", cnn_result.get('formality', 'N/A'))
            ]
            for col, (label, val) in zip(mcols, attrs):
                with col:
                    st.metric(label, val)

            st.markdown('</div>', unsafe_allow_html=True)

            # Gemini vision + Remix
            st.markdown('<div class="lux-card">', unsafe_allow_html=True)
            st.markdown("### ✨ Gemini Vision Analysis")
            st.markdown(gemini_vis.get('analysis', ''))
            st.markdown("---")
            st.markdown("### 🎨 Remixed Design")
            st.markdown(remix_result.get('description', ''))
            st.markdown('</div>', unsafe_allow_html=True)

            # Products filtered by sidebar budget
            if st.session_state.get('img_products'):
                budget_note = f" — Budget: {budget}" if budget else ""
                st.markdown(f"### 🛒 Recommended Products{budget_note}")
                pcols = st.columns(min(4, len(st.session_state['img_products'])))
                for i, prod in enumerate(st.session_state['img_products'][:4]):
                    with pcols[i]:
                        st.markdown(f"""
                        <div class="prod-card">
                            <img class="prod-img" src="{prod['image']}"
                                 onerror="this.src='https://via.placeholder.com/300x200?text=Fashion'">
                            <div class="prod-body">
                                <div class="prod-title">{prod['title'][:55]}...</div>
                                <div class="prod-price">{prod['price']}</div>
                                <div class="prod-source">📍 {prod['source']}</div>
                                <a class="prod-btn" href="{prod['link']}" target="_blank">Shop Now →</a>
                            </div>
                        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 3: OCCASION PLANNER
# ═══════════════════════════════════════════════════════════════════════════════
elif "Occasion" in mode:
    st.markdown("### 🎯 AI Occasion Outfit Planner")

    # ── Single column input form ────────────────────────────────────────────────
    st.markdown('<div class="lux-card">', unsafe_allow_html=True)
    occ = st.selectbox("Occasion:", [
        "Beach Wedding", "Corporate Interview", "First Date", "College Farewell",
        "Diwali Celebration", "New Year Party", "Casual Brunch", "Graduation Ceremony"
    ])
    pref = st.text_area("Your preferences:", placeholder="Colors you love, styles you prefer, what to avoid...", height=80)
    occ_budget = st.text_input("Budget:", placeholder="e.g. ₹3000", value=budget.replace("₹","") if budget else "")
    occ_season = st.selectbox("Season:", ["Spring", "Summer", "Autumn", "Winter", "Any"],
                               index=["Spring","Summer","Autumn","Winter","Any"].index(season) if season else 4)
    if st.button("🎯 Plan Complete Outfit"):
        with st.spinner("Planning your look..."):
            # Use occ_budget if typed, else fall back to sidebar budget
            effective_budget = f"₹{occ_budget}" if occ_budget.strip() else budget
            r = generate_occasion_outfit(occ, pref, effective_budget, occ_season)
            st.session_state['occ_result'] = r
            st.session_state['occ_occasion'] = occ
            # Build a richer search query from occasion + preferences + season
            kws = r.get('keywords', '')
            if not kws:
                kws = f"{occ} {pref} {occ_season} outfit"
            st.session_state['occ_products'] = rank_products_by_relevance(
                search_fashion_products(kws, effective_budget),
                r.get('description', ''), kws.split(',')
            )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Curated look — displayed BELOW the form ─────────────────────────────────
    if 'occ_result' in st.session_state:
        st.markdown('<div class="lux-card">', unsafe_allow_html=True)
        st.markdown(f"### 👗 Complete Look for {st.session_state.get('occ_occasion', occ)}")
        st.markdown(st.session_state['occ_result'].get('description', ''))
        st.markdown('</div>', unsafe_allow_html=True)

    if 'occ_products' in st.session_state and st.session_state['occ_products']:
        st.markdown("### 🛒 Recommended Products")

        # Hard-filter by effective budget
        import re as _re
        eff_budget = f"₹{occ_budget}" if 'occ_budget' in dir() and str(occ_budget).strip() else budget
        disp_prods = st.session_state['occ_products']
        if eff_budget:
            bmax = int(_re.sub(r"[^\d]", "", eff_budget) or 999999)
            filtered = [p for p in disp_prods if int(_re.sub(r"[^\d]", "", p.get('price','0')) or 0) <= bmax]
            if filtered: disp_prods = filtered

        pcols = st.columns(min(4, len(disp_prods)))
        for i, prod in enumerate(disp_prods[:4]):
            with pcols[i]:
                st.markdown(f"""
                <div class="prod-card">
                    <img class="prod-img" src="{prod['image']}" onerror="this.src='https://via.placeholder.com/300x200?text=Fashion'">
                    <div class="prod-body">
                        <div class="prod-title">{prod['title'][:55]}...</div>
                        <div class="prod-price">{prod['price']}</div>
                        <div class="prod-source">📍 {prod['source']}</div>
                        <a class="prod-btn" href="{prod['link']}" target="_blank">Shop Now →</a>
                    </div>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 4: TREND EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif "Trend" in mode:
    st.markdown("""
    <div style="margin-bottom:32px;">
        <div style="font-size:0.7rem;letter-spacing:3px;text-transform:uppercase;color:#888;margin-bottom:6px;">
            AI-Powered · Gemini 2.5 Flash
        </div>
        <div style="font-family:'Cormorant Garamond',serif;font-size:2.2rem;font-weight:300;
                    letter-spacing:2px;color:#0A0A0A;">Trend Explorer</div>
        <div style="font-size:0.8rem;color:#888;margin-top:4px;">
            Select any trend to generate a full outfit with ML pipeline.
            Sidebar occasion & season filters apply to all generations.
        </div>
    </div>
    """, unsafe_allow_html=True)

    TREND_META = {
        "Mob Wife":           {"sub": "Bold fur, maximalist jewellery, power silhouettes", "season": "AW"},
        "Ballet Core":        {"sub": "Soft tulle, wrap tops, satin ribbons, blush tones", "season": "SS"},
        "Quiet Luxury":       {"sub": "Minimal branding, neutral palette, premium fabrics", "season": "AW"},
        "Dopamine Dressing":  {"sub": "Clashing brights, playful prints, joyful colour",   "season": "SS"},
        "Old Money":          {"sub": "Tailored classics, muted tones, understated elegance","season": "AW"},
        "Dark Academia":      {"sub": "Plaid, corduroy, layered knits, gothic undertones",  "season": "AW"},
        "Coastal Grandmother":{"sub": "Linen, natural fibres, relaxed beach-town aesthetic","season": "SS"},
        "Gorpcore":           {"sub": "Outdoor tech-wear, functional layers, earth tones",  "season": "AW"},
    }

    for i, trend in enumerate(TRENDING):
        meta = TREND_META.get(trend, {"sub": "", "season": ""})
        has_result = f'trend_r_{i}' in st.session_state

        st.markdown(f"""
        <div class="lux-card" style="margin-bottom:16px;padding:24px 28px;">
            <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:24px;">
                <div style="flex:1;">
                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
                        <span style="font-family:'Cormorant Garamond',serif;font-size:1.5rem;
                                     font-weight:400;letter-spacing:1px;color:#0A0A0A;">{trend}</span>
                        <span style="font-size:0.65rem;font-weight:600;letter-spacing:2px;
                                     text-transform:uppercase;color:#888;background:#F2F2F2;
                                     padding:3px 8px;border-radius:2px;">{meta['season']}</span>
                    </div>
                    <div style="font-size:0.82rem;color:#666;line-height:1.5;">{meta['sub']}</div>
                </div>
                <div style="font-size:0.65rem;font-weight:700;letter-spacing:2px;
                            text-transform:uppercase;color:#0A0A0A;min-width:80px;text-align:right;">
                    Trend {str(i+1).zfill(2)}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        btn_col, result_col = st.columns([1, 3])
        with btn_col:
            filters_note = []
            if occasion: filters_note.append(occasion)
            if season:   filters_note.append(season)
            if budget:   filters_note.append(budget)
            label = "Generate Design" + (f"  [{', '.join(filters_note)}]" if filters_note else "")
            if st.button(label, key=f"tr_{i}"):
                with st.spinner(f"Generating {trend} outfit..."):
                    prompt_text = f"Create a complete {trend} aesthetic fashion outfit"
                    if occasion: prompt_text += f" suitable for {occasion}"
                    if season:   prompt_text += f" in {season}"
                    r = generate_design_description(prompt_text, occasion=occasion, budget=budget)
                    # Also search products for this trend
                    trend_kw = f"{trend} outfit {occasion or ''} {season or ''}".strip()
                    prods = rank_products_by_relevance(
                        search_fashion_products(trend_kw, budget),
                        r.get('description', ''), [trend]
                    )
                    st.session_state[f'trend_r_{i}'] = r
                    st.session_state[f'trend_p_{i}'] = prods

        with result_col:
            if has_result:
                r = st.session_state[f'trend_r_{i}']
                desc = r.get('description', '')
                with st.expander("View Generated Design", expanded=False):
                    st.markdown(desc[:600] + ("..." if len(desc) > 600 else ""))

        # Product cards for this trend
        if f'trend_p_{i}' in st.session_state and st.session_state[f'trend_p_{i}']:
            prods = st.session_state[f'trend_p_{i}'][:4]
            pcols = st.columns(len(prods))
            for j, prod in enumerate(prods):
                with pcols[j]:
                    st.markdown(f"""
                    <div class="prod-card">
                        <img class="prod-img" src="{prod['image']}"
                             onerror="this.src='https://via.placeholder.com/300x200?text=Fashion'">
                        <div class="prod-body">
                            <div class="prod-title">{prod['title'][:50]}...</div>
                            <div class="prod-price">{prod['price']}</div>
                            <div class="prod-source">📍 {prod['source']}</div>
                            <a class="prod-btn" href="{prod['link']}" target="_blank">Shop Now →</a>
                        </div>
                    </div>""", unsafe_allow_html=True)

        st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 5: ML ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif "Analytics" in mode:
    import pandas as pd
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        _has_plotly = True
    except ImportError:
        _has_plotly = False
        st.info("💡 Install plotly for interactive charts: `pip install plotly`")

    st.markdown("""
    <div style="margin-bottom:32px;">
        <div style="font-size:0.7rem;letter-spacing:3px;text-transform:uppercase;color:#888;margin-bottom:6px;">
            Model Transparency · 9 Algorithms
        </div>
        <div style="font-family:'Cormorant Garamond',serif;font-size:2.2rem;font-weight:300;
                    letter-spacing:2px;color:#0A0A0A;">ML Analytics Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top KPI metrics ────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Total Algorithms", "9", "Active in pipeline")
    with k2: st.metric("Deep Learning", "3", "CLIP · MobileNetV2 · BERT")
    with k3: st.metric("Classical ML", "3", "KMeans · TF-IDF · FAISS")
    with k4: st.metric("CV Operations", "2", "GrabCut · Gaussian Blur")

    st.markdown("---")

    if _has_plotly:
        # ── Row 1: Category donut + Performance bar ────────────────────────────────
        c1, c2 = st.columns([1, 1.6], gap="large")

    with c1:
        st.markdown('<div class="lux-card">', unsafe_allow_html=True)
        st.markdown("**Algorithm Category Mix**")
        fig_donut = go.Figure(go.Pie(
            labels=["Deep Learning", "Classical ML", "Computer Vision", "Generative AI"],
            values=[3, 3, 2, 1],
            hole=0.62,
            marker_colors=["#1a1a1a", "#E8733A", "#5B8DB8", "#8BC34A"],
            textfont_size=11,
            textinfo="label+percent",
        ))
        fig_donut.update_layout(
            showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
            height=260,
            paper_bgcolor="#2B2B2B", plot_bgcolor="#2B2B2B",
            annotations=[dict(text="9<br><span style='font-size:10px'>models</span>",
                              x=0.5, y=0.5, font_size=18, showarrow=False)]
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="lux-card">', unsafe_allow_html=True)
        st.markdown("**Model Embedding Dimensions**")
        models =    ["CLIP ViT-B/32", "MobileNetV2", "Sentence-BERT", "TF-IDF", "K-Means", "FAISS"]
        dims =      [512,             1280,           384,             500,       5,          512]
        colors_bar = ["#1a1a1a", "#E8733A", "#5B8DB8", "#AAAAAA", "#8BC34A", "#1a1a1a"]
        fig_bar = go.Figure(go.Bar(
            x=models, y=dims,
            marker_color=colors_bar,
            text=dims, textposition="outside",
            textfont=dict(size=10, color="#000000"),
        ))
        fig_bar.update_layout(
            height=260, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="#2B2B2B",
plot_bgcolor="#2B2B2B",
            yaxis=dict(showgrid=True, gridcolor="#F2F2F2", zeroline=False, title="Dimensions"),
            xaxis=dict(showgrid=False),
            font=dict(family="DM Sans", size=11, color="#333333"),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 2: Pipeline stage timing + BERT ranking signal weights ─────────────
    c3, c4 = st.columns([1.4, 1], gap="large")

    with c3:
        st.markdown('<div class="lux-card">', unsafe_allow_html=True)
        st.markdown("**Pipeline Stage — Estimated Latency (ms)**")
        stages = ["NLP / KeyBERT", "CLIP Embed", "Gemini 2.5 Flash", "K-Means Colors",
                  "MobileNetV2", "FAISS Index", "BERT Ranking"]
        latency = [120, 85, 1800, 60, 140, 20, 95]
        colors_lat = ["#E8733A" if l == max(latency) else ("#555555" if l > 100 else "#AAAAAA") for l in latency]
        fig_lat = go.Figure(go.Bar(
            y=stages, x=latency, orientation="h",
            marker_color=colors_lat,
            text=[f"{l}ms" for l in latency], textposition="outside",
            textfont=dict(size=10),
        ))
        fig_lat.update_layout(
            height=280, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="#2B2B2B",
plot_bgcolor="#2B2B2B",
            xaxis=dict(showgrid=True, gridcolor="#F2F2F2", zeroline=False, title="ms"),
            yaxis=dict(showgrid=False),
            font=dict(family="DM Sans", size=11, color="#333333"),
        )
        st.plotly_chart(fig_lat, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="lux-card">', unsafe_allow_html=True)
        st.markdown("**BERT Product Ranking — Signal Weights**")
        signals = ["Semantic<br>Similarity", "Style Tag<br>Overlap", "Rating<br>Score"]
        weights = [50, 30, 20]
        fig_rank = go.Figure(go.Bar(
            x=signals, y=weights,
            marker_color=["#1a1a1a", "#5B8DB8", "#BBBBBB"],
            text=[f"{w}%" for w in weights], textposition="outside",
            textfont=dict(size=12, color="#333"),
            width=0.5,
        ))
        fig_rank.update_layout(
            height=280, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="#2B2B2B",
plot_bgcolor="#2B2B2B",
            yaxis=dict(showgrid=True, gridcolor="#F2F2F2", zeroline=False,
                       title="Weight %", range=[0, 65]),
            xaxis=dict(showgrid=False),
            font=dict(family="DM Sans", size=11, color="#333333"),
        )
        st.plotly_chart(fig_rank, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 3: Live session stats (if designs exist) ───────────────────────────
    if st.session_state.saved_designs:
        st.markdown("---")
        st.markdown('<div class="lux-card">', unsafe_allow_html=True)
        st.markdown("**Session Portfolio Analysis**")
        sc1, sc2 = st.columns([1, 1])
        with sc1:
            # Style tag frequency
            from collections import Counter
            all_tags = []
            for d in st.session_state.saved_designs:
                all_tags.extend(d.get('style_tags', []))
            if all_tags:
                tag_counts = Counter(all_tags).most_common(8)
                fig_tags = go.Figure(go.Bar(
                    x=[t[0] for t in tag_counts],
                    y=[t[1] for t in tag_counts],
                    marker_color="#0A0A0A",
                ))
                fig_tags.update_layout(
                    height=220, title="Top Style Keywords in Your Designs",
                    margin=dict(t=40, b=10, l=10, r=10),
                    paper_bgcolor="#2B2B2B",
plot_bgcolor="#2B2B2B",
                    yaxis=dict(showgrid=True, gridcolor="#F2F2F2", zeroline=False),
                    font=dict(family="DM Sans", size=10, color="#333333"),
                )
                st.plotly_chart(fig_tags, use_container_width=True, config={"displayModeBar": False})
            else:
                st.caption("Save designs to see keyword analysis.")
        with sc2:
            # Color palette frequency
            all_colors = []
            for d in st.session_state.saved_designs:
                all_colors.extend(d.get('colors', []))
            if all_colors:
                color_counts = Counter(all_colors).most_common(6)
                fig_colors = go.Figure(go.Bar(
                    x=[c[0] for c in color_counts],
                    y=[c[1] for c in color_counts],
                    marker_color=px.colors.qualitative.Set2[:len(color_counts)],
                ))
                fig_colors.update_layout(
                    height=220, title="Dominant Colors Across Portfolio",
                    margin=dict(t=40, b=10, l=10, r=10),
                    paper_bgcolor="#2B2B2B",
plot_bgcolor="#2B2B2B",
                    yaxis=dict(showgrid=True, gridcolor="#F2F2F2", zeroline=False),
                    font=dict(family="DM Sans", size=10, color="#333333"),
                )
                st.plotly_chart(fig_colors, use_container_width=True, config={"displayModeBar": False})
            else:
                st.caption("Save designs to see color analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Algorithm Reference Table — always shown ───────────────────────────────
    st.markdown("---")
    st.markdown('<div class="lux-card">', unsafe_allow_html=True)
    st.markdown("**Algorithm Reference**")
    algo_data = {
        "Algorithm": ["CLIP ViT-B/32", "K-Means (k=5)", "MobileNetV2", "Sentence-BERT",
                      "FAISS IndexFlatL2", "KeyBERT + MMR", "TF-IDF Cosine", "GrabCut", "Gemini 2.5 Flash"],
        "Category":  ["Deep Learning", "Classical ML", "Deep Learning (CNN)", "Deep Learning (NLP)",
                      "Approx. NN", "DL + Classical", "Classical ML", "Computer Vision", "Generative AI"],
        "Purpose":   ["Zero-shot style classification", "Dominant color extraction",
                      "Garment attribute prediction", "Product semantic ranking",
                      "Design similarity search", "Keyword extraction w/ diversity",
                      "Portfolio keyword matching", "Background isolation", "Design text generation"],
        "Output":    ["512-d embedding", "k=5 RGB centroids", "1280-d features", "384-d embedding",
                      "Top-k L2 neighbors", "Ranked keyword list", "Cosine score", "FG/BG mask",
                      "Structured text"],
    }
    df = pd.DataFrame(algo_data)
    st.table(df)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center;color:#CCC;font-size:0.8rem;padding:24px 0;margin-top:20px;">
✦ AI Fashion Studio &nbsp;|&nbsp; Gemini 2.5 Flash · CLIP · MobileNetV2 · FAISS · BERT · KMeans · Streamlit ✦
</div>
""", unsafe_allow_html=True)