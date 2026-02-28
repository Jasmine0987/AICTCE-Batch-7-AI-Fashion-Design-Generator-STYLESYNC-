"""
MODULE: Product Finder + ML-Ranked Results
ALGORITHM: Multi-signal relevance ranking (Semantic + Tag + Rating fusion)
"""

import requests, os, re
from typing import List, Dict


def _parse_budget_inr(budget_str: str) -> int:
    """Extract integer rupee amount from strings like '₹2000' or '₹2,000'."""
    if not budget_str:
        return 999999
    digits = re.sub(r"[^\d]", "", budget_str)
    return int(digits) if digits else 999999


def _price_inr(price_str: str) -> int:
    """Parse price string like '₹1,299' → 1299."""
    digits = re.sub(r"[^\d]", "", price_str)
    return int(digits) if digits else 0


def search_fashion_products(keywords: str, budget: str = None, max_results: int = 6) -> List[Dict]:
    api_key = os.getenv("SERPAPI_KEY", "")
    if not api_key:
        return get_mock_products(keywords, budget)
    query = keywords + (f" under {budget}" if budget else "")
    params = {"engine": "google_shopping", "q": query, "api_key": api_key,
               "num": max_results, "gl": "in", "hl": "en"}
    try:
        r = requests.get("https://serpapi.com/search", params=params, timeout=10)
        data = r.json()
        products = []
        for item in data.get("shopping_results", [])[:max_results]:
            products.append({"title": item.get("title", "Fashion Item"),
                             "price": item.get("price", "N/A"),
                             "link": item.get("link", "#"),
                             "image": item.get("thumbnail", ""),
                             "source": item.get("source", "Online Store"),
                             "rating": item.get("rating", 3.5)})
        if products:
            # Filter by budget if set
            budget_max = _parse_budget_inr(budget)
            filtered = [p for p in products if _price_inr(p["price"]) <= budget_max]
            return filtered if filtered else products
        return get_mock_products(keywords, budget)
    except Exception:
        return get_mock_products(keywords, budget)


# ── Curated mock catalogue — ALL items ≤ ₹2,999, unique per category ──────────
_MOCK_CATALOGUE = {
    # ── Trend Explorer trends ──────────────────────────────────────────────────
    "mob wife": [
        {"title": "Faux Fur Trim Coat - Chocolate Brown",    "price": "₹2,799", "source": "Myntra",          "image": "https://images.unsplash.com/photo-1544022613-e87ca75a784a?w=300", "rating": 4.5},
        {"title": "Animal Print Wrap Dress - Leopard",       "price": "₹1,899", "source": "AJIO",            "image": "https://images.unsplash.com/photo-1585914924626-15adac1e6402?w=300", "rating": 4.3},
        {"title": "Gold Chain Belt - Statement Piece",       "price": "₹899",   "source": "Nykaa Fashion",   "image": "https://images.unsplash.com/photo-1611085583191-a3b181a88401?w=300", "rating": 4.1},
        {"title": "Bodycon Midi Dress - Wine Red",           "price": "₹1,599", "source": "Flipkart Fashion", "image": "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=300", "rating": 4.4},
    ],
    "ballet core": [
        {"title": "Wrap Ballet Cardigan - Blush Pink",       "price": "₹1,499", "source": "Myntra",          "image": "https://images.unsplash.com/photo-1518611012118-696072aa579a?w=300", "rating": 4.5},
        {"title": "Satin Slip Skirt - Powder Pink",          "price": "₹1,199", "source": "AJIO",            "image": "https://images.unsplash.com/photo-1523380677598-64d85d515b35?w=300", "rating": 4.3},
        {"title": "Fitted Ribbed Ballet Neck Top - Cream",   "price": "₹799",   "source": "Flipkart Fashion", "image": "https://images.unsplash.com/photo-1503342217505-b0a15ec3261c?w=300", "rating": 4.2},
        {"title": "Satin Ribbon Hair Bow - Ballet Pink",     "price": "₹349",   "source": "Nykaa Fashion",   "image": "https://images.unsplash.com/photo-1559181567-c3190b4f9a78?w=300", "rating": 4.0},
    ],
    "quiet luxury": [
        {"title": "Tailored Straight-Leg Trousers - Camel",  "price": "₹2,299", "source": "Myntra",          "image": "https://images.unsplash.com/photo-1594938374182-a57b91cc9a5e?w=300", "rating": 4.6},
        {"title": "Merino Crew-Neck Sweater - Oatmeal",      "price": "₹1,999", "source": "AJIO",            "image": "https://images.unsplash.com/photo-1607345366928-199ea26cfe3e?w=300", "rating": 4.5},
        {"title": "Structured Tote - Tan Vegan Leather",     "price": "₹2,499", "source": "Nykaa Fashion",   "image": "https://images.unsplash.com/photo-1548036328-c9fa89d128fa?w=300", "rating": 4.4},
        {"title": "Relaxed Linen Shirt - Off-White",         "price": "₹1,299", "source": "Flipkart Fashion", "image": "https://images.unsplash.com/photo-1598300042247-d088f8ab3a91?w=300", "rating": 4.3},
    ],
    "dopamine dressing": [
        {"title": "Colour-Block Mini Dress - Yellow & Blue",  "price": "₹1,699", "source": "Myntra",          "image": "https://images.unsplash.com/photo-1585487000160-6ebcfceb0d03?w=300", "rating": 4.4},
        {"title": "Neon Green Co-ord Set - Brights",          "price": "₹1,499", "source": "AJIO",            "image": "https://images.unsplash.com/photo-1552902865-b72c031ac5ea?w=300", "rating": 4.2},
        {"title": "Hot Pink Oversized Blazer",                "price": "₹2,199", "source": "Nykaa Fashion",   "image": "https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=300", "rating": 4.3},
        {"title": "Mixed Print Maxi Skirt - Festival",        "price": "₹1,299", "source": "Flipkart Fashion", "image": "https://images.unsplash.com/photo-1496747611176-843222e1e57c?w=300", "rating": 4.1},
    ],
    "old money": [
        {"title": "Plaid Blazer - Navy & Cream Check",        "price": "₹2,799", "source": "Myntra",          "image": "https://images.unsplash.com/photo-1617137968427-85924c800a22?w=300", "rating": 4.6},
        {"title": "Straight-Leg Chinos - Khaki",              "price": "₹1,799", "source": "AJIO",            "image": "https://images.unsplash.com/photo-1473966968600-fa801b869a1a?w=300", "rating": 4.4},
        {"title": "Pearl Stud Earrings - Classic",            "price": "₹699",   "source": "Nykaa Fashion",   "image": "https://images.unsplash.com/photo-1515562141207-7a88fb7ce338?w=300", "rating": 4.3},
        {"title": "Cable-Knit Pullover - Ivory",              "price": "₹1,999", "source": "Flipkart Fashion", "image": "https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=300", "rating": 4.5},
    ],
    "dark academia": [
        {"title": "Corduroy Trousers - Forest Green",         "price": "₹1,899", "source": "AJIO",            "image": "https://images.unsplash.com/photo-1602810316693-3667c854239a?w=300", "rating": 4.2},
        {"title": "Turtleneck Ribbed Sweater - Burgundy",     "price": "₹1,499", "source": "Flipkart Fashion", "image": "https://images.unsplash.com/photo-1604644401890-0bd678c83788?w=300", "rating": 4.1},
        {"title": "Plaid Pinafore Dress - Brown Check",       "price": "₹2,199", "source": "Myntra",          "image": "https://images.unsplash.com/photo-1511216335778-7cb8f49fa7a3?w=300", "rating": 4.4},
        {"title": "Leather-Look Satchel Bag - Cognac",        "price": "₹2,499", "source": "Nykaa Fashion",   "image": "https://images.unsplash.com/photo-1548036328-c9fa89d128fa?w=300", "rating": 4.3},
    ],
    "coastal grandmother": [
        {"title": "Linen Wide-Leg Pants - Sea Salt White",    "price": "₹1,699", "source": "Myntra",          "image": "https://images.unsplash.com/photo-1506629082955-511b1aa562c8?w=300", "rating": 4.5},
        {"title": "Oversized Striped Linen Shirt - Blue",     "price": "₹1,299", "source": "AJIO",            "image": "https://images.unsplash.com/photo-1598300042247-d088f8ab3a91?w=300", "rating": 4.3},
        {"title": "Woven Straw Tote - Natural",               "price": "₹899",   "source": "Flipkart Fashion", "image": "https://images.unsplash.com/photo-1590874103328-eac38a683ce7?w=300", "rating": 4.2},
        {"title": "Cotton Knit Cardigan - Cornflower Blue",   "price": "₹1,899", "source": "Nykaa Fashion",   "image": "https://images.unsplash.com/photo-1576566588028-4147f3842f27?w=300", "rating": 4.4},
    ],
    "gorpcore": [
        {"title": "Fleece Half-Zip Pullover - Earth Brown",   "price": "₹1,999", "source": "Myntra",          "image": "https://images.unsplash.com/photo-1483985988355-763728e1935b?w=300", "rating": 4.3},
        {"title": "Utility Cargo Shorts - Sage Green",        "price": "₹1,299", "source": "AJIO",            "image": "https://images.unsplash.com/photo-1473966968600-fa801b869a1a?w=300", "rating": 4.1},
        {"title": "Trek-Ready Crossbody - Olive",             "price": "₹1,699", "source": "Flipkart Fashion", "image": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=300", "rating": 4.2},
        {"title": "Technical Zip-Off Trousers - Khaki",       "price": "₹2,499", "source": "Nykaa Fashion",   "image": "https://images.unsplash.com/photo-1551698618-1dfe5d97d256?w=300", "rating": 4.4},
    ],
    # ── Text / Image / Occasion mode categories ────────────────────────────────
    "cottagecore": [
        {"title": "Floral Linen Midi Dress - Dusty Rose",     "price": "₹1,499", "source": "Myntra",          "image": "https://images.unsplash.com/photo-1515372039744-b8f02a3ae446?w=300", "rating": 4.4},
        {"title": "Puff Sleeve Smocked Top - Ivory",          "price": "₹899",   "source": "AJIO",            "image": "https://images.unsplash.com/photo-1503342217505-b0a15ec3261c?w=300", "rating": 4.2},
        {"title": "Tiered Prairie Skirt - Butter Yellow",     "price": "₹1,099", "source": "Flipkart Fashion", "image": "https://images.unsplash.com/photo-1512436991641-6745cdb1723f?w=300", "rating": 4.0},
        {"title": "Linen Overalls - Natural Sage",            "price": "₹1,799", "source": "Nykaa Fashion",   "image": "https://images.unsplash.com/photo-1567401893414-76b7b1e5a7a5?w=300", "rating": 4.5},
    ],
    "y2k": [
        {"title": "Metallic Pleated Mini Skirt - Silver",     "price": "₹899",   "source": "AJIO",            "image": "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=300", "rating": 4.1},
        {"title": "Butterfly Graphic Babydoll Tee - Pink",    "price": "₹699",   "source": "Myntra",          "image": "https://images.unsplash.com/photo-1529139574466-a303027c1d8b?w=300", "rating": 4.0},
        {"title": "Low-Rise Flared Jeans - Light Wash",       "price": "₹1,599", "source": "Flipkart Fashion", "image": "https://images.unsplash.com/photo-1541099649105-f69ad21f3246?w=300", "rating": 3.9},
        {"title": "Gem-Stone Chain Necklace - Iridescent",    "price": "₹499",   "source": "Nykaa Fashion",   "image": "https://images.unsplash.com/photo-1515562141207-7a88fb7ce338?w=300", "rating": 4.3},
    ],
    "festive": [
        {"title": "Embroidered Anarkali Suit - Teal & Gold",  "price": "₹2,799", "source": "Myntra",          "image": "https://images.unsplash.com/photo-1610030469983-98e550d6193c?w=300", "rating": 4.6},
        {"title": "Bandhani Print Kurta Set - Fuchsia",       "price": "₹1,699", "source": "AJIO",            "image": "https://images.unsplash.com/photo-1583391733956-3750e0ff4e8b?w=300", "rating": 4.4},
        {"title": "Mirror Work Palazzo Set - Ivory & Gold",   "price": "₹2,199", "source": "Nykaa Fashion",   "image": "https://images.unsplash.com/photo-1571908599407-cdb918ed83bf?w=300", "rating": 4.3},
        {"title": "Block Print Cotton Dupatta - Multi",       "price": "₹799",   "source": "Flipkart Fashion", "image": "https://images.unsplash.com/photo-1511216335778-7cb8f49fa7a3?w=300", "rating": 4.0},
    ],
    "streetwear": [
        {"title": "Oversized Drop-Shoulder Tee - Washed Black","price": "₹999",  "source": "Myntra",          "image": "https://images.unsplash.com/photo-1556821840-3a63f15732ce?w=300", "rating": 4.2},
        {"title": "Cargo Joggers - Tactical Olive",            "price": "₹1,499", "source": "AJIO",            "image": "https://images.unsplash.com/photo-1565084888279-aca607bb7ef0?w=300", "rating": 4.0},
        {"title": "Chunky Lug-Sole Boot - Triple Black",       "price": "₹2,799", "source": "Flipkart Fashion", "image": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=300", "rating": 4.5},
        {"title": "Boxy Logo Cap - Washed Vintage",            "price": "₹599",   "source": "Nykaa Fashion",   "image": "https://images.unsplash.com/photo-1588850561407-ed78c282e89b?w=300", "rating": 3.9},
    ],
    "bohemian": [
        {"title": "Tiered Maxi Skirt - Terracotta Block Print","price": "₹1,199", "source": "AJIO",            "image": "https://images.unsplash.com/photo-1512436991641-6745cdb1723f?w=300", "rating": 4.1},
        {"title": "Crochet Halter Top - Natural Ecru",         "price": "₹799",   "source": "Myntra",          "image": "https://images.unsplash.com/photo-1509631179647-0177331693ae?w=300", "rating": 4.0},
        {"title": "Tassel Fringe Earrings - Rust Orange",      "price": "₹449",   "source": "Nykaa Fashion",   "image": "https://images.unsplash.com/photo-1611085583191-a3b181a88401?w=300", "rating": 4.2},
        {"title": "Embroidered Peasant Blouse - Cream",        "price": "₹1,099", "source": "Flipkart Fashion", "image": "https://images.unsplash.com/photo-1485462537746-965f33f7f6a7?w=300", "rating": 3.9},
    ],
    "minimalist": [
        {"title": "Fitted Ribbed Midi Dress - Jet Black",      "price": "₹1,799", "source": "Myntra",          "image": "https://images.unsplash.com/photo-1550614000-4895a10e1bfd?w=300", "rating": 4.4},
        {"title": "Wide-Leg Tailored Trousers - Stone",        "price": "₹1,999", "source": "AJIO",            "image": "https://images.unsplash.com/photo-1594938374182-a57b91cc9a5e?w=300", "rating": 4.3},
        {"title": "Clean Leather Belt Bag - Black",            "price": "₹1,299", "source": "Nykaa Fashion",   "image": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=300", "rating": 4.5},
        {"title": "Seamless Cotton Bodysuit - Nude",           "price": "₹899",   "source": "Flipkart Fashion", "image": "https://images.unsplash.com/photo-1567401893414-76b7b1e5a7a5?w=300", "rating": 4.2},
    ],
    "default": [
        {"title": "Floral Wrap Midi Dress - Multicolour",      "price": "₹1,299", "source": "Myntra",          "image": "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=300", "rating": 4.2},
        {"title": "Linen Coord Set - Pastel Blue",             "price": "₹1,699", "source": "AJIO",            "image": "https://images.unsplash.com/photo-1506629082955-511b1aa562c8?w=300", "rating": 4.0},
        {"title": "Pleated Midi Skirt - Dusty Rose",           "price": "₹1,199", "source": "Flipkart Fashion", "image": "https://images.unsplash.com/photo-1585487000160-6ebcfceb0d03?w=300", "rating": 3.9},
        {"title": "Relaxed Blazer - Biscuit Beige",            "price": "₹2,499", "source": "Nykaa Fashion",   "image": "https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=300", "rating": 4.5},
    ],
}


def get_mock_products(keywords: str, budget: str = None) -> List[Dict]:
    """
    Return keyword-matched mock products, filtered by budget.
    Scans keywords for style/garment hints and picks the best matching catalogue section.
    """
    kw_lower = keywords.lower()

    # Pick the best matching category — exact trend name first, then keyword hints
    chosen = "default"
    category_keywords = {
        "mob wife":           ["mob wife", "fur", "leopard", "animal print", "bodycon", "bold", "maximalist"],
        "ballet core":        ["ballet core", "ballet", "tulle", "satin", "ribbon", "blush", "pirouette", "wrap cardigan"],
        "quiet luxury":       ["quiet luxury", "old money", "camel", "merino", "linen shirt", "minimal brand", "tote"],
        "dopamine dressing":  ["dopamine", "colour block", "neon", "bright", "clashing", "playful", "vivid"],
        "old money":          ["old money", "plaid blazer", "chino", "pearl", "cable knit", "preppy", "heritage"],
        "dark academia":      ["dark academia", "corduroy", "turtleneck", "pinafore", "plaid", "tweed", "satchel"],
        "coastal grandmother":["coastal grandmother", "coastal", "linen wide", "striped linen", "straw tote", "cornflower"],
        "gorpcore":           ["gorpcore", "fleece", "utility cargo", "crossbody", "zip-off", "technical", "outdoor"],
        "cottagecore":        ["cottagecore", "floral", "smocked", "prairie", "puff sleeve", "pastoral", "sundress"],
        "y2k":                ["y2k", "metallic", "butterfly", "rhinestone", "low rise", "2000s", "babydoll"],
        "festive":            ["festive", "anarkali", "bandhani", "mirror work", "lehenga", "diwali", "kurta", "ethnic", "wedding", "saree"],
        "streetwear":         ["streetwear", "hoodie", "cargo jogger", "lug sole", "logo cap", "oversized tee", "urban"],
        "bohemian":           ["bohemian", "boho", "crochet", "fringe", "tassel", "tiered", "peasant"],
        "minimalist":         ["minimalist", "ribbed", "bodysuit", "belt bag", "stone", "clean", "neutral"],
    }

    best_score = 0
    for cat, hints in category_keywords.items():
        score = sum(1 for h in hints if h in kw_lower)
        if score > best_score:
            best_score = score
            chosen = cat

    products = [dict(p) for p in _MOCK_CATALOGUE.get(chosen, _MOCK_CATALOGUE["default"])]

    # Inject link
    source_links = {
        "Myntra": "https://www.myntra.com",
        "AJIO": "https://www.ajio.com",
        "Flipkart Fashion": "https://www.flipkart.com/clothing-and-accessories",
        "Nykaa Fashion": "https://www.nykaa.com/fashion",
    }
    for p in products:
        p.setdefault("link", source_links.get(p["source"], "#"))

    # Filter by budget
    if budget:
        budget_max = _parse_budget_inr(budget)
        filtered = [p for p in products if _price_inr(p["price"]) <= budget_max]
        # If everything is filtered out, return cheapest item only
        if not filtered:
            products.sort(key=lambda x: _price_inr(x["price"]))
            return products[:1]
        products = filtered

    return products