import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


def build_product_vector(product, all_categories):
    """
    Convert a product into a numeric feature vector:
    [category_encoding..., normalized_price, normalized_rating]
    """
    # One-hot encode category
    category_vector = [1 if product.get("category") == cat else 0 for cat in all_categories]

    # Normalize price (0-1 scale, cap at 10000)
    price = min(product.get("price", 0), 10000) / 10000

    # Normalize rating (0-1 scale, max 5)
    rating = product.get("rating", 0) / 5

    return np.array(category_vector + [price, rating], dtype=float)


def smart_score(reviews, all_products):
    """
    Smart scoring: weight categories and price range by user ratings.
    Returns dict of { product_id: score }
    """
    category_scores = defaultdict(float)
    prices = []
    reviewed_ids = set()

    for r in reviews:
        pid = str(r["product"]["_id"])
        reviewed_ids.add(pid)
        cat = r["product"].get("category")
        rating = r.get("rating", 0)
        price = r["product"].get("price", 0)

        if cat:
            category_scores[cat] += rating
        if price:
            prices.append(price)

    avg_price = sum(prices) / len(prices) if prices else 500
    price_low = avg_price * 0.5
    price_high = avg_price * 1.8

    scores = {}
    for p in all_products:
        pid = str(p["_id"])
        if pid in reviewed_ids:
            continue
        if p.get("stock", 0) <= 0:
            continue

        score = 0
        cat = p.get("category")
        price = p.get("price", 0)
        rating = p.get("rating", 0)

        # Category match score
        score += category_scores.get(cat, 0)

        # Price range bonus
        if price_low <= price <= price_high:
            score += 5

        # Product rating bonus
        score += rating * 2

        scores[pid] = score

    return scores, reviewed_ids


def ml_score(reviews, all_products, reviewed_ids):
    """
    ML scoring: cosine similarity between user taste vector
    and each product vector.
    Returns dict of { product_id: similarity_score }
    """
    if not reviews or not all_products:
        return {}

    all_categories = list(set(p.get("category", "") for p in all_products))

    # Build user taste vector = weighted average of reviewed product vectors
    user_vector = np.zeros(len(all_categories) + 2)
    total_weight = 0

    for r in reviews:
        rating = r.get("rating", 1)
        vec = build_product_vector(r["product"], all_categories)
        user_vector += vec * rating
        total_weight += rating

    if total_weight > 0:
        user_vector /= total_weight

    # Score each unreviewed product by cosine similarity
    similarity_scores = {}
    for p in all_products:
        pid = str(p["_id"])
        if pid in reviewed_ids or p.get("stock", 0) <= 0:
            continue

        product_vec = build_product_vector(p, all_categories)

        # Cosine similarity
        dot = np.dot(user_vector, product_vec)
        norm = np.linalg.norm(user_vector) * np.linalg.norm(product_vec)
        sim = dot / norm if norm > 0 else 0

        similarity_scores[pid] = sim

    return similarity_scores


def recommend(reviews, all_products, top_n=4):
    """
    Combine smart scoring + ML cosine similarity.
    Returns list of top_n product IDs.
    """
    if not reviews:
        # No reviews: return top rated products
        top = sorted(
            [p for p in all_products if p.get("stock", 0) > 0],
            key=lambda p: p.get("rating", 0),
            reverse=True
        )
        return [str(p["_id"]) for p in top[:top_n]]

    # Get both scores
    smart_scores, reviewed_ids = smart_score(reviews, all_products)
    ml_scores = ml_score(reviews, all_products, reviewed_ids)

    # Normalize smart scores to 0-1
    max_smart = max(smart_scores.values(), default=1)
    if max_smart == 0:
        max_smart = 1
    norm_smart = {
        pid: score / max_smart
        for pid, score in smart_scores.items()
    }

    # Normalize ml scores to 0-1
    max_ml = max(ml_scores.values(), default=1)
    if max_ml == 0:
        max_ml = 1
    norm_ml = {
        pid: score / max_ml
        for pid, score in ml_scores.items()
    }

    # Combine: 50% smart + 50% ML
    all_ids = set(list(norm_smart.keys()) + list(norm_ml.keys()))
    combined = {}
    for pid in all_ids:
        s = norm_smart.get(pid, 0)
        m = norm_ml.get(pid, 0)
        combined[pid] = (s * 0.5) + (m * 0.5)

    # Sort by combined score
    sorted_ids = sorted(combined, key=lambda x: combined[x], reverse=True)
    top_ids = sorted_ids[:top_n]

    # Fill if not enough
    if len(top_ids) < top_n:
        fallback = sorted(
            [p for p in all_products
             if str(p["_id"]) not in reviewed_ids
             and str(p["_id"]) not in top_ids
             and p.get("stock", 0) > 0],
            key=lambda p: p.get("rating", 0),
            reverse=True
        )
        for p in fallback:
            if len(top_ids) >= top_n:
                break
            top_ids.append(str(p["_id"]))

    return top_ids

if __name__ == "__main__":
    print("Running recommendation engine tests...")
    
    import json
    
    mock_reviews = [
        {
            "user": "user1",
            "rating": 5,
            "product": {
                "_id": "p1",
                "category": "Roses",
                "price": 500,
                "rating": 4.5
            }
        },
        {
            "user": "user1",
            "rating": 4,
            "product": {
                "_id": "p2",
                "category": "Lilies",
                "price": 800,
                "rating": 4.2
            }
        }
    ]
    
    mock_products = [
        {"_id": "p1", "category": "Roses", "price": 500, "rating": 4.5, "stock": 10},
        {"_id": "p2", "category": "Lilies", "price": 800, "rating": 4.2, "stock": 5},
        {"_id": "p3", "category": "Roses", "price": 600, "rating": 4.8, "stock": 15},
        {"_id": "p4", "category": "Tulips", "price": 400, "rating": 4.0, "stock": 20},
        {"_id": "p5", "category": "Lilies", "price": 850, "rating": 4.7, "stock": 2},
        {"_id": "p6", "category": "Sunflowers", "price": 300, "rating": 4.1, "stock": 0} # Out of stock
    ]
    
    print("Mock Reviews:")
    print(json.dumps(mock_reviews, indent=2))
    
    print("\nMock Products:")
    print(json.dumps(mock_products, indent=2))
    
    print("\nGenerating recommendations...")
    recommendations = recommend(mock_reviews, mock_products, top_n=3)
    
    print("\nRecommended Product IDs:")
    print(recommendations)
