from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
import os
import traceback
from dotenv import load_dotenv
from recommender import recommend

load_dotenv()

app = Flask(__name__)
CORS(app)

# MongoDB connection
client = MongoClient(os.getenv("MONGO_URI"))
db = client.get_database("test")


def serialize(doc):
    """Convert MongoDB ObjectId to string recursively."""
    if isinstance(doc, list):
        return [serialize(d) for d in doc]
    if isinstance(doc, dict):
        return {k: serialize(v) for k, v in doc.items()}
    if isinstance(doc, ObjectId):
        return str(doc)
    return doc


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "flower-recommender"})


@app.route("/recommend", methods=["POST"])
def get_recommendations():
    try:
        data = request.get_json()
        print("📦 Full received data:", data)

        user_id = data.get("userId")
        print("👤 userId received:", user_id)

        if not user_id:
            print("❌ No userId provided")
            return jsonify({"error": "userId is required"}), 400

        # 1. Fetch user reviews from MongoDB
        print("🔍 Fetching reviews for user:", user_id)
        reviews_cursor = db.reviews.find(
            {"user": ObjectId(user_id)}
        ).sort("createdAt", -1).limit(20)

        reviews = []
        for r in reviews_cursor:
            product = db.products.find_one({"_id": r["product"]})
            if product:
                r["product"] = product
                reviews.append(serialize(r))

        print("📝 Reviews found:", len(reviews))

        # 2. Fetch all available products
        all_products_cursor = db.products.find({"stock": {"$gt": 0}})
        all_products = serialize(list(all_products_cursor))
        print("🌸 Products found:", len(all_products))

        # 3. Run recommendation engine
        print("🤖 Running recommendation engine...")
        suggested_ids = recommend(reviews, all_products, top_n=4)
        print("✅ Suggested IDs:", suggested_ids)

        # 4. Fetch full product details for suggested IDs
        suggested_products = []
        for pid in suggested_ids:
            product = db.products.find_one({"_id": ObjectId(pid)})
            if product:
                suggested_products.append(serialize(product))

        print("🎁 Final suggestions count:", len(suggested_products))

        return jsonify({
            "suggestions": suggested_products,
            "total_reviews": len(reviews),
            "method": "smart+ml"
        })

    except Exception as e:
        print("❌ ERROR:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    print(f"🌸 Flower Recommender running on port {port}")
    app.run(debug=True, port=port)