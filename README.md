# 🌸 Flower Recommender - Python Microservice

A Flask-based recommendation engine using smart scoring + ML cosine similarity.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
Edit `.env` and set your MongoDB URI:
```env
MONGO_URI=mongodb://localhost:27017/your_db_name
PORT=5001
```

### 3. Run the service
```bash
python app.py
```

Service runs on http://localhost:5001

---

## API Endpoints

### GET /health
Check if service is running.
```json
{ "status": "ok", "service": "flower-recommender" }
```

### POST /recommend
Get product recommendations for a user.

**Request:**
```json
{ "userId": "64abc123..." }
```

**Response:**
```json
{
  "suggestions": [...],
  "total_reviews": 5,
  "method": "smart+ml"
}
```

---

## How It Works

1. Fetches user's reviews from MongoDB
2. Smart scoring — weights categories and price range by star rating
3. ML scoring — cosine similarity between user taste vector and product vectors
4. Combines both scores (50/50) and returns top 4 products
5. Falls back to top-rated products if not enough results

---

## Running All Services

```bash
# Terminal 1 — Node.js backend (port 5000)
cd ../backend && npm run dev

# Terminal 2 — Python service (port 5001)
python app.py

# Terminal 3 — Next.js frontend (port 3000)
cd ../frontend && npm run dev
```
# pythonforproject
