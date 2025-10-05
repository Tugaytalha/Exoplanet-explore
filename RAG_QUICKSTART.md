# 🚀 RAG Quick Start

## Setup (2 minutes)

### 1. Get API Key
Visit https://aistudio.google.com/app/apikey → Create API Key → Copy it

### 2. Install Dependencies
```bash
pip install google-generativeai sentence-transformers faiss-cpu
```

### 3. Set API Key
```bash
export GEMINI_API_KEY="your-api-key-here"
```

### 4. Test RAG System
```bash
python test_rag.py
```

### 5. Start API
```bash
uvicorn api:app --reload
```

---

## Usage Examples

### Check Status
```bash
curl http://localhost:8000/rag/status
```

### Ask a Question
```bash
curl -X POST "http://localhost:8000/rag/ask" \
  -F "question=How many exoplanets has Kepler discovered?"
```

### Python Example
```python
import requests

response = requests.post(
    'http://localhost:8000/rag/ask',
    data={'question': 'What is the habitable zone?'}
)

print(response.json()['answer'])
```

### JavaScript Example
```javascript
const formData = new FormData();
formData.append('question', 'How does Kepler detect planets?');

fetch('http://localhost:8000/rag/ask', {
  method: 'POST',
  body: formData
})
.then(res => res.json())
.then(data => console.log(data.answer));
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/rag/ask` | POST | Ask a question |
| `/rag/status` | GET | Check RAG status |
| `/rag/rebuild` | POST | Rebuild index |

---

## Parameters

**POST /rag/ask**
- `question` (required): Your question
- `top_k` (optional, default=5): Number of sources
- `temperature` (optional, default=0.7): Creativity (0.0-1.0)
- `include_sources` (optional, default=false): Include source docs

---

## Example Questions

✅ "How many exoplanets has Kepler discovered?"  
✅ "What is the habitable zone?"  
✅ "How does the transit method work?"  
✅ "What percentage of planets are Earth-sized?"  
✅ "Which systems have the most planets?"  
✅ "What is a super-Earth?"  
✅ "How are planet sizes categorized?"  
✅ "What is the average planet radius?"  

---

## Troubleshooting

### "RAG dependencies not available"
```bash
pip install google-generativeai sentence-transformers faiss-cpu
```

### "Gemini API key not configured"
```bash
export GEMINI_API_KEY="your-key-here"
```

### "Index not found"
The index is built automatically on first use. Or:
```bash
curl -X POST http://localhost:8000/rag/rebuild
```

---

## Full Documentation

See `RAG_GUIDE.md` for complete documentation including:
- Advanced usage
- Frontend integration
- Security best practices
- Performance optimization

---

## What is RAG?

**RAG = Retrieval-Augmented Generation**

1. 🔍 **Retrieval**: Finds relevant info from your dataset
2. 🤖 **Generation**: Uses AI (Gemini) to answer naturally

**Benefits:**
- Answers based on YOUR data
- No hallucinations
- Up-to-date information
- Natural language interface

---

## Tech Stack

- **AI Model**: Google Gemini 2.0 Flash
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Search**: FAISS
- **API**: FastAPI

---

## Next Steps

1. ✅ Test with `python test_rag.py`
2. ✅ Start API with `uvicorn api:app --reload`
3. ✅ Try example questions
4. ✅ Integrate into your frontend
5. ✅ Read full guide in `RAG_GUIDE.md`

**Need Help?**
- API Docs: http://localhost:8000/docs
- Status Check: http://localhost:8000/rag/status
- Full Guide: RAG_GUIDE.md

Happy exploring! 🌌✨

