# 🤖 RAG System Guide - Exoplanet AI Assistant

## Overview

The Exoplanet API now includes a **RAG (Retrieval-Augmented Generation)** system powered by **Google Gemini 2.0 Flash**. This allows you to ask natural language questions about exoplanets and get AI-generated answers based on your actual dataset.

### What is RAG?

RAG combines:
1. **Vector Search**: Finds relevant information from your exoplanet database
2. **AI Generation**: Uses Google Gemini to generate natural, contextual answers

This ensures answers are:
- ✅ Grounded in your actual data
- ✅ Accurate and specific
- ✅ Natural and conversational

---

## 🚀 Quick Start

### 1. Get Your Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

**Free tier includes:**
- 15 requests per minute
- 1 million tokens per day
- Perfect for development and testing

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `google-generativeai` - Official Google Gemini SDK
- `sentence-transformers` - Text embeddings for semantic search
- `faiss-cpu` - Fast vector similarity search

### 3. Set API Key

**Option A: Environment Variable (Recommended)**
```bash
# Linux/Mac
export GEMINI_API_KEY="your-api-key-here"

# Windows PowerShell
$env:GEMINI_API_KEY="your-api-key-here"

# Windows CMD
set GEMINI_API_KEY=your-api-key-here
```

**Option B: .env File**
```bash
# Create .env file in project root
echo "GEMINI_API_KEY=your-api-key-here" > .env
```

**Option C: Docker**
```yaml
# In docker-compose.yml
services:
  exoplanet-api:
    environment:
      - GEMINI_API_KEY=your-api-key-here
```

### 4. Start the API

```bash
uvicorn api:app --reload
```

### 5. Check RAG Status

```bash
curl http://localhost:8000/api/rag/status
```

**Response:**
```json
{
  "rag_available": true,
  "gemini_api_key_set": true,
  "index_exists": true,
  "rag_initialized": false,
  "indexed_documents": 0
}
```

---

## 💬 Asking Questions

### Using cURL

```bash
curl -X POST "http://localhost:8000/api/rag/ask" \
  -F "question=How many exoplanets has Kepler discovered?" \
  -F "top_k=5" \
  -F "temperature=0.7"
```

### Using Python

```python
import requests

response = requests.post(
    'http://localhost:8000/api/rag/ask',
    data={
        'question': 'How many exoplanets has Kepler discovered?',
        'top_k': 5,
        'temperature': 0.7,
        'include_sources': False
    }
)

result = response.json()
print(f"Question: {result['query']}")
print(f"Answer: {result['answer']}")
print(f"Sources used: {result['num_sources']}")
```

### Using JavaScript

```javascript
const formData = new FormData();
formData.append('question', 'What is the habitable zone?');
formData.append('top_k', '5');
formData.append('temperature', '0.7');

fetch('http://localhost:8000/api/rag/ask', {
  method: 'POST',
  body: formData
})
.then(res => res.json())
.then(data => {
  console.log('Answer:', data.answer);
  console.log('Sources:', data.num_sources);
});
```

---

## 📊 Parameters

### `question` (required)
Your question in natural language.

**Examples:**
- "How many exoplanets has Kepler discovered?"
- "What is the habitable zone?"
- "How does Kepler detect planets?"
- "What's the average size of confirmed planets?"
- "Tell me about multi-planet systems"

### `top_k` (optional, default: 5)
Number of relevant documents to retrieve for context.

- **Lower (1-3)**: Faster, more focused
- **Higher (5-10)**: More comprehensive, slower

### `temperature` (optional, default: 0.7)
Controls response creativity:

- **0.0-0.3**: Factual, deterministic
- **0.4-0.7**: Balanced (recommended)
- **0.8-1.0**: Creative, exploratory

### `include_sources` (optional, default: false)
Include retrieved source documents in response.

Useful for:
- Fact-checking
- Understanding context
- Debugging

---

## 📖 Example Questions

### General Statistics
```bash
curl -X POST "http://localhost:8000/api/rag/ask" \
  -F "question=How many KOIs are there in total?"
```

**Expected Answer:** "The dataset contains 59,316 Kepler Objects of Interest..."

### Scientific Concepts
```bash
curl -X POST "http://localhost:8000/api/rag/ask" \
  -F "question=What is the transit method?"
```

**Expected Answer:** "The transit method is how Kepler detects exoplanets..."

### Data Analysis
```bash
curl -X POST "http://localhost:8000/api/rag/ask" \
  -F "question=What percentage of planets are Earth-sized?"
```

**Expected Answer:** "According to the data, approximately X% of planets..."

### Multi-Planet Systems
```bash
curl -X POST "http://localhost:8000/api/rag/ask" \
  -F "question=Which systems have the most planets?"
```

**Expected Answer:** "The systems with the most confirmed planets include..."

### Habitability
```bash
curl -X POST "http://localhost:8000/api/rag/ask" \
  -F "question=How many potentially habitable planets are there?"
```

**Expected Answer:** "There are X KOIs with equilibrium temperatures in the habitable zone..."

---

## 🔧 Advanced Usage

### With Source Documents

```python
response = requests.post(
    'http://localhost:8000/api/rag/ask',
    data={
        'question': 'What are the different planet size categories?',
        'include_sources': True
    }
)

result = response.json()
print(f"Answer: {result['answer']}\n")

if 'sources' in result:
    print("Sources used:")
    for i, source in enumerate(result['sources'], 1):
        print(f"\n{i}. Score: {source['score']:.4f}")
        print(f"   {source['text'][:200]}...")
```

### Batch Questions

```python
questions = [
    "How many exoplanets has Kepler discovered?",
    "What is the average planet radius?",
    "How does the transit method work?",
    "What are super-Earths?",
]

for q in questions:
    response = requests.post(
        'http://localhost:8000/api/rag/ask',
        data={'question': q, 'temperature': 0.5}
    )
    result = response.json()
    print(f"\nQ: {q}")
    print(f"A: {result['answer']}")
    print("-" * 80)
```

### Custom Temperature for Different Tasks

```python
# Factual answers (low temperature)
factual = requests.post('http://localhost:8000/api/rag/ask', data={
    'question': 'How many confirmed planets are there?',
    'temperature': 0.2
}).json()

# Exploratory answers (higher temperature)
exploratory = requests.post('http://localhost:8000/api/rag/ask', data={
    'question': 'What makes an exoplanet interesting?',
    'temperature': 0.9
}).json()
```

---

## 🔄 Managing the Knowledge Base

### Check Status

```bash
curl http://localhost:8000/api/rag/status
```

### Rebuild Index

Rebuild the RAG index after updating your dataset:

```bash
curl -X POST http://localhost:8000/api/rag/rebuild
```

**Response:**
```json
{
  "status": "success",
  "message": "RAG index rebuilt successfully",
  "indexed_documents": 25,
  "index_path": "rag_index"
}
```

### What Gets Indexed?

The RAG system automatically creates a knowledge base including:

1. **General Statistics**
   - Total KOI count
   - Disposition breakdown
   - Mission overview

2. **Parameter Descriptions**
   - Orbital period, radius, temperature, etc.
   - Statistical summaries (mean, median, min, max)

3. **Multi-Planet Systems**
   - Systems with multiple planets
   - Top systems by planet count

4. **Habitability Analysis**
   - Potentially habitable zone candidates
   - Temperature ranges

5. **Planet Size Categories**
   - Earth-sized, Super-Earth, Neptune, Jupiter
   - Distribution percentages

6. **Scientific Context**
   - Transit method explanation
   - Kepler mission details
   - Discovery limitations

---

## 🎨 Frontend Integration

### React Component

```jsx
import { useState } from 'react';

function ExoplanetAI() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);

  const askQuestion = async () => {
    setLoading(true);
    
    const formData = new FormData();
    formData.append('question', question);
    formData.append('temperature', '0.7');
    
    const response = await fetch('http://localhost:8000/api/rag/ask', {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    setAnswer(result.answer);
    setLoading(false);
  };

  return (
    <div>
      <h2>Ask About Exoplanets</h2>
      <input
        type="text"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="e.g., How many planets has Kepler found?"
      />
      <button onClick={askQuestion} disabled={loading}>
        {loading ? 'Thinking...' : 'Ask'}
      </button>
      
      {answer && (
        <div className="answer">
          <h3>Answer:</h3>
          <p>{answer}</p>
        </div>
      )}
    </div>
  );
}
```

### HTML + Vanilla JS

```html
<!DOCTYPE html>
<html>
<head>
    <title>Exoplanet AI</title>
    <style>
        .chat-container { max-width: 800px; margin: 0 auto; padding: 20px; }
        .question-input { width: 100%; padding: 10px; font-size: 16px; }
        .ask-button { padding: 10px 20px; font-size: 16px; margin-top: 10px; }
        .answer-box { background: #f0f0f0; padding: 15px; margin-top: 20px; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="chat-container">
        <h1>🌌 Ask About Exoplanets</h1>
        <input type="text" id="question" class="question-input" 
               placeholder="e.g., How many exoplanets has Kepler discovered?">
        <button class="ask-button" onclick="askQuestion()">Ask</button>
        <div id="answer" class="answer-box" style="display:none;"></div>
    </div>

    <script>
        async function askQuestion() {
            const question = document.getElementById('question').value;
            const answerDiv = document.getElementById('answer');
            
            answerDiv.style.display = 'block';
            answerDiv.innerHTML = '🤔 Thinking...';
            
            const formData = new FormData();
            formData.append('question', question);
            formData.append('temperature', '0.7');
            
            const response = await fetch('http://localhost:8000/api/rag/ask', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            answerDiv.innerHTML = `
                <strong>Q:</strong> ${result.query}<br><br>
                <strong>A:</strong> ${result.answer}<br><br>
                <small>📚 Used ${result.num_sources} sources</small>
            `;
        }
    </script>
</body>
</html>
```

---

## ⚠️ Troubleshooting

### "RAG system not available"

**Cause:** Dependencies not installed

**Solution:**
```bash
pip install google-generativeai sentence-transformers faiss-cpu
```

### "Gemini API key not configured"

**Cause:** GEMINI_API_KEY environment variable not set

**Solution:**
```bash
export GEMINI_API_KEY="your-key-here"
```

Get your key at: https://aistudio.google.com/app/apikey

### "Index not found"

**Cause:** RAG index hasn't been built yet

**Solution:**
The index is built automatically on first use. Or manually rebuild:
```bash
curl -X POST http://localhost:8000/api/rag/rebuild
```

### Slow Responses

**Causes:**
- Large `top_k` value
- Complex questions
- Network latency to Gemini API

**Solutions:**
- Reduce `top_k` to 3-5
- Use lower `temperature` (0.3-0.5)
- Cache frequently asked questions

### Rate Limits

**Free Tier Limits:**
- 15 requests per minute
- 1 million tokens per day

**Solutions:**
- Implement request caching
- Add rate limiting on your end
- Consider upgrading to paid tier

---

## 📈 Performance Tips

### 1. Index Management

Build index once, reuse many times:
```python
# Build during deployment
curl -X POST http://localhost:8000/api/rag/rebuild

# Reuse for all queries (fast!)
```

### 2. Optimal Parameters

For best speed/quality balance:
```python
{
    'top_k': 5,           # Good balance
    'temperature': 0.7,   # Balanced creativity
    'include_sources': False  # Faster response
}
```

### 3. Caching

Cache common questions:
```python
cache = {}

def ask_with_cache(question):
    if question in cache:
        return cache[question]
    
    response = requests.post('http://localhost:8000/api/rag/ask',
                           data={'question': question})
    result = response.json()
    cache[question] = result
    return result
```

---

## 🔒 Security Best Practices

### 1. Protect API Keys

✅ **DO:**
- Store in environment variables
- Use secret management services (AWS Secrets Manager, etc.)
- Rotate keys regularly

❌ **DON'T:**
- Commit keys to Git
- Hardcode in source files
- Share publicly

### 2. Rate Limiting

Implement rate limiting to prevent abuse:

```python
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/rag/ask")
@limiter.limit("10/minute")
async def ask_rag_question(request: Request, ...):
    ...
```

### 3. Input Validation

Sanitize user inputs:

```python
MAX_QUESTION_LENGTH = 500

if len(question) > MAX_QUESTION_LENGTH:
    raise HTTPException(400, "Question too long")

if not question.strip():
    raise HTTPException(400, "Question cannot be empty")
```

---

## 📚 Additional Resources

- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [Sentence Transformers](https://www.sbert.net/)
- [RAG Overview](https://www.pinecone.io/learn/retrieval-augmented-generation/)

---

## 🎯 Next Steps

1. **Get your API key** from Google AI Studio
2. **Install dependencies** with pip
3. **Set environment variable** with your key
4. **Start the API** and test with `/api/rag/status`
5. **Ask your first question!**

```bash
curl -X POST "http://localhost:8000/api/rag/ask" \
  -F "question=Tell me about exoplanets!"
```

Happy exploring! 🚀🌌

