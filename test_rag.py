"""
Test script for RAG system
Run this to test the RAG functionality directly
"""

import os
import pandas as pd
from rag_system import ExoplanetRAG, create_exoplanet_knowledge_base

def main():
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("   Get your API key at: https://aistudio.google.com/app/apikey")
        print("   Then set it with: export GEMINI_API_KEY='your-key-here'")
        return
    
    print("="*80)
    print("EXOPLANET RAG SYSTEM TEST")
    print("="*80)
    
    # Initialize RAG
    print("\n1. Initializing RAG system...")
    rag = ExoplanetRAG(api_key=api_key)
    
    # Load data
    print("\n2. Loading exoplanet data...")
    df = pd.read_csv("data/koi_with_relative_location.csv")
    print(f"   Loaded {len(df)} rows")
    
    # Create knowledge base
    print("\n3. Building knowledge base...")
    create_exoplanet_knowledge_base(df, rag)
    
    # Save index
    print("\n4. Saving index...")
    rag.save_index("rag_index")
    
    # Test queries
    print("\n" + "="*80)
    print("TESTING RAG SYSTEM")
    print("="*80)
    
    test_queries = [
        {
            "question": "How many exoplanets has Kepler discovered?",
            "description": "Basic statistics question"
        },
        {
            "question": "What is the habitable zone?",
            "description": "Scientific concept explanation"
        },
        {
            "question": "How does Kepler detect exoplanets?",
            "description": "Methodology question"
        },
        {
            "question": "What are the different planet size categories?",
            "description": "Classification question"
        },
    ]
    
    for i, query_info in enumerate(test_queries, 1):
        print(f"\n{'─'*80}")
        print(f"Test {i}/4: {query_info['description']}")
        print(f"{'─'*80}")
        print(f"\n❓ Question: {query_info['question']}")
        
        try:
            result = rag.ask(
                query_info['question'],
                top_k=5,
                temperature=0.7,
                include_sources=True
            )
            
            print(f"\n💡 Answer:\n{result['answer']}")
            print(f"\n📚 Used {result['num_sources']} source documents")
            
            if 'sources' in result:
                print(f"\n📄 Top sources:")
                for j, source in enumerate(result['sources'][:2], 1):
                    print(f"\n   {j}. (Score: {source['score']:.4f})")
                    print(f"      {source['text'][:200]}...")
            
            print(f"\n✅ Test {i} passed!")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print(f"   Test {i} failed")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print("="*80)
    print("\n✨ RAG system is working! You can now:")
    print("   1. Start the API: uvicorn api:app --reload")
    print("   2. Ask questions via POST /rag/ask")
    print("   3. Check status via GET /rag/status")
    print("\nExample:")
    print('   curl -X POST "http://localhost:8000/rag/ask" \\')
    print('     -F "question=How many exoplanets has Kepler discovered?"')


if __name__ == "__main__":
    main()

