"""
RAG (Retrieval-Augmented Generation) System for Exoplanet Data
Uses Google Gemini 2.0 Flash for natural language responses
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, BinaryIO
from pathlib import Path
import re

try:
    import google.generativeai as genai
    from sentence_transformers import SentenceTransformer
    import faiss
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("⚠️  RAG dependencies not available. Install with: pip install google-generativeai sentence-transformers faiss-cpu")

try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️  PDF dependencies not available. Install with: pip install pypdf2 pdfplumber")


class ExoplanetRAG:
    """
    RAG system for answering questions about exoplanet data using Gemini 2.0 Flash
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize the RAG system
        
        Args:
            api_key: Google AI Studio API key (or set GEMINI_API_KEY env var)
            model_name: Gemini model to use (default: gemini-2.0-flash-exp)
        """
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("RAG dependencies not installed. Run: pip install google-generativeai sentence-transformers faiss-cpu")
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable or pass api_key parameter.\n"
                "Get your API key at: https://aistudio.google.com/app/apikey"
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        
        # Initialize embedding model for document retrieval
        print("📥 Loading sentence transformer model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize FAISS index
        self.index = None
        self.documents = []
        self.document_metadata = []
        
        print(f"✅ RAG system initialized with {model_name}")
    
    def index_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Index documents for retrieval
        
        Args:
            documents: List of text documents to index
            metadata: Optional metadata for each document
        """
        print(f"📊 Indexing {len(documents)} documents...")
        
        self.documents = documents
        self.document_metadata = metadata or [{} for _ in documents]
        
        # Create embeddings
        embeddings = self.embedder.encode(documents, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        print(f"✅ Indexed {len(documents)} documents (dimension: {dimension})")
    
    def retrieve_relevant_docs(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve most relevant documents for a query
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of dicts with 'text', 'score', and 'metadata'
        """
        if self.index is None:
            return []
        
        # Encode query
        query_embedding = self.embedder.encode([query])[0].astype('float32')
        query_embedding = np.array([query_embedding])
        
        # Search
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    'text': self.documents[idx],
                    'score': float(dist),
                    'metadata': self.document_metadata[idx]
                })
        
        return results
    
    def generate_response(self, query: str, context: Optional[str] = None, temperature: float = 0.7) -> str:
        """
        Generate response using Gemini with optional context
        
        Args:
            query: User question
            context: Optional context to include in prompt
            temperature: Response creativity (0.0-1.0)
            
        Returns:
            Generated response text
        """
        # Build prompt
        if context:
            prompt = f"""You are an expert in exoplanet science and astronomy. Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer (be specific and cite relevant data from the context):"""
        else:
            prompt = f"""You are an expert in exoplanet science and astronomy. Answer the following question accurately and concisely.

Question: {query}

Answer:"""
        
        # Generate response
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=1024,
                )
            )
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def ask(self, query: str, top_k: int = 5, temperature: float = 0.7, include_sources: bool = False) -> Dict:
        """
        Ask a question using RAG (retrieval + generation)
        
        Args:
            query: User question
            top_k: Number of documents to retrieve for context
            temperature: Response creativity
            include_sources: Include retrieved documents in response
            
        Returns:
            Dict with 'answer', 'sources' (if requested), 'query'
        """
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_docs(query, top_k=top_k)
        
        # Build context from retrieved documents
        context = "\n\n".join([
            f"[Document {i+1}]:\n{doc['text']}"
            for i, doc in enumerate(relevant_docs)
        ])
        
        # Generate response
        answer = self.generate_response(query, context=context if context else None, temperature=temperature)
        
        # Prepare response
        result = {
            'query': query,
            'answer': answer,
            'num_sources': len(relevant_docs)
        }
        
        if include_sources:
            result['sources'] = [
                {
                    'text': doc['text'][:500] + ('...' if len(doc['text']) > 500 else ''),
                    'score': doc['score'],
                    'metadata': doc['metadata']
                }
                for doc in relevant_docs
            ]
        
        return result
    
    def save_index(self, path: str = "rag_index"):
        """Save FAISS index and documents to disk"""
        path = Path(path)
        path.mkdir(exist_ok=True)
        
        if self.index is not None:
            faiss.write_index(self.index, str(path / "faiss.index"))
        
        with open(path / "documents.json", 'w') as f:
            json.dump({
                'documents': self.documents,
                'metadata': self.document_metadata
            }, f, indent=2)
        
        print(f"✅ Index saved to {path}")
    
    def load_index(self, path: str = "rag_index"):
        """Load FAISS index and documents from disk"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Index directory not found: {path}")
        
        # Load FAISS index
        if (path / "faiss.index").exists():
            self.index = faiss.read_index(str(path / "faiss.index"))
        
        # Load documents
        with open(path / "documents.json", 'r') as f:
            data = json.load(f)
            self.documents = data['documents']
            self.document_metadata = data['metadata']
        
        print(f"✅ Loaded index with {len(self.documents)} documents")
    
    def extract_text_from_pdf(self, pdf_path: Union[str, Path, BinaryIO], method: str = 'pdfplumber') -> str:
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to PDF file or file-like object
            method: Extraction method ('pdfplumber' or 'pypdf2')
            
        Returns:
            Extracted text as string
        """
        if not PDF_AVAILABLE:
            raise ImportError("PDF dependencies not installed. Run: pip install pypdf2 pdfplumber")
        
        try:
            if method == 'pdfplumber':
                # pdfplumber is better for complex layouts
                if isinstance(pdf_path, (str, Path)):
                    with pdfplumber.open(pdf_path) as pdf:
                        text = ""
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n\n"
                else:
                    # File-like object
                    with pdfplumber.open(pdf_path) as pdf:
                        text = ""
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n\n"
            else:
                # PyPDF2 fallback
                if isinstance(pdf_path, (str, Path)):
                    with open(pdf_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page in reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n\n"
                else:
                    # File-like object
                    reader = PyPDF2.PdfReader(pdf_path)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
            
            # Clean up text
            text = self._clean_extracted_text(text)
            return text
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted text from PDF"""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove page numbers (common patterns)
        text = re.sub(r'\n\d+\n', '\n', text)
        
        # Fix common OCR issues
        text = text.replace('ﬁ', 'fi')
        text = text.replace('ﬂ', 'fl')
        text = text.replace('–', '-')
        text = text.replace('—', '-')
        
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Number of overlapping characters between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending
                for delimiter in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    last_delimiter = text[start:end].rfind(delimiter)
                    if last_delimiter != -1:
                        end = start + last_delimiter + len(delimiter)
                        break
            
            chunks.append(text[start:end].strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def index_pdf(self, pdf_path: Union[str, Path, BinaryIO], 
                  filename: str = None,
                  chunk_size: int = 1000, 
                  overlap: int = 200,
                  metadata: Optional[Dict] = None) -> int:
        """
        Extract text from PDF, chunk it, and add to index
        
        Args:
            pdf_path: Path to PDF file or file-like object
            filename: Name of the file (for metadata)
            chunk_size: Maximum characters per chunk
            overlap: Overlapping characters between chunks
            metadata: Additional metadata for the document
            
        Returns:
            Number of chunks added
        """
        print(f"📄 Processing PDF: {filename or 'uploaded file'}...")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text or len(text.strip()) < 100:
            raise ValueError("PDF appears to be empty or text extraction failed")
        
        print(f"   Extracted {len(text)} characters")
        
        # Chunk text
        chunks = self.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        print(f"   Created {len(chunks)} chunks")
        
        # Prepare metadata
        base_metadata = metadata or {}
        base_metadata.update({
            'type': 'pdf',
            'filename': filename or 'unknown',
            'total_chunks': len(chunks),
        })
        
        # Add chunks to documents
        new_documents = []
        new_metadata = []
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'chunk_total': len(chunks),
            })
            new_documents.append(chunk)
            new_metadata.append(chunk_metadata)
        
        # Combine with existing documents
        all_documents = self.documents + new_documents
        all_metadata = self.document_metadata + new_metadata
        
        # Reindex everything
        print(f"   Reindexing with {len(all_documents)} total documents...")
        self.index_documents(all_documents, all_metadata)
        
        print(f"✅ Added {len(chunks)} chunks from PDF")
        return len(chunks)


def create_exoplanet_knowledge_base(df, rag_system: ExoplanetRAG):
    """
    Create a knowledge base from exoplanet DataFrame
    
    Args:
        df: Pandas DataFrame with exoplanet data
        rag_system: Initialized RAG system
    """
    documents = []
    metadata = []
    
    print("📚 Building knowledge base from exoplanet data...")
    
    # 1. Add general statistics
    total_count = len(df)
    disposition_counts = df.get('disposition', df.get('koi_disposition', pd.Series())).value_counts().to_dict()
    
    stats_doc = f"""General Exoplanet Statistics:
- Total number of Kepler Objects of Interest (KOIs): {total_count}
- Disposition breakdown: {disposition_counts}
- Data source: NASA Kepler Mission
- The Kepler mission discovered thousands of exoplanet candidates by detecting periodic dips in stellar brightness caused by planets transiting in front of their host stars.
"""
    documents.append(stats_doc)
    metadata.append({'type': 'statistics', 'category': 'general'})
    
    # 2. Add information about key columns
    column_info = {
        'koi_period': 'Orbital period in days - how long it takes the planet to complete one orbit',
        'koi_prad': 'Planet radius in Earth radii - size of the planet compared to Earth',
        'koi_teq': 'Equilibrium temperature in Kelvin - estimated surface temperature',
        'koi_insol': 'Insolation flux in Earth units - amount of stellar energy received',
        'koi_depth': 'Transit depth in parts per million - how much star brightness dims',
        'koi_duration': 'Transit duration in hours - how long the transit lasts',
        'st_teff': 'Stellar effective temperature in Kelvin - temperature of the host star',
        'st_rad': 'Stellar radius in Solar radii - size of the host star',
        'st_mass': 'Stellar mass in Solar masses - mass of the host star',
        'sy_dist': 'Distance to system in parsecs',
    }
    
    for col, description in column_info.items():
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 0:
                doc = f"""Exoplanet Parameter: {col}
Description: {description}
Statistics:
- Mean: {values.mean():.2f}
- Median: {values.median():.2f}
- Min: {values.min():.2f}
- Max: {values.max():.2f}
- Available for {len(values)} out of {len(df)} KOIs
"""
                documents.append(doc)
                metadata.append({'type': 'parameter', 'column': col})
    
    # 3. Add information about interesting systems
    # Group by stellar system if possible
    if 'hostname' in df.columns:
        multi_planet_systems = df.groupby('hostname').size()
        multi_planet_systems = multi_planet_systems[multi_planet_systems > 1].sort_values(ascending=False)
        
        if len(multi_planet_systems) > 0:
            doc = f"""Multi-Planet Systems:
- Total systems with multiple confirmed planets: {len(multi_planet_systems)}
- Top 10 systems by planet count:
"""
            for system, count in multi_planet_systems.head(10).items():
                doc += f"  - {system}: {count} planets\n"
            
            doc += "\nMulti-planet systems are particularly interesting as they provide insights into planetary system formation and dynamics."
            documents.append(doc)
            metadata.append({'type': 'systems', 'category': 'multi_planet'})
    
    # 4. Add information about habitable zone candidates
    if 'koi_teq' in df.columns:
        # Habitable zone temperature range (approximately)
        habitable = df[(df['koi_teq'] >= 200) & (df['koi_teq'] <= 320)]
        if len(habitable) > 0:
            doc = f"""Potentially Habitable Zone Planets:
- Number of KOIs with equilibrium temperature in habitable range (200-320 K): {len(habitable)}
- Percentage of total: {len(habitable)/len(df)*100:.1f}%
- The habitable zone is the region around a star where liquid water could exist on a planet's surface
- Temperature range is approximate and depends on atmospheric composition and other factors
"""
            documents.append(doc)
            metadata.append({'type': 'science', 'category': 'habitable_zone'})
    
    # 5. Add planet size categories
    if 'koi_prad' in df.columns:
        sizes = df['koi_prad'].dropna()
        size_categories = {
            'Earth-sized (0.8-1.25 R⊕)': len(sizes[(sizes >= 0.8) & (sizes < 1.25)]),
            'Super-Earth (1.25-2 R⊕)': len(sizes[(sizes >= 1.25) & (sizes < 2)]),
            'Neptune-sized (2-6 R⊕)': len(sizes[(sizes >= 2) & (sizes < 6)]),
            'Jupiter-sized (6+ R⊕)': len(sizes[sizes >= 6]),
        }
        
        doc = f"""Planet Size Distribution:
"""
        for category, count in size_categories.items():
            percentage = count / len(sizes) * 100
            doc += f"- {category}: {count} planets ({percentage:.1f}%)\n"
        
        doc += "\nR⊕ = Earth radii. Planet size is a key indicator of composition - smaller planets are more likely to be rocky, while larger ones are typically gas giants."
        documents.append(doc)
        metadata.append({'type': 'statistics', 'category': 'planet_sizes'})
    
    # 6. Add discovery method information
    doc = """Kepler Mission Discovery Method:
The Kepler spacecraft used the transit method to discover exoplanets:
1. Monitor brightness of over 150,000 stars continuously
2. Detect periodic dips in brightness when a planet passes in front of its star
3. The depth of the dip indicates planet size
4. The period between dips indicates orbital period
5. Duration of transit helps determine orbital distance

Key advantages:
- Can detect small, Earth-sized planets
- Provides accurate measurements of planet size and orbital period
- Can discover multiple planets in same system

Limitations:
- Only detects planets whose orbits are aligned edge-on to Earth
- Cannot directly measure planet mass (requires follow-up observations)
- Limited to relatively nearby stars
"""
    documents.append(doc)
    metadata.append({'type': 'methodology', 'category': 'kepler_mission'})
    
    # Index all documents
    rag_system.index_documents(documents, metadata)
    
    return rag_system


# Example usage
if __name__ == "__main__":
    import pandas as pd
    
    # Initialize RAG system
    rag = ExoplanetRAG()
    
    # Load data
    df = pd.read_csv("data/koi_with_relative_location.csv")
    
    # Create knowledge base
    create_exoplanet_knowledge_base(df, rag)
    
    # Save index
    rag.save_index()
    
    # Test queries
    test_queries = [
        "How many exoplanets has Kepler discovered?",
        "What is the habitable zone?",
        "How does Kepler detect exoplanets?",
        "What are the different planet size categories?",
    ]
    
    print("\n" + "="*80)
    print("Testing RAG System")
    print("="*80 + "\n")
    
    for query in test_queries:
        print(f"❓ Query: {query}")
        result = rag.ask(query, top_k=3, include_sources=False)
        print(f"💡 Answer: {result['answer']}")
        print(f"📚 Used {result['num_sources']} source documents")
        print("-" * 80 + "\n")

