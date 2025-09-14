import google.generativeai as genai
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from dotenv import load_dotenv
import time
import os
import re
from typing import List, Optional, Any, Dict
from enum import Enum

from langchain_cohere import CohereRerank
import pickle

from pinecone import Pinecone

# Load environment variables
load_dotenv()


class QueryType(Enum):
    """Types of queries the system can handle"""
    FACTUAL = "factual"
    COMPARISON = "comparison"
    DEFINITION = "definition"
    PROCEDURE = "procedure"
    ANALYSIS = "analysis"
    GENERAL = "general"


class CitationEnhancer:
    """Enhanced citation system with passage highlighting and confidence scoring"""
    
    def __init__(self):
        # Initialize Gemini for citation analysis
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.citation_model = genai.GenerativeModel("gemini-2.5-flash")
    
    def extract_citations(self, query: str, response: str, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract and enhance citations from response and documents"""
        citations = []
        
        for i, doc in enumerate(documents):
            try:
                # Calculate relevance score based on document metadata
                base_score = doc.metadata.get('score', 0)
                quality_score = doc.metadata.get('quality_score', 0.5)
                
                # Enhanced confidence calculation
                confidence_score = self._calculate_confidence(query, response, doc.page_content, base_score, quality_score)
                
                # Find relevant passages
                relevant_passages = self._find_relevant_passages(query, response, doc.page_content)
                
                # Build citation object
                citation = {
                    'citation_id': i + 1,
                    'document_id': doc.metadata.get('id', f'doc-{i}'),
                    'filename': doc.metadata.get('filename', 'Unknown'),
                    'page_number': doc.metadata.get('page_number'),
                    'document_category': doc.metadata.get('document_category', 'general'),
                    'content_type': doc.metadata.get('content_type', 'standard'),
                    'source_type': doc.metadata.get('source_type', 'primary'),
                    'relevance_score': base_score,
                    'quality_score': quality_score,
                    'confidence_score': confidence_score,
                    'relevant_passages': relevant_passages,
                    'full_content': doc.page_content,
                    'content_preview': doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                }
                
                # Add expanded query info if available
                if doc.metadata.get('expanded_query'):
                    citation['expanded_query'] = doc.metadata['expanded_query']
                
                citations.append(citation)
                
            except Exception as e:
                print(f"Error processing citation {i}: {e}")
                continue
        
        # Sort citations by confidence score
        citations.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        return citations
    
    def _calculate_confidence(self, query: str, response: str, content: str, base_score: float, quality_score: float) -> float:
        """Calculate confidence score for citation"""
        try:
            # Normalize base score to 0-1 range (assuming Pinecone scores are typically 0-1)
            normalized_base = max(0, min(1, base_score))
            
            # Text similarity factors
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            response_words = set(response.lower().split())
            
            # Calculate word overlap
            query_content_overlap = len(query_words.intersection(content_words)) / max(len(query_words), 1)
            response_content_overlap = len(response_words.intersection(content_words)) / max(len(response_words), 1)
            
            # Length factor (prefer substantial content)
            length_factor = min(1.0, len(content) / 500)  # Normalize to 500 chars
            
            # Combine factors
            confidence = (
                normalized_base * 0.4 +           # Vector similarity
                quality_score * 0.2 +             # Document quality
                query_content_overlap * 0.2 +     # Query relevance
                response_content_overlap * 0.1 +   # Response relevance
                length_factor * 0.1                # Content completeness
            )
            
            return round(min(1.0, confidence), 3)
            
        except Exception as e:
            print(f"Error calculating confidence: {e}")
            return base_score
    
    def _find_relevant_passages(self, query: str, response: str, content: str, max_passages: int = 3) -> List[Dict[str, Any]]:
        """Find and highlight relevant passages in the content"""
        try:
            # Split content into sentences
            sentences = self._split_into_sentences(content)
            if not sentences:
                return []
            
            # Score sentences for relevance
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            
            scored_sentences = []
            for i, sentence in enumerate(sentences):
                sentence_words = set(sentence.lower().split())
                
                # Calculate relevance score
                query_overlap = len(query_words.intersection(sentence_words)) / max(len(query_words), 1)
                response_overlap = len(response_words.intersection(sentence_words)) / max(len(response_words), 1)
                length_factor = min(1.0, len(sentence) / 200)  # Prefer substantial sentences
                
                relevance_score = query_overlap * 0.6 + response_overlap * 0.3 + length_factor * 0.1
                
                scored_sentences.append({
                    'sentence': sentence.strip(),
                    'position': i,
                    'relevance_score': relevance_score,
                    'word_count': len(sentence.split())
                })
            
            # Sort by relevance and take top passages
            scored_sentences.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            relevant_passages = []
            for sentence_data in scored_sentences[:max_passages]:
                if sentence_data['relevance_score'] > 0.1:  # Minimum relevance threshold
                    # Highlight key terms
                    highlighted_sentence = self._highlight_terms(
                        sentence_data['sentence'], 
                        query_words.union(response_words)
                    )
                    
                    relevant_passages.append({
                        'text': sentence_data['sentence'],
                        'highlighted_text': highlighted_sentence,
                        'position': sentence_data['position'],
                        'relevance_score': round(sentence_data['relevance_score'], 3),
                        'word_count': sentence_data['word_count']
                    })
            
            return relevant_passages
            
        except Exception as e:
            print(f"Error finding relevant passages: {e}")
            return [{'text': content[:200] + "...", 'highlighted_text': content[:200] + "...", 'relevance_score': 0.5}]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        # Simple sentence splitting (could be enhanced with NLTK)
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]  # Filter out very short sentences
    
    def _highlight_terms(self, text: str, terms: set) -> str:
        """Highlight relevant terms in text"""
        highlighted = text
        for term in terms:
            if len(term) > 2:  # Only highlight meaningful terms
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                highlighted = pattern.sub(f'**{term}**', highlighted)
        return highlighted
    
    def generate_citation_summary(self, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for citations"""
        if not citations:
            return {}
        
        total_citations = len(citations)
        avg_confidence = sum(c['confidence_score'] for c in citations) / total_citations
        
        # Count by categories
        categories = {}
        content_types = {}
        source_types = {}
        
        for citation in citations:
            cat = citation.get('document_category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
            
            ct = citation.get('content_type', 'unknown')
            content_types[ct] = content_types.get(ct, 0) + 1
            
            st = citation.get('source_type', 'primary')
            source_types[st] = source_types.get(st, 0) + 1
        
        return {
            'total_citations': total_citations,
            'average_confidence': round(avg_confidence, 3),
            'high_confidence_count': len([c for c in citations if c['confidence_score'] > 0.7]),
            'categories': categories,
            'content_types': content_types,
            'source_types': source_types,
            'top_citation': citations[0] if citations else None
        }


class QueryIntelligence:
    """Smart query analysis and enhancement system"""
    
    def __init__(self):
        self.query_patterns = {
            QueryType.FACTUAL: [
                r'\bwhat\s+is\b',
                r'\bwho\s+is\b',
                r'\bwhen\s+did\b',
                r'\bwhere\s+is\b',
                r'\bhow\s+many\b',
                r'\bhow\s+much\b'
            ],
            QueryType.COMPARISON: [
                r'\bcompare\b',
                r'\bdifference\s+between\b',
                r'\bvs\b',
                r'\bversus\b',
                r'\bbetter\s+than\b',
                r'\bsimilar\s+to\b'
            ],
            QueryType.DEFINITION: [
                r'\bdefine\b',
                r'\bdefinition\s+of\b',
                r'\bwhat\s+does\s+.+\s+mean\b',
                r'\bexplain\s+.+\s+term\b'
            ],
            QueryType.PROCEDURE: [
                r'\bhow\s+to\b',
                r'\bsteps\s+to\b',
                r'\bprocess\s+of\b',
                r'\bmethod\s+for\b',
                r'\bprocedure\b'
            ],
            QueryType.ANALYSIS: [
                r'\banalyze\b',
                r'\bevaluate\b',
                r'\bassess\b',
                r'\bwhy\s+is\b',
                r'\bcause\s+of\b',
                r'\breason\s+for\b'
            ]
        }
        
        # Initialize Gemini for query expansion
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.expansion_model = genai.GenerativeModel("gemini-2.5-flash")
    
    def detect_query_type(self, query: str) -> QueryType:
        """Detect the type of query for intelligent processing"""
        query_lower = query.lower()
        
        # Score each query type based on pattern matches
        type_scores = {}
        for query_type, patterns in self.query_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches
            type_scores[query_type] = score
        
        # Return the highest scoring type, or GENERAL if no clear match
        if max(type_scores.values()) > 0:
            return max(type_scores, key=type_scores.get)
        return QueryType.GENERAL
    
    def expand_query(self, query: str, query_type: QueryType) -> List[str]:
        """Generate expanded queries based on query type"""
        try:
            # Create type-specific expansion prompts
            expansion_prompts = {
                QueryType.FACTUAL: f"Generate 2-3 alternative factual questions that seek the same information as: '{query}'. Focus on different ways to ask for facts.",
                QueryType.COMPARISON: f"Generate 2-3 alternative comparison questions related to: '{query}'. Include different aspects to compare.",
                QueryType.DEFINITION: f"Generate 2-3 alternative ways to ask for definitions or explanations related to: '{query}'.",
                QueryType.PROCEDURE: f"Generate 2-3 alternative questions about processes or methods related to: '{query}'.",
                QueryType.ANALYSIS: f"Generate 2-3 alternative analytical questions related to: '{query}'. Include different analytical angles.",
                QueryType.GENERAL: f"Generate 2-3 alternative questions that seek similar information as: '{query}'."
            }
            
            prompt = expansion_prompts.get(query_type, expansion_prompts[QueryType.GENERAL])
            prompt += "\n\nReturn only the questions, one per line, without numbering or bullet points."
            
            response = self.expansion_model.generate_content(prompt)
            
            # Parse the response to extract individual queries
            expanded_queries = []
            if response.text:
                lines = response.text.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    # Remove numbering, bullets, and other formatting
                    line = re.sub(r'^[-*•\d\.]+\s*', '', line)
                    if line and line != query:
                        expanded_queries.append(line)
            
            return expanded_queries[:3]  # Limit to 3 expansions
            
        except Exception as e:
            print(f"Error in query expansion: {e}")
            return []
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract key terms from the query"""
        # Remove stop words and extract meaningful terms
        stop_words = {
            'what', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'how', 'can', 'could', 'should', 'would', 'will',
            'does', 'do', 'did', 'has', 'have', 'had', 'been', 'being', 'am', 'are', 'was', 'were'
        }
        
        # Extract words (letters and numbers)
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:10]  # Limit to top 10 keywords
    
    def generate_metadata_filters(self, query: str, query_type: QueryType) -> Dict[str, Any]:
        """Generate metadata filters based on query analysis"""
        filters = {}
        
        # Detect document category preferences
        category_keywords = {
            'technical': ['technical', 'engineering', 'programming', 'code', 'software', 'hardware'],
            'business': ['business', 'marketing', 'sales', 'strategy', 'management', 'finance'],
            'academic': ['research', 'study', 'academic', 'theory', 'analysis', 'scholarly'],
            'legal': ['legal', 'law', 'contract', 'policy', 'regulation', 'compliance'],
            'medical': ['medical', 'health', 'treatment', 'diagnosis', 'clinical', 'patient']
        }
        
        query_lower = query.lower()
        for category, keywords in category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                filters['document_category'] = category
                break
        
        # Content type preferences based on query type
        if query_type == QueryType.DEFINITION:
            filters['content_type'] = 'informational'
        elif query_type == QueryType.PROCEDURE:
            filters['content_type'] = 'instructional'
        elif query_type == QueryType.ANALYSIS:
            filters['content_type'] = 'analytical'
        
        return filters
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Complete query analysis including type detection, expansion, and filtering"""
        query_type = self.detect_query_type(query)
        expanded_queries = self.expand_query(query, query_type)
        keywords = self.extract_keywords(query)
        metadata_filters = self.generate_metadata_filters(query, query_type)
        
        return {
            'original_query': query,
            'query_type': query_type.value,
            'expanded_queries': expanded_queries,
            'keywords': keywords,
            'metadata_filters': metadata_filters,
            'complexity_score': len(keywords) / 10.0,  # Simple complexity measure
            'analysis_timestamp': time.time()
        }


# Custom Gemini LLM wrapper for LangChain
class GeminiLLM(LLM):
    """Custom Gemini LLM wrapper for LangChain"""
    
    def __init__(self, model_name: str = "gemini-2.5-flash", **kwargs):
        super().__init__(**kwargs)
        self._model_name = model_name
        # Configure the API key
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self._model = genai.GenerativeModel(model_name)
    
    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Gemini API"""
        try:
            response = self._model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return f"Error: {str(e)}"

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "quickstart"))

# Custom Pinecone retriever using llama-text-embed-v2 with enhanced metadata support
class PineconeLlamaRetriever(BaseRetriever):
    def __init__(self, index, top_k: int = 4, query_intelligence: Optional[QueryIntelligence] = None):
        super().__init__()
        self._index = index
        self._top_k = top_k
        self._query_intelligence = query_intelligence or QueryIntelligence()
    
    def _matches_filters(self, hit_metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if hit metadata matches the provided filters (post-search filtering)"""
        for key, value in filters.items():
            if key in hit_metadata:
                hit_value = hit_metadata[key]
                if hit_value != value:
                    return False
            else:
                # If the filter key is not in metadata, consider it a non-match
                return False
        return True
    
    def _build_metadata_filter(self, metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build Pinecone metadata filter from query analysis (kept for compatibility but not used)"""
        # Note: This method is kept for compatibility but not used with auto-embedding API
        return {}
    
    def _get_relevant_documents(self, query: str, metadata_filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Enhanced retrieval with query intelligence (metadata filtering done post-search)"""
        try:
            # Analyze the query for intelligent processing
            query_analysis = self._query_intelligence.analyze_query(query)
            
            # Combine provided filters with intelligent filters
            combined_filters = metadata_filters or {}
            combined_filters.update(query_analysis.get('metadata_filters', {}))
            
            # Prepare search parameters (no filter support in auto-embedding API)
            search_params = {
                "namespace": "default",
                "query": {
                    "inputs": {"text": query},
                    "top_k": self._top_k * 2  # Get more results to filter later
                }
            }
            
            # Perform primary search
            results = self._index.search(**search_params)
            
            documents = []
            # Handle different response formats
            if hasattr(results, 'result') and hasattr(results.result, 'hits'):
                hits = results.result.hits
            elif hasattr(results, 'hits'):
                hits = results.hits
            else:
                print(f"Unexpected result format: {type(results)}")
                hits = []
            
            # Process primary results and apply post-search filtering
            for hit in hits:
                try:
                    # Safely extract text from fields dict
                    if hasattr(hit, 'fields') and isinstance(hit.fields, dict):
                        text = hit.fields.get('text', '')
                    else:
                        print(f"No fields dict found in hit: {hit}")
                        continue
                    
                    if not text:
                        continue
                    
                    # Get metadata from hit fields
                    hit_metadata = hit.fields if hasattr(hit, 'fields') and isinstance(hit.fields, dict) else {}
                    
                    # Apply post-search metadata filtering
                    if combined_filters and not self._matches_filters(hit_metadata, combined_filters):
                        continue
                    
                    # Enhanced metadata from Pinecone with safe defaults
                    # Note: Hit ID and score might be None in auto-embedding API
                    hit_id = getattr(hit, 'id', None) or f"hit-{len(documents)}"
                    hit_score = getattr(hit, 'score', None)
                    if hit_score is None:
                        hit_score = 1.0 - (len(documents) * 0.1)  # Assign decreasing scores
                    
                    metadata = {
                        'id': hit_id,
                        'score': float(hit_score),
                        'query_type': query_analysis.get('query_type', 'general'),
                        'source_type': 'primary'
                    }
                    
                    # Add available metadata fields with safe extraction
                    for field in ['filename', 'document_category', 'content_type', 'quality_score', 'page_number', 'title', 'word_count']:
                        if field in hit_metadata:
                            value = hit_metadata[field]
                            if value is not None:
                                metadata[field] = value
                    
                    doc = Document(page_content=text, metadata=metadata)
                    documents.append(doc)
                    
                except Exception as e:
                    print(f"Error processing hit: {e}")
                    continue
            
            # If we have expanded queries and insufficient results, search with them
            expanded_queries = query_analysis.get('expanded_queries', [])
            if len(documents) < self._top_k and expanded_queries:
                for expanded_query in expanded_queries[:2]:  # Limit to 2 expansions
                    try:
                        expanded_search_params = {
                            "namespace": "default",
                            "query": {
                                "inputs": {"text": expanded_query},
                                "top_k": max(1, self._top_k - len(documents))
                            }
                        }
                        
                        expanded_results = self._index.search(**expanded_search_params)
                        
                        # Handle different response formats
                        if hasattr(expanded_results, 'result') and hasattr(expanded_results.result, 'hits'):
                            expanded_hits = expanded_results.result.hits
                        elif hasattr(expanded_results, 'hits'):
                            expanded_hits = expanded_results.hits
                        else:
                            expanded_hits = []
                        
                        # Add unique results from expanded search
                        existing_ids = {doc.metadata.get('id') for doc in documents}
                        
                        for hit in expanded_hits:
                            try:
                                hit_id = getattr(hit, 'id', None) or f"expanded-hit-{len(documents)}"
                                if hit_id not in existing_ids:
                                    # Safely extract text from fields dict
                                    if hasattr(hit, 'fields') and isinstance(hit.fields, dict):
                                        text = hit.fields.get('text', '')
                                    else:
                                        continue
                                    
                                    if not text:
                                        continue
                                    
                                    # Get metadata from hit fields
                                    hit_metadata = hit.fields if hasattr(hit, 'fields') and isinstance(hit.fields, dict) else {}
                                    
                                    # Apply post-search metadata filtering
                                    if combined_filters and not self._matches_filters(hit_metadata, combined_filters):
                                        continue
                                    
                                    # Handle None score for expanded hits
                                    hit_score = getattr(hit, 'score', None)
                                    if hit_score is None:
                                        hit_score = 0.5 - (len(documents) * 0.05)  # Lower scores for expanded
                                    
                                    metadata = {
                                        'id': hit_id,
                                        'score': float(hit_score),
                                        'query_type': query_analysis.get('query_type', 'general'),
                                        'source_type': 'expanded',
                                        'expanded_query': expanded_query
                                    }
                                    
                                    # Add available metadata fields with safe extraction
                                    for field in ['filename', 'document_category', 'content_type', 'quality_score', 'page_number', 'title', 'word_count']:
                                        if field in hit_metadata:
                                            value = hit_metadata[field]
                                            if value is not None:
                                                metadata[field] = value
                                    
                                    doc = Document(page_content=text, metadata=metadata)
                                    documents.append(doc)
                                    existing_ids.add(hit_id)
                                    
                                    if len(documents) >= self._top_k:
                                        break
                            except Exception as e:
                                print(f"Error processing expanded hit: {e}")
                                continue
                        
                        if len(documents) >= self._top_k:
                            break
                            
                    except Exception as e:
                        print(f"Error in expanded query search: {e}")
                        continue
            
            # Sort by score (descending) - handle None values safely
            documents.sort(key=lambda x: float(x.metadata.get('score', 0) or 0), reverse=True)
            
            return documents[:self._top_k]
            
        except Exception as e:
            print(f"Error in enhanced Pinecone search: {e}")
            return []
    
    async def _aget_relevant_documents(self, query: str, metadata_filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        return self._get_relevant_documents(query, metadata_filters)

# Initialize enhanced components
query_intelligence = QueryIntelligence()
citation_enhancer = CitationEnhancer()
retriever_vector = PineconeLlamaRetriever(index=index, top_k=4, query_intelligence=query_intelligence)

# Load BM25 retriever from pickle
import os
bm25_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "load_index", "bm25_index.pkl")
with open(bm25_path, "rb") as f:
    bm25_retriever = pickle.load(f)

# Combine BM25 and vector retrievers
hybridRetriever = EnsembleRetriever(
    retrievers=[bm25_retriever, retriever_vector],
    weights=[0.3, 0.7]
)

# Add Cohere reranker
reranker = CohereRerank(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    top_n=2,
    model="rerank-english-v3.0"
)
final_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=hybridRetriever
)

# Define the prompt template properly
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the following question using only the context provided.
If the answer is not in the context, say "I don't know."

Question: {query}
Context: {context}
""")

# Initialize LLM (Google Gemini 2.5 Flash)
llm = GeminiLLM(model_name="gemini-2.5-flash")

# Define the chain
chain = (
    RunnableParallel({
        "query": RunnablePassthrough(),
        "context": final_retriever
    })
    | prompt
    | llm
    | StrOutputParser()
)


def intelligent_rag_query(query: str, include_analysis: bool = True, include_citations: bool = True) -> Dict[str, Any]:
    """
    Enhanced RAG query with intelligent analysis, metadata, and citations
    
    Args:
        query: The user's question
        include_analysis: Whether to include query analysis in response
        include_citations: Whether to include enhanced citations
    
    Returns:
        Dictionary containing response, analysis, citations, and metadata
    """
    try:
        start_time = time.time()
        
        # Analyze the query
        query_analysis = query_intelligence.analyze_query(query)
        
        # Get the response from the chain
        response = chain.invoke(query)
        
        # Get relevant documents with metadata for citation enhancement
        retrieved_docs = final_retriever.get_relevant_documents(query)
        
        # Build enhanced result
        result = {
            'response': response,
            'query': query,
            'retrieved_documents': len(retrieved_docs),
            'processing_time': time.time() - start_time,
            'timestamp': time.time()
        }
        
        # Add query analysis if requested
        if include_analysis:
            result['query_analysis'] = {
                'query_type': query_analysis['query_type'],
                'keywords': query_analysis['keywords'],
                'complexity_score': query_analysis['complexity_score'],
                'expanded_queries': query_analysis['expanded_queries'],
                'metadata_filters': query_analysis['metadata_filters']
            }
        
        # Add enhanced citations if requested
        if include_citations and retrieved_docs:
            citations = citation_enhancer.extract_citations(query, response, retrieved_docs)
            citation_summary = citation_enhancer.generate_citation_summary(citations)
            
            result['citations'] = citations
            result['citation_summary'] = citation_summary
            
            # Legacy document metadata for backward compatibility
            result['document_metadata'] = []
            for citation in citations:
                doc_meta = {
                    'score': citation['relevance_score'],
                    'filename': citation['filename'],
                    'document_category': citation['document_category'],
                    'content_type': citation['content_type'],
                    'source_type': citation['source_type'],
                    'page_number': citation['page_number'],
                    'content_preview': citation['content_preview'],
                    'confidence_score': citation['confidence_score']
                }
                if citation.get('expanded_query'):
                    doc_meta['expanded_query'] = citation['expanded_query']
                
                result['document_metadata'].append(doc_meta)
        
        return result
        
    except Exception as e:
        return {
            'response': f"Error processing query: {str(e)}",
            'query': query,
            'error': True,
            'error_message': str(e),
            'timestamp': time.time()
        }


# Test function to verify the enhanced pipeline works
def test_pipeline():
    """Test the enhanced RAG pipeline with intelligent query processing"""
    try:
        test_queries = [
            "What is retrieval augmented generation?",  # Factual
            "How to implement a RAG system?",           # Procedure
            "Compare vector search vs keyword search"   # Comparison
        ]
        
        print("🧪 Testing Enhanced RAG Pipeline with Query Intelligence")
        print("=" * 60)
        
        for i, test_query in enumerate(test_queries, 1):
            print(f"\n🔍 Test {i}: '{test_query}'")
            print("-" * 40)
            
            # Test intelligent RAG query
            result = intelligent_rag_query(test_query, include_analysis=True)
            
            if result.get('error'):
                print(f"❌ Error: {result.get('error_message')}")
                continue
            
            print(f"📝 Response: {result['response'][:200]}...")
            print(f"🧠 Query Type: {result['query_analysis']['query_type']}")
            print(f"🔑 Keywords: {', '.join(result['query_analysis']['keywords'][:5])}")
            print(f"📊 Complexity Score: {result['query_analysis']['complexity_score']:.2f}")
            print(f"� Retrieved Documents: {result['retrieved_documents']}")
            print(f"⏱️ Processing Time: {result['processing_time']:.2f}s")
            
            if result['query_analysis']['expanded_queries']:
                print(f"🔄 Expanded Queries: {len(result['query_analysis']['expanded_queries'])}")
                for j, expanded in enumerate(result['query_analysis']['expanded_queries'][:2], 1):
                    print(f"   {j}. {expanded}")
            
            if result['query_analysis']['metadata_filters']:
                print(f"🏷️ Filters Applied: {result['query_analysis']['metadata_filters']}")
        
        print(f"\n✅ Enhanced pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Uncomment the line below to test the pipeline when running this file
# test_pipeline()

