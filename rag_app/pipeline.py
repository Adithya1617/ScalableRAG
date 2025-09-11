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
from typing import List, Optional, Any

from langchain_cohere import CohereRerank
import pickle

from pinecone import Pinecone

# Load environment variables
load_dotenv()

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

# Custom Pinecone retriever using llama-text-embed-v2
class PineconeLlamaRetriever(BaseRetriever):
    def __init__(self, index, top_k: int = 4):
        super().__init__()
        self._index = index
        self._top_k = top_k
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Use Pinecone's native search with llama-text-embed-v2
        try:
            # For integrated embedding, use the correct search format
            results = self._index.search(
                query={
                    "inputs": {"text": query},
                    "top_k": self._top_k
                },
                namespace="default"
            )
            
            documents = []
            # Parse results based on Pinecone's actual response structure
            # The response has structure: {'result': {'hits': [...]}}
            hits = results.get('result', {}).get('hits', [])
            for hit in hits:
                # Extract text from the stored data
                text = hit.get('fields', {}).get('text', '')
                if text:
                    doc = Document(
                        page_content=text,
                        metadata={
                            'id': hit.get('_id', ''),
                            'score': hit.get('_score', 0)
                        }
                    )
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Error in Pinecone search: {e}")
            return []
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

# Initialize the Pinecone retriever using llama-text-embed-v2
retriever_vector = PineconeLlamaRetriever(index=index, top_k=4)

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

# Test function to verify the pipeline works
def test_pipeline():
    """Test the RAG pipeline with a sample query"""
    try:
        test_query = "What is retrieval augmented generation?"
        print(f"🧪 Testing pipeline with query: '{test_query}'")
        
        # Test the chain
        result = chain.invoke(test_query)
        print(f"✅ Pipeline test successful!")
        print(f"📝 Response: {result}")
        
        return result
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return None

# Uncomment the line below to test the pipeline when running this file
# test_pipeline()

