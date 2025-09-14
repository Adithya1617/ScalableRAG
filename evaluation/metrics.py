import time
import statistics
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime
import concurrent.futures
from langchain_core.documents import Document
import google.generativeai as genai
from rouge_score import rouge_scorer
import os
from dotenv import load_dotenv

# Import advanced metrics
try:
    from evaluation.advanced_metrics import AdvancedRAGEvaluator, BenchmarkComparator
    ADVANCED_METRICS_AVAILABLE = True
except ImportError as e:
    print(f"Advanced metrics not available: {e}")
    ADVANCED_METRICS_AVAILABLE = False

load_dotenv()

class RAGEvaluator:
    def __init__(self, chain, retrievers_dict):
        """
        Enhanced RAG evaluator with advanced metrics support
        """
        self.chain = chain
        self.retrievers = retrievers_dict
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize Gemini for LLM-as-judge
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.judge_model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Initialize advanced evaluator if available
        self.advanced_evaluator = None
        self.benchmark_comparator = None
        if ADVANCED_METRICS_AVAILABLE:
            try:
                self.advanced_evaluator = AdvancedRAGEvaluator()
                self.benchmark_comparator = BenchmarkComparator()
                print("✅ Advanced metrics system initialized")
            except Exception as e:
                print(f"⚠️ Could not initialize advanced metrics: {e}")
        
        self.results_history = []
        self.human_feedback_storage = []
    
    def _extract_doc_id(self, doc):
        """
        Extract document ID from a retrieved document
        Handles different ID formats in your system
        """
        # Try different ways to get document ID
        if hasattr(doc, 'metadata') and doc.metadata:
            # Try 'id' field first
            if 'id' in doc.metadata:
                return doc.metadata['id']
            # Try other possible ID fields
            for id_field in ['doc_id', '_id', 'source', 'chunk_id']:
                if id_field in doc.metadata:
                    return doc.metadata[id_field]
        
        # Fallback: use first 50 characters of content as ID
        return doc.page_content[:50].replace('\n', ' ').strip()
    
    def evaluate_retrieval_quality_fixed(self, test_queries: List[Dict]) -> Dict:
        """
        Fixed retrieval evaluation that accounts for fuzzy matching
        """
        results = {}
        
        for retriever_name, retriever in self.retrievers.items():
            print(f"Evaluating {retriever_name} retriever...")
            
            hit_rate_scores = []
            mrr_scores = []
            precision_scores = []
            
            for test_case in test_queries:
                query = test_case['query']
                relevant_content = test_case.get('document_content', '')
                relevant_ids = set(test_case.get('relevant_doc_ids', []))
                
                try:
                    # Get retrieved documents
                    retrieved_docs = retriever.get_relevant_documents(query)
                    
                    if not retrieved_docs:
                        hit_rate_scores.append(0)
                        mrr_scores.append(0)
                        precision_scores.append(0)
                        continue
                    
                    # Extract IDs from retrieved docs
                    retrieved_ids = set()
                    retrieved_contents = []
                    
                    for doc in retrieved_docs[:5]:  # Top 5 results
                        doc_id = self._extract_doc_id(doc)
                        retrieved_ids.add(doc_id)
                        retrieved_contents.append(doc.page_content)
                    
                    # Method 1: Exact ID matching (if possible)
                    exact_matches = len(relevant_ids.intersection(retrieved_ids))
                    
                    # Method 2: Content-based matching (more reliable)
                    content_matches = 0
                    if relevant_content:
                        for content in retrieved_contents:
                            # Simple overlap check - if >30% of relevant content appears in retrieved content
                            relevant_words = set(relevant_content.lower().split())
                            retrieved_words = set(content.lower().split())
                            
                            if len(relevant_words) > 0:
                                overlap = len(relevant_words.intersection(retrieved_words))
                                overlap_ratio = overlap / len(relevant_words)
                                if overlap_ratio > 0.3:  # 30% word overlap threshold
                                    content_matches = 1
                                    break
                    
                    # Use the better of the two matching methods
                    has_relevant = max(1 if exact_matches > 0 else 0, content_matches)
                    
                    # Calculate metrics
                    hit_rate_scores.append(has_relevant)
                    
                    # MRR - find position of first relevant document
                    mrr = 0
                    if relevant_content:
                        for i, content in enumerate(retrieved_contents):
                            relevant_words = set(relevant_content.lower().split())
                            retrieved_words = set(content.lower().split())
                            
                            if len(relevant_words) > 0:
                                overlap_ratio = len(relevant_words.intersection(retrieved_words)) / len(relevant_words)
                                if overlap_ratio > 0.3:
                                    mrr = 1 / (i + 1)
                                    break
                    elif exact_matches > 0:
                        mrr = 1.0  # Assume first position if we have exact matches
                    
                    mrr_scores.append(mrr)
                    
                    # Precision - ratio of relevant docs in top-k
                    total_relevant = max(exact_matches, content_matches)
                    precision = total_relevant / min(len(retrieved_docs), 5)
                    precision_scores.append(precision)
                    
                    # Debug info
                    print(f"   Query: {query[:50]}...")
                    print(f"   Retrieved: {len(retrieved_docs)} docs, Hit: {has_relevant}, MRR: {mrr:.3f}")
                    
                except Exception as e:
                    print(f"   Error evaluating query '{query[:30]}...': {e}")
                    hit_rate_scores.append(0)
                    mrr_scores.append(0)
                    precision_scores.append(0)
            
            # Calculate averages
            results[retriever_name] = {
                'hit_rate': sum(hit_rate_scores) / len(hit_rate_scores) if hit_rate_scores else 0,
                'mrr': sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0,
                'precision': sum(precision_scores) / len(precision_scores) if precision_scores else 0,
                'total_queries': len(test_queries)
            }
            
            print(f"   Results: Hit Rate={results[retriever_name]['hit_rate']:.3f}, MRR={results[retriever_name]['mrr']:.3f}")
        
        return results
    
    def evaluate_response_quality_advanced(self, test_queries: List[Dict]) -> Dict:
        """
        Enhanced response quality evaluation with advanced metrics
        """
        print("   Running advanced response quality evaluation...")
        
        rouge_scores = []
        llm_judge_scores = []
        advanced_scores = []
        response_times = []
        successful_responses = 0
        
        for i, test_case in enumerate(test_queries):
            query = test_case['query']
            expected_answer = test_case.get('expected_answer', '')
            
            print(f"   Evaluating response {i+1}/{len(test_queries)}: {query[:50]}...")
            
            # Measure response time
            start_time = time.time()
            try:
                # Get response and retrieved documents
                actual_response = self.chain.invoke(query)
                response_time = time.time() - start_time
                response_times.append(response_time)
                successful_responses += 1
                
                print(f"     Response time: {response_time:.2f}s")
                print(f"     Response preview: {actual_response[:100]}...")
                
                # Get retrieved documents for advanced evaluation
                retrieved_docs = []
                citations = []
                
                # Try to get documents from retrievers
                try:
                    if 'hybrid_reranked' in self.retrievers:
                        retrieved_docs = self.retrievers['hybrid_reranked'].get_relevant_documents(query)
                    elif 'vector' in self.retrievers:
                        retrieved_docs = self.retrievers['vector'].get_relevant_documents(query)
                except Exception as e:
                    print(f"     Could not get retrieved docs: {e}")
                
                # ROUGE evaluation (only if we have expected answer)
                if expected_answer and len(expected_answer.strip()) > 10:
                    rouge_score = self.rouge_scorer.score(expected_answer, actual_response)
                    rouge_scores.append({
                        'rouge1': rouge_score['rouge1'].fmeasure,
                        'rouge2': rouge_score['rouge2'].fmeasure,
                        'rougeL': rouge_score['rougeL'].fmeasure
                    })
                
                # LLM-as-judge evaluation
                judge_score = self._llm_judge_evaluation(query, actual_response, expected_answer)
                llm_judge_scores.append(judge_score)
                
                # Advanced evaluation if available
                if self.advanced_evaluator and expected_answer:
                    try:
                        advanced_result = self.advanced_evaluator.evaluate_comprehensive(
                            query, expected_answer, actual_response, retrieved_docs, citations
                        )
                        
                        # Get benchmark comparison
                        benchmark_comparison = None
                        if self.benchmark_comparator:
                            comparison = self.benchmark_comparator.compare_to_benchmarks(advanced_result)
                            benchmark_comparison = comparison.get('overall_score', 'unknown')
                        
                        advanced_score = {
                            'overall_score': advanced_result.overall_score,
                            'semantic_similarity': advanced_result.semantic_similarity,
                            'factual_consistency': advanced_result.factual_consistency,
                            'answer_completeness': advanced_result.answer_completeness,
                            'citation_precision': advanced_result.citation_precision,
                            'citation_recall': advanced_result.citation_recall,
                            'knowledge_coverage': advanced_result.knowledge_coverage,
                            'response_appropriateness': advanced_result.response_appropriateness,
                            'benchmark_comparison': benchmark_comparison,
                            'detailed_breakdown': advanced_result.detailed_breakdown
                        }
                        advanced_scores.append(advanced_score)
                        
                        print(f"     Advanced score: {advanced_result.overall_score:.3f}")
                        print(f"     Semantic similarity: {advanced_result.semantic_similarity:.3f}")
                        print(f"     Factual consistency: {advanced_result.factual_consistency:.3f}")
                        print(f"     Benchmark: {benchmark_comparison}")
                        
                    except Exception as e:
                        print(f"     Error in advanced evaluation: {e}")
                        # Create fallback advanced score
                        advanced_scores.append({
                            'overall_score': judge_score / 5.0,  # Convert 1-5 to 0-1
                            'semantic_similarity': 0.5,
                            'factual_consistency': 0.5,
                            'answer_completeness': 0.5,
                            'citation_precision': 0.5,
                            'citation_recall': 0.5,
                            'knowledge_coverage': 0.5,
                            'response_appropriateness': judge_score / 5.0,
                            'benchmark_comparison': 'error',
                            'detailed_breakdown': {'error': str(e)}
                        })
                
            except Exception as e:
                print(f"     Error: {e}")
                response_times.append(10.0)  # Penalty for failed queries
                if expected_answer and len(expected_answer.strip()) > 10:
                    rouge_scores.append({'rouge1': 0, 'rouge2': 0, 'rougeL': 0})
                llm_judge_scores.append(1.0)  # Minimum score for failed queries
                
                if self.advanced_evaluator and expected_answer:
                    advanced_scores.append({
                        'overall_score': 0.1,
                        'semantic_similarity': 0.0,
                        'factual_consistency': 0.0,
                        'answer_completeness': 0.0,
                        'citation_precision': 0.0,
                        'citation_recall': 0.0,
                        'knowledge_coverage': 0.0,
                        'response_appropriateness': 0.1,
                        'benchmark_comparison': 'failed',
                        'detailed_breakdown': {'error': str(e)}
                    })
        
        # Calculate averages
        avg_rouge = {}
        if rouge_scores:
            avg_rouge = {
                'rouge1': sum(r['rouge1'] for r in rouge_scores) / len(rouge_scores),
                'rouge2': sum(r['rouge2'] for r in rouge_scores) / len(rouge_scores),
                'rougeL': sum(r['rougeL'] for r in rouge_scores) / len(rouge_scores)
            }
        
        # Calculate advanced averages
        avg_advanced = {}
        if advanced_scores:
            avg_advanced = {
                'overall_score': sum(a['overall_score'] for a in advanced_scores) / len(advanced_scores),
                'semantic_similarity': sum(a['semantic_similarity'] for a in advanced_scores) / len(advanced_scores),
                'factual_consistency': sum(a['factual_consistency'] for a in advanced_scores) / len(advanced_scores),
                'answer_completeness': sum(a['answer_completeness'] for a in advanced_scores) / len(advanced_scores),
                'citation_precision': sum(a['citation_precision'] for a in advanced_scores) / len(advanced_scores),
                'citation_recall': sum(a['citation_recall'] for a in advanced_scores) / len(advanced_scores),
                'knowledge_coverage': sum(a['knowledge_coverage'] for a in advanced_scores) / len(advanced_scores),
                'response_appropriateness': sum(a['response_appropriateness'] for a in advanced_scores) / len(advanced_scores),
                'benchmark_comparisons': [a['benchmark_comparison'] for a in advanced_scores],
                'individual_scores': advanced_scores
            }
        
        result = {
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'p95_response_time': statistics.quantiles(response_times, n=20)[18] if len(response_times) > 1 else (response_times[0] if response_times else 0),
            'rouge_scores': avg_rouge,
            'avg_llm_judge_score': sum(llm_judge_scores) / len(llm_judge_scores) if llm_judge_scores else 0,
            'success_rate': successful_responses / len(test_queries) if test_queries else 0,
            'total_queries': len(test_queries),
            'advanced_metrics': avg_advanced if avg_advanced else None
        }
        
        return result
        """
        Evaluate end-to-end response quality
        """
        rouge_scores = []
        llm_judge_scores = []
        response_times = []
        successful_responses = 0
        
        for i, test_case in enumerate(test_queries):
            query = test_case['query']
            expected_answer = test_case.get('expected_answer', '')
            
            print(f"   Evaluating response {i+1}/{len(test_queries)}: {query[:50]}...")
            
            # Measure response time
            start_time = time.time()
            try:
                actual_response = self.chain.invoke(query)
                response_time = time.time() - start_time
                response_times.append(response_time)
                successful_responses += 1
                
                print(f"     Response time: {response_time:.2f}s")
                print(f"     Response preview: {actual_response[:100]}...")
                
                # ROUGE evaluation (only if we have expected answer)
                if expected_answer and len(expected_answer.strip()) > 10:
                    rouge_score = self.rouge_scorer.score(expected_answer, actual_response)
                    rouge_scores.append({
                        'rouge1': rouge_score['rouge1'].fmeasure,
                        'rouge2': rouge_score['rouge2'].fmeasure,
                        'rougeL': rouge_score['rougeL'].fmeasure
                    })
                
                # LLM-as-judge evaluation
                judge_score = self._llm_judge_evaluation(query, actual_response, expected_answer)
                llm_judge_scores.append(judge_score)
                
            except Exception as e:
                print(f"     Error: {e}")
                response_times.append(10.0)  # Penalty for failed queries
                if expected_answer and len(expected_answer.strip()) > 10:
                    rouge_scores.append({'rouge1': 0, 'rouge2': 0, 'rougeL': 0})
                llm_judge_scores.append(1.0)  # Minimum score for failed queries
        
        # Calculate averages
        avg_rouge = {}
        if rouge_scores:
            avg_rouge = {
                'rouge1': sum(r['rouge1'] for r in rouge_scores) / len(rouge_scores),
                'rouge2': sum(r['rouge2'] for r in rouge_scores) / len(rouge_scores),
                'rougeL': sum(r['rougeL'] for r in rouge_scores) / len(rouge_scores)
            }
        
        return {
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'p95_response_time': statistics.quantiles(response_times, n=20)[18] if len(response_times) > 1 else (response_times[0] if response_times else 0),
            'rouge_scores': avg_rouge,
            'avg_llm_judge_score': sum(llm_judge_scores) / len(llm_judge_scores) if llm_judge_scores else 0,
            'success_rate': successful_responses / len(test_queries) if test_queries else 0,
            'total_queries': len(test_queries)
        }
    
    def _llm_judge_evaluation(self, query: str, response: str, expected_answer: str = "") -> float:
        """
        Use LLM to judge response quality - simplified and more reliable
        """
        if len(response.strip()) < 5:  # Very short responses get low scores
            return 1.0
        
        judge_prompt = f"""
        Rate this AI response on a scale of 1-5 (1=poor, 5=excellent):
        
        Question: {query}
        AI Response: {response}
        
        Consider:
        - Does it answer the question?
        - Is the information coherent?
        - Is it helpful to the user?
        
        Respond with only a number from 1 to 5.
        """
        
        try:
            judge_response = self.judge_model.generate_content(judge_prompt)
            score_text = judge_response.text.strip()
            
            # Extract number
            import re
            numbers = re.findall(r'[1-5]', score_text)
            if numbers:
                score = float(numbers[0])
                return max(1.0, min(5.0, score))
            else:
                return 3.0  # Default neutral score
                
        except Exception as e:
            print(f"     LLM judge error: {e}")
            return 3.0
    
    def run_comprehensive_evaluation(self, test_queries: List[Dict], include_stress_test: bool = False) -> Dict:
        """
        Run comprehensive evaluation with fixes
        """
        print("Starting Fixed RAG Evaluation...")
        print("=" * 50)
        evaluation_start = time.time()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'test_dataset_size': len(test_queries),
        }
        
        # 1. Retrieval quality evaluation
        print("\n1. Evaluating retrieval quality (with fixes)...")
        results['retrieval_quality'] = self.evaluate_retrieval_quality_fixed(test_queries)
        
        # 2. Response quality evaluation (with advanced metrics)
        print("\n2. Evaluating response quality with advanced metrics...")
        results['response_quality'] = self.evaluate_response_quality_advanced(test_queries)
        
        results['total_evaluation_time'] = time.time() - evaluation_start
        
        # Save results
        self.save_results(results)
        
        return results
    
    def save_results(self, results: Dict):
        """Save evaluation results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results/rag_evaluation_fixed_{timestamp}.json"
        
        # Create directory if it doesn't exist
        os.makedirs("evaluation_results", exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to {filename}")
        self.results_history.append(results)
    
    def generate_report(self, results: Dict) -> str:
        """Generate a human-readable evaluation report"""
        report = []
        report.append("=" * 60)
        report.append("FIXED RAG SYSTEM EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {results['timestamp']}")
        report.append(f"Test Dataset Size: {results['test_dataset_size']}")
        report.append("")
        
        # Retrieval Quality
        if 'retrieval_quality' in results:
            report.append("RETRIEVAL QUALITY METRICS (Fixed)")
            report.append("-" * 40)
            for method, metrics in results['retrieval_quality'].items():
                report.append(f"{method.upper()}:")
                report.append(f"  Hit Rate@5: {metrics['hit_rate']:.3f}")
                report.append(f"  MRR: {metrics['mrr']:.3f}")
                report.append(f"  Precision@5: {metrics['precision']:.3f}")
                report.append("")
        
        # Response Quality
        if 'response_quality' in results:
            rq = results['response_quality']
            report.append("RESPONSE QUALITY METRICS")
            report.append("-" * 30)
            report.append(f"Success Rate: {rq.get('success_rate', 0):.1%}")
            report.append(f"Average Response Time: {rq['avg_response_time']:.2f}s")
            report.append(f"P95 Response Time: {rq['p95_response_time']:.2f}s")
            report.append(f"LLM Judge Score: {rq['avg_llm_judge_score']:.2f}/5")
            
            if 'rouge_scores' in rq and rq['rouge_scores']:
                rs = rq['rouge_scores']
                report.append(f"ROUGE-1: {rs['rouge1']:.3f}")
                report.append(f"ROUGE-2: {rs['rouge2']:.3f}")
                report.append(f"ROUGE-L: {rs['rougeL']:.3f}")
            report.append("")
        
        report.append(f"Total Evaluation Time: {results['total_evaluation_time']:.1f}s")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def collect_human_feedback(self, query: str, response: str, feedback_data: Dict[str, Any]) -> str:
        """
        Collect and store human feedback for response quality
        
        Args:
            query: The original query
            response: The system response
            feedback_data: Dictionary containing:
                - rating: 1-5 star rating
                - feedback_text: Optional text feedback
                - dimensions: Optional specific dimension ratings
                - user_id: Optional user identifier
                - session_id: Optional session identifier
        
        Returns:
            feedback_id: Unique identifier for this feedback entry
        """
        import uuid
        
        feedback_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        feedback_entry = {
            'feedback_id': feedback_id,
            'timestamp': timestamp,
            'query': query,
            'response': response,
            'rating': feedback_data.get('rating', 0),
            'feedback_text': feedback_data.get('feedback_text', ''),
            'dimensions': feedback_data.get('dimensions', {}),
            'user_id': feedback_data.get('user_id', 'anonymous'),
            'session_id': feedback_data.get('session_id', 'unknown'),
            'response_length': len(response.split()),
            'query_length': len(query.split())
        }
        
        # Store in memory
        self.human_feedback_storage.append(feedback_entry)
        
        # Save to file
        self._save_human_feedback(feedback_entry)
        
        print(f"✅ Human feedback collected: {feedback_id}")
        return feedback_id
    
    def _save_human_feedback(self, feedback_entry: Dict[str, Any]):
        """Save human feedback to persistent storage"""
        feedback_dir = "evaluation_results/human_feedback"
        os.makedirs(feedback_dir, exist_ok=True)
        
        # Save individual feedback
        feedback_file = f"{feedback_dir}/feedback_{feedback_entry['feedback_id']}.json"
        with open(feedback_file, 'w') as f:
            json.dump(feedback_entry, f, indent=2)
        
        # Append to aggregated feedback log
        log_file = f"{feedback_dir}/feedback_log.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(feedback_entry) + '\n')
    
    def analyze_human_feedback(self) -> Dict[str, Any]:
        """
        Analyze collected human feedback to identify patterns and insights
        """
        if not self.human_feedback_storage:
            return {'error': 'No human feedback available'}
        
        feedback_data = self.human_feedback_storage
        
        # Basic statistics
        ratings = [f['rating'] for f in feedback_data if f['rating'] > 0]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        # Rating distribution
        rating_distribution = {}
        for rating in range(1, 6):
            rating_distribution[rating] = len([r for r in ratings if r == rating])
        
        # Response length analysis
        response_lengths = [f['response_length'] for f in feedback_data]
        avg_response_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
        
        # Query type analysis
        query_patterns = self._analyze_query_patterns([f['query'] for f in feedback_data])
        
        # Sentiment analysis of feedback text
        feedback_sentiment = self._analyze_feedback_sentiment([f['feedback_text'] for f in feedback_data if f['feedback_text']])
        
        # Identify top issues
        low_rated_feedback = [f for f in feedback_data if f['rating'] <= 2]
        common_issues = self._extract_common_issues(low_rated_feedback)
        
        # High-performing responses
        high_rated_feedback = [f for f in feedback_data if f['rating'] >= 4]
        success_patterns = self._extract_success_patterns(high_rated_feedback)
        
        return {
            'total_feedback_entries': len(feedback_data),
            'average_rating': avg_rating,
            'rating_distribution': rating_distribution,
            'average_response_length': avg_response_length,
            'query_patterns': query_patterns,
            'feedback_sentiment': feedback_sentiment,
            'common_issues': common_issues,
            'success_patterns': success_patterns,
            'recommendations': self._generate_improvement_recommendations(feedback_data)
        }
    
    def _analyze_query_patterns(self, queries: List[str]) -> Dict[str, Any]:
        """Analyze patterns in user queries"""
        if not queries:
            return {}
        
        # Simple pattern analysis
        total_queries = len(queries)
        avg_query_length = sum(len(q.split()) for q in queries) / total_queries
        
        # Common question words
        question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which']
        question_word_counts = {}
        
        for word in question_words:
            count = sum(1 for q in queries if word.lower() in q.lower())
            question_word_counts[word] = count
        
        return {
            'total_queries': total_queries,
            'average_length': avg_query_length,
            'question_word_distribution': question_word_counts
        }
    
    def _analyze_feedback_sentiment(self, feedback_texts: List[str]) -> Dict[str, Any]:
        """Analyze sentiment of feedback text"""
        if not feedback_texts:
            return {'no_text_feedback': True}
        
        # Simple sentiment analysis based on keywords
        positive_words = ['good', 'great', 'excellent', 'helpful', 'accurate', 'clear', 'perfect']
        negative_words = ['bad', 'wrong', 'unhelpful', 'confusing', 'unclear', 'poor', 'terrible']
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for text in feedback_texts:
            text_lower = text.lower()
            pos_score = sum(1 for word in positive_words if word in text_lower)
            neg_score = sum(1 for word in negative_words if word in text_lower)
            
            if pos_score > neg_score:
                positive_count += 1
            elif neg_score > pos_score:
                negative_count += 1
            else:
                neutral_count += 1
        
        return {
            'positive_feedback': positive_count,
            'negative_feedback': negative_count,
            'neutral_feedback': neutral_count,
            'sentiment_score': (positive_count - negative_count) / len(feedback_texts)
        }
    
    def _extract_common_issues(self, low_rated_feedback: List[Dict]) -> List[str]:
        """Extract common issues from low-rated feedback"""
        if not low_rated_feedback:
            return []
        
        # Extract patterns from low-rated responses
        issues = []
        
        # Check for common problem indicators
        problem_indicators = {
            "I don't know": "System frequently says 'I don't know'",
            "wrong": "Factual inaccuracies reported",
            "incomplete": "Incomplete responses",
            "confusing": "Confusing or unclear responses",
            "too long": "Responses are too verbose",
            "too short": "Responses are too brief",
            "no sources": "Missing source citations"
        }
        
        for indicator, description in problem_indicators.items():
            count = sum(1 for f in low_rated_feedback 
                       if indicator.lower() in f['feedback_text'].lower() 
                       or indicator.lower() in f['response'].lower())
            if count > 0:
                issues.append(f"{description} ({count} instances)")
        
        return issues[:5]  # Top 5 issues
    
    def _extract_success_patterns(self, high_rated_feedback: List[Dict]) -> List[str]:
        """Extract success patterns from high-rated feedback"""
        if not high_rated_feedback:
            return []
        
        patterns = []
        
        # Analyze successful responses
        success_indicators = {
            "clear": "Clear and understandable responses",
            "accurate": "Accurate information provided", 
            "helpful": "Helpful and actionable responses",
            "detailed": "Appropriately detailed responses",
            "sources": "Good source attribution"
        }
        
        for indicator, description in success_indicators.items():
            count = sum(1 for f in high_rated_feedback 
                       if indicator.lower() in f['feedback_text'].lower())
            if count > 0:
                patterns.append(f"{description} ({count} instances)")
        
        return patterns[:5]  # Top 5 success patterns
    
    def _generate_improvement_recommendations(self, feedback_data: List[Dict]) -> List[str]:
        """Generate specific improvement recommendations based on feedback"""
        recommendations = []
        
        # Analyze rating patterns
        ratings = [f['rating'] for f in feedback_data if f['rating'] > 0]
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            
            if avg_rating < 3.0:
                recommendations.append("Overall system performance needs significant improvement")
            elif avg_rating < 4.0:
                recommendations.append("System performance is moderate, focus on consistency")
            
            # Check for specific issues
            low_rated = [f for f in feedback_data if f['rating'] <= 2]
            if len(low_rated) > len(feedback_data) * 0.3:  # More than 30% low ratings
                recommendations.append("High frequency of low ratings - investigate core system issues")
        
        # Response length analysis
        response_lengths = [f['response_length'] for f in feedback_data]
        if response_lengths:
            avg_length = sum(response_lengths) / len(response_lengths)
            if avg_length < 10:
                recommendations.append("Responses may be too brief - add more detail and context")
            elif avg_length > 200:
                recommendations.append("Responses may be too verbose - focus on conciseness")
        
        # Query understanding
        dont_know_responses = sum(1 for f in feedback_data 
                                 if "don't know" in f['response'].lower() 
                                 or "cannot" in f['response'].lower())
        if dont_know_responses > len(feedback_data) * 0.5:  # More than 50%
            recommendations.append("High frequency of 'don't know' responses - improve knowledge coverage")
        
        return recommendations[:10]  # Top 10 recommendations