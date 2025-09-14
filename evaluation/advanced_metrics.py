#!/usr/bin/env python3
"""
Advanced RAG evaluation metrics beyond basic ROUGE and LLM-judge
"""

import numpy as np
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter

@dataclass 
class AdvancedEvaluationResult:
    """Comprehensive evaluation result"""
    semantic_similarity: float
    factual_consistency: float
    answer_completeness: float
    citation_precision: float
    citation_recall: float
    knowledge_coverage: float
    response_appropriateness: float
    overall_score: float
    detailed_breakdown: Dict[str, Any]

class AdvancedRAGEvaluator:
    """Advanced RAG evaluation with multiple sophisticated metrics"""
    
    def __init__(self):
        # Load sentence transformer for semantic similarity
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Factual consistency patterns
        self.fact_patterns = {
            'numbers': r'\b\d+(?:\.\d+)?\b',
            'dates': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            'percentages': r'\b\d+(?:\.\d+)?%\b',
            'monetary': r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b',
            'proper_nouns': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        }
    
    def evaluate_comprehensive(self, 
                             query: str,
                             expected_answer: str,
                             actual_response: str,
                             retrieved_documents: List[Dict],
                             citations: List[Dict] = None) -> AdvancedEvaluationResult:
        """Perform comprehensive evaluation across all dimensions"""
        
        # 1. Semantic Similarity
        semantic_sim = self._evaluate_semantic_similarity(expected_answer, actual_response)
        
        # 2. Factual Consistency  
        factual_consistency = self._evaluate_factual_consistency(expected_answer, actual_response)
        
        # 3. Answer Completeness
        completeness = self._evaluate_answer_completeness(query, expected_answer, actual_response)
        
        # 4. Citation Quality
        cit_precision, cit_recall = self._evaluate_citation_quality(
            query, actual_response, retrieved_documents, citations
        )
        
        # 5. Knowledge Coverage
        knowledge_coverage = self._evaluate_knowledge_coverage(query, retrieved_documents)
        
        # 6. Response Appropriateness
        appropriateness = self._evaluate_response_appropriateness(query, actual_response)
        
        # Calculate overall score with weights
        weights = {
            'semantic': 0.25,
            'factual': 0.20,
            'completeness': 0.20,
            'citation_precision': 0.10,
            'citation_recall': 0.05,
            'knowledge_coverage': 0.10,
            'appropriateness': 0.10
        }
        
        overall_score = (
            semantic_sim * weights['semantic'] +
            factual_consistency * weights['factual'] +
            completeness * weights['completeness'] +
            cit_precision * weights['citation_precision'] +
            cit_recall * weights['citation_recall'] +
            knowledge_coverage * weights['knowledge_coverage'] +
            appropriateness * weights['appropriateness']
        )
        
        # Detailed breakdown
        breakdown = {
            'semantic_details': self._semantic_analysis_details(expected_answer, actual_response),
            'factual_details': self._factual_analysis_details(expected_answer, actual_response),
            'completeness_details': self._completeness_analysis_details(query, expected_answer, actual_response),
            'citation_details': self._citation_analysis_details(retrieved_documents, citations),
            'weights_used': weights
        }
        
        return AdvancedEvaluationResult(
            semantic_similarity=semantic_sim,
            factual_consistency=factual_consistency,
            answer_completeness=completeness,
            citation_precision=cit_precision,
            citation_recall=cit_recall,
            knowledge_coverage=knowledge_coverage,
            response_appropriateness=appropriateness,
            overall_score=overall_score,
            detailed_breakdown=breakdown
        )
    
    def _evaluate_semantic_similarity(self, expected: str, actual: str) -> float:
        """Evaluate semantic similarity using sentence embeddings"""
        try:
            expected_embedding = self.sentence_model.encode([expected])
            actual_embedding = self.sentence_model.encode([actual])
            
            similarity = cosine_similarity(expected_embedding, actual_embedding)[0][0]
            return max(0, similarity)  # Ensure non-negative
        except Exception as e:
            print(f"Error in semantic similarity: {e}")
            return 0.0
    
    def _evaluate_factual_consistency(self, expected: str, actual: str) -> float:
        """Evaluate factual consistency by comparing extracted facts"""
        expected_facts = self._extract_facts(expected)
        actual_facts = self._extract_facts(actual)
        
        if not expected_facts:
            return 1.0 if not actual_facts else 0.5  # No facts to check
        
        consistency_scores = []
        
        for fact_type, expected_values in expected_facts.items():
            actual_values = actual_facts.get(fact_type, [])
            
            if not expected_values:
                continue
                
            # Calculate overlap
            expected_set = set(expected_values)
            actual_set = set(actual_values)
            
            if expected_set:
                overlap = len(expected_set.intersection(actual_set))
                consistency = overlap / len(expected_set)
                consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _extract_facts(self, text: str) -> Dict[str, List[str]]:
        """Extract various types of facts from text"""
        facts = {}
        
        for fact_type, pattern in self.fact_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            facts[fact_type] = list(set(matches))  # Remove duplicates
        
        return facts
    
    def _evaluate_answer_completeness(self, query: str, expected: str, actual: str) -> float:
        """Evaluate how complete the answer is"""
        
        # Extract key concepts from expected answer
        expected_concepts = self._extract_key_concepts(expected)
        actual_concepts = self._extract_key_concepts(actual)
        
        if not expected_concepts:
            return 1.0
        
        # Check coverage of expected concepts
        covered_concepts = 0
        for concept in expected_concepts:
            if any(concept.lower() in actual_concept.lower() for actual_concept in actual_concepts):
                covered_concepts += 1
        
        concept_coverage = covered_concepts / len(expected_concepts)
        
        # Check length appropriateness (not too short, not too verbose)
        length_ratio = len(actual.split()) / max(len(expected.split()), 1)
        length_score = 1.0 if 0.5 <= length_ratio <= 2.0 else max(0.3, 1.0 - abs(length_ratio - 1.0))
        
        return (concept_coverage * 0.8 + length_score * 0.2)
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text (simplified implementation)"""
        # Remove common stop words and extract meaningful phrases
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words = re.findall(r'\b\w+\b', text.lower())
        concepts = []
        
        # Extract 2-3 word phrases and important single words
        for i, word in enumerate(words):
            if word not in stop_words and len(word) > 3:
                concepts.append(word)
                
                # Add 2-word phrases
                if i < len(words) - 1:
                    next_word = words[i + 1]
                    if next_word not in stop_words:
                        concepts.append(f"{word} {next_word}")
        
        return list(set(concepts))
    
    def _evaluate_citation_quality(self, query: str, response: str, 
                                 retrieved_docs: List[Dict], citations: List[Dict]) -> Tuple[float, float]:
        """Evaluate citation precision and recall"""
        
        if not citations:
            return 0.0, 0.0
        
        # Precision: How many cited sources are actually relevant?
        relevant_citations = 0
        for citation in citations:
            if self._is_citation_relevant(query, response, citation):
                relevant_citations += 1
        
        precision = relevant_citations / len(citations) if citations else 0.0
        
        # Recall: How many relevant documents were cited?
        total_relevant_docs = len([doc for doc in retrieved_docs if self._is_document_relevant(query, doc)])
        recall = relevant_citations / max(total_relevant_docs, 1)
        
        return precision, recall
    
    def _is_citation_relevant(self, query: str, response: str, citation: Dict) -> bool:
        """Check if a citation is relevant to the query and response"""
        citation_content = citation.get('content_preview', '') or citation.get('full_content', '')
        
        # Simple relevance check based on keyword overlap
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        citation_words = set(citation_content.lower().split())
        
        query_overlap = len(query_words.intersection(citation_words)) / max(len(query_words), 1)
        response_overlap = len(response_words.intersection(citation_words)) / max(len(response_words), 1)
        
        return (query_overlap > 0.2) or (response_overlap > 0.1)
    
    def _is_document_relevant(self, query: str, document: Dict) -> bool:
        """Check if a retrieved document is relevant to the query"""
        doc_content = document.get('page_content', '') or str(document)
        
        query_words = set(query.lower().split())
        doc_words = set(doc_content.lower().split())
        
        overlap = len(query_words.intersection(doc_words)) / max(len(query_words), 1)
        return overlap > 0.2
    
    def _evaluate_knowledge_coverage(self, query: str, retrieved_docs: List[Dict]) -> float:
        """Evaluate how well the retrieved knowledge covers the query"""
        
        if not retrieved_docs:
            return 0.0
        
        query_concepts = set(self._extract_key_concepts(query))
        
        covered_concepts = set()
        for doc in retrieved_docs:
            doc_content = doc.get('page_content', '') or str(doc)
            doc_concepts = set(self._extract_key_concepts(doc_content))
            covered_concepts.update(query_concepts.intersection(doc_concepts))
        
        coverage = len(covered_concepts) / max(len(query_concepts), 1)
        return min(coverage, 1.0)
    
    def _evaluate_response_appropriateness(self, query: str, response: str) -> float:
        """Evaluate if the response is appropriate for the query type"""
        
        # Check for common inappropriate responses
        inappropriate_patterns = [
            r"i don't know",
            r"i cannot",
            r"i'm not sure",
            r"sorry, i can't"
        ]
        
        response_lower = response.lower()
        
        # If query seems answerable but response is evasive
        if any(re.search(pattern, response_lower) for pattern in inappropriate_patterns):
            # Check if query seems like it should be answerable
            if len(query.split()) > 3 and not any(word in query.lower() for word in ['meaning of life', 'philosophy', 'opinion']):
                return 0.3  # Likely should have been answerable
            else:
                return 0.8  # Appropriate to say "I don't know"
        
        # Check for response length appropriateness
        if len(response.split()) < 5:
            return 0.4  # Too short
        elif len(response.split()) > 500:
            return 0.6  # Too verbose
        else:
            return 1.0  # Appropriate length
    
    def _semantic_analysis_details(self, expected: str, actual: str) -> Dict[str, Any]:
        """Detailed semantic analysis"""
        return {
            'expected_length': len(expected.split()),
            'actual_length': len(actual.split()),
            'length_ratio': len(actual.split()) / max(len(expected.split()), 1),
            'shared_concepts': len(set(self._extract_key_concepts(expected)).intersection(
                set(self._extract_key_concepts(actual))))
        }
    
    def _factual_analysis_details(self, expected: str, actual: str) -> Dict[str, Any]:
        """Detailed factual analysis"""
        expected_facts = self._extract_facts(expected)
        actual_facts = self._extract_facts(actual)
        
        return {
            'expected_facts': expected_facts,
            'actual_facts': actual_facts,
            'fact_overlaps': {
                fact_type: len(set(expected_facts.get(fact_type, [])).intersection(
                    set(actual_facts.get(fact_type, []))))
                for fact_type in self.fact_patterns.keys()
            }
        }
    
    def _completeness_analysis_details(self, query: str, expected: str, actual: str) -> Dict[str, Any]:
        """Detailed completeness analysis"""
        return {
            'query_concepts': self._extract_key_concepts(query),
            'expected_concepts': self._extract_key_concepts(expected),
            'actual_concepts': self._extract_key_concepts(actual),
            'concept_coverage_ratio': len(set(self._extract_key_concepts(expected)).intersection(
                set(self._extract_key_concepts(actual)))) / max(len(self._extract_key_concepts(expected)), 1)
        }
    
    def _citation_analysis_details(self, retrieved_docs: List[Dict], citations: List[Dict]) -> Dict[str, Any]:
        """Detailed citation analysis"""
        return {
            'total_retrieved': len(retrieved_docs),
            'total_cited': len(citations) if citations else 0,
            'citation_rate': (len(citations) / max(len(retrieved_docs), 1)) if citations else 0.0,
            'avg_citation_confidence': np.mean([c.get('confidence_score', 0) for c in citations]) if citations else 0.0
        }

class BenchmarkComparator:
    """Compare RAG system performance against benchmarks"""
    
    def __init__(self):
        self.industry_benchmarks = {
            'semantic_similarity': {'excellent': 0.8, 'good': 0.6, 'fair': 0.4, 'poor': 0.2},
            'factual_consistency': {'excellent': 0.9, 'good': 0.7, 'fair': 0.5, 'poor': 0.3},
            'citation_precision': {'excellent': 0.8, 'good': 0.6, 'fair': 0.4, 'poor': 0.2},
            'overall_score': {'excellent': 0.8, 'good': 0.6, 'fair': 0.4, 'poor': 0.2}
        }
    
    def compare_to_benchmarks(self, evaluation_result: AdvancedEvaluationResult) -> Dict[str, str]:
        """Compare results to industry benchmarks"""
        comparisons = {}
        
        metrics = {
            'semantic_similarity': evaluation_result.semantic_similarity,
            'factual_consistency': evaluation_result.factual_consistency,
            'citation_precision': evaluation_result.citation_precision,
            'overall_score': evaluation_result.overall_score
        }
        
        for metric, value in metrics.items():
            benchmarks = self.industry_benchmarks[metric]
            
            if value >= benchmarks['excellent']:
                comparisons[metric] = 'excellent'
            elif value >= benchmarks['good']:
                comparisons[metric] = 'good'
            elif value >= benchmarks['fair']:
                comparisons[metric] = 'fair'
            else:
                comparisons[metric] = 'poor'
        
        return comparisons
    
    def generate_improvement_recommendations(self, comparison: Dict[str, str]) -> List[str]:
        """Generate specific improvement recommendations"""
        recommendations = []
        
        if comparison.get('semantic_similarity') in ['poor', 'fair']:
            recommendations.append("Improve semantic similarity by enhancing document chunking and embedding quality")
        
        if comparison.get('factual_consistency') in ['poor', 'fair']:
            recommendations.append("Improve factual consistency by adding fact-checking mechanisms and better source validation")
        
        if comparison.get('citation_precision') in ['poor', 'fair']:
            recommendations.append("Enhance citation quality by improving relevance scoring and source ranking")
        
        if comparison.get('overall_score') in ['poor', 'fair']:
            recommendations.append("Focus on overall system integration and balance between retrieval and generation")
        
        return recommendations

if __name__ == "__main__":
    # Example usage
    evaluator = AdvancedRAGEvaluator()
    
    # Example evaluation
    query = "What is the probation period for new employees?"
    expected = "The probation period is 6 months."
    actual = "New employees have a 6-month probation period during which their performance is evaluated."
    retrieved_docs = [{"page_content": "Employee probation policies specify 6 months..."}]
    citations = [{"confidence_score": 0.8, "content_preview": "probation period 6 months"}]
    
    result = evaluator.evaluate_comprehensive(query, expected, actual, retrieved_docs, citations)
    
    print("Advanced Evaluation Results:")
    print(f"Overall Score: {result.overall_score:.3f}")
    print(f"Semantic Similarity: {result.semantic_similarity:.3f}")
    print(f"Factual Consistency: {result.factual_consistency:.3f}")
    print(f"Answer Completeness: {result.answer_completeness:.3f}")
    print(f"Citation Precision: {result.citation_precision:.3f}")
    
    # Benchmark comparison
    comparator = BenchmarkComparator()
    comparison = comparator.compare_to_benchmarks(result)
    print(f"\nBenchmark Comparison: {comparison}")
    
    recommendations = comparator.generate_improvement_recommendations(comparison)
    print(f"\nRecommendations:")
    for rec in recommendations:
        print(f"- {rec}")
