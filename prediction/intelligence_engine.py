"""
Predictive Intelligence Engine
==============================

Advanced predictive system that anticipates user needs, predicts outcomes,
and provides proactive intelligence capabilities.
"""

import asyncio
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import time
from collections import defaultdict, deque
from enum import Enum

logger = logging.getLogger(__name__)

class PredictionType(Enum):
    """Types of predictions"""
    USER_INTENT = "user_intent"
    RESPONSE_QUALITY = "response_quality"
    PERFORMANCE = "performance"
    BEHAVIOR_PATTERN = "behavior_pattern"
    OUTCOME = "outcome"
    TREND = "trend"

@dataclass
class Prediction:
    """Single prediction result"""
    prediction_type: PredictionType
    predicted_value: Any
    confidence: float
    probability_distribution: Dict[str, float]
    reasoning: str
    supporting_evidence: List[str]
    prediction_horizon: float  # Time horizon in seconds

@dataclass
class PredictionResult:
    """Comprehensive prediction result"""
    primary_prediction: Prediction
    alternative_predictions: List[Prediction]
    accuracy: float
    processing_time: float
    model_confidence: float
    prediction_path: List[str]

class PatternRecognizer:
    """Advanced pattern recognition for predictive analysis"""
    
    def __init__(self):
        self.patterns = defaultdict(list)
        self.pattern_weights = defaultdict(float)
        self.sequence_memory = deque(maxlen=1000)
        
    def add_sequence(self, sequence: List[Any], outcome: Any, weight: float = 1.0):
        """Add a sequence-outcome pair for pattern learning"""
        sequence_key = self._sequence_to_key(sequence)
        self.patterns[sequence_key].append({
            "outcome": outcome,
            "weight": weight,
            "timestamp": time.time()
        })
        
        # Update pattern weight
        self.pattern_weights[sequence_key] += weight
        
        # Store in memory
        self.sequence_memory.append({
            "sequence": sequence,
            "outcome": outcome,
            "weight": weight
        })
    
    def predict_from_pattern(self, sequence: List[Any]) -> Tuple[Any, float]:
        """Predict outcome based on learned patterns"""
        sequence_key = self._sequence_to_key(sequence)
        
        if sequence_key not in self.patterns:
            # Try partial matches
            best_match = self._find_best_partial_match(sequence)
            if best_match:
                sequence_key = best_match
            else:
                return None, 0.0
        
        # Calculate weighted prediction
        outcomes = self.patterns[sequence_key]
        if not outcomes:
            return None, 0.0
        
        # Weight recent outcomes more heavily
        current_time = time.time()
        weighted_outcomes = defaultdict(float)
        total_weight = 0.0
        
        for outcome_data in outcomes:
            age = current_time - outcome_data["timestamp"]
            time_weight = np.exp(-age / 3600.0)  # Exponential decay with 1-hour half-life
            final_weight = outcome_data["weight"] * time_weight
            
            weighted_outcomes[str(outcome_data["outcome"])] += final_weight
            total_weight += final_weight
        
        if total_weight == 0:
            return None, 0.0
        
        # Find most likely outcome
        best_outcome = max(weighted_outcomes, key=weighted_outcomes.get)
        confidence = weighted_outcomes[best_outcome] / total_weight
        
        return best_outcome, confidence
    
    def _sequence_to_key(self, sequence: List[Any]) -> str:
        """Convert sequence to string key"""
        return "|".join(str(item) for item in sequence)
    
    def _find_best_partial_match(self, sequence: List[Any]) -> Optional[str]:
        """Find best partial pattern match"""
        sequence_str = self._sequence_to_key(sequence)
        best_match = None
        best_score = 0.0
        
        for pattern_key in self.patterns.keys():
            # Calculate similarity score
            similarity = self._calculate_similarity(sequence_str, pattern_key)
            if similarity > best_score and similarity > 0.5:
                best_score = similarity
                best_match = pattern_key
        
        return best_match
    
    def _calculate_similarity(self, seq1: str, seq2: str) -> float:
        """Calculate similarity between two sequences"""
        parts1 = seq1.split("|")
        parts2 = seq2.split("|")
        
        # Simple overlap-based similarity
        overlap = len(set(parts1) & set(parts2))
        total = len(set(parts1) | set(parts2))
        
        return overlap / total if total > 0 else 0.0

class UserBehaviorPredictor:
    """Predicts user behavior patterns and preferences"""
    
    def __init__(self):
        self.user_sessions = deque(maxlen=1000)
        self.behavior_patterns = defaultdict(list)
        self.preference_model = {}
        
    def add_user_interaction(self, 
                           query: str,
                           response: str,
                           feedback_score: float,
                           context: Dict[str, Any] = None):
        """Add user interaction for behavior learning"""
        interaction = {
            "query": query,
            "response": response,
            "feedback_score": feedback_score,
            "context": context or {},
            "timestamp": time.time(),
            "query_length": len(query.split()),
            "response_length": len(response.split()),
            "query_type": self._classify_query_type(query)
        }
        
        self.user_sessions.append(interaction)
        self._update_behavior_patterns(interaction)
    
    def predict_user_intent(self, partial_query: str) -> Prediction:
        """Predict user intent from partial query"""
        # Analyze query characteristics
        query_features = self._extract_query_features(partial_query)
        
        # Find similar past queries
        similar_queries = self._find_similar_queries(partial_query)
        
        # Predict intent based on patterns
        if similar_queries:
            intent_scores = defaultdict(float)
            for query_data in similar_queries:
                intent = query_data["query_type"]
                similarity = self._calculate_query_similarity(partial_query, query_data["query"])
                intent_scores[intent] += similarity * query_data["feedback_score"]
            
            # Normalize scores
            total_score = sum(intent_scores.values())
            if total_score > 0:
                intent_probs = {intent: score/total_score for intent, score in intent_scores.items()}
            else:
                intent_probs = {"general": 1.0}
            
            best_intent = max(intent_probs, key=intent_probs.get)
            confidence = intent_probs[best_intent]
        else:
            best_intent = "general"
            confidence = 0.5
            intent_probs = {"general": 1.0}
        
        return Prediction(
            prediction_type=PredictionType.USER_INTENT,
            predicted_value=best_intent,
            confidence=confidence,
            probability_distribution=intent_probs,
            reasoning=f"Based on analysis of {len(similar_queries)} similar queries",
            supporting_evidence=[q["query"][:50] + "..." for q in similar_queries[:3]],
            prediction_horizon=0.0  # Immediate prediction
        )
    
    def predict_response_quality(self, query: str, proposed_response: str) -> Prediction:
        """Predict the quality of a proposed response"""
        # Extract features
        query_features = self._extract_query_features(query)
        response_features = self._extract_response_features(proposed_response)
        
        # Find similar query-response pairs
        similar_pairs = self._find_similar_query_response_pairs(query, proposed_response)
        
        if similar_pairs:
            quality_scores = [pair["feedback_score"] for pair in similar_pairs]
            predicted_quality = np.mean(quality_scores)
            confidence = 1.0 - np.std(quality_scores)  # Lower std = higher confidence
        else:
            # Heuristic-based quality estimation
            predicted_quality = self._estimate_response_quality_heuristic(query, proposed_response)
            confidence = 0.6
        
        # Create probability distribution
        quality_bins = {
            "excellent": max(0, predicted_quality - 0.8),
            "good": max(0, predicted_quality - 0.6),
            "fair": max(0, predicted_quality - 0.4),
            "poor": max(0, 0.4 - predicted_quality)
        }
        
        # Normalize
        total = sum(quality_bins.values())
        if total > 0:
            quality_probs = {k: v/total for k, v in quality_bins.items()}
        else:
            quality_probs = {"fair": 1.0}
        
        return Prediction(
            prediction_type=PredictionType.RESPONSE_QUALITY,
            predicted_value=predicted_quality,
            confidence=confidence,
            probability_distribution=quality_probs,
            reasoning=f"Based on {len(similar_pairs)} similar interactions",
            supporting_evidence=[f"Query similarity: {len(similar_pairs)} matches"],
            prediction_horizon=0.0
        )
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what", "how", "why", "when", "where"]):
            return "question"
        elif any(word in query_lower for word in ["create", "make", "generate", "build"]):
            return "creation"
        elif any(word in query_lower for word in ["explain", "describe", "tell me"]):
            return "explanation"
        elif any(word in query_lower for word in ["help", "assist", "support"]):
            return "assistance"
        else:
            return "general"
    
    def _extract_query_features(self, query: str) -> Dict[str, float]:
        """Extract features from query"""
        return {
            "length": len(query.split()),
            "question_marks": query.count("?"),
            "exclamation_marks": query.count("!"),
            "complexity": len(set(query.lower().split())) / max(len(query.split()), 1),
            "formality": self._calculate_formality(query)
        }
    
    def _extract_response_features(self, response: str) -> Dict[str, float]:
        """Extract features from response"""
        return {
            "length": len(response.split()),
            "informativeness": len(set(response.lower().split())) / max(len(response.split()), 1),
            "structure_score": response.count(".") + response.count(","),
            "technical_density": sum(1 for word in response.split() if len(word) > 8) / max(len(response.split()), 1)
        }
    
    def _calculate_formality(self, text: str) -> float:
        """Calculate formality score"""
        formal_words = {"therefore", "however", "furthermore", "consequently", "moreover"}
        informal_words = {"yeah", "ok", "cool", "awesome", "hey"}
        
        words = text.lower().split()
        formal_count = sum(1 for word in words if word in formal_words)
        informal_count = sum(1 for word in words if word in informal_words)
        
        if formal_count + informal_count == 0:
            return 0.5
        
        return formal_count / (formal_count + informal_count)
    
    def _find_similar_queries(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar queries from history"""
        query_words = set(query.lower().split())
        similar = []
        
        for interaction in self.user_sessions:
            other_words = set(interaction["query"].lower().split())
            similarity = len(query_words & other_words) / len(query_words | other_words) if query_words | other_words else 0
            
            if similarity > 0.3:
                similar.append({
                    **interaction,
                    "similarity": similarity
                })
        
        similar.sort(key=lambda x: x["similarity"], reverse=True)
        return similar[:limit]
    
    def _find_similar_query_response_pairs(self, query: str, response: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar query-response pairs"""
        similar_queries = self._find_similar_queries(query, limit * 2)
        
        # Filter by response similarity
        response_words = set(response.lower().split())
        similar_pairs = []
        
        for query_data in similar_queries:
            other_response_words = set(query_data["response"].lower().split())
            response_similarity = len(response_words & other_response_words) / len(response_words | other_response_words) if response_words | other_response_words else 0
            
            if response_similarity > 0.2:
                similar_pairs.append({
                    **query_data,
                    "response_similarity": response_similarity
                })
        
        similar_pairs.sort(key=lambda x: x["response_similarity"], reverse=True)
        return similar_pairs[:limit]
    
    def _calculate_query_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries"""
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _estimate_response_quality_heuristic(self, query: str, response: str) -> float:
        """Estimate response quality using heuristics"""
        quality = 0.5  # Base quality
        
        # Length appropriateness
        query_len = len(query.split())
        response_len = len(response.split())
        
        if query_len < 5 and 20 <= response_len <= 100:
            quality += 0.1
        elif query_len >= 5 and 50 <= response_len <= 200:
            quality += 0.1
        
        # Relevance (word overlap)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words & response_words)
        
        if overlap > 0:
            quality += min(overlap * 0.05, 0.2)
        
        # Informativeness
        unique_ratio = len(set(response.lower().split())) / max(len(response.split()), 1)
        quality += unique_ratio * 0.2
        
        return min(max(quality, 0.0), 1.0)
    
    def _update_behavior_patterns(self, interaction: Dict[str, Any]):
        """Update behavior patterns based on interaction"""
        # Track patterns by query type
        query_type = interaction["query_type"]
        self.behavior_patterns[query_type].append({
            "feedback_score": interaction["feedback_score"],
            "query_length": interaction["query_length"],
            "response_length": interaction["response_length"],
            "timestamp": interaction["timestamp"]
        })

class PerformancePredictor:
    """Predicts system performance and resource needs"""
    
    def __init__(self):
        self.performance_history = deque(maxlen=1000)
        self.resource_usage_patterns = defaultdict(list)
        
    def add_performance_data(self, 
                           processing_time: float,
                           memory_usage: float,
                           cpu_usage: float,
                           query_complexity: float):
        """Add performance data point"""
        data_point = {
            "processing_time": processing_time,
            "memory_usage": memory_usage,
            "cpu_usage": cpu_usage,
            "query_complexity": query_complexity,
            "timestamp": time.time()
        }
        
        self.performance_history.append(data_point)
    
    def predict_processing_time(self, query_complexity: float) -> Prediction:
        """Predict processing time based on query complexity"""
        if not self.performance_history:
            return Prediction(
                prediction_type=PredictionType.PERFORMANCE,
                predicted_value=1.0,
                confidence=0.3,
                probability_distribution={"fast": 0.3, "medium": 0.4, "slow": 0.3},
                reasoning="No historical data available",
                supporting_evidence=[],
                prediction_horizon=0.0
            )
        
        # Find similar complexity queries
        similar_data = [
            data for data in self.performance_history
            if abs(data["query_complexity"] - query_complexity) < 0.2
        ]
        
        if similar_data:
            processing_times = [data["processing_time"] for data in similar_data]
            predicted_time = np.mean(processing_times)
            confidence = 1.0 - (np.std(processing_times) / max(predicted_time, 0.1))
        else:
            # Linear regression on all data
            complexities = [data["query_complexity"] for data in self.performance_history]
            times = [data["processing_time"] for data in self.performance_history]
            
            if len(complexities) > 1:
                slope = np.corrcoef(complexities, times)[0, 1] * np.std(times) / np.std(complexities)
                intercept = np.mean(times) - slope * np.mean(complexities)
                predicted_time = slope * query_complexity + intercept
                confidence = 0.6
            else:
                predicted_time = times[0] if times else 1.0
                confidence = 0.3
        
        # Create probability distribution
        time_ranges = {
            "fast": max(0, 1.0 - predicted_time),
            "medium": 1.0 - abs(predicted_time - 1.0),
            "slow": max(0, predicted_time - 1.0)
        }
        
        total = sum(time_ranges.values())
        if total > 0:
            time_probs = {k: v/total for k, v in time_ranges.items()}
        else:
            time_probs = {"medium": 1.0}
        
        return Prediction(
            prediction_type=PredictionType.PERFORMANCE,
            predicted_value=predicted_time,
            confidence=confidence,
            probability_distribution=time_probs,
            reasoning=f"Based on {len(similar_data)} similar complexity queries",
            supporting_evidence=[f"Average time: {predicted_time:.3f}s"],
            prediction_horizon=0.0
        )

class TrendAnalyzer:
    """Analyzes trends and predicts future patterns"""
    
    def __init__(self):
        self.time_series_data = defaultdict(deque)
        self.trend_models = {}
        
    def add_time_series_point(self, metric_name: str, value: float, timestamp: float = None):
        """Add a time series data point"""
        if timestamp is None:
            timestamp = time.time()
        
        self.time_series_data[metric_name].append({
            "value": value,
            "timestamp": timestamp
        })
        
        # Keep only recent data (last 1000 points)
        if len(self.time_series_data[metric_name]) > 1000:
            self.time_series_data[metric_name].popleft()
    
    def predict_trend(self, metric_name: str, horizon: float = 3600.0) -> Prediction:
        """Predict future trend for a metric"""
        if metric_name not in self.time_series_data or len(self.time_series_data[metric_name]) < 3:
            return Prediction(
                prediction_type=PredictionType.TREND,
                predicted_value="stable",
                confidence=0.3,
                probability_distribution={"increasing": 0.33, "stable": 0.34, "decreasing": 0.33},
                reasoning="Insufficient data for trend analysis",
                supporting_evidence=[],
                prediction_horizon=horizon
            )
        
        # Extract time series
        data_points = list(self.time_series_data[metric_name])
        values = [point["value"] for point in data_points]
        timestamps = [point["timestamp"] for point in data_points]
        
        # Simple linear trend analysis
        if len(values) > 1:
            # Calculate slope
            time_diffs = np.array(timestamps) - timestamps[0]
            slope = np.corrcoef(time_diffs, values)[0, 1] * np.std(values) / np.std(time_diffs) if np.std(time_diffs) > 0 else 0
            
            # Determine trend
            if slope > 0.01:
                trend = "increasing"
                confidence = min(abs(slope) * 10, 0.9)
            elif slope < -0.01:
                trend = "decreasing"
                confidence = min(abs(slope) * 10, 0.9)
            else:
                trend = "stable"
                confidence = 0.7
            
            # Create probability distribution
            trend_probs = {
                "increasing": max(0, slope * 10 + 0.33),
                "stable": max(0, 0.34 - abs(slope) * 5),
                "decreasing": max(0, -slope * 10 + 0.33)
            }
            
            # Normalize
            total = sum(trend_probs.values())
            if total > 0:
                trend_probs = {k: v/total for k, v in trend_probs.items()}
            else:
                trend_probs = {"stable": 1.0}
        else:
            trend = "stable"
            confidence = 0.5
            trend_probs = {"stable": 1.0}
        
        return Prediction(
            prediction_type=PredictionType.TREND,
            predicted_value=trend,
            confidence=confidence,
            probability_distribution=trend_probs,
            reasoning=f"Linear trend analysis on {len(values)} data points",
            supporting_evidence=[f"Slope: {slope:.4f}" if 'slope' in locals() else "No slope calculated"],
            prediction_horizon=horizon
        )

class PredictiveIntelligence:
    """
    Main predictive intelligence engine for CopilotX
    
    Combines multiple prediction systems to provide comprehensive
    predictive capabilities and proactive intelligence.
    """
    
    def __init__(self):
        self.pattern_recognizer = PatternRecognizer()
        self.behavior_predictor = UserBehaviorPredictor()
        self.performance_predictor = PerformancePredictor()
        self.trend_analyzer = TrendAnalyzer()
        
        self.is_initialized = False
        self.prediction_history = deque(maxlen=1000)
        self.prediction_stats = {
            "total_predictions": 0,
            "prediction_accuracy": 0.0,
            "prediction_types": defaultdict(int),
            "average_confidence": 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize predictive intelligence engine"""
        try:
            logger.info("Initializing Predictive Intelligence Engine...")
            
            # Initialize all prediction components
            await self._initialize_predictors()
            
            self.is_initialized = True
            logger.info("Predictive Intelligence Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize predictive intelligence: {e}")
            return False
    
    async def _initialize_predictors(self):
        """Initialize all prediction components"""
        # All predictors initialize in their constructors
        pass
    
    async def predict_response(self, 
                             query: str,
                             reasoning_result: Dict[str, Any],
                             context: Dict[str, Any] = None) -> PredictionResult:
        """
        Predict optimal response characteristics
        
        Args:
            query: User query
            reasoning_result: Result from reasoning engine
            context: Additional context
            
        Returns:
            PredictionResult with response predictions
        """
        if not self.is_initialized:
            raise RuntimeError("Predictive intelligence not initialized")
        
        start_time = time.time()
        
        try:
            # Generate multiple predictions
            predictions = []
            prediction_path = []
            
            # Predict user intent
            intent_prediction = self.behavior_predictor.predict_user_intent(query)
            predictions.append(intent_prediction)
            prediction_path.append("user_intent")
            
            # Predict response quality for potential responses
            potential_response = self._generate_potential_response(query, reasoning_result)
            quality_prediction = self.behavior_predictor.predict_response_quality(query, potential_response)
            predictions.append(quality_prediction)
            prediction_path.append("response_quality")
            
            # Predict performance
            query_complexity = self._calculate_query_complexity(query, reasoning_result)
            performance_prediction = self.performance_predictor.predict_processing_time(query_complexity)
            predictions.append(performance_prediction)
            prediction_path.append("performance")
            
            # Select primary prediction (highest confidence)
            primary_prediction = max(predictions, key=lambda p: p.confidence)
            alternative_predictions = [p for p in predictions if p != primary_prediction]
            
            # Calculate overall accuracy and confidence
            accuracy = self._calculate_prediction_accuracy(predictions)
            model_confidence = np.mean([p.confidence for p in predictions])
            
            processing_time = time.time() - start_time
            
            # Create result
            result = PredictionResult(
                primary_prediction=primary_prediction,
                alternative_predictions=alternative_predictions,
                accuracy=accuracy,
                processing_time=processing_time,
                model_confidence=model_confidence,
                prediction_path=prediction_path
            )
            
            # Update statistics
            self._update_prediction_stats(result)
            
            # Store in history
            self.prediction_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def _generate_potential_response(self, query: str, reasoning_result: Dict[str, Any]) -> str:
        """Generate a potential response for quality prediction"""
        # Simple response generation for prediction purposes
        base_responses = [
            f"Based on your query about '{query[:30]}...', I can provide the following analysis:",
            f"To address your question regarding '{query[:30]}...', here are my insights:",
            f"I understand you're asking about '{query[:30]}...'. Let me explain:",
        ]
        
        base_response = base_responses[hash(query) % len(base_responses)]
        
        # Add reasoning information if available
        if reasoning_result and "conclusion" in reasoning_result:
            base_response += f" {reasoning_result['conclusion']}"
        
        return base_response
    
    def _calculate_query_complexity(self, query: str, reasoning_result: Dict[str, Any]) -> float:
        """Calculate query complexity score"""
        complexity = 0.0
        
        # Length-based complexity
        word_count = len(query.split())
        complexity += min(word_count / 50.0, 1.0) * 0.3
        
        # Vocabulary complexity
        unique_words = len(set(query.lower().split()))
        vocab_complexity = unique_words / max(word_count, 1)
        complexity += vocab_complexity * 0.3
        
        # Question complexity
        question_words = ["what", "how", "why", "when", "where", "which", "who"]
        question_count = sum(1 for word in question_words if word in query.lower())
        complexity += min(question_count / 3.0, 1.0) * 0.2
        
        # Reasoning complexity
        if reasoning_result:
            reasoning_depth = reasoning_result.get("reasoning_depth", 1)
            complexity += min(reasoning_depth / 10.0, 1.0) * 0.2
        
        return min(complexity, 1.0)
    
    def _calculate_prediction_accuracy(self, predictions: List[Prediction]) -> float:
        """Calculate overall prediction accuracy"""
        # Simple accuracy estimate based on confidence levels
        confidence_scores = [p.confidence for p in predictions]
        
        if not confidence_scores:
            return 0.5
        
        # Weighted average with bias toward higher confidence predictions
        weights = np.array(confidence_scores)
        weighted_accuracy = np.average(confidence_scores, weights=weights)
        
        return weighted_accuracy
    
    def _update_prediction_stats(self, result: PredictionResult):
        """Update prediction statistics"""
        self.prediction_stats["total_predictions"] += 1
        
        # Update average accuracy
        total = self.prediction_stats["total_predictions"]
        current_avg = self.prediction_stats["prediction_accuracy"]
        new_avg = ((current_avg * (total - 1)) + result.accuracy) / total
        self.prediction_stats["prediction_accuracy"] = new_avg
        
        # Update prediction type counts
        self.prediction_stats["prediction_types"][result.primary_prediction.prediction_type.value] += 1
        
        # Update average confidence
        current_conf = self.prediction_stats["average_confidence"]
        new_conf = ((current_conf * (total - 1)) + result.model_confidence) / total
        self.prediction_stats["average_confidence"] = new_conf
    
    async def add_interaction_feedback(self, 
                                     query: str,
                                     response: str,
                                     feedback_score: float,
                                     processing_time: float,
                                     context: Dict[str, Any] = None):
        """Add interaction feedback for learning"""
        # Update behavior predictor
        self.behavior_predictor.add_user_interaction(query, response, feedback_score, context)
        
        # Update performance predictor
        query_complexity = self._calculate_query_complexity(query, context or {})
        memory_usage = context.get("memory_usage", 0.5) if context else 0.5
        cpu_usage = context.get("cpu_usage", 0.5) if context else 0.5
        
        self.performance_predictor.add_performance_data(
            processing_time, memory_usage, cpu_usage, query_complexity
        )
        
        # Update trend analyzer
        self.trend_analyzer.add_time_series_point("feedback_score", feedback_score)
        self.trend_analyzer.add_time_series_point("processing_time", processing_time)
        self.trend_analyzer.add_time_series_point("query_complexity", query_complexity)
    
    async def get_trend_predictions(self, horizon: float = 3600.0) -> Dict[str, Prediction]:
        """Get trend predictions for key metrics"""
        metrics = ["feedback_score", "processing_time", "query_complexity"]
        trend_predictions = {}
        
        for metric in metrics:
            prediction = self.trend_analyzer.predict_trend(metric, horizon)
            trend_predictions[metric] = prediction
        
        return trend_predictions
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get comprehensive prediction statistics"""
        return {
            "is_initialized": self.is_initialized,
            "prediction_stats": dict(self.prediction_stats),
            "prediction_history_size": len(self.prediction_history),
            "pattern_count": len(self.pattern_recognizer.patterns),
            "behavior_sessions": len(self.behavior_predictor.user_sessions),
            "performance_history": len(self.performance_predictor.performance_history),
            "time_series_metrics": list(self.trend_analyzer.time_series_data.keys())
        }