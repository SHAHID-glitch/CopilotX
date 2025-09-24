"""
Adaptive Learning System
========================

Self-improving learning system that continuously adapts and evolves
based on interactions, feedback, and performance metrics.
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import time
from collections import deque, defaultdict
import pickle
import json

logger = logging.getLogger(__name__)

@dataclass
class LearningExperience:
    """Single learning experience"""
    query: str
    response: str
    feedback_score: float
    context: Dict[str, Any]
    timestamp: float
    processing_time: float
    success_indicators: Dict[str, float]

@dataclass
class LearningUpdate:
    """Update from learning process"""
    parameter_updates: Dict[str, torch.Tensor]
    confidence_change: float
    performance_improvement: float
    learning_rate_adjustment: float
    adaptation_strategy: str

class ExperienceBuffer:
    """Buffer for storing and managing learning experiences"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.experiences = deque(maxlen=max_size)
        self.positive_experiences = deque(maxlen=max_size // 2)
        self.negative_experiences = deque(maxlen=max_size // 2)
        
    def add_experience(self, experience: LearningExperience):
        """Add a new learning experience"""
        self.experiences.append(experience)
        
        # Categorize by feedback
        if experience.feedback_score > 0.7:
            self.positive_experiences.append(experience)
        elif experience.feedback_score < 0.4:
            self.negative_experiences.append(experience)
    
    def sample_experiences(self, n: int, strategy: str = "balanced") -> List[LearningExperience]:
        """Sample experiences for learning"""
        if strategy == "balanced":
            pos_count = min(n // 2, len(self.positive_experiences))
            neg_count = min(n // 2, len(self.negative_experiences))
            
            sampled = []
            if pos_count > 0:
                pos_indices = np.random.choice(len(self.positive_experiences), pos_count, replace=False)
                sampled.extend([self.positive_experiences[i] for i in pos_indices])
            
            if neg_count > 0:
                neg_indices = np.random.choice(len(self.negative_experiences), neg_count, replace=False)
                sampled.extend([self.negative_experiences[i] for i in neg_indices])
            
            return sampled
        
        elif strategy == "recent":
            return list(self.experiences)[-n:] if len(self.experiences) >= n else list(self.experiences)
        
        elif strategy == "random":
            if len(self.experiences) == 0:
                return []
            indices = np.random.choice(len(self.experiences), min(n, len(self.experiences)), replace=False)
            return [self.experiences[i] for i in indices]
        
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        if not self.experiences:
            return {"total": 0, "positive": 0, "negative": 0, "average_score": 0.0}
        
        scores = [exp.feedback_score for exp in self.experiences]
        return {
            "total": len(self.experiences),
            "positive": len(self.positive_experiences),
            "negative": len(self.negative_experiences),
            "average_score": np.mean(scores),
            "score_std": np.std(scores)
        }

class MetaLearner(nn.Module):
    """Meta-learning network for learning how to learn"""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Learning rate prediction network
        self.lr_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Adaptation strategy network
        self.strategy_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # 4 adaptation strategies
            nn.Softmax(dim=-1)
        )
        
        # Performance predictor
        self.performance_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, experience_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict learning parameters from experience features"""
        learning_rate = self.lr_predictor(experience_features) * 0.1  # Scale to reasonable range
        strategy_probs = self.strategy_predictor(experience_features)
        predicted_performance = self.performance_predictor(experience_features)
        
        return {
            "learning_rate": learning_rate,
            "strategy_probabilities": strategy_probs,
            "predicted_performance": predicted_performance
        }

class AdaptationEngine:
    """Engine for adapting system parameters based on learning"""
    
    def __init__(self):
        self.adaptation_strategies = {
            "conservative": self._conservative_adaptation,
            "aggressive": self._aggressive_adaptation,
            "balanced": self._balanced_adaptation,
            "experimental": self._experimental_adaptation
        }
        
        self.adaptation_history = []
    
    async def adapt_parameters(self, 
                             experiences: List[LearningExperience],
                             current_params: Dict[str, Any],
                             strategy: str = "balanced") -> Dict[str, Any]:
        """Adapt parameters based on experiences"""
        if strategy not in self.adaptation_strategies:
            strategy = "balanced"
        
        adaptation_func = self.adaptation_strategies[strategy]
        adapted_params = await adaptation_func(experiences, current_params)
        
        # Record adaptation
        self.adaptation_history.append({
            "timestamp": time.time(),
            "strategy": strategy,
            "experience_count": len(experiences),
            "parameter_changes": self._calculate_parameter_changes(current_params, adapted_params)
        })
        
        return adapted_params
    
    async def _conservative_adaptation(self, 
                                     experiences: List[LearningExperience],
                                     current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Conservative adaptation with small changes"""
        adapted_params = current_params.copy()
        
        # Calculate average feedback
        avg_feedback = np.mean([exp.feedback_score for exp in experiences])
        
        # Small adjustments based on feedback
        adjustment_factor = 0.01 * (avg_feedback - 0.5)  # Small changes
        
        for key, value in adapted_params.items():
            if isinstance(value, (int, float)):
                adapted_params[key] = value * (1 + adjustment_factor)
        
        return adapted_params
    
    async def _aggressive_adaptation(self, 
                                   experiences: List[LearningExperience],
                                   current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Aggressive adaptation with larger changes"""
        adapted_params = current_params.copy()
        
        # Calculate feedback variance for risk assessment
        feedback_scores = [exp.feedback_score for exp in experiences]
        avg_feedback = np.mean(feedback_scores)
        feedback_variance = np.var(feedback_scores)
        
        # Larger adjustments
        adjustment_factor = 0.1 * (avg_feedback - 0.5) * (1 + feedback_variance)
        
        for key, value in adapted_params.items():
            if isinstance(value, (int, float)):
                adapted_params[key] = value * (1 + adjustment_factor)
        
        return adapted_params
    
    async def _balanced_adaptation(self, 
                                 experiences: List[LearningExperience],
                                 current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Balanced adaptation approach"""
        adapted_params = current_params.copy()
        
        # Analyze experience patterns
        positive_count = sum(1 for exp in experiences if exp.feedback_score > 0.7)
        negative_count = sum(1 for exp in experiences if exp.feedback_score < 0.4)
        
        # Adjust based on success/failure ratio
        if positive_count > negative_count:
            adjustment_factor = 0.05  # Moderate positive adjustment
        elif negative_count > positive_count:
            adjustment_factor = -0.03  # Moderate negative adjustment
        else:
            adjustment_factor = 0.01  # Small positive bias
        
        for key, value in adapted_params.items():
            if isinstance(value, (int, float)):
                adapted_params[key] = value * (1 + adjustment_factor)
        
        return adapted_params
    
    async def _experimental_adaptation(self, 
                                     experiences: List[LearningExperience],
                                     current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Experimental adaptation with exploration"""
        adapted_params = current_params.copy()
        
        # Add controlled randomness for exploration
        for key, value in adapted_params.items():
            if isinstance(value, (int, float)):
                # Add random exploration factor
                exploration_factor = np.random.normal(0, 0.05)
                adapted_params[key] = value * (1 + exploration_factor)
        
        return adapted_params
    
    def _calculate_parameter_changes(self, 
                                   old_params: Dict[str, Any],
                                   new_params: Dict[str, Any]) -> Dict[str, float]:
        """Calculate the magnitude of parameter changes"""
        changes = {}
        
        for key in old_params:
            if key in new_params and isinstance(old_params[key], (int, float)):
                old_val = old_params[key]
                new_val = new_params[key]
                
                if old_val != 0:
                    change_percent = (new_val - old_val) / old_val * 100
                else:
                    change_percent = 0.0
                
                changes[key] = change_percent
        
        return changes

class KnowledgeGraph:
    """Dynamic knowledge graph for storing learned patterns"""
    
    def __init__(self):
        self.nodes = {}  # Concept nodes
        self.edges = {}  # Relationships between concepts
        self.node_weights = defaultdict(float)
        self.edge_weights = defaultdict(float)
    
    def add_concept(self, concept: str, properties: Dict[str, Any] = None):
        """Add a concept to the knowledge graph"""
        if concept not in self.nodes:
            self.nodes[concept] = properties or {}
            self.node_weights[concept] = 1.0
        else:
            # Update properties
            if properties:
                self.nodes[concept].update(properties)
            # Increase weight for reinforcement
            self.node_weights[concept] += 0.1
    
    def add_relationship(self, concept1: str, concept2: str, relationship: str, strength: float = 1.0):
        """Add a relationship between concepts"""
        edge_key = (concept1, concept2, relationship)
        
        if edge_key not in self.edges:
            self.edges[edge_key] = {"strength": strength, "count": 1}
        else:
            # Reinforce existing relationship
            self.edges[edge_key]["count"] += 1
            self.edges[edge_key]["strength"] += strength * 0.1
        
        self.edge_weights[edge_key] = self.edges[edge_key]["strength"]
    
    def query_related_concepts(self, concept: str, max_results: int = 5) -> List[Tuple[str, str, float]]:
        """Query concepts related to a given concept"""
        related = []
        
        for (c1, c2, rel), edge_data in self.edges.items():
            if c1 == concept:
                related.append((c2, rel, edge_data["strength"]))
            elif c2 == concept:
                related.append((c1, rel, edge_data["strength"]))
        
        # Sort by strength and return top results
        related.sort(key=lambda x: x[2], reverse=True)
        return related[:max_results]
    
    def get_concept_importance(self, concept: str) -> float:
        """Get the importance score of a concept"""
        return self.node_weights.get(concept, 0.0)

class AdaptiveLearningSystem:
    """
    Adaptive Learning System for CopilotX
    
    Continuously learns and adapts from interactions, improving
    performance and capabilities over time.
    """
    
    def __init__(self):
        self.experience_buffer = ExperienceBuffer()
        self.meta_learner = MetaLearner()
        self.adaptation_engine = AdaptationEngine()
        self.knowledge_graph = KnowledgeGraph()
        
        self.is_initialized = False
        self.learning_stats = {
            "total_experiences": 0,
            "successful_adaptations": 0,
            "learning_rate": 0.01,
            "adaptation_frequency": 0,
            "knowledge_growth_rate": 0.0
        }
        
        # Learning parameters
        self.learning_config = {
            "min_experiences_for_learning": 10,
            "learning_batch_size": 32,
            "adaptation_threshold": 0.1,
            "meta_learning_enabled": True,
            "knowledge_graph_enabled": True
        }
    
    async def initialize(self) -> bool:
        """Initialize adaptive learning system"""
        try:
            logger.info("Initializing Adaptive Learning System...")
            
            # Initialize meta-learner
            await self._initialize_meta_learner()
            
            # Load existing knowledge if available
            await self._load_knowledge()
            
            self.is_initialized = True
            logger.info("Adaptive Learning System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize learning system: {e}")
            return False
    
    async def _initialize_meta_learner(self):
        """Initialize meta-learning components"""
        # Set meta-learner to evaluation mode initially
        self.meta_learner.eval()
    
    async def _load_knowledge(self):
        """Load existing knowledge from storage"""
        # In a real implementation, this would load from persistent storage
        logger.info("Loading existing knowledge base...")
        
        # Initialize with some basic concepts
        self.knowledge_graph.add_concept("ai", {"type": "technology", "complexity": 0.9})
        self.knowledge_graph.add_concept("learning", {"type": "process", "complexity": 0.8})
        self.knowledge_graph.add_concept("intelligence", {"type": "concept", "complexity": 0.95})
        
        # Add relationships
        self.knowledge_graph.add_relationship("ai", "learning", "enables", 0.9)
        self.knowledge_graph.add_relationship("learning", "intelligence", "develops", 0.8)
    
    async def learn_from_interaction(self, 
                                   query: str,
                                   response: str,
                                   feedback_score: float = None,
                                   context: Dict[str, Any] = None) -> LearningUpdate:
        """
        Learn from a single interaction
        
        Args:
            query: User query
            response: System response
            feedback_score: Optional feedback score (0-1)
            context: Additional context information
            
        Returns:
            LearningUpdate with adaptation information
        """
        if not self.is_initialized:
            raise RuntimeError("Learning system not initialized")
        
        try:
            # Create learning experience
            experience = self._create_experience(query, response, feedback_score, context)
            
            # Add to experience buffer
            self.experience_buffer.add_experience(experience)
            
            # Update knowledge graph
            await self._update_knowledge_graph(experience)
            
            # Perform learning if enough experiences
            learning_update = await self._perform_learning()
            
            # Update statistics
            self._update_learning_stats(experience, learning_update)
            
            return learning_update
            
        except Exception as e:
            logger.error(f"Learning from interaction failed: {e}")
            raise
    
    def _create_experience(self, 
                          query: str,
                          response: str,
                          feedback_score: float = None,
                          context: Dict[str, Any] = None) -> LearningExperience:
        """Create a learning experience from interaction"""
        
        # Estimate feedback score if not provided
        if feedback_score is None:
            feedback_score = self._estimate_feedback_score(query, response, context)
        
        # Calculate success indicators
        success_indicators = self._calculate_success_indicators(query, response, context)
        
        return LearningExperience(
            query=query,
            response=response,
            feedback_score=feedback_score,
            context=context or {},
            timestamp=time.time(),
            processing_time=context.get('processing_time', 0.0) if context else 0.0,
            success_indicators=success_indicators
        )
    
    def _estimate_feedback_score(self, 
                                query: str,
                                response: str,
                                context: Dict[str, Any] = None) -> float:
        """Estimate feedback score based on response quality"""
        score = 0.5  # Base score
        
        # Response length heuristic
        if 50 <= len(response) <= 500:
            score += 0.1
        
        # Relevance heuristic (simple word overlap)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words))
        
        if overlap > 0:
            score += min(overlap * 0.05, 0.2)
        
        # Context-based adjustments
        if context:
            if context.get('confidence', 0.5) > 0.7:
                score += 0.1
            if context.get('processing_time', 1.0) < 0.5:
                score += 0.1
        
        return min(max(score, 0.0), 1.0)
    
    def _calculate_success_indicators(self, 
                                    query: str,
                                    response: str,
                                    context: Dict[str, Any] = None) -> Dict[str, float]:
        """Calculate various success indicators"""
        indicators = {
            "response_completeness": min(len(response) / 100.0, 1.0),
            "query_relevance": self._calculate_relevance(query, response),
            "coherence": self._calculate_coherence(response),
            "informativeness": self._calculate_informativeness(response)
        }
        
        if context:
            indicators["confidence"] = context.get("confidence", 0.5)
            indicators["processing_efficiency"] = 1.0 / (1.0 + context.get("processing_time", 1.0))
        
        return indicators
    
    def _calculate_relevance(self, query: str, response: str) -> float:
        """Calculate relevance score between query and response"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(response_words))
        relevance = overlap / len(query_words)
        
        return min(relevance, 1.0)
    
    def _calculate_coherence(self, text: str) -> float:
        """Calculate text coherence score"""
        sentences = text.split('.')
        if len(sentences) < 2:
            return 1.0
        
        # Simple coherence based on sentence length consistency
        lengths = [len(sent.split()) for sent in sentences if sent.strip()]
        if not lengths:
            return 0.5
        
        length_variance = np.var(lengths)
        coherence = 1.0 / (1.0 + length_variance / 10.0)
        
        return coherence
    
    def _calculate_informativeness(self, text: str) -> float:
        """Calculate informativeness score"""
        words = text.split()
        if not words:
            return 0.0
        
        unique_words = len(set(words))
        informativeness = unique_words / len(words)
        
        return informativeness
    
    async def _update_knowledge_graph(self, experience: LearningExperience):
        """Update knowledge graph based on experience"""
        if not self.learning_config["knowledge_graph_enabled"]:
            return
        
        # Extract concepts from query and response
        query_concepts = self._extract_concepts(experience.query)
        response_concepts = self._extract_concepts(experience.response)
        
        # Add concepts to knowledge graph
        for concept in query_concepts + response_concepts:
            self.knowledge_graph.add_concept(concept)
        
        # Add relationships between query and response concepts
        for q_concept in query_concepts:
            for r_concept in response_concepts:
                strength = experience.feedback_score
                self.knowledge_graph.add_relationship(
                    q_concept, r_concept, "addresses", strength
                )
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple concept extraction (can be enhanced with NLP)
        words = text.lower().split()
        
        # Filter for important words (length > 4, not common words)
        common_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'know', 'want', 'been'}
        concepts = [word for word in words if len(word) > 4 and word not in common_words]
        
        return concepts[:5]  # Limit to top 5 concepts
    
    async def _perform_learning(self) -> LearningUpdate:
        """Perform learning update if conditions are met"""
        buffer_stats = self.experience_buffer.get_stats()
        
        if buffer_stats["total"] < self.learning_config["min_experiences_for_learning"]:
            return LearningUpdate(
                parameter_updates={},
                confidence_change=0.0,
                performance_improvement=0.0,
                learning_rate_adjustment=0.0,
                adaptation_strategy="none"
            )
        
        # Sample experiences for learning
        experiences = self.experience_buffer.sample_experiences(
            self.learning_config["learning_batch_size"],
            strategy="balanced"
        )
        
        # Determine adaptation strategy
        adaptation_strategy = await self._select_adaptation_strategy(experiences)
        
        # Perform adaptation
        current_params = {"learning_rate": self.learning_stats["learning_rate"]}
        adapted_params = await self.adaptation_engine.adapt_parameters(
            experiences, current_params, adaptation_strategy
        )
        
        # Calculate updates
        parameter_updates = self._calculate_parameter_updates(current_params, adapted_params)
        performance_improvement = self._estimate_performance_improvement(experiences)
        confidence_change = self._calculate_confidence_change(experiences)
        
        # Apply updates
        self.learning_stats["learning_rate"] = adapted_params.get("learning_rate", 
                                                                self.learning_stats["learning_rate"])
        
        return LearningUpdate(
            parameter_updates=parameter_updates,
            confidence_change=confidence_change,
            performance_improvement=performance_improvement,
            learning_rate_adjustment=adapted_params.get("learning_rate", 0.01) - current_params.get("learning_rate", 0.01),
            adaptation_strategy=adaptation_strategy
        )
    
    async def _select_adaptation_strategy(self, experiences: List[LearningExperience]) -> str:
        """Select the best adaptation strategy"""
        if not self.learning_config["meta_learning_enabled"]:
            return "balanced"
        
        # Calculate experience features
        avg_feedback = np.mean([exp.feedback_score for exp in experiences])
        feedback_variance = np.var([exp.feedback_score for exp in experiences])
        
        # Simple strategy selection based on performance
        if avg_feedback > 0.8:
            return "conservative"  # Don't change much if doing well
        elif avg_feedback < 0.4:
            return "aggressive"    # Make bigger changes if doing poorly
        elif feedback_variance > 0.1:
            return "experimental"  # Explore if results are inconsistent
        else:
            return "balanced"      # Default balanced approach
    
    def _calculate_parameter_updates(self, 
                                   old_params: Dict[str, Any],
                                   new_params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Calculate parameter updates as tensors"""
        updates = {}
        
        for key in old_params:
            if key in new_params:
                old_val = old_params[key]
                new_val = new_params[key]
                
                if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                    update = torch.tensor(new_val - old_val, dtype=torch.float32)
                    updates[key] = update
        
        return updates
    
    def _estimate_performance_improvement(self, experiences: List[LearningExperience]) -> float:
        """Estimate performance improvement from experiences"""
        if len(experiences) < 2:
            return 0.0
        
        # Sort by timestamp
        sorted_experiences = sorted(experiences, key=lambda x: x.timestamp)
        
        # Compare early vs late performance
        early_scores = [exp.feedback_score for exp in sorted_experiences[:len(sorted_experiences)//2]]
        late_scores = [exp.feedback_score for exp in sorted_experiences[len(sorted_experiences)//2:]]
        
        early_avg = np.mean(early_scores) if early_scores else 0.5
        late_avg = np.mean(late_scores) if late_scores else 0.5
        
        return late_avg - early_avg
    
    def _calculate_confidence_change(self, experiences: List[LearningExperience]) -> float:
        """Calculate change in confidence based on experiences"""
        confidence_scores = []
        
        for exp in experiences:
            if 'confidence' in exp.success_indicators:
                confidence_scores.append(exp.success_indicators['confidence'])
        
        if not confidence_scores:
            return 0.0
        
        # Calculate trend in confidence
        if len(confidence_scores) > 1:
            return confidence_scores[-1] - confidence_scores[0]
        
        return confidence_scores[0] - 0.5  # Compare to neutral
    
    def _update_learning_stats(self, experience: LearningExperience, update: LearningUpdate):
        """Update learning statistics"""
        self.learning_stats["total_experiences"] += 1
        
        if update.performance_improvement > 0:
            self.learning_stats["successful_adaptations"] += 1
        
        # Update adaptation frequency
        if update.adaptation_strategy != "none":
            self.learning_stats["adaptation_frequency"] += 1
        
        # Update knowledge growth rate
        current_knowledge_size = len(self.knowledge_graph.nodes)
        self.learning_stats["knowledge_growth_rate"] = current_knowledge_size / max(self.learning_stats["total_experiences"], 1)
    
    async def save_knowledge(self, filepath: str):
        """Save learned knowledge to file"""
        try:
            knowledge_data = {
                "nodes": dict(self.knowledge_graph.nodes),
                "edges": {str(k): v for k, v in self.knowledge_graph.edges.items()},
                "node_weights": dict(self.knowledge_graph.node_weights),
                "edge_weights": {str(k): v for k, v in self.knowledge_graph.edge_weights.items()},
                "learning_stats": self.learning_stats.copy(),
                "learning_config": self.learning_config.copy()
            }
            
            with open(filepath, 'w') as f:
                json.dump(knowledge_data, f, indent=2)
            
            logger.info(f"Knowledge saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save knowledge: {e}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        buffer_stats = self.experience_buffer.get_stats()
        knowledge_stats = {
            "total_concepts": len(self.knowledge_graph.nodes),
            "total_relationships": len(self.knowledge_graph.edges),
            "average_concept_importance": np.mean(list(self.knowledge_graph.node_weights.values())) if self.knowledge_graph.node_weights else 0.0
        }
        
        return {
            "is_initialized": self.is_initialized,
            "learning_stats": self.learning_stats.copy(),
            "experience_buffer": buffer_stats,
            "knowledge_graph": knowledge_stats,
            "learning_config": self.learning_config.copy(),
            "adaptation_history_size": len(self.adaptation_engine.adaptation_history)
        }