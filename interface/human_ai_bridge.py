"""
Human-AI Bridge Interface
=========================

Advanced interface system for seamless human-AI interaction with
empathy modeling, context awareness, and adaptive communication.
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

class CommunicationStyle(Enum):
    """Communication styles for adaptive interaction"""
    FORMAL = "formal"
    CASUAL = "casual" 
    TECHNICAL = "technical"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    EMPATHETIC = "empathetic"

class EmotionalState(Enum):
    """Detected emotional states"""
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    FRUSTRATED = "frustrated"
    EXCITED = "excited"
    CONFUSED = "confused"
    SATISFIED = "satisfied"

@dataclass
class UserProfile:
    """User profile for personalized interaction"""
    user_id: str
    preferred_style: CommunicationStyle
    expertise_level: float  # 0.0 = beginner, 1.0 = expert
    interaction_history: List[Dict[str, Any]]
    emotional_patterns: Dict[str, float]
    preferences: Dict[str, Any]
    satisfaction_score: float

@dataclass
class InteractionContext:
    """Context for current interaction"""
    user_profile: Optional[UserProfile]
    emotional_state: EmotionalState
    communication_style: CommunicationStyle
    urgency_level: float
    complexity_preference: float
    session_history: List[Dict[str, Any]]

class EmotionalIntelligenceEngine:
    """Engine for understanding and responding to emotions"""
    
    def __init__(self):
        self.emotion_patterns = {
            EmotionalState.FRUSTRATED: [
                "this doesn't work", "not working", "frustrated", "annoying",
                "stupid", "hate", "terrible", "awful", "why won't", "can't get"
            ],
            EmotionalState.EXCITED: [
                "amazing", "awesome", "fantastic", "love", "brilliant",
                "perfect", "excellent", "wonderful", "incredible"
            ],
            EmotionalState.CONFUSED: [
                "don't understand", "confused", "what does", "how do",
                "not sure", "unclear", "lost", "help me understand"
            ],
            EmotionalState.SATISFIED: [
                "thank you", "thanks", "helpful", "good", "works",
                "solved", "fixed", "appreciate", "great job"
            ]
        }
        
        self.empathy_responses = {
            EmotionalState.FRUSTRATED: [
                "I understand this can be frustrating. Let me help you work through this step by step.",
                "I can see you're having difficulties. Let's tackle this together.",
                "Frustration is completely understandable. I'm here to help make this easier."
            ],
            EmotionalState.CONFUSED: [
                "No worries! Let me explain this more clearly.",
                "I understand this might be confusing. Let me break it down for you.",
                "It's okay to be confused - let's work through this together."
            ],
            EmotionalState.EXCITED: [
                "I'm glad you're excited about this! Let me help you make the most of it.",
                "Your enthusiasm is wonderful! I'm happy to help you explore this further.",
                "Great energy! Let's channel that into achieving your goals."
            ]
        }
    
    def detect_emotion(self, text: str, context: Dict[str, Any] = None) -> EmotionalState:
        """Detect emotional state from text"""
        text_lower = text.lower()
        emotion_scores = {emotion: 0.0 for emotion in EmotionalState}
        
        # Pattern matching
        for emotion, patterns in self.emotion_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            emotion_scores[emotion] = score
        
        # Contextual adjustments
        if context:
            # Adjust based on interaction history
            if context.get("previous_failures", 0) > 2:
                emotion_scores[EmotionalState.FRUSTRATED] += 1
            
            # Consider response time
            if context.get("response_time", 1.0) > 3.0:
                emotion_scores[EmotionalState.FRUSTRATED] += 0.5
        
        # Punctuation analysis
        exclamation_count = text.count("!")
        question_count = text.count("?")
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        if exclamation_count > 1 or caps_ratio > 0.3:
            emotion_scores[EmotionalState.FRUSTRATED] += 1
        
        if question_count > 2:
            emotion_scores[EmotionalState.CONFUSED] += 1
        
        # Find dominant emotion
        max_emotion = max(emotion_scores, key=emotion_scores.get)
        
        # Return neutral if no strong emotion detected
        if emotion_scores[max_emotion] < 1:
            return EmotionalState.NEUTRAL
        
        return max_emotion
    
    def generate_empathetic_response(self, emotion: EmotionalState) -> str:
        """Generate empathetic response based on emotion"""
        if emotion in self.empathy_responses:
            responses = self.empathy_responses[emotion]
            return responses[hash(str(time.time())) % len(responses)]
        
        return "I understand. Let me help you with this."

class CommunicationStyleAdapter:
    """Adapts communication style based on user preferences"""
    
    def __init__(self):
        self.style_templates = {
            CommunicationStyle.FORMAL: {
                "greeting": "Good day. How may I assist you today?",
                "response_prefix": "Based on my analysis,",
                "conclusion": "I trust this information addresses your inquiry.",
                "tone_markers": ["furthermore", "however", "consequently"]
            },
            CommunicationStyle.CASUAL: {
                "greeting": "Hey! What can I help you with?",
                "response_prefix": "So basically,",
                "conclusion": "Hope that helps!",
                "tone_markers": ["yeah", "cool", "awesome", "got it"]
            },
            CommunicationStyle.TECHNICAL: {
                "greeting": "Please specify your technical requirements.",
                "response_prefix": "Technical analysis indicates:",
                "conclusion": "Implementation details follow these specifications.",
                "tone_markers": ["algorithm", "implementation", "architecture", "optimization"]
            },
            CommunicationStyle.FRIENDLY: {
                "greeting": "Hi there! I'm excited to help you today!",
                "response_prefix": "I'd be happy to explain that",
                "conclusion": "Feel free to ask if you need anything else!",
                "tone_markers": ["happy", "excited", "wonderful", "great"]
            },
            CommunicationStyle.EMPATHETIC: {
                "greeting": "I'm here to support you. What's on your mind?",
                "response_prefix": "I understand your concern, and",
                "conclusion": "I'm here if you need any further support.",
                "tone_markers": ["understand", "support", "care", "help"]
            }
        }
    
    def adapt_response(self, response: str, style: CommunicationStyle, emotion: EmotionalState) -> str:
        """Adapt response to match communication style and emotion"""
        if style not in self.style_templates:
            return response
        
        template = self.style_templates[style]
        
        # Add style-appropriate prefix if response doesn't already have one
        if not any(marker in response.lower() for marker in template["tone_markers"]):
            adapted_response = f"{template['response_prefix']} {response}"
        else:
            adapted_response = response
        
        # Adjust for emotional state
        if emotion == EmotionalState.FRUSTRATED:
            adapted_response = self._add_patience_markers(adapted_response)
        elif emotion == EmotionalState.CONFUSED:
            adapted_response = self._add_clarity_markers(adapted_response)
        elif emotion == EmotionalState.EXCITED:
            adapted_response = self._add_enthusiasm_markers(adapted_response)
        
        return adapted_response
    
    def _add_patience_markers(self, response: str) -> str:
        """Add patience markers for frustrated users"""
        patience_markers = [
            "Let's take this step by step: ",
            "I'll walk you through this carefully: ",
            "Don't worry, we'll figure this out: "
        ]
        
        marker = patience_markers[hash(response) % len(patience_markers)]
        return f"{marker}{response}"
    
    def _add_clarity_markers(self, response: str) -> str:
        """Add clarity markers for confused users"""
        clarity_markers = [
            "To clarify: ",
            "In simple terms: ",
            "Let me explain this clearly: "
        ]
        
        marker = clarity_markers[hash(response) % len(clarity_markers)]
        return f"{marker}{response}"
    
    def _add_enthusiasm_markers(self, response: str) -> str:
        """Add enthusiasm markers for excited users"""
        enthusiasm_markers = [
            "That's great! ",
            "Excellent question! ",
            "I love your enthusiasm! "
        ]
        
        marker = enthusiasm_markers[hash(response) % len(enthusiasm_markers)]
        return f"{marker}{response}"

class PersonalizationEngine:
    """Engine for personalizing interactions based on user profiles"""
    
    def __init__(self):
        self.user_profiles = {}
        self.default_profile = UserProfile(
            user_id="default",
            preferred_style=CommunicationStyle.FRIENDLY,
            expertise_level=0.5,
            interaction_history=[],
            emotional_patterns={},
            preferences={},
            satisfaction_score=0.8
        )
    
    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Get existing user profile or create new one"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                preferred_style=CommunicationStyle.FRIENDLY,
                expertise_level=0.5,
                interaction_history=[],
                emotional_patterns={},
                preferences={},
                satisfaction_score=0.8
            )
        
        return self.user_profiles[user_id]
    
    def update_profile(self, 
                      user_id: str,
                      interaction_data: Dict[str, Any],
                      feedback_score: float = None):
        """Update user profile based on interaction"""
        profile = self.get_or_create_profile(user_id)
        
        # Add to interaction history
        interaction_record = {
            **interaction_data,
            "timestamp": time.time(),
            "feedback_score": feedback_score
        }
        profile.interaction_history.append(interaction_record)
        
        # Keep only recent interactions
        if len(profile.interaction_history) > 100:
            profile.interaction_history = profile.interaction_history[-100:]
        
        # Update emotional patterns
        if "emotion" in interaction_data:
            emotion = interaction_data["emotion"]
            if emotion not in profile.emotional_patterns:
                profile.emotional_patterns[emotion] = 0.0
            profile.emotional_patterns[emotion] += 0.1
        
        # Update satisfaction score
        if feedback_score is not None:
            profile.satisfaction_score = (profile.satisfaction_score * 0.9) + (feedback_score * 0.1)
        
        # Adapt communication style based on patterns
        self._adapt_communication_style(profile)
        
        # Adjust expertise level
        self._update_expertise_level(profile, interaction_data)
    
    def _adapt_communication_style(self, profile: UserProfile):
        """Adapt communication style based on user patterns"""
        # Analyze emotional patterns
        if profile.emotional_patterns:
            dominant_emotion = max(profile.emotional_patterns, key=profile.emotional_patterns.get)
            
            # Adjust style based on dominant emotion
            if dominant_emotion == "frustrated":
                profile.preferred_style = CommunicationStyle.EMPATHETIC
            elif dominant_emotion == "confused":
                profile.preferred_style = CommunicationStyle.TECHNICAL
            elif dominant_emotion == "excited":
                profile.preferred_style = CommunicationStyle.FRIENDLY
        
        # Adjust based on satisfaction score
        if profile.satisfaction_score < 0.6:
            profile.preferred_style = CommunicationStyle.EMPATHETIC
        elif profile.satisfaction_score > 0.8:
            profile.preferred_style = CommunicationStyle.CASUAL
    
    def _update_expertise_level(self, profile: UserProfile, interaction_data: Dict[str, Any]):
        """Update user expertise level"""
        # Simple heuristic based on query complexity
        query_complexity = interaction_data.get("query_complexity", 0.5)
        
        # If user asks complex questions, increase expertise
        if query_complexity > 0.7:
            profile.expertise_level = min(profile.expertise_level + 0.05, 1.0)
        elif query_complexity < 0.3:
            profile.expertise_level = max(profile.expertise_level - 0.02, 0.0)

class ContextManager:
    """Manages interaction context and session state"""
    
    def __init__(self):
        self.active_sessions = {}
        self.context_history = {}
    
    def create_interaction_context(self, 
                                 user_id: str,
                                 query: str,
                                 session_id: str = None) -> InteractionContext:
        """Create interaction context"""
        if session_id is None:
            session_id = f"{user_id}_{int(time.time())}"
        
        # Get session history
        session_history = self.active_sessions.get(session_id, [])
        
        # Analyze query for context clues
        urgency_level = self._detect_urgency(query)
        complexity_preference = self._detect_complexity_preference(query)
        
        # Create context (user_profile will be injected by HumanAIBridge)
        context = InteractionContext(
            user_profile=None,  # Will be set by HumanAIBridge
            emotional_state=EmotionalState.NEUTRAL,  # Will be detected
            communication_style=CommunicationStyle.FRIENDLY,  # Default
            urgency_level=urgency_level,
            complexity_preference=complexity_preference,
            session_history=session_history
        )
        
        return context
    
    def update_session(self, session_id: str, interaction_data: Dict[str, Any]):
        """Update session with new interaction"""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = []
        
        self.active_sessions[session_id].append({
            **interaction_data,
            "timestamp": time.time()
        })
        
        # Keep only recent interactions
        if len(self.active_sessions[session_id]) > 50:
            self.active_sessions[session_id] = self.active_sessions[session_id][-50:]
    
    def _detect_urgency(self, query: str) -> float:
        """Detect urgency level from query"""
        urgency_indicators = [
            "urgent", "asap", "quickly", "immediately", "emergency",
            "right now", "help!", "deadline", "critical"
        ]
        
        urgency_count = sum(1 for indicator in urgency_indicators if indicator in query.lower())
        urgency_level = min(urgency_count * 0.3, 1.0)
        
        # Punctuation indicators
        if "!" in query:
            urgency_level += 0.2
        
        return min(urgency_level, 1.0)
    
    def _detect_complexity_preference(self, query: str) -> float:
        """Detect preferred complexity level"""
        simple_indicators = ["simple", "basic", "easy", "beginner", "quick"]
        complex_indicators = ["detailed", "comprehensive", "advanced", "technical", "in-depth"]
        
        simple_count = sum(1 for indicator in simple_indicators if indicator in query.lower())
        complex_count = sum(1 for indicator in complex_indicators if indicator in query.lower())
        
        if complex_count > simple_count:
            return 0.8  # High complexity preference
        elif simple_count > complex_count:
            return 0.2  # Low complexity preference
        else:
            return 0.5  # Medium complexity preference

class HumanAIBridge:
    """
    Main Human-AI Bridge Interface for CopilotX
    
    Orchestrates all interface components to provide seamless,
    empathetic, and adaptive human-AI interaction.
    """
    
    def __init__(self):
        self.emotional_intelligence = EmotionalIntelligenceEngine()
        self.style_adapter = CommunicationStyleAdapter()
        self.personalization_engine = PersonalizationEngine()
        self.context_manager = ContextManager()
        
        self.is_initialized = False
        self.interface_stats = {
            "total_interactions": 0,
            "emotional_states_detected": defaultdict(int),
            "communication_styles_used": defaultdict(int),
            "average_satisfaction": 0.0,
            "adaptation_count": 0
        }
    
    async def initialize(self) -> bool:
        """Initialize Human-AI Bridge"""
        try:
            logger.info("Initializing Human-AI Bridge...")
            
            # Initialize all interface components
            await self._initialize_components()
            
            self.is_initialized = True
            logger.info("Human-AI Bridge initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Human-AI Bridge: {e}")
            return False
    
    async def _initialize_components(self):
        """Initialize all interface components"""
        # All components initialize in their constructors
        pass
    
    async def process_interaction(self, 
                                user_id: str,
                                query: str,
                                response: str,
                                session_id: str = None,
                                feedback_score: float = None) -> str:
        """
        Process human-AI interaction with full context awareness
        
        Args:
            user_id: Unique user identifier
            query: User query
            response: AI-generated response
            session_id: Optional session identifier
            feedback_score: Optional feedback score
            
        Returns:
            Adapted response optimized for the user
        """
        if not self.is_initialized:
            raise RuntimeError("Human-AI Bridge not initialized")
        
        try:
            # Create interaction context
            context = self.context_manager.create_interaction_context(user_id, query, session_id)
            
            # Get user profile
            user_profile = self.personalization_engine.get_or_create_profile(user_id)
            context.user_profile = user_profile
            
            # Detect emotional state
            emotional_state = self.emotional_intelligence.detect_emotion(query, {
                "previous_failures": len([h for h in user_profile.interaction_history 
                                        if h.get("feedback_score", 0.8) < 0.5]),
                "response_time": 1.0  # Could be actual response time
            })
            context.emotional_state = emotional_state
            
            # Determine communication style
            communication_style = self._determine_communication_style(context)
            context.communication_style = communication_style
            
            # Generate empathetic response if needed
            empathy_response = ""
            if emotional_state != EmotionalState.NEUTRAL:
                empathy_response = self.emotional_intelligence.generate_empathetic_response(emotional_state)
            
            # Adapt response style
            adapted_response = self.style_adapter.adapt_response(
                response, communication_style, emotional_state
            )
            
            # Combine empathy and adapted response
            if empathy_response:
                final_response = f"{empathy_response} {adapted_response}"
            else:
                final_response = adapted_response
            
            # Update user profile
            interaction_data = {
                "query": query,
                "response": final_response,
                "emotion": emotional_state.value,
                "communication_style": communication_style.value,
                "query_complexity": context.complexity_preference,
                "urgency_level": context.urgency_level
            }
            
            self.personalization_engine.update_profile(user_id, interaction_data, feedback_score)
            
            # Update session
            if session_id:
                self.context_manager.update_session(session_id, interaction_data)
            
            # Update statistics
            self._update_interface_stats(context, feedback_score)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Interaction processing failed: {e}")
            return response  # Return original response on error
    
    def _determine_communication_style(self, context: InteractionContext) -> CommunicationStyle:
        """Determine appropriate communication style"""
        # Start with user preference
        if context.user_profile:
            style = context.user_profile.preferred_style
        else:
            style = CommunicationStyle.FRIENDLY
        
        # Adjust based on emotional state
        if context.emotional_state == EmotionalState.FRUSTRATED:
            style = CommunicationStyle.EMPATHETIC
        elif context.emotional_state == EmotionalState.CONFUSED:
            # Use technical style for confused users to provide clarity
            style = CommunicationStyle.TECHNICAL
        
        # Adjust based on urgency
        if context.urgency_level > 0.7:
            style = CommunicationStyle.PROFESSIONAL
        
        # Adjust based on complexity preference
        if context.complexity_preference > 0.7:
            style = CommunicationStyle.TECHNICAL
        elif context.complexity_preference < 0.3:
            style = CommunicationStyle.CASUAL
        
        return style
    
    def _update_interface_stats(self, context: InteractionContext, feedback_score: float = None):
        """Update interface statistics"""
        self.interface_stats["total_interactions"] += 1
        self.interface_stats["emotional_states_detected"][context.emotional_state.value] += 1
        self.interface_stats["communication_styles_used"][context.communication_style.value] += 1
        
        if feedback_score is not None:
            # Update average satisfaction
            total = self.interface_stats["total_interactions"]
            current_avg = self.interface_stats["average_satisfaction"]
            new_avg = ((current_avg * (total - 1)) + feedback_score) / total
            self.interface_stats["average_satisfaction"] = new_avg
        
        # Count adaptations (when style differs from default)
        if context.communication_style != CommunicationStyle.FRIENDLY:
            self.interface_stats["adaptation_count"] += 1
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile"""
        return self.personalization_engine.user_profiles.get(user_id)
    
    def get_interface_stats(self) -> Dict[str, Any]:
        """Get comprehensive interface statistics"""
        return {
            "is_initialized": self.is_initialized,
            "interface_stats": dict(self.interface_stats),
            "emotional_states_detected": dict(self.interface_stats["emotional_states_detected"]),
            "communication_styles_used": dict(self.interface_stats["communication_styles_used"]),
            "total_user_profiles": len(self.personalization_engine.user_profiles),
            "active_sessions": len(self.context_manager.active_sessions)
        }