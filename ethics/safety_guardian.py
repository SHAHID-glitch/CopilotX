"""
Safety Guardian System
======================

Comprehensive AI safety and ethics system ensuring responsible AI behavior,
bias detection, privacy protection, and ethical decision-making.
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import re
import time
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    """Safety assessment levels"""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    BLOCKED = "blocked"

class BiasType(Enum):
    """Types of bias to detect"""
    GENDER = "gender"
    RACIAL = "racial"
    AGE = "age"
    RELIGIOUS = "religious"
    CULTURAL = "cultural"
    SOCIOECONOMIC = "socioeconomic"
    POLITICAL = "political"

@dataclass
class SafetyResult:
    """Result from safety assessment"""
    is_safe: bool
    safety_level: SafetyLevel
    reason: str
    confidence: float
    detected_issues: List[str]
    recommendations: List[str]

@dataclass
class BiasDetectionResult:
    """Result from bias detection"""
    bias_detected: bool
    bias_types: List[BiasType]
    severity_score: float
    evidence: List[str]
    mitigation_suggestions: List[str]

class ContentSafetyFilter:
    """Filter for detecting harmful or inappropriate content"""
    
    def __init__(self):
        # Harmful content patterns
        self.harmful_patterns = {
            "violence": [
                r"\b(kill|murder|attack|hurt|harm|violence|weapon|bomb|gun)\b",
                r"\b(fight|punch|kick|hit|stab|shoot)\b"
            ],
            "hate_speech": [
                r"\b(hate|racist|discrimination|bigot|supremacist)\b",
                r"\b(inferior|superior) (race|people|group)\b"
            ],
            "inappropriate": [
                r"\b(explicit|nsfw|adult|sexual|pornographic)\b",
                r"\b(drug|illegal|criminal|fraud|scam)\b"
            ],
            "misinformation": [
                r"\b(fake news|conspiracy|hoax|lie|false claim)\b",
                r"\b(proven false|debunked|misinformation)\b"
            ]
        }
        
        # Sensitive topics requiring careful handling
        self.sensitive_topics = {
            "medical": [
                r"\b(medical advice|diagnosis|treatment|medication|surgery)\b",
                r"\b(cancer|depression|anxiety|mental health|suicide)\b"
            ],
            "legal": [
                r"\b(legal advice|lawsuit|court|lawyer|attorney)\b",
                r"\b(illegal|law|regulation|compliance)\b"
            ],
            "financial": [
                r"\b(financial advice|investment|stock|trading|crypto)\b",
                r"\b(loan|debt|bankruptcy|tax|money)\b"
            ]
        }
    
    def assess_content_safety(self, text: str) -> SafetyResult:
        """Assess content safety"""
        text_lower = text.lower()
        detected_issues = []
        safety_level = SafetyLevel.SAFE
        
        # Check for harmful patterns
        for category, patterns in self.harmful_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    detected_issues.append(f"Potential {category} content detected")
                    safety_level = SafetyLevel.BLOCKED
        
        # Check for sensitive topics
        sensitive_topics_found = []
        for category, patterns in self.sensitive_topics.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    sensitive_topics_found.append(category)
                    if safety_level == SafetyLevel.SAFE:
                        safety_level = SafetyLevel.CAUTION
        
        # Generate recommendations
        recommendations = []
        if sensitive_topics_found:
            recommendations.append("Provide appropriate disclaimers for sensitive topics")
            recommendations.append("Suggest consulting relevant professionals")
        
        if detected_issues:
            recommendations.append("Content should be reviewed and potentially blocked")
            recommendations.append("Consider alternative phrasings")
        
        is_safe = safety_level in [SafetyLevel.SAFE, SafetyLevel.CAUTION]
        confidence = 0.8 if detected_issues else 0.9
        
        reason = f"Content assessment: {len(detected_issues)} issues found" if detected_issues else "Content appears safe"
        
        return SafetyResult(
            is_safe=is_safe,
            safety_level=safety_level,
            reason=reason,
            confidence=confidence,
            detected_issues=detected_issues,
            recommendations=recommendations
        )

class BiasDetector:
    """Advanced bias detection system"""
    
    def __init__(self):
        # Bias indicators by type
        self.bias_indicators = {
            BiasType.GENDER: {
                "stereotypes": [
                    r"\b(women are|men are|girls are|boys are) (better|worse|naturally)\b",
                    r"\b(feminine|masculine) (job|role|profession)\b"
                ],
                "exclusionary": [
                    r"\b(only (men|women)|just for (boys|girls))\b",
                    r"\b(not suitable for (women|men))\b"
                ]
            },
            BiasType.RACIAL: {
                "stereotypes": [
                    r"\b(people of|racial|ethnic) (group|background).*(are|tend to|usually)\b",
                    r"\b(culture|race).*(inferior|superior|better|worse)\b"
                ],
                "profiling": [
                    r"\b(looks like|sounds like|typical of)\b",
                    r"\b(racial|ethnic) (characteristic|trait|feature)\b"
                ]
            },
            BiasType.AGE: {
                "stereotypes": [
                    r"\b(young people|old people|elderly|seniors) (are|can't|cannot)\b",
                    r"\b(too (young|old) (for|to))\b"
                ],
                "discrimination": [
                    r"\b(age limit|age requirement|age restriction)\b"
                ]
            }
        }
        
        # Inclusive language alternatives
        self.inclusive_alternatives = {
            "guys": ["everyone", "team", "folks"],
            "manpower": ["workforce", "personnel", "staff"],
            "blacklist": ["blocklist", "denylist"],
            "whitelist": ["allowlist", "safelist"],
            "master/slave": ["primary/secondary", "main/replica"]
        }
    
    def detect_bias(self, text: str, context: Dict[str, Any] = None) -> BiasDetectionResult:
        """Detect potential bias in text"""
        text_lower = text.lower()
        detected_biases = []
        evidence = []
        severity_scores = []
        
        # Check each bias type
        for bias_type, categories in self.bias_indicators.items():
            bias_found = False
            for category, patterns in categories.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text_lower, re.IGNORECASE)
                    if matches:
                        detected_biases.append(bias_type)
                        evidence.append(f"{bias_type.value} {category}: {matches[0][:50]}...")
                        severity_scores.append(self._calculate_bias_severity(pattern, matches))
                        bias_found = True
                        break
                if bias_found:
                    break
        
        # Check for non-inclusive language
        inclusive_issues = self._check_inclusive_language(text)
        if inclusive_issues:
            evidence.extend(inclusive_issues)
            severity_scores.append(0.3)  # Lower severity for language issues
        
        # Calculate overall severity
        overall_severity = np.mean(severity_scores) if severity_scores else 0.0
        
        # Generate mitigation suggestions
        suggestions = self._generate_mitigation_suggestions(detected_biases, evidence)
        
        return BiasDetectionResult(
            bias_detected=len(detected_biases) > 0 or len(inclusive_issues) > 0,
            bias_types=list(set(detected_biases)),
            severity_score=overall_severity,
            evidence=evidence,
            mitigation_suggestions=suggestions
        )
    
    def _calculate_bias_severity(self, pattern: str, matches: List[str]) -> float:
        """Calculate severity score for detected bias"""
        # Simple heuristic based on pattern type and match count
        base_severity = 0.5
        
        # Increase severity for explicit stereotypes
        if "stereotype" in pattern or "are" in pattern:
            base_severity += 0.3
        
        # Increase severity for exclusionary language
        if "only" in pattern or "not suitable" in pattern:
            base_severity += 0.4
        
        # Adjust for number of matches
        severity = base_severity + (len(matches) * 0.1)
        
        return min(severity, 1.0)
    
    def _check_inclusive_language(self, text: str) -> List[str]:
        """Check for non-inclusive language"""
        issues = []
        text_lower = text.lower()
        
        for non_inclusive, alternatives in self.inclusive_alternatives.items():
            if non_inclusive in text_lower:
                issues.append(f"Consider replacing '{non_inclusive}' with: {', '.join(alternatives)}")
        
        return issues
    
    def _generate_mitigation_suggestions(self, 
                                       bias_types: List[BiasType], 
                                       evidence: List[str]) -> List[str]:
        """Generate suggestions for mitigating detected bias"""
        suggestions = []
        
        if BiasType.GENDER in bias_types:
            suggestions.append("Use gender-neutral language when possible")
            suggestions.append("Avoid assumptions about gender roles or capabilities")
        
        if BiasType.RACIAL in bias_types:
            suggestions.append("Focus on individual characteristics rather than group generalizations")
            suggestions.append("Be aware of cultural sensitivity in descriptions")
        
        if BiasType.AGE in bias_types:
            suggestions.append("Avoid age-related assumptions about abilities")
            suggestions.append("Consider age-inclusive alternatives")
        
        # General suggestions
        suggestions.extend([
            "Review content for unconscious bias",
            "Consider diverse perspectives in examples",
            "Use inclusive language guidelines"
        ])
        
        return suggestions[:5]  # Limit to top 5 suggestions

class PrivacyProtector:
    """Privacy protection and data anonymization"""
    
    def __init__(self):
        # Patterns for detecting PII
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-?\d{2}-?\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            "address": r'\b\d+\s+[\w\s]+\s+(street|st|avenue|ave|road|rd|drive|dr|lane|ln|way|court|ct)\b'
        }
        
        self.anonymization_cache = {}
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect personally identifiable information"""
        detected_pii = defaultdict(list)
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected_pii[pii_type].extend(matches)
        
        return dict(detected_pii)
    
    def anonymize_text(self, text: str, preserve_format: bool = True) -> Tuple[str, Dict[str, str]]:
        """Anonymize text by replacing PII with placeholders"""
        anonymized_text = text
        anonymization_map = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                original_value = match.group()
                
                # Generate consistent anonymized value
                if original_value in self.anonymization_cache:
                    anonymized_value = self.anonymization_cache[original_value]
                else:
                    anonymized_value = self._generate_anonymized_value(pii_type, original_value, preserve_format)
                    self.anonymization_cache[original_value] = anonymized_value
                
                anonymized_text = anonymized_text.replace(original_value, anonymized_value)
                anonymization_map[original_value] = anonymized_value
        
        return anonymized_text, anonymization_map
    
    def _generate_anonymized_value(self, pii_type: str, original: str, preserve_format: bool) -> str:
        """Generate anonymized value for PII"""
        # Create hash-based anonymization for consistency
        hash_input = f"{pii_type}_{original}".encode()
        hash_value = hashlib.md5(hash_input).hexdigest()[:8]
        
        if pii_type == "email":
            return f"user{hash_value}@example.com" if preserve_format else f"[EMAIL_{hash_value}]"
        elif pii_type == "phone":
            return f"555-{hash_value[:3]}-{hash_value[3:7]}" if preserve_format else f"[PHONE_{hash_value}]"
        elif pii_type == "ssn":
            return f"XXX-XX-{hash_value[:4]}" if preserve_format else f"[SSN_{hash_value}]"
        elif pii_type == "credit_card":
            return f"XXXX-XXXX-XXXX-{hash_value[:4]}" if preserve_format else f"[CARD_{hash_value}]"
        elif pii_type == "ip_address":
            return f"192.168.1.{hash(hash_value) % 255}" if preserve_format else f"[IP_{hash_value}]"
        elif pii_type == "address":
            return f"[ADDRESS_{hash_value}]"
        else:
            return f"[{pii_type.upper()}_{hash_value}]"

class EthicalDecisionEngine:
    """Engine for making ethical decisions in AI responses"""
    
    def __init__(self):
        # Ethical principles
        self.ethical_principles = {
            "beneficence": "Maximize benefits and well-being",
            "non_maleficence": "Do no harm",
            "autonomy": "Respect individual choice and freedom",
            "justice": "Treat all individuals fairly",
            "transparency": "Be open and honest about capabilities and limitations",
            "accountability": "Take responsibility for AI decisions and outcomes"
        }
        
        # Ethical dilemma scenarios
        self.dilemma_handlers = {
            "medical_advice": self._handle_medical_advice_request,
            "legal_advice": self._handle_legal_advice_request,
            "harmful_information": self._handle_harmful_information_request,
            "privacy_violation": self._handle_privacy_violation,
            "bias_amplification": self._handle_bias_amplification
        }
    
    def evaluate_ethical_implications(self, 
                                    query: str, 
                                    proposed_response: str,
                                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate ethical implications of a proposed response"""
        evaluation = {
            "ethical_score": 0.8,  # Default score
            "principle_violations": [],
            "recommendations": [],
            "approved": True
        }
        
        # Check for specific ethical dilemmas
        for scenario, handler in self.dilemma_handlers.items():
            if self._detect_scenario(query, proposed_response, scenario):
                scenario_evaluation = handler(query, proposed_response, context)
                evaluation = self._merge_evaluations(evaluation, scenario_evaluation)
        
        # General ethical checks
        general_evaluation = self._perform_general_ethical_checks(query, proposed_response)
        evaluation = self._merge_evaluations(evaluation, general_evaluation)
        
        return evaluation
    
    def _detect_scenario(self, query: str, response: str, scenario: str) -> bool:
        """Detect if a specific ethical scenario applies"""
        combined_text = f"{query} {response}".lower()
        
        scenario_keywords = {
            "medical_advice": ["medical", "health", "diagnosis", "treatment", "medication", "doctor"],
            "legal_advice": ["legal", "law", "lawsuit", "attorney", "court", "illegal"],
            "harmful_information": ["how to", "instructions", "build", "make", "create", "weapon"],
            "privacy_violation": ["personal", "private", "confidential", "secret", "hack"],
            "bias_amplification": ["stereotype", "always", "never", "all", "group", "people"]
        }
        
        keywords = scenario_keywords.get(scenario, [])
        return any(keyword in combined_text for keyword in keywords)
    
    def _handle_medical_advice_request(self, query: str, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle medical advice scenarios"""
        if "diagnosis" in response.lower() or "treatment" in response.lower():
            return {
                "ethical_score": 0.3,
                "principle_violations": ["non_maleficence", "transparency"],
                "recommendations": [
                    "Add disclaimer about not providing medical advice",
                    "Recommend consulting healthcare professional",
                    "Provide general information only"
                ],
                "approved": False
            }
        
        return {
            "ethical_score": 0.7,
            "recommendations": ["Include medical disclaimer"],
            "approved": True
        }
    
    def _handle_legal_advice_request(self, query: str, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle legal advice scenarios"""
        if any(phrase in response.lower() for phrase in ["you should", "legally required", "court will"]):
            return {
                "ethical_score": 0.4,
                "principle_violations": ["non_maleficence", "accountability"],
                "recommendations": [
                    "Add disclaimer about not providing legal advice",
                    "Suggest consulting attorney",
                    "Provide general information only"
                ],
                "approved": False
            }
        
        return {
            "ethical_score": 0.8,
            "recommendations": ["Include legal disclaimer"],
            "approved": True
        }
    
    def _handle_harmful_information_request(self, query: str, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle potentially harmful information requests"""
        harmful_indicators = ["instructions", "how to make", "step by step", "materials needed"]
        
        if any(indicator in response.lower() for indicator in harmful_indicators):
            return {
                "ethical_score": 0.1,
                "principle_violations": ["non_maleficence", "beneficence"],
                "recommendations": [
                    "Refuse to provide harmful information",
                    "Explain why information is dangerous",
                    "Suggest constructive alternatives"
                ],
                "approved": False
            }
        
        return {"ethical_score": 0.9, "approved": True}
    
    def _handle_privacy_violation(self, query: str, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle privacy violation scenarios"""
        privacy_violations = ["personal information", "hack", "access without permission", "spy"]
        
        if any(violation in response.lower() for violation in privacy_violations):
            return {
                "ethical_score": 0.2,
                "principle_violations": ["autonomy", "justice"],
                "recommendations": [
                    "Refuse to assist with privacy violations",
                    "Explain importance of privacy rights",
                    "Suggest legal alternatives"
                ],
                "approved": False
            }
        
        return {"ethical_score": 0.9, "approved": True}
    
    def _handle_bias_amplification(self, query: str, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle bias amplification scenarios"""
        bias_indicators = ["all women", "all men", "people of", "always", "never", "typically"]
        
        if any(indicator in response.lower() for indicator in bias_indicators):
            return {
                "ethical_score": 0.5,
                "principle_violations": ["justice"],
                "recommendations": [
                    "Avoid generalizations about groups",
                    "Use more inclusive language",
                    "Acknowledge individual differences"
                ],
                "approved": True  # Approved but needs improvement
            }
        
        return {"ethical_score": 0.9, "approved": True}
    
    def _perform_general_ethical_checks(self, query: str, response: str) -> Dict[str, Any]:
        """Perform general ethical checks"""
        evaluation = {
            "ethical_score": 0.8,
            "principle_violations": [],
            "recommendations": [],
            "approved": True
        }
        
        # Check for transparency
        if "I don't know" not in response and "uncertain" not in response:
            if len(response) > 500:  # Long responses should acknowledge limitations
                evaluation["recommendations"].append("Consider acknowledging limitations or uncertainties")
        
        # Check for balanced perspective
        if any(word in response.lower() for word in ["definitely", "certainly", "absolutely", "always", "never"]):
            evaluation["recommendations"].append("Consider using more nuanced language")
            evaluation["ethical_score"] -= 0.1
        
        return evaluation
    
    def _merge_evaluations(self, eval1: Dict[str, Any], eval2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two ethical evaluations"""
        merged = eval1.copy()
        
        # Take minimum ethical score
        merged["ethical_score"] = min(eval1["ethical_score"], eval2["ethical_score"])
        
        # Combine violations and recommendations
        merged["principle_violations"].extend(eval2.get("principle_violations", []))
        merged["recommendations"].extend(eval2.get("recommendations", []))
        
        # AND logic for approval
        merged["approved"] = eval1["approved"] and eval2.get("approved", True)
        
        return merged

class SafetyGuardian:
    """
    Main Safety Guardian System for CopilotX
    
    Orchestrates all safety and ethics components to ensure responsible
    AI behavior and protect users from harm.
    """
    
    def __init__(self):
        self.content_filter = ContentSafetyFilter()
        self.bias_detector = BiasDetector()
        self.privacy_protector = PrivacyProtector()
        self.ethical_engine = EthicalDecisionEngine()
        
        self.is_initialized = False
        self.safety_stats = {
            "total_assessments": 0,
            "blocked_content": 0,
            "bias_detections": 0,
            "privacy_violations": 0,
            "ethical_violations": 0,
            "safety_score": 0.95
        }
    
    async def initialize(self) -> bool:
        """Initialize Safety Guardian system"""
        try:
            logger.info("Initializing Safety Guardian System...")
            
            # Initialize all safety components
            await self._initialize_safety_components()
            
            self.is_initialized = True
            logger.info("Safety Guardian System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Safety Guardian: {e}")
            return False
    
    async def _initialize_safety_components(self):
        """Initialize all safety components"""
        # All components initialize in their constructors
        pass
    
    async def validate_query(self, query: str) -> SafetyResult:
        """Validate user query for safety and ethics"""
        if not self.is_initialized:
            raise RuntimeError("Safety Guardian not initialized")
        
        try:
            # Content safety assessment
            safety_result = self.content_filter.assess_content_safety(query)
            
            # Update statistics
            self.safety_stats["total_assessments"] += 1
            if not safety_result.is_safe:
                self.safety_stats["blocked_content"] += 1
            
            return safety_result
            
        except Exception as e:
            logger.error(f"Query validation failed: {e}")
            return SafetyResult(
                is_safe=False,
                safety_level=SafetyLevel.WARNING,
                reason="Safety validation error",
                confidence=0.5,
                detected_issues=["Validation error occurred"],
                recommendations=["Please try again or rephrase query"]
            )
    
    async def validate_response(self, 
                              query: str,
                              response: str,
                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive response validation"""
        if not self.is_initialized:
            raise RuntimeError("Safety Guardian not initialized")
        
        try:
            validation_results = {
                "approved": True,
                "overall_safety_score": 1.0,
                "issues_found": [],
                "recommendations": [],
                "anonymized_response": response
            }
            
            # Content safety check
            content_safety = self.content_filter.assess_content_safety(response)
            if not content_safety.is_safe:
                validation_results["approved"] = False
                validation_results["issues_found"].extend(content_safety.detected_issues)
                validation_results["recommendations"].extend(content_safety.recommendations)
            
            validation_results["overall_safety_score"] *= 0.8 if not content_safety.is_safe else 1.0
            
            # Bias detection
            bias_result = self.bias_detector.detect_bias(response, context)
            if bias_result.bias_detected:
                validation_results["issues_found"].append(f"Bias detected: {', '.join([b.value for b in bias_result.bias_types])}")
                validation_results["recommendations"].extend(bias_result.mitigation_suggestions)
                validation_results["overall_safety_score"] *= (1.0 - bias_result.severity_score * 0.5)
                self.safety_stats["bias_detections"] += 1
            
            # Privacy protection
            pii_detected = self.privacy_protector.detect_pii(response)
            if pii_detected:
                anonymized_response, anonymization_map = self.privacy_protector.anonymize_text(response)
                validation_results["anonymized_response"] = anonymized_response
                validation_results["issues_found"].append(f"PII detected: {', '.join(pii_detected.keys())}")
                validation_results["recommendations"].append("Personal information has been anonymized")
                self.safety_stats["privacy_violations"] += 1
            
            # Ethical evaluation
            ethical_eval = self.ethical_engine.evaluate_ethical_implications(query, response, context)
            if not ethical_eval["approved"]:
                validation_results["approved"] = False
                validation_results["issues_found"].append("Ethical concerns identified")
                validation_results["recommendations"].extend(ethical_eval["recommendations"])
                self.safety_stats["ethical_violations"] += 1
            
            validation_results["overall_safety_score"] *= ethical_eval["ethical_score"]
            
            # Update overall safety score
            self._update_safety_score(validation_results["overall_safety_score"])
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Response validation failed: {e}")
            return {
                "approved": False,
                "overall_safety_score": 0.5,
                "issues_found": ["Validation error occurred"],
                "recommendations": ["Response could not be properly validated"],
                "anonymized_response": response
            }
    
    def _update_safety_score(self, current_score: float):
        """Update overall system safety score"""
        # Exponential moving average
        alpha = 0.1
        self.safety_stats["safety_score"] = (
            (1 - alpha) * self.safety_stats["safety_score"] + 
            alpha * current_score
        )
    
    def get_safety_stats(self) -> Dict[str, Any]:
        """Get comprehensive safety statistics"""
        total_assessments = max(self.safety_stats["total_assessments"], 1)
        
        return {
            "is_initialized": self.is_initialized,
            "safety_stats": self.safety_stats.copy(),
            "safety_metrics": {
                "content_block_rate": self.safety_stats["blocked_content"] / total_assessments,
                "bias_detection_rate": self.safety_stats["bias_detections"] / total_assessments,
                "privacy_violation_rate": self.safety_stats["privacy_violations"] / total_assessments,
                "ethical_violation_rate": self.safety_stats["ethical_violations"] / total_assessments,
                "overall_safety_score": self.safety_stats["safety_score"]
            }
        }