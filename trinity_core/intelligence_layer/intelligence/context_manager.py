#!/usr/bin/env python3
"""
TARA Core Intelligence - Context Manager
Maintains conversation context, memory, and continuity for intelligent responses
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import hashlib

class TARAContextManager:
    """
    TARA's context management system for maintaining conversation continuity
    Tracks conversation history, user preferences, and contextual patterns
    """
    
    def __init__(self, max_history_length: int = 50):
        # Conversation context storage
        self.conversation_history = deque(maxlen=max_history_length)
        self.current_session = {
            "session_id": self._generate_session_id(),
            "start_time": datetime.now(),
            "user_profile": {},
            "conversation_themes": [],
            "active_domains": [],
            "emotional_trajectory": []
        }
        
        # Context analysis patterns
        self.context_patterns = {
            "topic_transitions": {
                "smooth": ["also", "additionally", "furthermore", "speaking of", "that reminds me"],
                "abrupt": ["anyway", "by the way", "changing topics", "different question", "new topic"],
                "return": ["back to", "returning to", "as we discussed", "earlier you mentioned"]
            },
            
            "reference_patterns": {
                "temporal": ["earlier", "before", "previously", "just now", "a moment ago", "recently"],
                "conversational": ["you said", "you mentioned", "as you explained", "like you told me"],
                "contextual": ["this", "that", "it", "they", "those", "these"]
            },
            
            "continuation_cues": {
                "elaboration": ["tell me more", "explain further", "can you elaborate", "go deeper"],
                "clarification": ["what do you mean", "can you clarify", "I don't understand", "confused"],
                "confirmation": ["is that right", "correct me if", "am I understanding", "do you mean"]
            }
        }
        
        # User preference tracking
        self.user_preferences = {
            "communication_style": {
                "formality_level": "moderate",  # formal, moderate, casual
                "detail_preference": "balanced",  # brief, balanced, detailed
                "empathy_preference": "high",     # low, moderate, high
                "technical_level": "moderate"     # beginner, moderate, advanced
            },
            
            "domain_interests": {},  # Track which domains user engages with most
            "response_patterns": {},  # Track what types of responses user prefers
            "time_preferences": {},   # Track when user is most active
            "language_preferences": {
                "primary_language": "english",
                "complexity_level": "standard",
                "cultural_context": "general"
            }
        }
        
        # Context memory types
        self.memory_types = {
            "short_term": {
                "capacity": 10,
                "retention_time": timedelta(minutes=30),
                "content": deque(maxlen=10)
            },
            "medium_term": {
                "capacity": 50,
                "retention_time": timedelta(hours=24),
                "content": deque(maxlen=50)
            },
            "long_term": {
                "capacity": 200,
                "retention_time": timedelta(days=30),
                "content": deque(maxlen=200)
            }
        }
        
        # Contextual intelligence
        self.contextual_intelligence = {
            "topic_coherence": 0.0,    # How well topics flow together
            "emotional_consistency": 0.0,  # Emotional continuity
            "user_engagement": 0.0,    # Level of user engagement
            "conversation_depth": 0.0,  # How deep the conversation goes
            "domain_focus": 0.0        # Focus on specific domains
        }
        
        # Performance tracking
        self.context_stats = {
            "conversations_managed": 0,
            "context_switches": 0,
            "successful_references": 0,
            "preference_updates": 0
        }
    
    def _generate_session_id(self) -> str:
        """Generate a unique session identifier"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]
    
    async def add_conversation_turn(self, user_input: str, assistant_response: str,
                                  detected_emotion: Dict[str, Any], domain: str,
                                  urgency_level: str = "low") -> Dict[str, Any]:
        """
        Add a conversation turn to the context history
        
        Args:
            user_input: The user's message
            assistant_response: TARA's response
            detected_emotion: Emotion analysis results
            domain: Detected domain
            urgency_level: Urgency level
            
        Returns:
            Dict containing context analysis and updates
        """
        
        # Create conversation turn record
        turn_record = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "assistant_response": assistant_response,
            "detected_emotion": detected_emotion,
            "domain": domain,
            "urgency_level": urgency_level,
            "turn_number": len(self.conversation_history) + 1
        }
        
        # Analyze context before adding
        context_analysis = await self._analyze_conversation_context(turn_record)
        
        # Add to conversation history
        self.conversation_history.append(turn_record)
        
        # Update session information
        await self._update_session_context(turn_record, context_analysis)
        
        # Update user preferences
        await self._update_user_preferences(turn_record)
        
        # Update contextual intelligence
        await self._update_contextual_intelligence(context_analysis)
        
        # Store in appropriate memory
        await self._store_in_memory(turn_record, context_analysis)
        
        result = {
            "context_analysis": context_analysis,
            "session_update": {
                "total_turns": len(self.conversation_history),
                "current_domain": domain,
                "emotional_state": detected_emotion.get("primary_emotion", "neutral"),
                "conversation_flow": context_analysis.get("flow_type", "normal")
            },
            "user_preferences": self.user_preferences,
            "contextual_intelligence": self.contextual_intelligence,
            "success": True
        }
        
        await self._update_context_stats()
        
        return result
    
    async def _analyze_conversation_context(self, turn_record: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the context of the current conversation turn"""
        
        user_input = turn_record["user_input"].lower()
        current_domain = turn_record["domain"]
        
        # Analyze topic flow
        topic_analysis = await self._analyze_topic_flow(user_input, current_domain)
        
        # Analyze references to previous conversation
        reference_analysis = await self._analyze_references(user_input)
        
        # Analyze conversation continuity
        continuity_analysis = await self._analyze_continuity(user_input)
        
        # Analyze emotional context
        emotional_context = await self._analyze_emotional_context(turn_record)
        
        # Determine context type
        context_type = await self._determine_context_type(
            topic_analysis, reference_analysis, continuity_analysis
        )
        
        return {
            "topic_analysis": topic_analysis,
            "reference_analysis": reference_analysis,
            "continuity_analysis": continuity_analysis,
            "emotional_context": emotional_context,
            "context_type": context_type,
            "flow_type": topic_analysis.get("flow_type", "normal"),
            "requires_context_awareness": reference_analysis.get("has_references", False)
        }
    
    async def _analyze_topic_flow(self, user_input: str, current_domain: str) -> Dict[str, Any]:
        """Analyze how topics flow in the conversation"""
        
        if not self.conversation_history:
            return {
                "flow_type": "initial",
                "topic_change": False,
                "domain_change": False,
                "transition_type": "new_conversation"
            }
        
        # Get last conversation turn
        last_turn = self.conversation_history[-1]
        last_domain = last_turn["domain"]
        
        # Check for domain change
        domain_change = current_domain != last_domain
        
        # Analyze transition type
        transition_type = "continuation"
        
        for transition_type_name, indicators in self.context_patterns["topic_transitions"].items():
            for indicator in indicators:
                if indicator in user_input:
                    transition_type = transition_type_name
                    break
        
        # Determine flow type
        if domain_change and transition_type == "abrupt":
            flow_type = "abrupt_change"
        elif domain_change and transition_type == "smooth":
            flow_type = "smooth_transition"
        elif transition_type == "return":
            flow_type = "topic_return"
        else:
            flow_type = "normal_flow"
        
        return {
            "flow_type": flow_type,
            "topic_change": domain_change,
            "domain_change": domain_change,
            "transition_type": transition_type,
            "previous_domain": last_domain,
            "current_domain": current_domain
        }
    
    async def _analyze_references(self, user_input: str) -> Dict[str, Any]:
        """Analyze references to previous conversation elements"""
        
        references_found = {
            "temporal": [],
            "conversational": [],
            "contextual": []
        }
        
        has_references = False
        
        for reference_type, patterns in self.context_patterns["reference_patterns"].items():
            for pattern in patterns:
                if pattern in user_input:
                    references_found[reference_type].append(pattern)
                    has_references = True
        
        # Analyze what user might be referring to
        potential_references = []
        if has_references and self.conversation_history:
            # Look for potential references in recent conversation
            for turn in list(self.conversation_history)[-5:]:  # Last 5 turns
                potential_references.append({
                    "turn_number": turn["turn_number"],
                    "domain": turn["domain"],
                    "key_topics": self._extract_key_topics(turn["user_input"]),
                    "timestamp": turn["timestamp"]
                })
        
        return {
            "has_references": has_references,
            "reference_types": references_found,
            "potential_references": potential_references,
            "context_dependency": "high" if has_references else "low"
        }
    
    async def _analyze_continuity(self, user_input: str) -> Dict[str, Any]:
        """Analyze conversation continuity cues"""
        
        continuation_cues = {
            "elaboration": [],
            "clarification": [],
            "confirmation": []
        }
        
        for cue_type, patterns in self.context_patterns["continuation_cues"].items():
            for pattern in patterns:
                if pattern in user_input:
                    continuation_cues[cue_type].append(pattern)
        
        # Determine continuity level
        total_cues = sum(len(cues) for cues in continuation_cues.values())
        
        if total_cues > 2:
            continuity_level = "high"
        elif total_cues > 0:
            continuity_level = "medium"
        else:
            continuity_level = "low"
        
        return {
            "continuity_cues": continuation_cues,
            "continuity_level": continuity_level,
            "seeks_elaboration": len(continuation_cues["elaboration"]) > 0,
            "seeks_clarification": len(continuation_cues["clarification"]) > 0,
            "seeks_confirmation": len(continuation_cues["confirmation"]) > 0
        }
    
    async def _analyze_emotional_context(self, turn_record: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze emotional context and trajectory"""
        
        current_emotion = turn_record["detected_emotion"].get("primary_emotion", "neutral")
        current_intensity = turn_record["detected_emotion"].get("emotion_intensity", 0.5)
        
        # Compare with recent emotional history
        emotional_trajectory = []
        if len(self.conversation_history) >= 2:
            recent_emotions = [
                turn["detected_emotion"].get("primary_emotion", "neutral")
                for turn in list(self.conversation_history)[-3:]
            ]
            emotional_trajectory = recent_emotions
        
        # Determine emotional pattern
        if len(emotional_trajectory) >= 2:
            if all(emotion == emotional_trajectory[0] for emotion in emotional_trajectory):
                emotional_pattern = "consistent"
            elif len(set(emotional_trajectory)) == len(emotional_trajectory):
                emotional_pattern = "varied"
            else:
                emotional_pattern = "mixed"
        else:
            emotional_pattern = "initial"
        
        return {
            "current_emotion": current_emotion,
            "current_intensity": current_intensity,
            "emotional_trajectory": emotional_trajectory,
            "emotional_pattern": emotional_pattern,
            "emotional_stability": self._calculate_emotional_stability(emotional_trajectory)
        }
    
    def _calculate_emotional_stability(self, emotional_trajectory: List[str]) -> float:
        """Calculate emotional stability based on trajectory"""
        if len(emotional_trajectory) < 2:
            return 1.0
        
        # Simple stability calculation based on emotional changes
        changes = 0
        for i in range(1, len(emotional_trajectory)):
            if emotional_trajectory[i] != emotional_trajectory[i-1]:
                changes += 1
        
        stability = 1.0 - (changes / (len(emotional_trajectory) - 1))
        return stability
    
    async def _determine_context_type(self, topic_analysis: Dict[str, Any],
                                    reference_analysis: Dict[str, Any],
                                    continuity_analysis: Dict[str, Any]) -> str:
        """Determine the overall context type"""
        
        if topic_analysis["flow_type"] == "initial":
            return "new_conversation"
        elif reference_analysis["has_references"]:
            return "context_dependent"
        elif continuity_analysis["continuity_level"] == "high":
            return "continuation"
        elif topic_analysis["domain_change"]:
            return "topic_change"
        else:
            return "normal_flow"
    
    def _extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics from text (simplified approach)"""
        # This is a simplified topic extraction
        # In production, would use more sophisticated NLP
        words = text.lower().split()
        
        # Filter out common words
        common_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        key_words = [word for word in words if word not in common_words and len(word) > 3]
        
        return key_words[:5]  # Return top 5 key words
    
    async def _update_session_context(self, turn_record: Dict[str, Any], 
                                    context_analysis: Dict[str, Any]):
        """Update session context information"""
        
        # Update active domains
        domain = turn_record["domain"]
        if domain not in self.current_session["active_domains"]:
            self.current_session["active_domains"].append(domain)
        
        # Update conversation themes
        key_topics = self._extract_key_topics(turn_record["user_input"])
        self.current_session["conversation_themes"].extend(key_topics)
        
        # Update emotional trajectory
        emotion = turn_record["detected_emotion"].get("primary_emotion", "neutral")
        self.current_session["emotional_trajectory"].append({
            "emotion": emotion,
            "intensity": turn_record["detected_emotion"].get("emotion_intensity", 0.5),
            "timestamp": turn_record["timestamp"]
        })
    
    async def _update_user_preferences(self, turn_record: Dict[str, Any]):
        """Update user preferences based on conversation patterns"""
        
        domain = turn_record["domain"]
        
        # Update domain interests
        if domain not in self.user_preferences["domain_interests"]:
            self.user_preferences["domain_interests"][domain] = 0
        self.user_preferences["domain_interests"][domain] += 1
        
        # Update communication style preferences (simplified)
        user_input = turn_record["user_input"].lower()
        
        # Detect formality preference
        formal_indicators = ["please", "thank you", "could you", "would you"]
        informal_indicators = ["hey", "yeah", "ok", "cool"]
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in user_input)
        informal_count = sum(1 for indicator in informal_indicators if indicator in user_input)
        
        if formal_count > informal_count:
            self.user_preferences["communication_style"]["formality_level"] = "formal"
        elif informal_count > formal_count:
            self.user_preferences["communication_style"]["formality_level"] = "casual"
    
    async def _update_contextual_intelligence(self, context_analysis: Dict[str, Any]):
        """Update contextual intelligence metrics"""
        
        # Update topic coherence
        if context_analysis["topic_analysis"]["flow_type"] in ["normal_flow", "smooth_transition"]:
            self.contextual_intelligence["topic_coherence"] = min(
                self.contextual_intelligence["topic_coherence"] + 0.1, 1.0
            )
        
        # Update emotional consistency
        emotional_pattern = context_analysis["emotional_context"]["emotional_pattern"]
        if emotional_pattern == "consistent":
            self.contextual_intelligence["emotional_consistency"] = min(
                self.contextual_intelligence["emotional_consistency"] + 0.1, 1.0
            )
        
        # Update conversation depth
        if context_analysis["continuity_analysis"]["seeks_elaboration"]:
            self.contextual_intelligence["conversation_depth"] = min(
                self.contextual_intelligence["conversation_depth"] + 0.1, 1.0
            )
    
    async def _store_in_memory(self, turn_record: Dict[str, Any], 
                             context_analysis: Dict[str, Any]):
        """Store conversation turn in appropriate memory"""
        
        memory_item = {
            "turn_record": turn_record,
            "context_analysis": context_analysis,
            "storage_timestamp": datetime.now().isoformat()
        }
        
        # Store in short-term memory
        self.memory_types["short_term"]["content"].append(memory_item)
        
        # Store in medium-term if important
        if (turn_record["urgency_level"] in ["high", "critical"] or 
            context_analysis["reference_analysis"]["has_references"]):
            self.memory_types["medium_term"]["content"].append(memory_item)
        
        # Store in long-term if very important
        if (turn_record["urgency_level"] == "critical" or
            context_analysis["continuity_analysis"]["continuity_level"] == "high"):
            self.memory_types["long_term"]["content"].append(memory_item)
    
    async def _update_context_stats(self):
        """Update context management performance statistics"""
        self.context_stats["conversations_managed"] += 1
        
        if self.conversation_history:
            last_turn = self.conversation_history[-1]
            if len(self.conversation_history) > 1:
                prev_turn = self.conversation_history[-2]
                if last_turn["domain"] != prev_turn["domain"]:
                    self.context_stats["context_switches"] += 1
    
    async def get_relevant_context(self, current_input: str, 
                                 max_context_items: int = 5) -> Dict[str, Any]:
        """Get relevant context for the current input"""
        
        # Analyze current input for context needs
        input_lower = current_input.lower()
        
        # Check for reference patterns
        has_references = any(
            pattern in input_lower 
            for patterns in self.context_patterns["reference_patterns"].values()
            for pattern in patterns
        )
        
        relevant_context = {
            "recent_conversation": [],
            "relevant_domains": [],
            "emotional_context": {},
            "user_preferences": self.user_preferences,
            "session_info": self.current_session
        }
        
        # Get recent conversation if references detected
        if has_references and self.conversation_history:
            relevant_context["recent_conversation"] = [
                {
                    "turn_number": turn["turn_number"],
                    "domain": turn["domain"],
                    "user_input": turn["user_input"][:100] + "..." if len(turn["user_input"]) > 100 else turn["user_input"],
                    "timestamp": turn["timestamp"]
                }
                for turn in list(self.conversation_history)[-max_context_items:]
            ]
        
        # Get relevant domains
        if self.user_preferences["domain_interests"]:
            relevant_context["relevant_domains"] = sorted(
                self.user_preferences["domain_interests"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]  # Top 3 domains
        
        # Get emotional context
        if self.current_session["emotional_trajectory"]:
            recent_emotions = self.current_session["emotional_trajectory"][-3:]
            relevant_context["emotional_context"] = {
                "recent_emotions": recent_emotions,
                "dominant_emotion": max(
                    set(emotion["emotion"] for emotion in recent_emotions),
                    key=lambda x: sum(1 for emotion in recent_emotions if emotion["emotion"] == x)
                ) if recent_emotions else "neutral"
            }
        
        return relevant_context
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """Get context management performance statistics"""
        return {
            "performance_stats": self.context_stats,
            "session_info": {
                "session_duration": str(datetime.now() - self.current_session["start_time"]),
                "total_turns": len(self.conversation_history),
                "active_domains": len(self.current_session["active_domains"]),
                "conversation_themes": len(set(self.current_session["conversation_themes"]))
            },
            "memory_usage": {
                memory_type: len(config["content"])
                for memory_type, config in self.memory_types.items()
            },
            "contextual_intelligence": self.contextual_intelligence,
            "user_preferences": self.user_preferences
        }

# Convenience function for easy integration
async def manage_conversation_context(user_input: str, assistant_response: str,
                                    detected_emotion: Dict[str, Any], domain: str,
                                    context_manager: TARAContextManager = None) -> Dict[str, Any]:
    """
    Convenience function to manage conversation context
    
    Usage:
        context_manager = TARAContextManager()
        result = await manage_conversation_context(
            user_input="I'm still confused about what you said earlier",
            assistant_response="Let me clarify that for you...",
            detected_emotion={"primary_emotion": "confusion", "emotion_intensity": 0.6},
            domain="education",
            context_manager=context_manager
        )
    """
    if context_manager is None:
        context_manager = TARAContextManager()
    
    return await context_manager.add_conversation_turn(
        user_input, assistant_response, detected_emotion, domain
    ) 