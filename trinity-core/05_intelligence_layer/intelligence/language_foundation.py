#!/usr/bin/env python3
"""
TARA Core Intelligence - Language Foundation
Core knowledge of letters, numbers, special characters, and languages
The fundamental building blocks of all human communication
"""

import re
import string
import unicodedata
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
import asyncio

class TARALanguageFoundation:
    """
    TARA's foundational understanding of language elements
    Letters, numbers, special characters, and linguistic patterns
    """
    
    def __init__(self):
        # Character knowledge base
        self.character_knowledge = {
            "letters": {
                "latin": {
                    "uppercase": set(string.ascii_uppercase),
                    "lowercase": set(string.ascii_lowercase),
                    "accented": self._get_accented_letters(),
                    "total_count": 52  # 26 uppercase + 26 lowercase
                },
                "numbers": {
                    "digits": set(string.digits),
                    "roman_numerals": {"I", "V", "X", "L", "C", "D", "M"},
                    "total_count": 10
                },
                "special_characters": {
                    "punctuation": set(string.punctuation),
                    "whitespace": set(string.whitespace),
                    "symbols": self._get_common_symbols(),
                    "total_count": len(string.punctuation) + len(string.whitespace)
                }
            }
        }
        
        # Language patterns and structures
        self.language_patterns = {
            "sentence_structures": {
                "declarative": r"^[A-Z][^.!?]*[.!?]$",
                "interrogative": r"^[A-Z][^?]*\?$",
                "exclamatory": r"^[A-Z][^!]*!$",
                "imperative": r"^[A-Z][^.!?]*[.!]$"
            },
            
            "word_patterns": {
                "simple_word": r"^[a-zA-Z]+$",
                "compound_word": r"^[a-zA-Z]+-[a-zA-Z]+$",
                "acronym": r"^[A-Z]{2,}$",
                "abbreviation": r"^[A-Z][a-z]*\.$"
            },
            
            "number_patterns": {
                "integer": r"^-?\d+$",
                "decimal": r"^-?\d+\.\d+$",
                "percentage": r"^-?\d+(\.\d+)?%$",
                "currency": r"^\$-?\d+(\.\d{2})?$",
                "phone": r"^\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$",
                "date": r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$"
            }
        }
        
        # Linguistic intelligence
        self.linguistic_intelligence = {
            "communication_intent": {
                "question_indicators": ["what", "how", "why", "when", "where", "who", "which", "can", "could", "would", "should"],
                "request_indicators": ["please", "could you", "would you", "can you", "help me", "I need"],
                "emotion_indicators": ["feel", "feeling", "emotion", "happy", "sad", "angry", "excited", "worried"],
                "urgency_indicators": ["urgent", "emergency", "asap", "immediately", "now", "help", "crisis"]
            },
            
            "politeness_markers": {
                "formal": ["please", "thank you", "excuse me", "pardon", "sir", "madam", "respectfully"],
                "informal": ["thanks", "hey", "hi", "yeah", "okay", "cool", "awesome"],
                "apologetic": ["sorry", "apologize", "my mistake", "I'm afraid", "unfortunately"]
            },
            
            "complexity_indicators": {
                "simple": ["basic", "simple", "easy", "quick", "brief"],
                "moderate": ["detailed", "explain", "understand", "learn"],
                "complex": ["advanced", "sophisticated", "comprehensive", "in-depth", "technical"]
            }
        }
        
        # Multi-language support patterns
        self.language_support = {
            "detected_languages": {
                "english": {
                    "common_words": ["the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"],
                    "articles": ["a", "an", "the"],
                    "pronouns": ["I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"],
                    "greeting_patterns": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
                },
                "spanish": {
                    "common_words": ["el", "la", "los", "las", "y", "o", "pero", "en", "de", "con", "por", "para"],
                    "articles": ["el", "la", "los", "las", "un", "una", "unos", "unas"],
                    "greeting_patterns": ["hola", "buenos días", "buenas tardes", "buenas noches"]
                },
                "french": {
                    "common_words": ["le", "la", "les", "et", "ou", "mais", "dans", "de", "avec", "par", "pour"],
                    "articles": ["le", "la", "les", "un", "une", "des"],
                    "greeting_patterns": ["bonjour", "bonsoir", "salut"]
                }
            }
        }
        
        # Text analysis capabilities
        self.analysis_capabilities = {
            "readability": {
                "flesch_kincaid_weights": {"sentence_length": 1.015, "syllable_count": 84.6, "constant": 206.835},
                "complexity_thresholds": {"simple": 90, "moderate": 70, "complex": 50}
            },
            
            "sentiment_indicators": {
                "positive": ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "love", "like", "enjoy"],
                "negative": ["bad", "terrible", "awful", "hate", "dislike", "problem", "issue", "wrong", "error"],
                "neutral": ["okay", "fine", "normal", "standard", "typical", "average", "regular"]
            }
        }
        
        # Performance tracking
        self.foundation_stats = {
            "text_analyses": 0,
            "language_detections": 0,
            "pattern_matches": 0,
            "character_classifications": 0
        }
    
    def _get_accented_letters(self) -> Set[str]:
        """Get common accented letters"""
        accented = set()
        for char in "àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ":
            accented.add(char)
            accented.add(char.upper())
        return accented
    
    def _get_common_symbols(self) -> Set[str]:
        """Get common symbols beyond punctuation"""
        return {"@", "#", "$", "%", "&", "*", "+", "=", "~", "|", "\\", "/", "<", ">", "^", "_", "`"}
    
    async def analyze_text_foundation(self, text: str) -> Dict[str, Any]:
        """
        Analyze text at the foundational level - characters, patterns, structure
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict containing foundational analysis results
        """
        
        # Character analysis
        char_analysis = await self._analyze_characters(text)
        
        # Pattern recognition
        pattern_analysis = await self._recognize_patterns(text)
        
        # Language detection
        language_analysis = await self._detect_language(text)
        
        # Structure analysis
        structure_analysis = await self._analyze_structure(text)
        
        # Communication intent analysis
        intent_analysis = await self._analyze_communication_intent(text)
        
        # Readability analysis
        readability_analysis = await self._analyze_readability(text)
        
        result = {
            "text_length": len(text),
            "character_analysis": char_analysis,
            "pattern_analysis": pattern_analysis,
            "language_analysis": language_analysis,
            "structure_analysis": structure_analysis,
            "intent_analysis": intent_analysis,
            "readability_analysis": readability_analysis,
            "foundation_timestamp": datetime.now().isoformat(),
            "success": True
        }
        
        # Update performance stats
        await self._update_foundation_stats()
        
        return result
    
    async def _analyze_characters(self, text: str) -> Dict[str, Any]:
        """Analyze character composition of text"""
        
        char_counts = {
            "letters": 0,
            "digits": 0,
            "punctuation": 0,
            "whitespace": 0,
            "special_symbols": 0,
            "accented_letters": 0,
            "total_characters": len(text)
        }
        
        character_details = {
            "uppercase_letters": 0,
            "lowercase_letters": 0,
            "unique_characters": set(),
            "character_frequency": {},
            "non_ascii_characters": 0
        }
        
        for char in text:
            character_details["unique_characters"].add(char)
            
            # Count character frequency
            if char not in character_details["character_frequency"]:
                character_details["character_frequency"][char] = 0
            character_details["character_frequency"][char] += 1
            
            # Classify character
            if char.isalpha():
                char_counts["letters"] += 1
                if char.isupper():
                    character_details["uppercase_letters"] += 1
                else:
                    character_details["lowercase_letters"] += 1
                
                # Check for accented letters
                if char in self.character_knowledge["letters"]["latin"]["accented"]:
                    char_counts["accented_letters"] += 1
                    
            elif char.isdigit():
                char_counts["digits"] += 1
            elif char in string.punctuation:
                char_counts["punctuation"] += 1
            elif char.isspace():
                char_counts["whitespace"] += 1
            else:
                char_counts["special_symbols"] += 1
            
            # Check for non-ASCII
            if ord(char) > 127:
                character_details["non_ascii_characters"] += 1
        
        # Calculate character ratios
        total_chars = len(text)
        character_ratios = {
            "letter_ratio": char_counts["letters"] / total_chars if total_chars > 0 else 0,
            "digit_ratio": char_counts["digits"] / total_chars if total_chars > 0 else 0,
            "punctuation_ratio": char_counts["punctuation"] / total_chars if total_chars > 0 else 0,
            "whitespace_ratio": char_counts["whitespace"] / total_chars if total_chars > 0 else 0
        }
        
        return {
            "character_counts": char_counts,
            "character_details": {
                **character_details,
                "unique_characters": len(character_details["unique_characters"]),
                "character_frequency": dict(sorted(character_details["character_frequency"].items(), 
                                                 key=lambda x: x[1], reverse=True)[:10])  # Top 10
            },
            "character_ratios": character_ratios
        }
    
    async def _recognize_patterns(self, text: str) -> Dict[str, Any]:
        """Recognize linguistic and structural patterns in text"""
        
        pattern_matches = {
            "sentence_types": {},
            "word_patterns": {},
            "number_patterns": {}
        }
        
        # Sentence pattern recognition
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                for pattern_type, pattern in self.language_patterns["sentence_structures"].items():
                    if re.match(pattern, sentence):
                        if pattern_type not in pattern_matches["sentence_types"]:
                            pattern_matches["sentence_types"][pattern_type] = 0
                        pattern_matches["sentence_types"][pattern_type] += 1
        
        # Word pattern recognition
        words = re.findall(r'\b\w+\b', text)
        for word in words:
            for pattern_type, pattern in self.language_patterns["word_patterns"].items():
                if re.match(pattern, word):
                    if pattern_type not in pattern_matches["word_patterns"]:
                        pattern_matches["word_patterns"][pattern_type] = 0
                    pattern_matches["word_patterns"][pattern_type] += 1
        
        # Number pattern recognition
        for pattern_type, pattern in self.language_patterns["number_patterns"].items():
            matches = re.findall(pattern, text)
            if matches:
                pattern_matches["number_patterns"][pattern_type] = len(matches)
        
        return pattern_matches
    
    async def _detect_language(self, text: str) -> Dict[str, Any]:
        """Detect the primary language of the text"""
        
        text_lower = text.lower()
        language_scores = {}
        
        # Simple language detection based on common words
        for language, config in self.language_support["detected_languages"].items():
            score = 0
            word_matches = 0
            
            # Check common words
            for common_word in config["common_words"]:
                if f" {common_word} " in f" {text_lower} ":
                    score += 2
                    word_matches += 1
            
            # Check articles
            for article in config["articles"]:
                if f" {article} " in f" {text_lower} ":
                    score += 1
                    word_matches += 1
            
            # Check greeting patterns
            for greeting in config["greeting_patterns"]:
                if greeting in text_lower:
                    score += 3
                    word_matches += 1
            
            language_scores[language] = {
                "score": score,
                "word_matches": word_matches,
                "confidence": min(score / 10, 1.0)  # Normalize to 0-1
            }
        
        # Determine primary language
        if language_scores:
            primary_language = max(language_scores.items(), key=lambda x: x[1]["score"])
            detected_language = primary_language[0]
            confidence = primary_language[1]["confidence"]
        else:
            detected_language = "unknown"
            confidence = 0.0
        
        return {
            "detected_language": detected_language,
            "confidence": confidence,
            "all_language_scores": language_scores,
            "is_multilingual": len([lang for lang, data in language_scores.items() if data["score"] > 0]) > 1
        }
    
    async def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the structural elements of text"""
        
        # Basic structure counts
        structure_counts = {
            "sentences": len(re.split(r'[.!?]+', text)) - 1,  # -1 for empty last element
            "words": len(re.findall(r'\b\w+\b', text)),
            "paragraphs": len([p for p in text.split('\n\n') if p.strip()]),
            "lines": len(text.split('\n'))
        }
        
        # Average calculations
        structure_averages = {
            "words_per_sentence": 0,
            "characters_per_word": 0,
            "sentences_per_paragraph": 0
        }
        
        if structure_counts["sentences"] > 0:
            structure_averages["words_per_sentence"] = structure_counts["words"] / structure_counts["sentences"]
        
        if structure_counts["words"] > 0:
            structure_averages["characters_per_word"] = len(re.sub(r'\s+', '', text)) / structure_counts["words"]
        
        if structure_counts["paragraphs"] > 0:
            structure_averages["sentences_per_paragraph"] = structure_counts["sentences"] / structure_counts["paragraphs"]
        
        return {
            "structure_counts": structure_counts,
            "structure_averages": structure_averages,
            "text_density": structure_counts["words"] / len(text) if len(text) > 0 else 0
        }
    
    async def _analyze_communication_intent(self, text: str) -> Dict[str, Any]:
        """Analyze the communication intent of the text"""
        
        text_lower = text.lower()
        intent_indicators = {
            "question_intent": 0,
            "request_intent": 0,
            "emotion_intent": 0,
            "urgency_intent": 0
        }
        
        # Check for question indicators
        for indicator in self.linguistic_intelligence["communication_intent"]["question_indicators"]:
            if indicator in text_lower:
                intent_indicators["question_intent"] += 1
        
        # Check for request indicators
        for indicator in self.linguistic_intelligence["communication_intent"]["request_indicators"]:
            if indicator in text_lower:
                intent_indicators["request_intent"] += 1
        
        # Check for emotion indicators
        for indicator in self.linguistic_intelligence["communication_intent"]["emotion_indicators"]:
            if indicator in text_lower:
                intent_indicators["emotion_intent"] += 1
        
        # Check for urgency indicators
        for indicator in self.linguistic_intelligence["communication_intent"]["urgency_indicators"]:
            if indicator in text_lower:
                intent_indicators["urgency_intent"] += 1
        
        # Determine primary intent
        primary_intent = max(intent_indicators.items(), key=lambda x: x[1])
        
        return {
            "intent_indicators": intent_indicators,
            "primary_intent": primary_intent[0],
            "intent_strength": primary_intent[1],
            "has_question": "?" in text,
            "has_exclamation": "!" in text,
            "politeness_level": self._assess_politeness(text_lower)
        }
    
    def _assess_politeness(self, text_lower: str) -> str:
        """Assess the politeness level of the text"""
        
        formal_count = sum(1 for marker in self.linguistic_intelligence["politeness_markers"]["formal"] 
                          if marker in text_lower)
        informal_count = sum(1 for marker in self.linguistic_intelligence["politeness_markers"]["informal"] 
                           if marker in text_lower)
        
        if formal_count > informal_count:
            return "formal"
        elif informal_count > formal_count:
            return "informal"
        else:
            return "neutral"
    
    async def _analyze_readability(self, text: str) -> Dict[str, Any]:
        """Analyze text readability and complexity"""
        
        # Basic readability metrics
        sentences = len(re.split(r'[.!?]+', text)) - 1
        words = len(re.findall(r'\b\w+\b', text))
        syllables = self._count_syllables(text)
        
        # Flesch-Kincaid readability score (simplified)
        if sentences > 0 and words > 0:
            avg_sentence_length = words / sentences
            avg_syllables_per_word = syllables / words
            
            flesch_score = (206.835 - 
                          (1.015 * avg_sentence_length) - 
                          (84.6 * avg_syllables_per_word))
        else:
            flesch_score = 0
        
        # Determine complexity level
        if flesch_score >= 90:
            complexity_level = "very_easy"
        elif flesch_score >= 80:
            complexity_level = "easy"
        elif flesch_score >= 70:
            complexity_level = "fairly_easy"
        elif flesch_score >= 60:
            complexity_level = "standard"
        elif flesch_score >= 50:
            complexity_level = "fairly_difficult"
        elif flesch_score >= 30:
            complexity_level = "difficult"
        else:
            complexity_level = "very_difficult"
        
        return {
            "flesch_score": round(flesch_score, 2),
            "complexity_level": complexity_level,
            "average_sentence_length": round(words / sentences, 2) if sentences > 0 else 0,
            "average_syllables_per_word": round(syllables / words, 2) if words > 0 else 0,
            "total_syllables": syllables
        }
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in text (simplified approach)"""
        
        vowels = "aeiouy"
        syllable_count = 0
        words = re.findall(r'\b\w+\b', text.lower())
        
        for word in words:
            word_syllables = 0
            prev_was_vowel = False
            
            for char in word:
                if char in vowels:
                    if not prev_was_vowel:
                        word_syllables += 1
                    prev_was_vowel = True
                else:
                    prev_was_vowel = False
            
            # Handle silent 'e'
            if word.endswith('e') and word_syllables > 1:
                word_syllables -= 1
            
            # Every word has at least one syllable
            if word_syllables == 0:
                word_syllables = 1
            
            syllable_count += word_syllables
        
        return syllable_count
    
    async def _update_foundation_stats(self):
        """Update language foundation performance statistics"""
        self.foundation_stats["text_analyses"] += 1
        self.foundation_stats["language_detections"] += 1
        self.foundation_stats["pattern_matches"] += 1
        self.foundation_stats["character_classifications"] += 1
    
    def get_foundation_statistics(self) -> Dict[str, Any]:
        """Get language foundation performance statistics"""
        return {
            "performance_stats": self.foundation_stats,
            "knowledge_base_size": {
                "supported_languages": len(self.language_support["detected_languages"]),
                "pattern_types": len(self.language_patterns),
                "character_categories": len(self.character_knowledge["letters"])
            },
            "capabilities": {
                "character_analysis": True,
                "pattern_recognition": True,
                "language_detection": True,
                "structure_analysis": True,
                "intent_analysis": True,
                "readability_analysis": True
            }
        }

# Convenience function for easy integration
async def analyze_text_foundation(text: str) -> Dict[str, Any]:
    """
    Convenience function to analyze text at the foundational level
    
    Usage:
        result = await analyze_text_foundation("Hello, how are you feeling today?")
        language = result["language_analysis"]["detected_language"]  # "english"
        intent = result["intent_analysis"]["primary_intent"]  # "question_intent"
        complexity = result["readability_analysis"]["complexity_level"]  # "easy"
    """
    foundation = TARALanguageFoundation()
    return await foundation.analyze_text_foundation(text) 