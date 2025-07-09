import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
import random
import string
import uuid # New import
import os # New import
from datetime import datetime # New import

# Trinity Architecture imports
from trinity_core.agents.coordination.lightweight_mcp_v2 import LightweightMCPv2, MCPMessage
from trinity_core.intelligence_layer.intelligence.comprehensive_intelligence import TARAComprehensiveIntelligence
from trinity_core.core_components.config_manager import SmartTrinityConfigManager

# Initialize logger
logger = logging.getLogger(__name__)

class TrinityDataGenerator:
    """
    Intelligent Data Generator Agent - Trinity Architecture Optimization
    Responsible for generating high-quality, domain-specific training data.
    """
    def __init__(self, hub: Any): # Changed to Any to avoid circular import with IntelligenceHub for now
        self.hub = hub
        self.config_manager = hub.config_manager
        self.config = self.config_manager.get_config_dict()
        self.mcp = hub.mcp
        self.intelligence = hub.intelligence

    async def _generate_synthetic_data_for_domain(self, domain: str, category: str, sample_count: int) -> List[Dict[str, Any]]: # Changed return type
        """
        Generates synthetically realistic, domain-specific training data.
        This method will generate data that "looks" realistic for the given domain
        by mimicking real-world conversations and domain-related interactions.
        The output will be in a structured JSON format, similar to our existing training data.
        """
        logger.info(f"Generating {sample_count} synthetic conversations for domain: {domain} (Category: {category})")
        synthetic_conversations = []

        # Define templates for generating realistic conversations based on domain/category
        # These templates will be expanded to be more dynamic and varied.
        if domain == "healthcare":
            conversation_templates = [
                {
                    "scenario": "patient_symptoms",
                    "primary_emotion": "concerned",
                    "turns": [
                        {"role": "user", "content": "I've been having a persistent cough and fever for three days.", "emotion": "concerned", "intent": "symptom_inquiry"},
                        {"role": "assistant", "content": "I understand your concern. While I can't diagnose, persistent cough and fever warrant medical attention. Have you considered consulting a doctor?", "domain": "healthcare", "response_pattern": "medical_referral"},
                        {"role": "user", "content": "Should I go to an urgent care clinic or schedule an appointment with my GP?", "emotion": "uncertain", "intent": "action_guidance"},
                        {"role": "assistant", "content": "For persistent symptoms like yours, an urgent care clinic can provide immediate assessment. For ongoing care, your GP is ideal. Which aligns better with your current situation?", "domain": "healthcare", "response_pattern": "option_guidance"}
                    ]
                },
                {
                    "scenario": "medication_query",
                    "primary_emotion": "curious",
                    "turns": [
                        {"role": "user", "content": "What are the side effects of Ibuprofen?", "emotion": "curious", "intent": "medication_info"},
                        {"role": "assistant", "content": "Common side effects of Ibuprofen include stomach upset and dizziness. Please consult the medication leaflet or your pharmacist for comprehensive information.", "domain": "healthcare", "response_pattern": "info_and_referral"}
                    ]
                }
            ]
        elif domain == "shopping":
            conversation_templates = [
                {
                    "scenario": "product_inquiry",
                    "primary_emotion": "interested",
                    "turns": [
                        {"role": "user", "content": "Do you have the new XYZ smartphone in stock?", "emotion": "interested", "intent": "product_availability"},
                        {"role": "assistant", "content": "Yes, the XYZ smartphone is currently in stock. Would you like to know about its features or pricing?", "domain": "shopping", "response_pattern": "confirm_and_offer"}
                    ]
                },
                {
                    "scenario": "return_policy",
                    "primary_emotion": "frustrated",
                    "turns": [
                        {"role": "user", "content": "I need to return an item I bought last week. What's your return policy?", "emotion": "frustrated", "intent": "return_policy_query"},
                        {"role": "assistant", "content": "Our return policy allows returns within 30 days with a valid receipt. Was the item faulty or simply unwanted?", "domain": "shopping", "response_pattern": "policy_and_inquiry"}
                    ]
                }
            ]
        elif domain == "education":
             conversation_templates = [
                {
                    "scenario": "course_assistance",
                    "primary_emotion": "stressed",
                    "turns": [
                        {"role": "user", "content": "I'm struggling with the concepts in my advanced physics course.", "emotion": "stressed", "intent": "academic_help"},
                        {"role": "assistant", "content": "Physics can be challenging. Let's break down the concepts you're finding difficult. Are there specific topics you'd like to review?", "domain": "education", "response_pattern": "offer_assistance"}
                    ]
                }
            ]
        elif domain == "mental_health": # Mimicking the structure from DIALOGPT_LEARNING_PROCESS.md
            conversation_templates = [
                {
                    "scenario": "anxiety_support",
                    "primary_emotion": "anxious",
                    "turns": [
                        {"role": "user", "content": "I'm feeling really anxious about my presentation tomorrow.", "emotion": "anxious", "intent": "emotional_support"},
                        {"role": "assistant", "content": "I understand anxiety about presentations is very common. Your feelings are completely valid. Consider reaching out to a mental health professional for personalized coping strategies.", "domain": "mental_health", "response_pattern": "empathy_and_referral"},
                        {"role": "user", "content": "What are some quick relaxation techniques I can try?", "emotion": "seeking_solution", "intent": "technique_query"},
                        {"role": "assistant", "content": "Deep breathing exercises, progressive muscle relaxation, and mindfulness meditation can help. Would you like me to guide you through one?", "domain": "mental_health", "response_pattern": "suggest_techniques"}
                    ]
                },
                {
                    "scenario": "stress_management",
                    "primary_emotion": "stressed",
                    "turns": [
                        {"role": "user", "content": "I'm completely overwhelmed with my workload and feeling stressed.", "emotion": "stressed", "intent": "stress_relief"},
                        {"role": "assistant", "content": "It sounds like you're under a lot of pressure. Taking small breaks and prioritizing tasks can sometimes help. Have you tried any stress-reducing activities lately?", "domain": "mental_health", "response_pattern": "empathy_and_suggestion"}
                    ]
                }
            ]
        else: # Generic templates for other domains
            conversation_templates = [
                {
                    "scenario": "general_inquiry",
                    "primary_emotion": "neutral",
                    "turns": [
                        {"role": "user", "content": f"Tell me something interesting about {domain}.", "emotion": "curious", "intent": "information_request"},
                        {"role": "assistant", "content": f"The {domain} domain encompasses many fascinating aspects, from its historical development to its modern applications. What specifically are you curious about?", "domain": domain, "response_pattern": "general_info"}
                    ]
                }
            ]

        for i in range(sample_count):
            template = random.choice(conversation_templates)
            conversation_data = {
                "conversation_id": str(uuid.uuid4()),
                "domain": domain,
                "scenario": template["scenario"],
                "primary_emotion": template["primary_emotion"],
                "turns": template["turns"]
            }
            synthetic_conversations.append(conversation_data)
        
        logger.info(f"Generated {len(synthetic_conversations)} structured synthetic conversations for '{domain}'.")
        return synthetic_conversations

    async def generate_intelligent_data(self, request: Dict[str, Any], 
                                      intelligence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrates intelligent data generation based on the request,
        leveraging simulation, real data loading, or synthetic generation.
        """
        domain = request.get("domain")
        sample_count = request.get("sample_count", 5000)
        quality_target = request.get("quality_target", 0.99)
        is_simulation = request.get("simulation", False)
        generate_synthetic = request.get("generate_synthetic", False) # New flag
        
        domain_config = self.config_manager.get_tara_proven_params(domain) # Corrected method call
        category = domain_config.get("category") if domain_config else "N/A"

        # Determine the base output directory based on simulation status
        # If it's a simulation, use 'dev'. Otherwise, use 'production' (for real data or synthetic in non-simulation mode).
        if is_simulation:
            base_output_data_dir = Path(self.config["paths"]["data_training_base_dir"]) / "dev" / "training"
        else:
            base_output_data_dir = Path(self.config["paths"]["data_training_base_dir"]) / "production" / "training"

        logger.info(f"Hub preparing data generation for domain: {domain}")
        start_time = time.time()
        
        data_result = {}

        if generate_synthetic:
            logger.info(f"Attempting to generate SYNTHETIC training data for {domain}...")
            # This method now returns a list of conversation dicts
            raw_training_data = await self._generate_synthetic_data_for_domain(domain, category, sample_count)
            source_type = "synthetic"
            # Define a path for the raw synthetic data to be saved for validation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # For synthetic, save to dev/training/<category>/<domain> (optional domain subfolder)
            domain_dir = base_output_data_dir / category / domain # Added domain subfolder for synthetic/dev
            os.makedirs(domain_dir, exist_ok=True)
            raw_data_file_path = domain_dir / f"{domain}_raw_structured_data_{timestamp}.json"
            with open(raw_data_file_path, 'w', encoding='utf-8') as f:
                json.dump(raw_training_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Raw synthetic data saved for validation: {raw_data_file_path}")
            source_path = str(raw_data_file_path) # Use this as the source path for validation

        elif is_simulation:
            logger.info(f"Attempting to generate SIMULATED training data for {domain}...")
            raw_training_data = await self._generate_simulated_training_examples(domain, intelligence.get("data_needs", {}), sample_count)
            source_type = "simulated"
            # Define a path for the raw simulated data to be saved for validation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # For simulated, save to dev/training/<category>/<domain> (optional domain subfolder)
            domain_dir = base_output_data_dir / category / domain # Added domain subfolder for simulated/dev
            os.makedirs(domain_dir, exist_ok=True)
            raw_data_file_path = domain_dir / f"{domain}_simulated_raw_data_{timestamp}.json"
            with open(raw_data_file_path, 'w', encoding='utf-8') as f:
                json.dump(raw_training_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Raw simulated data saved for validation: {raw_data_file_path}")
            source_path = str(raw_data_file_path) # Use this as the source path for validation

        else:
            logger.info(f"Attempting to load REAL training data for {domain} from production path...")
            # For real data, load from production/training/<category>/<domain>
            real_data_path = base_output_data_dir / category / domain / f"colab_{domain}_training_data.json" # Production path
            
            # Fallback for 'shopping' domain to its specific colab filename if category path fails
            if not real_data_path.exists() and domain == "shopping":
                real_data_path = base_output_data_dir / category / domain / f"colab_shopping_training_data.json"
                logger.info(f"Adjusted path for shopping domain: {real_data_path}")

            try:
                if not real_data_path.exists():
                    raise FileNotFoundError(f"Real data file not found at: {real_data_path}. Set simulation=True or generate_synthetic=True if real data is not available.")
                
                with open(real_data_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                # Ensure raw_training_data is always a list of conversation dictionaries
                if isinstance(raw_data, dict) and "samples" in raw_data:
                    raw_training_data = raw_data["samples"]
                elif isinstance(raw_data, list):
                    raw_training_data = raw_data
                else:
                    raise ValueError(f"Unexpected data format in {real_data_path}. Expected a list or a dictionary with 'samples' key.")

                source_type = "real"
                raw_data_file_path = str(real_data_path) # Path to the real data file
                source_path = raw_data_file_path # Use this as the source path for validation

            except FileNotFoundError as e:
                logger.error(f"Error loading real data for {domain}: {e}")
                # Re-raise the FileNotFoundError if real data is expected and not found
                raise e
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {real_data_path}: {e}")
                raise ValueError(f"Invalid JSON format in {real_data_path}") from e
            except Exception as e:
                logger.error(f"An unexpected error occurred while loading real data for {domain}: {e}")
                raise

        # Ensure raw_training_data is always a list
        if not isinstance(raw_training_data, list):
            logger.error(f"Data generation for {domain} resulted in non-list type: {type(raw_training_data)}. Expected List[Dict].")
            raw_training_data = [] # Ensure it's an empty list to avoid further errors

        logger.debug(f"DEBUG (data_generator): Type of raw_training_data before _clean_generated_data: {type(raw_training_data)}")
        logger.debug(f"DEBUG (data_generator): Value of raw_training_data before _clean_generated_data: {raw_training_data[0] if raw_training_data else 'Empty'}")
        
        # Pass all conversation objects to cleaning, which will also handle formatting
        cleaned_examples = self._clean_generated_data(raw_training_data)
        
        logger.debug(f"DEBUG (data_generator): Type of cleaned_examples after _clean_generated_data: {type(cleaned_examples)}")
        logger.debug(f"DEBUG (data_generator): Value of cleaned_examples after _clean_generated_data: {cleaned_examples[0] if cleaned_examples else 'Empty'}")

        if not cleaned_examples:
            logger.error(f"No training examples generated for {domain} after cleaning. Source: {source_type} from {source_path}")
            data_result = {
                "success": False,
                "domain": domain,
                "error": "No training examples generated after cleaning.",
                "source_type": source_type,
                "source_path": source_path,
                "sample_count": 0,
                "quality_score": 0.0,
                "processing_time": time.time() - start_time
            }
            return data_result

        # Analyze quality of generated data
        quality_metrics = await self._assess_quality_metrics(domain) # This needs to be updated to use cleaned_examples
        quality_score = quality_metrics.get("overall_quality_score", 0.0)

        # Filter based on quality (example: only use 40% of generated data for training)
        # This is where the '2000 out of 5000 samples are valid' logic will come in.
        # For now, let's implement a simple pass-through. The actual filtering logic
        # will be implemented within _clean_generated_data or here based on quality_metrics.
        final_training_data = cleaned_examples # Placeholder for actual filtering

        logger.info(f"Successfully generated/loaded and cleaned {len(final_training_data)} training examples for '{domain}'.")
        
        data_result = {
            "success": True,
            "domain": domain,
            "source_type": source_type,
            "source_path": source_path, # Path to the raw data file (either generated or loaded)
            "sample_count": len(final_training_data),
            "quality_score": quality_score,
            "processing_time": time.time() - start_time,
            "insights": await self._extract_insights({"generated_data": final_training_data, "quality_metrics": quality_metrics}),
            "recommendations": await self._generate_recommendations({"generated_data": final_training_data, "quality_metrics": quality_metrics}),
            "training_examples": final_training_data # The actual data to be passed on
        }

        # Send message to MCP if needed (example: data generation complete)
        # await self.mcp.send_message(MCPMessage("data_generation_complete", data_result))
        return data_result

    async def _generate_simulated_training_examples(self, domain: str, user_needs: Dict[str, Any], target_sample_count: int) -> List[Dict[str, Any]]:
        """
        Generates simulated training examples, now structured as conversations.
        This is for internal testing when real or synthetic data is not used.
        """
        logger.info(f"Generating {target_sample_count} simulated conversations for domain: {domain}")
        simulated_conversations = []

        for i in range(target_sample_count):
            conversation_id = str(uuid.uuid4())
            user_content = f"Simulated user input for {domain} example {i+1}. {user_needs.get('context', '')}"
            assistant_content = f"Simulated assistant response for {domain} example {i+1}. Data generated for testing purposes."

            conversation = {
                "conversation_id": conversation_id,
                "domain": domain,
                "scenario": "simulation_test",
                "primary_emotion": "neutral",
                "turns": [
                    {"role": "user", "content": user_content, "emotion": "neutral", "intent": "simulated_query"},
                    {"role": "assistant", "content": assistant_content, "emotion": "neutral", "intent": "simulated_response"}
                ]
            }
            simulated_conversations.append(conversation)
        
        logger.info(f"Generated {len(simulated_conversations)} simulated conversations for '{domain}'.")
        return simulated_conversations

    def _clean_generated_data(self, conversations: List[Dict[str, Any]]) -> List[str]: # Changed input and output type
        """
        Cleans and formats generated conversation data into a flat list of strings
        (e.g., 'User: ... Assistant: ...') as expected by the training pipeline.
        This method also performs quality filtering to ensure only valid samples are returned.
        
        Quality checks include:
        - Each conversation must have at least one turn.
        - Each turn must have 'role' and 'content' and non-empty content.
        - Conversations should alternate between 'user' and 'assistant' roles.
        - Minimum length for each turn's content.
        - Basic check for repetitive content (future enhancement).
        """
        logger.info(f"Cleaning and validating {len(conversations)} generated conversations.")
        cleaned_and_formatted_samples = []
        min_turn_length = 5 # Minimum characters for a turn's content

        for convo in conversations:
            convo_id = convo.get("conversation_id", "unknown_id")
            turns = convo.get("turns", [])
            formatted_parts = []
            is_valid_convo = True
            last_role = None

            if not turns:
                logger.warning(f"Invalid conversation {convo_id}: No turns found.")
                is_valid_convo = False
                continue
            
            for i, turn in enumerate(turns):
                role = turn.get("role")
                content = turn.get("content", "").strip()

                if not role or not content or len(content) < min_turn_length:
                    logger.warning(f"Invalid turn in {convo_id}: Missing role/content or content too short. Turn: {turn}")
                    is_valid_convo = False
                    break

                # Check for alternating roles (simple check, can be made more robust)
                if i > 0 and role == last_role:
                    logger.warning(f"Invalid turn sequence in {convo_id}: Consecutive turns with same role '{role}'. Turn: {turn}")
                    is_valid_convo = False
                    break
                
                if role == "user":
                    formatted_parts.append(f"User: {content}")
                elif role == "assistant":
                    formatted_parts.append(f"Assistant: {content}")
                else:
                    logger.warning(f"Invalid role in {convo_id}: Unknown role '{role}'. Turn: {turn}")
                    is_valid_convo = False
                    break
                last_role = role
            
            if is_valid_convo and formatted_parts:
                # Add special tokens as per DIALOGPT_LEARNING_PROCESS.md
                formatted_text = "\n".join(formatted_parts)
                # Assuming tokenizer.bos_token and tokenizer.eos_token are available if this were integrated with a tokenizer
                # For now, use simple markers as strings
                final_formatted_sample = f"<|beginoftext|>{formatted_text}<|endoftext|>"
                cleaned_and_formatted_samples.append(final_formatted_sample)
            elif is_valid_convo: # This case means formatted_parts was empty due to some edge case not caught by other checks
                logger.warning(f"Conversation {convo_id} passed checks but had no formatted parts.")

        logger.info(f"Finished cleaning. {len(cleaned_and_formatted_samples)} valid samples out of {len(conversations)} original conversations.")
        return cleaned_and_formatted_samples

    async def _assess_quality_metrics(self, domain: str) -> Dict[str, Any]:
        """
        Assesses the quality metrics of generated data. This is a placeholder
        and would involve more sophisticated NLP-based quality assessments
        in a production system.
        """
        # For now, return a placeholder score. This would be dynamically calculated
        # based on criteria like relevance, coherence, diversity, and adherence to domain patterns.
        # In a real system, this might involve a separate model or rule-based system.
        return {
            "overall_quality_score": random.uniform(0.7, 0.99), # Simulate a quality score
            "domain_relevance_score": random.uniform(0.8, 0.99),
            "coherence_score": random.uniform(0.75, 0.95),
            "diversity_score": random.uniform(0.6, 0.9),
            "validation_pass_rate": len(self.hub.shared_context.get("cleaned_samples", [])) / max(1, len(self.hub.shared_context.get("raw_samples", [])))
        }

    async def _suggest_optimizations(self, intelligence: Dict[str, Any]) -> List[str]:
        """
        Suggests optimizations for data generation based on quality metrics and intelligence.
        """
        suggestions = []
        quality_score = intelligence.get("quality_metrics", {}).get("overall_quality_score", 0.0)

        if quality_score < 0.8:
            suggestions.append("Increase diversity in synthetic data templates.")
            suggestions.append("Refine prompt engineering for LLM-based generation.")
        if intelligence.get("quality_metrics", {}).get("domain_relevance_score", 0.0) < 0.9:
            suggestions.append("Ensure domain-specific terminology is more prevalent.")
        
        return suggestions if suggestions else ["Data generation looks optimized for current settings."]

    async def _extract_insights(self, generated_data: Dict[str, Any]) -> List[str]:
        """
        Extracts insights from the generated data.
        """
        insights = []
        if generated_data.get("quality_metrics", {}).get("overall_quality_score", 0.0) > 0.9:
            insights.append("High quality synthetic data generated, indicating robust templates.")
        insights.append(f"Generated {len(generated_data.get("generated_data", []))} samples for training.")
        return insights

    async def _generate_recommendations(self, generated_data: Dict[str, Any]) -> List[str]:
        """
        Generates recommendations for future data generation and training.
        """
        recommendations = []
        recommendations.append("Monitor model performance with this data and iterate on templates.")
        if generated_data.get("quality_metrics", {}).get("validation_pass_rate", 0.0) < 0.5:
            recommendations.append("Review cleaning rules for potential false negatives.")
        return recommendations

    def _generate_cache_key(self, request: Dict[str, Any], intelligence: Dict[str, Any]) -> str:
        """
        Generates a cache key for data generation requests.
        """
        domain = request.get("domain", "general")
        sample_count = request.get("sample_count", 0)
        simulation = request.get("simulation", False)
        synthetic = request.get("generate_synthetic", False)
        return f"data_gen_{domain}_{sample_count}_{simulation}_{synthetic}" 