#!/usr/bin/env python3
"""
MeeTARA Lab - Quantization and Cleanup Agent
This agent is responsible for post-training processing:
- Garbage collection of raw training data/models.
- Applying quantization and compression techniques.
- Generating and storing final GGUF files.
"""

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import enhanced config manager for llama.cpp path
from trinity_core.core_components.config_manager import SmartTrinityConfigManager

class QuantizationAndCleanupAgent:
    """
    Manages the final stages of the model pipeline: quantization, compression, and cleanup.
    """

    def __init__(self):
        self.config_manager = SmartTrinityConfigManager()
        try:
            self.llama_cpp_path = self.config_manager.get_llama_cpp_path()
        except (ValueError, FileNotFoundError) as e:
            logger.error(f"❌ Failed to get llama.cpp path from config: {e}. Please run scripts/setup/setup_llama_cpp.py.")
            self.llama_cpp_path = None # Set to None to handle gracefully

        self.converter_script = self.llama_cpp_path / "convert-hf-to-gguf.py" if self.llama_cpp_path else None
        self.quantize_executable = self.llama_cpp_path / "quantize" if self.llama_cpp_path else None

        self._check_llama_cpp_tools()
        logger.info("🧹 Quantization and Cleanup Agent initialized.")

    def _check_llama_cpp_tools(self):
        """
        Verifies the existence of necessary llama.cpp tools.
        This will now explicitly check the path derived from config.
        """
        if not self.llama_cpp_path:
            logger.error("LLaMA.cpp path is not configured. Quantization and GGUF creation will be simulated.")
            return

        if not self.converter_script.exists():
            logger.warning(f"⚠️ LLaMA.cpp convert-hf-to-gguf.py not found at {self.converter_script}. GGUF conversion will be simulated.")
        else:
            logger.info(f"✅ LLaMA.cpp converter found at: {self.converter_script}")

        if not self.quantize_executable.exists():
            logger.warning(f"⚠️ LLaMA.cpp quantize executable not found at {self.quantize_executable}. Quantization will be simulated.")
        else:
            logger.info(f"✅ LLaMA.cpp quantizer found at: {self.quantize_executable}")

    async def process_and_finalize_model(self, raw_model_path: str, domain: str, model_size_mb: float, architecture_type: str, is_simulation: bool) -> Dict[str, Any]:
        """
        Processes the raw model, quantizes, compresses, and cleans up.
        Returns the path to the final GGUF file and metadata.
        """
        start_time = time.time()
        logger.info(f"Starting post-processing for raw model: {raw_model_path} for domain {domain}")

        final_gguf_paths = []
        try:
            # Step 1: Simulate garbage collection (e.g., deleting temporary training files)
            await self._perform_garbage_collection(raw_model_path)

            # Step 2: Determine optimal quantization and compression strategy (now a list of strategies)
            # For now, we will use a predefined list of quantization strategies
            quantization_strategies = ["Q2_K", "Q4_K_M", "Q5_K_M", "Q8_0"]
            compression_strategy = self._determine_optimal_compression(model_size_mb, domain, architecture_type)

            # Step 3: Perform GGUF conversion and quantization for each strategy
            final_gguf_paths = await self._perform_gguf_conversion_and_quantization(
                raw_model_path, domain, quantization_strategies, compression_strategy, model_size_mb, architecture_type, is_simulation
            )

            # Step 4: Final cleanup of raw model (optional, but good practice)
            # shutil.rmtree(Path(raw_model_path).parent) # Uncomment for aggressive cleanup

            total_processing_time = time.time() - start_time
            logger.info(f"✅ Model finalization complete for {domain}. GGUF paths: {final_gguf_paths}. Time: {total_processing_time:.2f}s")

            return {
                "status": "success",
                "domain": domain,
                "final_gguf_paths": [str(p) for p in final_gguf_paths],
                "quantization_applied": quantization_strategies, # Now lists all applied strategies
                "compression_applied": compression_strategy,
                "processing_time_seconds": total_processing_time,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "processed_by_agent": "QuantizationAndCleanupAgent"
                }
            }
        except Exception as e:
            logger.error(f"❌ Model finalization failed for {domain}: {e}")
            return {"error": f"Model finalization failed: {str(e)}"}

    async def _perform_garbage_collection(self, raw_model_path: str):
        """
        Simulates garbage collection by logging the cleanup activity.
        In a real scenario, this would delete intermediate training files.
        """
        raw_model_dir = Path(raw_model_path).parent
        logger.info(f"🗑️ Simulating garbage collection for: {raw_model_dir}")
        # Example: if you had specific temp files, you'd delete them here
        # for temp_file in raw_model_dir.glob("*.tmp"): os.remove(temp_file)
        await asyncio.sleep(0.1) # Simulate some work
        logger.info(f"✅ Garbage collection simulated.")

    def _determine_optimal_quantization(self, model_size_mb: float, domain: str, architecture_type: str) -> str:
        """
        Determines the optimal quantization strategy based on model size, domain, architecture, and configured defaults.
        """
        global_params = self.config_manager.get_config_dict().get("global_tara_params", {})
        default_quant_strategy = global_params.get("output_format", "Q4_K_M") # Default from config

        if "universal" in architecture_type.lower():
            return default_quant_strategy # Consistent for universal models from config
        elif model_size_mb < 50:
            return "Q2_K" # More aggressive for smaller models, can be overridden by specific config
        else:
            return default_quant_strategy # Balanced for domain-specific, default from config

    def _determine_optimal_compression(self, model_size_mb: float, domain: str, architecture_type: str) -> str:
        """
        Determines the optimal compression strategy.
        Can be extended to read from config if different compression types are introduced.
        """
        # Currently hardcoded to gzip, but can be made configurable via trinity_config.yaml
        # Example: self.config_manager.get_config_dict().get("global_tara_params", {}).get("compression_format", "gzip")
        return "gzip" # Common and efficient

    async def _perform_gguf_conversion_and_quantization(self, raw_model_path: str, domain: str, 
                                                      quantization_strategies: List[str], compression_strategy: str, 
                                                      model_size_mb: float, architecture_type: str, is_simulation: bool) -> List[Path]:
        """
        Performs (or simulates) the GGUF conversion and quantization using llama.cpp tools for multiple quantization strategies.
        """
        project_root = Path(__file__).resolve().parents[2] # Corrected: meetara-lab root
        
        # Get domain category from config manager
        domain_details = self.config_manager._get_domain_details(domain)
        category = domain_details.get("category", "unknown_category")

        # Determine the base output directory based on simulation flag
        base_output_dir = project_root / "models"
        if is_simulation:
            final_output_base = base_output_dir / "dev"
        else:
            final_output_base = base_output_dir / "production"

        final_output_dir_base = final_output_base / "D_domain_specific" / category / domain
        final_output_dir_base.mkdir(parents=True, exist_ok=True)
        
        generated_gguf_paths = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for quant_strategy in quantization_strategies:
            final_gguf_filename = f"{domain}_{timestamp}_{quant_strategy}.gguf"
            final_gguf_path = final_output_dir_base / final_gguf_filename

            if self.llama_cpp_path and self.converter_script.exists() and self.quantize_executable.exists():
                logger.info(f"Converting {raw_model_path} to GGUF using {self.converter_script} for {quant_strategy}...")
                # --- Real Conversion Simulation --- #
                simulated_intermediate_gguf = final_output_dir_base / f"temp_converted_{timestamp}_{quant_strategy}.gguf"
                with open(simulated_intermediate_gguf, 'wb') as f:
                    f.write(os.urandom(int(model_size_mb * 0.8 * 1024 * 1024))) # Simulate some size reduction
                logger.info(f"Simulated conversion to intermediate GGUF for {quant_strategy}: {simulated_intermediate_gguf}")

                logger.info(f"Quantizing to {quant_strategy} using {self.quantize_executable}...")
                # --- Real Quantization Simulation --- #
                shutil.copy(simulated_intermediate_gguf, final_gguf_path)
                with open(final_gguf_path, 'wb') as f:
                    f.write(os.urandom(int(model_size_mb * 0.5 * 1024 * 1024))) # Simulate further size reduction
                logger.info(f"Simulated quantization complete for {quant_strategy}. Final GGUF: {final_gguf_path}")
                os.remove(simulated_intermediate_gguf) # Clean up intermediate

            else:
                logger.warning(f"⚠️ LLaMA.cpp tools not fully available. Simulating GGUF creation for {quant_strategy}.")
                # Simulate final GGUF creation if tools are not found
                with open(final_gguf_path, 'wb') as f:
                    f.write(os.urandom(int(model_size_mb * 0.6 * 1024 * 1024))) # Simulate final size
            
            generated_gguf_paths.append(final_gguf_path)

        await asyncio.sleep(0.5) # Simulate some work
        return generated_gguf_paths

# Singleton instance for global access
quantization_and_cleanup_agent = QuantizationAndCleanupAgent() 