#!/usr/bin/env python3
"""
MeeTARA Lab - Simple GGUF Converter
DEMONSTRATES SMART AGENT ARCHITECTURE

🎯 DESIGN PRINCIPLE:
"Agents are smart, scripts are simple"

This script is intentionally MINIMAL - all intelligence lives in the agent.
The script just:
1. Loads data
2. Calls intelligent agent
3. Reports results

NO hardcoded values, NO complex logic, NO compression/quantization decisions.
The IntelligentModelFactory agent handles ALL the intelligence.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add trinity-core to path
sys.path.append(str(Path(__file__).parent.parent.parent / "trinity-core"))

from agents.super_agents.intelligent_model_factory import IntelligentModelFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleGGUFConverter:
    """
    Simple GGUF Converter - All intelligence delegated to agent
    
    This wrapper is intentionally simple:
    - No hardcoded parameters
    - No complex logic
    - No decision making
    - Just data loading and agent delegation
    """
    
    def __init__(self):
        # Create intelligent agent (agent handles all configuration)
        self.intelligent_agent = IntelligentModelFactory()
        
        logger.info("🚀 Simple GGUF Converter initialized")
        logger.info("   → All intelligence delegated to IntelligentModelFactory")
        logger.info("   → No hardcoded values in this script")
    
    async def convert_data_to_gguf(self, domain: str, data_path: str = None, 
                                 training_data: List[Dict] = None) -> Dict[str, Any]:
        """
        Convert training data to GGUF - Simple delegation to intelligent agent
        
        Args:
            domain: Domain name (e.g., "healthcare", "business")
            data_path: Path to training data file (optional)
            training_data: Direct training data (optional)
        
        Returns:
            Result from intelligent agent
        """
        
        logger.info(f"🔄 Converting {domain} data to GGUF")
        
        # Step 1: Load data (simple, no complex logic)
        if training_data:
            data = training_data
            logger.info(f"   → Using provided training data: {len(data)} samples")
        elif data_path:
            data = self._load_data_file(data_path)
            logger.info(f"   → Loaded from {data_path}: {len(data)} samples")
        else:
            data = []
            logger.info(f"   → No training data provided, agent will use domain predictions")
        
        # Step 2: Create simple request (no configuration decisions)
        request = {
            "domain": domain,
            "training_data": data,
            "source": data_path if data_path else "direct_input"
        }
        
        # Step 3: Delegate ALL intelligence to the agent
        logger.info("🧠 Delegating to IntelligentModelFactory...")
        result = await self.intelligent_agent.create_intelligent_model(request)
        
        # Step 4: Simple result reporting
        self._report_results(result)
        
        return result
    
    def _load_data_file(self, data_path: str) -> List[Dict]:
        """Load data file - Simple, no complex logic"""
        path = Path(data_path)
        
        if not path.exists():
            logger.warning(f"Data file not found: {data_path}")
            return []
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Ensure data is a list
            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                logger.warning(f"Unexpected data format in {data_path}")
                return []
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")
            return []
    
    def _report_results(self, result: Dict[str, Any]) -> None:
        """Report results - Simple, no complex analysis"""
        
        if result.get("status") == "success":
            logger.info("✅ GGUF conversion completed successfully")
            logger.info(f"   → Domain: {result.get('domain', 'unknown')}")
            logger.info(f"   → Output: {result.get('output_path', 'unknown')}")
            logger.info(f"   → Size: {result.get('model_size_mb', 0):.1f}MB")
            logger.info(f"   → Quantization: {result.get('quantization_level', 'unknown')}")
            logger.info(f"   → Compression: {result.get('compression_method', 'unknown')}")
            logger.info(f"   → Quality Score: {result.get('quality_score', 0):.2f}")
            logger.info(f"   → Confidence: {result.get('confidence_score', 0):.2f}")
            logger.info(f"   → Risk Level: {result.get('risk_level', 'unknown')}")
            logger.info(f"   → Creation Time: {result.get('creation_time', 0):.2f}s")
        else:
            logger.error("❌ GGUF conversion failed")
            logger.error(f"   → Error: {result.get('error', 'unknown')}")

# Simple command-line interface
async def main():
    """Simple main function - minimal logic"""
    
    if len(sys.argv) < 2:
        print("Usage: python simple_gguf_converter.py <domain> [data_path]")
        print("Example: python simple_gguf_converter.py healthcare data/training/healthcare/")
        return
    
    domain = sys.argv[1]
    data_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Create converter and run
    converter = SimpleGGUFConverter()
    result = await converter.convert_data_to_gguf(domain, data_path)
    
    # Simple success/failure exit codes
    sys.exit(0 if result.get("status") == "success" else 1)

if __name__ == "__main__":
    asyncio.run(main()) 