#!/usr/bin/env python3
"""
MeeTARA Lab - Universal GGUF Model Generation Pipeline
Focused on creating intelligent GGUF models for MeeTARA application.

Usage:
    python src/main.py generate-data --domains healthcare,finance,education
    python src/main.py train-colab --config config/training_params.yaml
    python src/main.py create-gguf --output universal_intelligence_v1.0.0.gguf
    python src/main.py validate-gguf --file output/universal_intelligence_v1.0.0.gguf
"""

import argparse
import sys
from pathlib import Path
import logging
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_training_data(domains: List[str], output_dir: str = "data/training"):
    """Generate training data for specified domains"""
    try:
        from src.data import generate_training_data as generate_data
        
        logger.info(f"📊 Generating training data for domains: {', '.join(domains)}")
        
        for domain in domains:
            logger.info(f"🔄 Processing domain: {domain}")
            generate_data(domain=domain, output_dir=output_dir)
            
        logger.info("✅ Training data generation completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Training data generation failed: {e}")
        return False

def prepare_colab_training(config_path: str):
    """Prepare training setup for Google Colab"""
    try:
        from src.training import prepare_colab_environment
        
        logger.info("🚀 Preparing Google Colab training environment...")
        
        # Create Colab-ready training package
        prepare_colab_environment(config_path=config_path)
        
        logger.info("📋 Google Colab training prepared!")
        logger.info("Next steps:")
        logger.info("1. Upload notebooks/colab_training_pipeline.ipynb to Google Colab")
        logger.info("2. Run the training pipeline in Colab with GPU acceleration")
        logger.info("3. Download trained model weights")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Colab preparation failed: {e}")
        return False

def create_universal_gguf(output_file: str, model_weights_path: str = None):
    """Create universal GGUF file from trained model"""
    try:
        from src.gguf import create_universal_gguf as create_gguf
        
        logger.info(f"🏭 Creating universal GGUF file: {output_file}")
        
        # Create GGUF with all domain intelligence
        result = create_gguf(
            output_path=output_file,
            model_weights_path=model_weights_path,
            target_size="8.3MB",
            quality_threshold=0.95
        )
        
        if result:
            logger.info(f"✅ Universal GGUF created successfully: {output_file}")
            logger.info("🎯 Ready for deployment to MeeTARA application!")
        else:
            logger.error("❌ GGUF creation failed")
            
        return result
        
    except Exception as e:
        logger.error(f"❌ GGUF creation failed: {e}")
        return False

def validate_gguf_quality(gguf_file: str):
    """Validate GGUF quality and intelligence capabilities"""
    try:
        from src.gguf import validate_gguf_quality as validate_quality
        
        logger.info(f"🔍 Validating GGUF quality: {gguf_file}")
        
        results = validate_quality(gguf_file)
        
        logger.info("📊 Validation Results:")
        for metric, score in results.items():
            status = "✅" if score >= 0.95 else "⚠️" if score >= 0.85 else "❌"
            logger.info(f"  {status} {metric}: {score:.2%}")
            
        overall_quality = sum(results.values()) / len(results)
        if overall_quality >= 0.95:
            logger.info("🎯 GGUF quality excellent - ready for MeeTARA deployment!")
            return True
        else:
            logger.warning("⚠️ GGUF quality needs improvement")
            return False
            
    except Exception as e:
        logger.error(f"❌ GGUF validation failed: {e}")
        return False

def deploy_to_meetara(gguf_file: str, meetara_repo_path: str):
    """Deploy GGUF to MeeTARA application repository"""
    try:
        import shutil
        from pathlib import Path
        
        logger.info(f"🚀 Deploying GGUF to MeeTARA: {meetara_repo_path}")
        
        # Copy GGUF to MeeTARA models directory
        meetara_models_dir = Path(meetara_repo_path) / "models"
        meetara_models_dir.mkdir(exist_ok=True)
        
        destination = meetara_models_dir / Path(gguf_file).name
        shutil.copy2(gguf_file, destination)
        
        logger.info(f"✅ GGUF deployed successfully to: {destination}")
        logger.info("🎯 MeeTARA application ready to use new intelligence!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        return False

def main():
    """Main entry point for MeeTARA Lab model generation pipeline"""
    parser = argparse.ArgumentParser(description="MeeTARA Lab - Universal GGUF Model Factory")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate training data command
    data_parser = subparsers.add_parser('generate-data', help='Generate training data for domains')
    data_parser.add_argument('--domains', required=True, 
                           help='Comma-separated list of domains (healthcare,finance,education)')
    data_parser.add_argument('--output-dir', default='data/training', 
                           help='Output directory for training data')
    
    # Prepare Colab training command
    colab_parser = subparsers.add_parser('prepare-colab', help='Prepare Google Colab training')
    colab_parser.add_argument('--config', required=True, 
                            help='Path to training configuration file')
    
    # Create GGUF command
    gguf_parser = subparsers.add_parser('create-gguf', help='Create universal GGUF file')
    gguf_parser.add_argument('--output', required=True, 
                           help='Output GGUF file path')
    gguf_parser.add_argument('--model-weights', 
                           help='Path to trained model weights from Colab')
    
    # Validate GGUF command
    validate_parser = subparsers.add_parser('validate-gguf', help='Validate GGUF quality')
    validate_parser.add_argument('--file', required=True, 
                               help='GGUF file to validate')
    
    # Deploy to MeeTARA command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy GGUF to MeeTARA repo')
    deploy_parser.add_argument('--gguf-file', required=True, 
                             help='GGUF file to deploy')
    deploy_parser.add_argument('--meetara-repo', required=True, 
                             help='Path to MeeTARA repository')
    
    args = parser.parse_args()
    
    if args.command == 'generate-data':
        domains = [d.strip() for d in args.domains.split(',')]
        success = generate_training_data(domains, args.output_dir)
        sys.exit(0 if success else 1)
        
    elif args.command == 'prepare-colab':
        success = prepare_colab_training(args.config)
        sys.exit(0 if success else 1)
        
    elif args.command == 'create-gguf':
        success = create_universal_gguf(args.output, args.model_weights)
        sys.exit(0 if success else 1)
        
    elif args.command == 'validate-gguf':
        success = validate_gguf_quality(args.file)
        sys.exit(0 if success else 1)
        
    elif args.command == 'deploy':
        success = deploy_to_meetara(args.gguf_file, args.meetara_repo)
        sys.exit(0 if success else 1)
        
    else:
        parser.print_help()
        logger.info("\n🎯 MeeTARA Lab Pipeline:")
        logger.info("1. generate-data  → Create training datasets")
        logger.info("2. prepare-colab  → Setup Google Colab training")
        logger.info("3. [Manual]       → Run training in Google Colab")
        logger.info("4. create-gguf    → Generate universal GGUF")
        logger.info("5. validate-gguf  → Validate model quality")
        logger.info("6. deploy         → Deploy to MeeTARA app")
        sys.exit(1)

if __name__ == "__main__":
    main() 
