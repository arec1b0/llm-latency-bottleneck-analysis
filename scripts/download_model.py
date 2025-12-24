"""
Model Download Script

Downloads and caches the LLM model from HuggingFace.
Validates model integrity and displays size information.
"""

import os
import sys
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_directory_size(path: Path) -> float:
    """
    Calculate total size of directory in GB.
    
    Args:
        path: Directory path
    
    Returns:
        Size in GB
    """
    total_size = 0
    for file in path.rglob("*"):
        if file.is_file():
            total_size += file.stat().st_size
    return total_size / (1024 ** 3)


def download_model(
    model_name: str,
    cache_dir: str,
    download_tokenizer_only: bool = False,
    use_fast_tokenizer: bool = True,
) -> None:
    """
    Download model and tokenizer from HuggingFace.
    
    Args:
        model_name: HuggingFace model identifier
        cache_dir: Directory to cache model files
        download_tokenizer_only: If True, only download tokenizer
        use_fast_tokenizer: If False, use slow tokenizer (more compatible)
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("MODEL DOWNLOAD")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Cache directory: {cache_path.absolute()}")
    logger.info("")
    
    try:
        # Download tokenizer
        logger.info("📥 Downloading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
                use_fast=use_fast_tokenizer,
            )
        except Exception as e:
            if use_fast_tokenizer:
                logger.warning(f"Fast tokenizer failed: {e}")
                logger.info("Retrying with slow tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    use_fast=False,
                )
            else:
                raise
        
        logger.info("✅ Tokenizer downloaded successfully")
        logger.info(f"   Vocabulary size: {tokenizer.vocab_size:,}")
        logger.info(f"   Tokenizer type: {'Fast' if tokenizer.is_fast else 'Slow'}")
        logger.info("")
        
        if download_tokenizer_only:
            logger.info("Tokenizer-only mode: Skipping model download")
            return
        
        # Download model
        logger.info("📥 Downloading model weights...")
        logger.info("   This may take several minutes depending on your connection...")
        logger.info("   Model size: ~13-14 GB for Mistral-7B")
        logger.info("")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        logger.info("✅ Model downloaded successfully")
        logger.info("")
        
        # Display model information
        logger.info("📊 Model Information:")
        logger.info(f"   Architecture: {model.config.model_type}")
        logger.info(f"   Hidden size: {model.config.hidden_size}")
        logger.info(f"   Layers: {model.config.num_hidden_layers}")
        logger.info(f"   Attention heads: {model.config.num_attention_heads}")
        logger.info(f"   Vocabulary: {model.config.vocab_size:,}")
        
        # Calculate parameter count
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
        logger.info("")
        
        # Check disk usage
        cache_size = get_directory_size(cache_path)
        logger.info(f"💾 Cache size: {cache_size:.2f} GB")
        logger.info("")
        
        # Verify model can be loaded
        logger.info("🔍 Verifying model integrity...")
        test_input = tokenizer("Test", return_tensors="pt")
        model.eval()
        with torch.no_grad():
            _ = model(**test_input)
        logger.info("✅ Model verification successful")
        logger.info("")
        
        logger.info("=" * 80)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Model cached at: {cache_path.absolute()}")
        logger.info(f"Total size: {cache_size:.2f} GB")
        logger.info("")
        logger.info("You can now start the API server:")
        logger.info("  uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Download interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        logger.error("")
        logger.error("Troubleshooting:")
        logger.error("  1. Check your internet connection")
        logger.error("  2. Verify HuggingFace model name")
        logger.error("  3. Ensure sufficient disk space (~20GB)")
        logger.error("  4. Check HuggingFace authentication (if required)")
        logger.error("  5. Try updating transformers: pip install --upgrade transformers")
        logger.error("")
        logger.error("For models requiring authentication:")
        logger.error("  huggingface-cli login")
        sys.exit(1)


def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    model_name = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
    cache_dir = os.getenv("MODEL_CACHE_DIR", "./models")
    
    # Parse command line arguments
    tokenizer_only = "--tokenizer-only" in sys.argv
    use_slow = "--use-slow" in sys.argv
    
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        model_name = sys.argv[1]
    
    # Confirm download
    if not tokenizer_only:
        logger.info("This will download approximately 13-14 GB of model weights.")
        logger.info(f"Model: {model_name}")
        logger.info(f"Cache directory: {cache_dir}")
        logger.info("")
        
        response = input("Continue? [Y/n]: ").strip().lower()
        if response and response != "y":
            logger.info("Download cancelled")
            sys.exit(0)
        logger.info("")
    
    # Download model
    download_model(
        model_name, 
        cache_dir, 
        tokenizer_only,
        use_fast_tokenizer=not use_slow
    )


if __name__ == "__main__":
    main()
