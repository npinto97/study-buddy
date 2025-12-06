from langchain_huggingface import HuggingFaceEmbeddings
from study_buddy.config import CONFIG, logger
import torch

_embeddings_instance = None

def get_embeddings():
    """Initializes embeddings with GPU support optimized for RTX 5080 (Lazy Loading)."""
    global _embeddings_instance
    if _embeddings_instance is not None:
        return _embeddings_instance

    logger.info(f"Initializing Embeddings with model: {CONFIG.embeddings.model}")
    
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
        logger.info("Using GPU for embeddings...")
    else:
        device = 'cpu'
        logger.warning("GPU not available, using CPU")
    
    model_kwargs = {
        'device': device,
        'trust_remote_code': True,
    }
    
    encode_kwargs = {
        'normalize_embeddings': True,
        'batch_size': 32,
        'convert_to_numpy': True
    }
    
    _embeddings_instance = HuggingFaceEmbeddings(
        model_name=CONFIG.embeddings.model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        show_progress=True
    )
    
    logger.info(f"Embeddings with model {CONFIG.embeddings.model} successfully initialized on {device.upper()}.")
    return _embeddings_instance
