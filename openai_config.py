"""
OpenAI Model Configuration
This file contains all the latest OpenAI models and their configurations.
"""

# Latest OpenAI Models Available (as of 2024)

# Embedding Models
EMBEDDING_MODELS = {
    "text-embedding-3-large": {
        "dimensions": 3072,
        "description": "Latest and most capable embedding model",
        "cost_per_1k_tokens": 0.00013,
        "recommended": True
    },
    "text-embedding-3-small": {
        "dimensions": 1536,
        "description": "Fast and cost-effective embedding model",
        "cost_per_1k_tokens": 0.00002,
        "recommended": False
    }
}

# Vision Models (Multimodal)
VISION_MODELS = {
    "gpt-4o": {
        "description": "Latest multimodal model with advanced vision capabilities",
        "max_tokens": 128000,
        "vision_support": True,
        "cost_per_1k_input": 0.005,
        "cost_per_1k_output": 0.015,
        "recommended": True
    },
    "gpt-4o-mini": {
        "description": "Faster and cheaper multimodal model",
        "max_tokens": 128000,
        "vision_support": True,
        "cost_per_1k_input": 0.00015,
        "cost_per_1k_output": 0.0006,
        "recommended": False
    }
}

# Text Models
TEXT_MODELS = {
    "gpt-4o": {
        "description": "Latest and most capable text model",
        "max_tokens": 128000,
        "cost_per_1k_input": 0.005,
        "cost_per_1k_output": 0.015,
        "recommended": True
    },
    "gpt-4o-mini": {
        "description": "Faster and cheaper text model",
        "max_tokens": 128000,
        "cost_per_1k_input": 0.00015,
        "cost_per_1k_output": 0.0006,
        "recommended": False
    }
}

# Default Model Selection
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"  # Compatible with existing 1536-dimension indexes
DEFAULT_VISION_MODEL = "gpt-4o"
DEFAULT_TEXT_MODEL = "gpt-4o"

# Model Capabilities
MODEL_CAPABILITIES = {
    "text-embedding-3-large": {
        "text_embedding": True,
        "image_embedding": True,
        "multilingual": True,
        "context_length": 8192
    },
    "gpt-4o": {
        "text_generation": True,
        "vision_analysis": True,
        "code_generation": True,
        "reasoning": True,
        "multilingual": True,
        "context_length": 128000
    }
}

# Cost Optimization
def get_cost_estimate(embedding_model: str, vision_model: str, num_images: int, avg_tokens_per_image: int = 1000):
    """
    Calculate estimated cost for processing images.
    
    Args:
        embedding_model: The embedding model to use
        vision_model: The vision model to use
        num_images: Number of images to process
        avg_tokens_per_image: Average tokens per image for vision analysis
    
    Returns:
        dict: Cost breakdown
    """
    embedding_cost = EMBEDDING_MODELS[embedding_model]["cost_per_1k_tokens"] * num_images * 3.072  # 3072 dimensions
    vision_cost = VISION_MODELS[vision_model]["cost_per_1k_input"] * num_images * avg_tokens_per_image / 1000
    
    total_cost = embedding_cost + vision_cost
    
    return {
        "embedding_cost": embedding_cost,
        "vision_cost": vision_cost,
        "total_cost": total_cost,
        "cost_per_image": total_cost / num_images if num_images > 0 else 0
    }

# Model Recommendations
def get_model_recommendations(use_case: str, budget: str = "medium"):
    """
    Get model recommendations based on use case and budget.
    
    Args:
        use_case: "production", "development", "testing", "cost_optimized"
        budget: "low", "medium", "high"
    
    Returns:
        dict: Recommended models
    """
    if use_case == "production" and budget in ["medium", "high"]:
        return {
            "embedding": "text-embedding-3-small",  # Compatible with existing indexes
            "vision": "gpt-4o",
            "text": "gpt-4o",
            "reason": "Best compatibility with existing Pinecone indexes while maintaining quality"
        }
    elif use_case == "development" or budget == "low":
        return {
            "embedding": "text-embedding-3-small",
            "vision": "gpt-4o-mini",
            "text": "gpt-4o-mini",
            "reason": "Cost-effective and compatible with existing indexes"
        }
    elif use_case == "cost_optimized":
        return {
            "embedding": "text-embedding-3-small",
            "vision": "gpt-4o-mini",
            "text": "gpt-4o-mini",
            "reason": "Minimal cost while maintaining compatibility"
        }
    else:
        return {
            "embedding": "text-embedding-3-small",  # Default to compatible model
            "vision": "gpt-4o",
            "text": "gpt-4o",
            "reason": "Balanced approach with existing index compatibility"
        }
