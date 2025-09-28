import os
from typing import Dict, Any

class Config:
    HF_API_KEY = os.getenv("HF_API_KEY", "")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    GROQ_MODEL = "llama-3.3-70b-versatile"

    DEFAULT_TEMPERATURE = 0.4
    MAX_TOKENS = 1500

    DEFAULT_OUTPUT_DIR = "results"
    SAMPLE_DATA_DIR = "sample_data"

    SCORE_THRESHOLDS = {
        "excellent": 0.8,
        "very_good": 0.7,
        "good": 0.6,
        "fair": 0.5,
        "poor": 0.4
    }

    SCORE_INTERPRETATIONS = {
        (0.8, 1.0): "Excellent Match",
        (0.7, 0.8): "Very Good Match",
        (0.6, 0.7): "Good Match",
        (0.5, 0.6): "Fair Match",
        (0.4, 0.5): "Poor Match",
        (0.0, 0.4): "Very Poor Match"
    }

    @classmethod
    def validate_api_keys(cls) -> bool:
        return bool(cls.HF_API_KEY and cls.GROQ_API_KEY)

    @classmethod
    def get_score_interpretation(cls, score: float) -> str:
        for (min_score, max_score), interpretation in cls.SCORE_INTERPRETATIONS.items():
            if min_score <= score < max_score:
                return interpretation
        return "Invalid Score"

    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        return {
            "embedding_model": cls.EMBEDDING_MODEL,
            "groq_model": cls.GROQ_MODEL,
            "temperature": cls.DEFAULT_TEMPERATURE,
            "max_tokens": cls.MAX_TOKENS,
            "score_thresholds": cls.SCORE_THRESHOLDS
        }
