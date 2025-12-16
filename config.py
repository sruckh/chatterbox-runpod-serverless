import os

# Environment Variables
HF_TOKEN = os.environ.get("HF_TOKEN")  # Required for model access

# HuggingFace cache configuration (IMPORTANT: Use chatterbox subdirectory)
HF_HOME = os.environ.get("HF_HOME", "/runpod-volume/chatterbox/hf_home")
HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE", "/runpod-volume/chatterbox/hf_cache")
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE

# S3 Configuration
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")

# Runpod volume structure
RUNPOD_VOLUME = "/runpod-volume"
CHATTERBOX_DIR = f"{RUNPOD_VOLUME}/chatterbox"
MODEL_CACHE_DIR = f"{CHATTERBOX_DIR}/models"
OUTPUT_DIR = f"{CHATTERBOX_DIR}/output"
AUDIO_PROMPTS_DIR = f"{CHATTERBOX_DIR}/audio_prompts"  # For voice cloning reference audio

# Application Configuration
DEFAULT_LANGUAGE = os.environ.get("DEFAULT_LANGUAGE", "en")
DEFAULT_SAMPLE_RATE = int(os.environ.get("DEFAULT_SAMPLE_RATE", "22050"))
MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", "2000"))

# Audio configuration
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac", ".opus"}
MIN_AUDIO_DURATION = 3.0  # seconds
MAX_AUDIO_DURATION = 30.0  # seconds

# Supported languages for ChatterBox Multilingual
# Arabic (ar) • Danish (da) • German (de) • Greek (el) • English (en) • Spanish (es)
# Finnish (fi) • French (fr) • Hebrew (he) • Hindi (hi) • Italian (it) • Japanese (ja)
# Korean (ko) • Malay (ms) • Dutch (nl) • Norwegian (no) • Polish (pl) • Portuguese (pt)
# Russian (ru) • Swedish (sv) • Swahili (sw) • Turkish (tr) • Chinese (zh)
SUPPORTED_LANGUAGES = {
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
    "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"
}

# ChatterBox generation parameters (from actual API)
DEFAULT_EXAGGERATION = 0.5  # Emotion/expressiveness (0.0-1.0)
DEFAULT_CFG_WEIGHT = 0.5    # Classifier-free guidance weight (0.0-1.0)
DEFAULT_TEMPERATURE = 0.8    # Sampling temperature
DEFAULT_REPETITION_PENALTY = 2.0  # Repetition penalty
DEFAULT_MIN_P = 0.05        # Minimum probability threshold
DEFAULT_TOP_P = 1.0         # Top-p sampling

# Parameter validation ranges
MIN_EXAGGERATION = 0.0
MAX_EXAGGERATION = 1.0
MIN_CFG_WEIGHT = 0.0
MAX_CFG_WEIGHT = 1.0
MIN_TEMPERATURE = 0.1
MAX_TEMPERATURE = 2.0
MIN_TOP_P = 0.0
MAX_TOP_P = 1.0
