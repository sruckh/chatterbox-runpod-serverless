import sys
import os
import torch
import logging
import soundfile as sf
from pathlib import Path

# Add ChatterBox to path (it's cloned at runtime in bootstrap.sh)
sys.path.insert(0, '/runpod-volume/chatterbox/chatterbox')

try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
except ImportError:
    # This might fail during build if not cloned yet, but will work at runtime
    pass

import config

log = logging.getLogger(__name__)

class ChatterBoxInference:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Load ChatterBox Multilingual model"""
        if self.model is not None:
            return self.model

        log.info(f"Loading ChatterBox Multilingual model on {self.device}...")
        try:
            # Import here to avoid build-time errors
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            log.info(f"Model loaded successfully (sample rate: {self.model.sr})")
            return self.model
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            raise

    def process_audio_prompt(self, audio_prompt_path: str) -> str:
        """Process and validate audio reference for voice cloning"""
        if not audio_prompt_path:
            return None

        # Validate file path (security)
        full_path = Path(config.AUDIO_PROMPTS_DIR) / audio_prompt_path

        # Security check: ensure path is within AUDIO_PROMPTS_DIR
        try:
            full_path = full_path.resolve()
            prompts_dir = Path(config.AUDIO_PROMPTS_DIR).resolve()
            # If directory doesn't exist yet (e.g. unit testing locally without volume), create it or mock it
            # But in production it exists.
            if not prompts_dir.exists():
                # fallback for safety checks if dir doesn't exist
                pass
            elif not str(full_path).startswith(str(prompts_dir)):
                # Double check with simple string manipulation if resolve logic is tricky with symlinks
                # But resolve() is generally safer.
                raise ValueError("Invalid audio_prompt path: Path traversal detected")
        except Exception as e:
            # If resolution fails (e.g. file doesn't exist yet), check simple string prefix
            if not str(full_path).startswith(str(config.AUDIO_PROMPTS_DIR)):
                raise ValueError("Invalid audio_prompt path")

        # Check file exists and extension
        if not full_path.exists():
            raise ValueError(f"Audio prompt not found: {audio_prompt_path}")

        if full_path.suffix.lower() not in config.AUDIO_EXTS:
            raise ValueError(f"Unsupported audio format: {full_path.suffix}")

        # Validate audio duration
        try:
            audio_info = sf.info(str(full_path))
            duration = audio_info.duration

            if duration < config.MIN_AUDIO_DURATION:
                log.warning(f"Audio duration {duration:.2f}s is below recommended minimum of {config.MIN_AUDIO_DURATION}s")
            elif duration > config.MAX_AUDIO_DURATION:
                log.warning(f"Audio duration {duration:.2f}s exceeds recommended maximum of {config.MAX_AUDIO_DURATION}s")
            else:
                log.info(f"Audio prompt duration: {duration:.2f}s (within recommended range)")
        except Exception as e:
            log.warning(f"Could not validate audio duration: {e}")

        return str(full_path)

    def generate(
        self,
        text,
        language="en",
        audio_prompt=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=2.0,
        min_p=0.05,
        top_p=1.0,
    ):
        """Generate audio from text using ChatterboxMultilingualTTS

        Args:
            text: Text to synthesize
            language: Language code (e.g., "en", "fr", "es", etc.)
            audio_prompt: Path to audio reference file for voice cloning
            exaggeration: Emotion/expressiveness level (0.0-1.0, default: 0.5)
            cfg_weight: Classifier-free guidance weight (0.0-1.0, default: 0.5)
            temperature: Sampling temperature (default: 0.8)
            repetition_penalty: Repetition penalty (default: 2.0)
            min_p: Minimum probability threshold (default: 0.05)
            top_p: Top-p sampling (default: 1.0)

        Returns:
            wav: Generated audio as tensor (with watermark applied)
        """
        if self.model is None:
            self.load_model()

        log.info(f"Generating audio for text: {text[:50]}... (language: {language})")

        # Process audio prompt if provided
        audio_prompt_path = self.process_audio_prompt(audio_prompt)

        # ChatterBox requires either audio_prompt_path or pre-prepared conditionals
        if not audio_prompt_path and self.model.conds is None:
            raise ValueError(
                "Either 'audio_prompt' must be provided for voice cloning, "
                "or model must have pre-prepared conditionals"
            )

        with torch.no_grad():
            wav = self.model.generate(
                text=text,
                language_id=language,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )

        return wav
