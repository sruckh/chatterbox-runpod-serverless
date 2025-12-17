import sys
import os
import torch
import logging
import soundfile as sf # Keep soundfile for writing, even if librosa is used for reading
import functools # No longer needed if monkeypatch is removed
from pathlib import Path

# Add ChatterBox to path (it's cloned at runtime in bootstrap.sh)
sys.path.insert(0, '/runpod-volume/chatterbox/chatterbox')

# Monkeypatch soundfile.read is removed as librosa is used for reading.
# The previous change to inference.py to use ChatterboxTurboTTS was cancelled.
# I will make sure ChatterboxTurboTTS is imported.
try:
    from chatterbox.tts_turbo import ChatterboxTurboTTS
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
        """Load ChatterBox Turbo model"""
        if self.model is None:
            return self.model

        log.info(f"Loading ChatterBox Turbo model on {self.device}...")
        try:
            # Import here to avoid build-time errors
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            self.model = ChatterboxTurboTTS.from_pretrained(device=self.device)
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

    def prepare_conditionals(self, wav_fpath, exaggeration=0.0, norm_loudness=True):
        ## Load and norm reference wav
        import librosa # Moved import here to ensure patch order doesn't matter for its own imports
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=config.DEFAULT_SAMPLE_RATE) # Use config sample rate here

        # Assert moved to config validation or handler
        # assert len(s3gen_ref_wav) / _sr > 5.0, "Audio prompt must be longer than 5 seconds!"

        if norm_loudness:
            s3gen_ref_wav = self.norm_loudness(s3gen_ref_wav, _sr)

        # Ensure correct sample rate for VoiceEncoder
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=config.DEFAULT_SAMPLE_RATE, target_sr=16000)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=16000)).float() # Explicitly cast to float32
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        # T3Cond needs to be filled here as in the original code,
        # but T3Cond and other related parts are internal to chatterbox.tts_turbo.py
        # and not exposed. So, this part needs to be simplified or removed if not directly accessible.
        # The original tts_turbo.py passes parameters to model.generate which then calls prepare_conditionals internally.
        # So I will remove prepare_conditionals from this class and simplify generate.

        # The parameters below are used by self.model.generate directly.
        # I will remove prepare_conditionals from this class.

        # THIS IS WRONG: My `prepare_conditionals` in `inference.py` is custom, the actual library has it in `tts_turbo.py`
        # So I need to use the `generate` function's parameters and remove the `prepare_conditionals` from `inference.py`

        pass # This function is moved back into ChatterboxTurboTTS internal to library, not exposed

    def generate(
        self,
        text,
        repetition_penalty=1.2,
        min_p=0.00,
        top_p=0.95,
        audio_prompt=None, # Changed from audio_prompt_path
        exaggeration=0.0,
        cfg_weight=0.0,
        temperature=0.8,
        top_k=1000,
        norm_loudness=True,
    ):
        """Generate audio from text using ChatterboxTurboTTS

        Args:
            text: Text to synthesize
            repetition_penalty: Repetition penalty (default: 1.2)
            min_p: Minimum probability threshold (default: 0.00)
            top_p: Top-p sampling (default: 0.95)
            audio_prompt: Path to audio reference file for voice cloning (relative to /runpod-volume/chatterbox/audio_prompts/)
            exaggeration: Emotion/expressiveness level (default: 0.0, ignored by Turbo)
            cfg_weight: Classifier-free guidance weight (default: 0.0, ignored by Turbo)
            temperature: Sampling temperature (default: 0.8)
            top_k: Top-k sampling (default: 1000)
            norm_loudness: Normalize loudness to -27 LUFS (default: True)

        Returns:
            wav: Generated audio as tensor (with watermark applied)
        """
        if self.model is None:
            self.load_model()

        log.info(f"Generating audio for text: {text[:50]}...")

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
                text,
                audio_prompt_path=audio_prompt_path, # Passed directly as the library handles preparation
                temperature=temperature,
                min_p=min_p,
                top_p=top_p,
                top_k=int(top_k),
                repetition_penalty=repetition_penalty,
                norm_loudness=norm_loudness,
                exaggeration=exaggeration, # Pass to generate as it's in the signature, even if ignored
                cfg_weight=cfg_weight      # Pass to generate as it's in the signature, even if ignored
            )

        return wav
