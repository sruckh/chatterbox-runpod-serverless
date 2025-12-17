import sys
import os
import torch
import logging
import soundfile as sf
from pathlib import Path

# Add ChatterBox to path (it's cloned at runtime in bootstrap.sh)
sys.path.insert(0, '/runpod-volume/chatterbox/chatterbox')

import config

log = logging.getLogger(__name__)

# Monkeypatch flag
_MONKEYPATCH_APPLIED = False

def apply_monkeypatches():
    """Apply monkeypatches to fix Float64/Float32 type mismatches"""
    global _MONKEYPATCH_APPLIED

    if _MONKEYPATCH_APPLIED:
        log.debug("Monkeypatches already applied, skipping")
        return

    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals
        from chatterbox.models.s3gen import S3GEN_SR
        from chatterbox.models.s3tokenizer import S3_SR
        from chatterbox.models.t3.modules.cond_enc import T3Cond
        import librosa

        # Monkeypatch prepare_conditionals to fix Float64/Float32 mismatch
        original_prepare_conditionals = ChatterboxTurboTTS.prepare_conditionals

        def patched_prepare_conditionals(self, wav_fpath, exaggeration=0.0, norm_loudness=True):
            ## Load and norm reference wav
            s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR, dtype='float32')

            # Assert removed or handled gracefully
            if len(s3gen_ref_wav) / _sr <= 3.0:
                 log.warning("Audio prompt is shorter than recommended 3 seconds.")

            if norm_loudness:
                s3gen_ref_wav = self.norm_loudness(s3gen_ref_wav, _sr)

            ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR).astype('float32')

            s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
            s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

            # Speech cond prompt tokens
            t3_cond_prompt_tokens = None
            if hasattr(self.t3.hp, 'speech_cond_prompt_len') and (plen := self.t3.hp.speech_cond_prompt_len):
                s3_tokzr = self.s3gen.tokenizer
                t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
                t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

            # Voice-encoder speaker embedding
            # FIX: Explicitly cast to float32 to ensure float32 instead of double
            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR)).to(dtype=torch.float32)
            ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

            # FIX: Create emotion_adv tensor with explicit float32 dtype
            emotion_adv_tensor = torch.tensor([[[float(exaggeration)]]], dtype=torch.float32, device=self.device)

            t3_cond = T3Cond(
                speaker_emb=ve_embed,
                cond_prompt_speech_tokens=t3_cond_prompt_tokens,
                emotion_adv=emotion_adv_tensor,
            ).to(device=self.device)
            self.conds = Conditionals(t3_cond, s3gen_ref_dict)

        # Apply the patch
        ChatterboxTurboTTS.prepare_conditionals = patched_prepare_conditionals
        _MONKEYPATCH_APPLIED = True
        log.info("✓ Monkeypatched ChatterboxTurboTTS.prepare_conditionals to fix Float32/Float64 type mismatches")

    except ImportError as e:
        log.warning(f"Could not apply monkeypatch (ChatterBox not yet available): {e}")
    except Exception as e:
        log.error(f"Error applying monkeypatch: {e}", exc_info=True)

class ChatterBoxInference:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Load ChatterBox Turbo model"""
        if self.model is not None:
            return self.model

        log.info(f"Loading ChatterBox Turbo model on {self.device}...")
        try:
            # Apply monkeypatches before loading model
            apply_monkeypatches()

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

    def generate(
        self,
        text,
        repetition_penalty=1.2,
        min_p=0.00,
        top_p=0.95,
        audio_prompt=None, 
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
                audio_prompt_path=audio_prompt_path, 
                temperature=temperature,
                min_p=min_p,
                top_p=top_p,
                top_k=int(top_k),
                repetition_penalty=repetition_penalty,
                norm_loudness=norm_loudness,
                exaggeration=exaggeration, 
                cfg_weight=cfg_weight      
            )

        return wav