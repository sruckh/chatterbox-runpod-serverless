import sys
import os
import torch
import numpy as np
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

            # FIX: Create emotion_adv tensor with explicit float32 dtype on CPU, then move to device
            # Creating directly on device before .to() can cause tensor corruption
            emotion_adv_tensor = torch.tensor([[[float(exaggeration)]]], dtype=torch.float32)

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
        self.max_chunk_chars = config.MAX_CHUNK_CHARS  # Maximum characters per chunk for stable generation

    def _smart_chunk_text(self, text: str, max_chars: int = None) -> list[str]:
        """Split text into chunks at natural boundaries (sentences, clauses)

        Args:
            text: Text to chunk
            max_chars: Maximum characters per chunk (default: self.max_chunk_chars)

        Returns:
            List of text chunks
        """
        if max_chars is None:
            max_chars = self.max_chunk_chars

        # If text is short enough, return as-is
        if len(text) <= max_chars:
            return [text]

        chunks = []

        # Primary split on sentence endings
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        # Secondary split on clause boundaries
        clause_boundaries = [', ', '; ', ': ', ' - ', ' — ']

        def split_at_boundaries(text_segment: str, boundaries: list[str]) -> list[str]:
            """Split text at specified boundaries"""
            parts = []
            current = ""

            i = 0
            while i < len(text_segment):
                current += text_segment[i]

                # Check if we're at a boundary
                for boundary in boundaries:
                    if text_segment[i:i+len(boundary)] == boundary:
                        if len(current) > 0:
                            parts.append(current)
                            current = ""
                        i += len(boundary) - 1
                        break
                i += 1

            if current:
                parts.append(current)

            return parts

        # First try splitting on sentences
        segments = split_at_boundaries(text, sentence_endings)

        # Combine segments into chunks respecting max_chars
        current_chunk = ""

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            # If single segment is too long, split on clause boundaries
            if len(segment) > max_chars:
                # First save current chunk if exists
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Split long segment on clauses
                clauses = split_at_boundaries(segment, clause_boundaries)

                for clause in clauses:
                    clause = clause.strip()
                    if not clause:
                        continue

                    # If single clause is still too long, hard split by words
                    if len(clause) > max_chars:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = ""

                        words = clause.split()
                        for word in words:
                            if len(current_chunk) + len(word) + 1 <= max_chars:
                                current_chunk += " " + word if current_chunk else word
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                current_chunk = word
                    else:
                        # Clause fits, check if we can add to current chunk
                        if len(current_chunk) + len(clause) + 1 <= max_chars:
                            current_chunk += " " + clause if current_chunk else clause
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = clause
            else:
                # Segment fits in max_chars, try to add to current chunk
                if len(current_chunk) + len(segment) + 1 <= max_chars:
                    current_chunk += " " + segment if current_chunk else segment
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = segment

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        # Filter empty chunks
        chunks = [c for c in chunks if c.strip()]

        if not chunks:
            return [text]  # Fallback to original text

        log.info(f"Split text into {len(chunks)} chunks (max {max_chars} chars each)")
        return chunks

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
        """Generate audio from text using ChatterboxTurboTTS with smart chunking

        Args:
            text: Text to synthesize (automatically chunked if > 550 chars)
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

        log.info(f"Generating audio for text ({len(text)} chars): {text[:50]}...")

        # Process audio prompt if provided
        audio_prompt_path = self.process_audio_prompt(audio_prompt)

        # ChatterBox requires either audio_prompt_path or pre-prepared conditionals
        if not audio_prompt_path and self.model.conds is None:
            raise ValueError(
                "Either 'audio_prompt' must be provided for voice cloning, "
                "or model must have pre-prepared conditionals"
            )

        # Smart chunk text if it's too long
        chunks = self._smart_chunk_text(text, self.max_chunk_chars)

        if len(chunks) == 1:
            # Single chunk, generate directly
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
        else:
            # Multiple chunks, generate and concatenate
            log.info(f"Processing {len(chunks)} chunks...")
            audio_chunks = []

            with torch.no_grad():
                for i, chunk_text in enumerate(chunks, 1):
                    log.info(f"Generating chunk {i}/{len(chunks)} ({len(chunk_text)} chars)...")

                    chunk_wav = self.model.generate(
                        chunk_text,
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

                    # Remove batch dimension and convert to numpy for concatenation
                    chunk_audio = chunk_wav.squeeze(0).cpu().numpy()
                    audio_chunks.append(chunk_audio)

            # Concatenate all audio chunks
            concatenated_audio = np.concatenate(audio_chunks, axis=0)

            # Convert back to tensor with batch dimension
            final_wav = torch.from_numpy(concatenated_audio).unsqueeze(0)

            log.info(f"Concatenated {len(audio_chunks)} chunks → {final_wav.shape[1]} samples ({final_wav.shape[1]/self.model.sr:.1f}s)")

            return final_wav