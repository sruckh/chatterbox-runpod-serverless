import runpod
import os
import logging
import base64
import io
import uuid
import soundfile as sf
import boto3
from botocore.exceptions import NoCredentialsError
import time
from pathlib import Path

from inference import ChatterBoxInference
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Initialize model loader
inference_engine = ChatterBoxInference()

def cleanup_old_files(directory, days=2):
    """Delete files older than specified days from directory

    Args:
        directory: Path to directory to clean
        days: Age threshold in days (default: 2)
    """
    try:
        output_dir = Path(directory)
        if not output_dir.exists():
            log.debug(f"Output directory {directory} does not exist, skipping cleanup")
            return

        current_time = time.time()
        cutoff_time = current_time - (days * 24 * 60 * 60)  # Convert days to seconds

        deleted_count = 0
        for file_path in output_dir.glob('*'):
            if file_path.is_file():
                file_age = file_path.stat().st_mtime
                if file_age < cutoff_time:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        log.debug(f"Deleted old file: {file_path.name}")
                    except Exception as e:
                        log.warning(f"Failed to delete {file_path.name}: {e}")

        if deleted_count > 0:
            log.info(f"Cleaned up {deleted_count} files older than {days} days from {directory}")
    except Exception as e:
        log.error(f"Cleanup failed: {e}")

def upload_to_s3(audio_buffer, filename):
    """Upload generated audio to S3 and return URL"""
    if not config.S3_BUCKET_NAME:
        log.warning("S3_BUCKET_NAME not set, returning base64 audio")
        return None

    try:
        s3 = boto3.client(
            's3',
            endpoint_url=config.S3_ENDPOINT_URL,
            aws_access_key_id=config.S3_ACCESS_KEY_ID,
            aws_secret_access_key=config.S3_SECRET_ACCESS_KEY,
            region_name=config.S3_REGION
        )
        
        s3.upload_fileobj(
            audio_buffer,
            config.S3_BUCKET_NAME,
            filename,
            ExtraArgs={'ContentType': 'audio/ogg'}
        )
        
        # Generate presigned URL
        url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': config.S3_BUCKET_NAME, 'Key': filename},
            ExpiresIn=3600  # 1 hour
        )
        return url
    except Exception as e:
        log.error(f"S3 upload failed: {e}")
        return None

def handler(job):
    """Runpod serverless handler

    Expected input format:
    {
        "text": str (required) - Text to synthesize
        "audio_prompt": str (optional) - Path to audio reference file for voice cloning
                                        (relative to /runpod-volume/chatterbox/audio_prompts/)
        "stream": bool (optional) - Enable streaming mode (default: false)
        "output_format": str (optional) - Output format for streaming: 'pcm_16' (default for CF workers)
        "exaggeration": float (optional) - Emotion/expressiveness level (0.0-1.0, default: 0.0, ignored by Turbo)
        "cfg_weight": float (optional) - Classifier-free guidance weight (0.0-1.0, default: 0.0, ignored by Turbo)
        "temperature": float (optional) - Sampling temperature (0.05-2.0, default: 0.8)
        "repetition_penalty": float (optional) - Repetition penalty (1.0-2.0, default: 1.2)
        "min_p": float (optional) - Minimum probability threshold (0.0-1.0, default: 0.00)
        "top_p": float (optional) - Top-p nucleus sampling (0.0-1.0, default: 0.95)
        "top_k": int (optional) - Top-k sampling (0-1000, default: 1000)
        "norm_loudness": bool (optional) - Normalize loudness (default: True)
    }

    Note: session_id is auto-generated internally for tracking and file naming.

    Batch mode Returns:
    {
        "status": "success",
        "sample_rate": int,
        "duration_sec": float,
        "audio_url": str (if S3 configured) OR "audio_base64": str (fallback)
    }

    Streaming mode Yields:
    {
        "status": "streaming",
        "chunk": int,
        "format": "pcm_16",
        "audio_chunk": str (base64 encoded),
        "sample_rate": 48000
    }
    {
        "status": "complete",
        "format": "pcm_16",
        "total_chunks": int,
        "elapsed_time_seconds": float
    }
    """
    job_input = job.get("input", {})

    # Extract streaming parameters first
    stream = job_input.get("stream", False)
    output_format = job_input.get("output_format", "pcm_16")

    # For streaming mode, use generator
    if stream:
        log.info(f"[Handler] Streaming mode requested: format={output_format}")
        yield from handler_stream(job_input, output_format)
        return

    # For batch mode, use original logic (yield result for consistency)
    yield handler_batch(job)


def _extract_and_validate_params(job_input: dict) -> tuple:
    """Extract and validate parameters from job input.

    Returns:
        tuple: (params_dict, error_dict) - error_dict is None if validation passes
    """
    # Extract parameters
    text = job_input.get("text")
    if not text:
        return None, {"error": "Missing 'text' parameter"}

    audio_prompt = job_input.get("audio_prompt")
    session_id = job_input.get("session_id", str(uuid.uuid4()))

    # ChatterBox Turbo generation parameters
    exaggeration = float(job_input.get("exaggeration", config.DEFAULT_EXAGGERATION))
    cfg_weight = float(job_input.get("cfg_weight", config.DEFAULT_CFG_WEIGHT))
    temperature = float(job_input.get("temperature", config.DEFAULT_TEMPERATURE))
    repetition_penalty = float(job_input.get("repetition_penalty", config.DEFAULT_REPETITION_PENALTY))
    min_p = float(job_input.get("min_p", config.DEFAULT_MIN_P))
    top_p = float(job_input.get("top_p", config.DEFAULT_TOP_P))
    top_k = int(job_input.get("top_k", config.DEFAULT_TOP_K))
    norm_loudness = job_input.get("norm_loudness", config.DEFAULT_NORM_LOUDNESS)

    # Validate input
    if len(text) > config.MAX_TEXT_LENGTH:
        return None, {"error": f"Text length exceeds maximum of {config.MAX_TEXT_LENGTH}"}

    # Validate generation parameters
    if not (config.MIN_EXAGGERATION <= exaggeration <= config.MAX_EXAGGERATION):
        return None, {"error": f"exaggeration must be between {config.MIN_EXAGGERATION} and {config.MAX_EXAGGERATION}"}

    if not (config.MIN_CFG_WEIGHT <= cfg_weight <= config.MAX_CFG_WEIGHT):
        return None, {"error": f"cfg_weight must be between {config.MIN_CFG_WEIGHT} and {config.MAX_CFG_WEIGHT}"}

    if not (config.MIN_TEMPERATURE <= temperature <= config.MAX_TEMPERATURE):
        return None, {"error": f"temperature must be between {config.MIN_TEMPERATURE} and {config.MAX_TEMPERATURE}"}

    if not (config.MIN_TOP_P <= top_p <= config.MAX_TOP_P):
        return None, {"error": f"top_p must be between {config.MIN_TOP_P} and {config.MAX_TOP_P}"}

    if not (config.MIN_TOP_K <= top_k <= config.MAX_TOP_K):
        return None, {"error": f"top_k must be between {config.MIN_TOP_K} and {config.MAX_TOP_K}"}

    params = {
        "text": text,
        "audio_prompt": audio_prompt,
        "session_id": session_id,
        "exaggeration": exaggeration,
        "cfg_weight": cfg_weight,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "min_p": min_p,
        "top_p": top_p,
        "top_k": top_k,
        "norm_loudness": norm_loudness,
    }

    return params, None


def handler_batch(job):
    """Batch mode handler - generates complete audio and returns URL/base64"""
    # Clean up old output files (older than 2 days)
    cleanup_old_files(config.OUTPUT_DIR, days=2)

    job_input = job.get("input", {})

    # Extract and validate parameters
    params, error = _extract_and_validate_params(job_input)
    if error:
        return error

    text = params["text"]
    audio_prompt = params["audio_prompt"]
    session_id = params["session_id"]
    exaggeration = params["exaggeration"]
    cfg_weight = params["cfg_weight"]
    temperature = params["temperature"]
    repetition_penalty = params["repetition_penalty"]
    min_p = params["min_p"]
    top_p = params["top_p"]
    top_k = params["top_k"]
    norm_loudness = params["norm_loudness"]

    try:
        # Generate audio
        wav = inference_engine.generate(
            text=text,
            audio_prompt=audio_prompt,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            min_p=min_p,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            norm_loudness=norm_loudness
        )

        # Get the model's sample rate (ChatterBox models have built-in sample rate)
        if inference_engine.model is None:
            inference_engine.load_model()
        sample_rate = inference_engine.model.sr
        log.info(f"Using model sample rate: {sample_rate}")

        # Convert to audio bytes (OGG Vorbis)
        log.info("Converting audio tensor to numpy...")
        audio_buffer = io.BytesIO()
        # Ensure wav is cpu numpy
        if hasattr(wav, 'cpu'):
            wav = wav.cpu().numpy()
        if hasattr(wav, 'numpy'):
            wav = wav.numpy()

        # ChatterBox likely returns (channels, samples) or just (samples)
        # Soundfile expects (samples, channels) usually, or just samples for mono
        if len(wav.shape) > 1 and wav.shape[0] < wav.shape[1]:
             wav = wav.T

        log.info(f"Writing audio to buffer (shape: {wav.shape})...")
        sf.write(audio_buffer, wav, sample_rate, format='OGG', subtype='VORBIS')
        audio_buffer.seek(0)

        # Upload to S3 or return base64
        filename = f"{session_id}_{uuid.uuid4()}.ogg"

        # Local output path for persistence in volume
        output_path = os.path.join(config.OUTPUT_DIR, filename)
        # Ensure output directory exists (it should be created by bootstrap, but good for safety)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        log.info(f"Saving audio locally to {output_path}...")
        with open(output_path, "wb") as f:
            f.write(audio_buffer.getbuffer())

        # Reset buffer for S3 upload
        audio_buffer.seek(0)

        log.info("Uploading to S3 (if configured)...")
        s3_url = upload_to_s3(audio_buffer, filename)

        response = {
            "status": "success",
            "sample_rate": sample_rate,
            "duration_sec": len(wav) / sample_rate
        }

        if s3_url:
            response["audio_url"] = s3_url
        else:
            # Fallback to base64
            audio_buffer.seek(0)
            b64_audio = base64.b64encode(audio_buffer.read()).decode("utf-8")
            response["audio_base64"] = b64_audio

        log.info("Handler completed successfully.")
        return response

    except Exception as e:
        log.error(f"Inference failed: {e}")
        return {"error": str(e)}


def handler_stream(job_input: dict, output_format: str):
    """Streaming mode handler - yields audio chunks as they're generated"""
    # Extract and validate parameters
    params, error = _extract_and_validate_params(job_input)
    if error:
        yield error
        return

    text = params["text"]
    audio_prompt = params["audio_prompt"]
    exaggeration = params["exaggeration"]
    cfg_weight = params["cfg_weight"]
    temperature = params["temperature"]
    repetition_penalty = params["repetition_penalty"]
    min_p = params["min_p"]
    top_p = params["top_p"]
    top_k = params["top_k"]
    norm_loudness = params["norm_loudness"]

    try:
        # Route based on output format
        if output_format == 'pcm_16':
            # Stream decoded audio chunks (for Cloudflare Workers)
            log.info("[Handler] Streaming decoded audio (pcm_16)")
            yield from inference_engine.generate_audio_stream_decoded(
                text=text,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                audio_prompt=audio_prompt,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                top_k=top_k,
                norm_loudness=norm_loudness,
            )
        else:
            # Unknown format
            yield {"error": f"Unknown output_format: {output_format}"}

    except Exception as e:
        log.error(f"Streaming inference failed: {e}")
        yield {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
