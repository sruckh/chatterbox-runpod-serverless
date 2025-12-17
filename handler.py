import runpod
import os
import logging
import base64
import io
import uuid
import soundfile as sf
import boto3
from botocore.exceptions import NoCredentialsError

from inference import ChatterBoxInference
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Initialize model loader
inference_engine = ChatterBoxInference()

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

    Returns:
    {
        "status": "success",
        "sample_rate": int,
        "duration_sec": float,
        "audio_url": str (if S3 configured) OR "audio_base64": str (fallback)
    }
    """
    job_input = job.get("input", {})

    # Extract parameters
    text = job_input.get("text")
    if not text:
        return {"error": "Missing 'text' parameter"}

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
        return {"error": f"Text length exceeds maximum of {config.MAX_TEXT_LENGTH}"}

    # Validate generation parameters
    if not (config.MIN_EXAGGERATION <= exaggeration <= config.MAX_EXAGGERATION):
        return {"error": f"exaggeration must be between {config.MIN_EXAGGERATION} and {config.MAX_EXAGGERATION}"}

    if not (config.MIN_CFG_WEIGHT <= cfg_weight <= config.MAX_CFG_WEIGHT):
        return {"error": f"cfg_weight must be between {config.MIN_CFG_WEIGHT} and {config.MAX_CFG_WEIGHT}"}

    if not (config.MIN_TEMPERATURE <= temperature <= config.MAX_TEMPERATURE):
        return {"error": f"temperature must be between {config.MIN_TEMPERATURE} and {config.MAX_TEMPERATURE}"}

    if not (config.MIN_TOP_P <= top_p <= config.MAX_TOP_P):
        return {"error": f"top_p must be between {config.MIN_TOP_P} and {config.MAX_TOP_P}"}
        
    if not (config.MIN_TOP_K <= top_k <= config.MAX_TOP_K):
        return {"error": f"top_k must be between {config.MIN_TOP_K} and {config.MAX_TOP_K}"}

    try:
        # Generate audio
        # wav is expected to be a tensor or numpy array
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

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
