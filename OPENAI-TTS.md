# OpenAI TTS API Implementation Plan

## Document Purpose

This document provides a complete, detailed implementation plan for adding OpenAI Text-to-Speech API compatibility to the ChatterBox Runpod Serverless deployment. This plan is designed to be used by a new AI agent in a fresh session with no prior context.

**Status:** Planning Document
**Target Implementation:** Single container deployment (dual handler approach)
**Backward Compatibility:** Required - existing custom API must continue to work

---

## Table of Contents

1. [Overview & Goals](#overview--goals)
2. [Current Architecture](#current-architecture)
3. [Target Architecture](#target-architecture)
4. [OpenAI API Specification](#openai-api-specification)
5. [Voice Mapping System](#voice-mapping-system)
6. [Implementation Details](#implementation-details)
7. [Testing Plan](#testing-plan)
8. [Deployment](#deployment)
9. [Future Enhancements](#future-enhancements)

---

## Overview & Goals

### What We're Building

Add OpenAI Text-to-Speech API compatibility to the existing ChatterBox Runpod serverless deployment, allowing the service to be used as a drop-in replacement for OpenAI's TTS API while maintaining the existing custom API.

### Why

- **Broad Tool Compatibility**: Many tools and libraries expect OpenAI TTS API format (LangChain, OpenAI SDKs, etc.)
- **Standardization**: OpenAI TTS is a de facto standard interface
- **Ease of Use**: Simpler interface for basic use cases (just voice name + text)
- **Cost Flexibility**: Use cheaper/custom TTS while maintaining OpenAI-compatible interface

### Success Criteria

- ✅ Accept OpenAI TTS API requests (`/v1/audio/speech` endpoint)
- ✅ Return raw audio bytes (not JSON) with proper Content-Type headers
- ✅ Support voice name mapping (alloy, echo, fable, onyx, nova, shimmer)
- ✅ Support MP3 response format (primary format)
- ✅ Maintain backward compatibility with existing custom API
- ✅ No container rebuild required to add new voices
- ✅ Single container deployment (no additional infrastructure)

### Out of Scope (Future Enhancements)

- Speed control (0.25x - 4.0x)
- Additional response formats (opus, aac, flac, wav, pcm)
- Streaming audio response
- OpenAI model parameter (tts-1 vs tts-1-hd)

---

## Current Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Repository                         │
│                  (Push code changes)                         │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Runpod Serverless                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Container (CUDA 12.8 + Python 3.12)                   │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │ handler.py                                       │  │ │
│  │  │ • Receives job from Runpod                       │  │ │
│  │  │ • Extracts input parameters                      │  │ │
│  │  │ • Calls inference_engine.generate()              │  │ │
│  │  │ • Returns JSON response                          │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │ inference.py                                     │  │ │
│  │  │ • ChatterBoxInference class                      │  │ │
│  │  │ • Smart text chunking (300 chars max)            │  │ │
│  │  │ • Audio generation with watermark                │  │ │
│  │  │ • Monkeypatches for float32 fixes                │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│       Network Volume (/runpod-volume/chatterbox/)           │
│  • models/ - HuggingFace cache                              │
│  • audio_prompts/ - Voice reference files                   │
│  • output/ - Generated audio files                          │
└─────────────────────────────────────────────────────────────┘
```

### Current API Request Format

**Endpoint:** `POST /run` or `/runsync`

**Request:**
```json
{
  "input": {
    "text": "Text to synthesize",
    "audio_prompt": "reference_voice.wav",
    "temperature": 0.8,
    "top_k": 1000,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
    "min_p": 0.00,
    "exaggeration": 0.0,
    "cfg_weight": 0.0,
    "norm_loudness": true
  }
}
```

**Response:**
```json
{
  "status": "success",
  "sample_rate": 24000,
  "duration_sec": 3.45,
  "audio_url": "https://presigned-s3-url.com/audio.ogg"
}
```

OR (if S3 not configured):
```json
{
  "status": "success",
  "sample_rate": 24000,
  "duration_sec": 3.45,
  "audio_base64": "base64_encoded_audio_data"
}
```

### Current File Structure

```
chatterbox-runpod-serverless/
├── handler.py              # Runpod entry point (custom API handler)
├── inference.py            # ChatterBox inference engine
├── config.py               # Configuration management
├── bootstrap.sh            # Runtime setup script
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
├── README.md               # Documentation
├── CLAUDE.md               # Developer guide
└── .gitignore             # Git ignore rules
```

### Key Configuration Variables (config.py)

```python
# Application Configuration
MAX_TEXT_LENGTH = 2000
DEFAULT_SAMPLE_RATE = 24000
MAX_CHUNK_CHARS = 300  # Smart chunking limit

# ChatterBox Turbo generation parameters
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 1000
DEFAULT_TOP_P = 0.95
DEFAULT_REPETITION_PENALTY = 1.2
DEFAULT_MIN_P = 0.00
DEFAULT_NORM_LOUDNESS = True

# S3 Configuration
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
# ... etc
```

### Current handler.py Flow

```python
def handler(job):
    """Runpod serverless handler"""
    job_input = job.get("input", {})

    # Extract parameters
    text = job_input.get("text")
    audio_prompt = job_input.get("audio_prompt")
    temperature = float(job_input.get("temperature", config.DEFAULT_TEMPERATURE))
    # ... extract all parameters

    # Generate audio
    wav = inference_engine.generate(
        text=text,
        audio_prompt=audio_prompt,
        temperature=temperature,
        # ... all parameters
    )

    # Save to buffer/S3
    # Return JSON response
```

---

## Target Architecture

### Dual Handler Design

```
                        ┌─────────────────────┐
                        │   Runpod Request    │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │    handler.py       │
                        │  (Main Entry Point) │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │   Detect Request Format     │
                    │   (Check for "model" key    │
                    │    or OpenAI-specific keys) │
                    └──────────────┬──────────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 │                                   │
                 ▼                                   ▼
      ┌──────────────────────┐         ┌──────────────────────┐
      │  openai_handler()    │         │  custom_handler()    │
      │  (New)               │         │  (Existing logic)    │
      │                      │         │                      │
      │ • Parse OpenAI req   │         │ • Current behavior   │
      │ • Map voice→audio    │         │ • JSON response      │
      │ • Call inference     │         │ • S3 or base64       │
      │ • Return raw audio   │         │                      │
      └───────────┬──────────┘         └──────────┬───────────┘
                  │                                │
                  └────────────┬───────────────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │  inference_engine.     │
                  │  generate()            │
                  │  (Shared)              │
                  └────────────────────────┘
```

### Voice Mapping System

**Storage:** JSON file on network volume at `/runpod-volume/chatterbox/voices.json`

**Structure:**
```json
{
  "alloy": {
    "audio_file": "voice_alloy.wav",
    "description": "Neutral, balanced voice",
    "enabled": true
  },
  "echo": {
    "audio_file": "voice_echo.wav",
    "description": "Clear, professional voice",
    "enabled": true
  },
  "fable": {
    "audio_file": "voice_fable.wav",
    "description": "Expressive, storytelling voice",
    "enabled": true
  },
  "onyx": {
    "audio_file": "voice_onyx.wav",
    "description": "Deep, authoritative voice",
    "enabled": true
  },
  "nova": {
    "audio_file": "voice_nova.wav",
    "description": "Warm, friendly voice",
    "enabled": true
  },
  "shimmer": {
    "audio_file": "voice_shimmer.wav",
    "description": "Bright, energetic voice",
    "enabled": true
  }
}
```

**Default Fallback:** If voices.json doesn't exist, use a hardcoded default mapping in code.

---

## OpenAI API Specification

### Official OpenAI TTS API

**Endpoint:** `POST /v1/audio/speech`

**Request Headers:**
```
Content-Type: application/json
Authorization: Bearer sk-... (optional - we'll ignore it, Runpod handles auth)
```

**Request Body:**
```json
{
  "model": "tts-1",
  "input": "The quick brown fox jumped over the lazy dog.",
  "voice": "alloy",
  "response_format": "mp3",
  "speed": 1.0
}
```

**Parameters:**

| Parameter | Type | Required | Default | Valid Values | Description |
|-----------|------|----------|---------|--------------|-------------|
| `model` | string | Yes | - | `tts-1`, `tts-1-hd` | TTS model to use |
| `input` | string | Yes | - | Any string | Text to synthesize (max 4096 chars) |
| `voice` | string | Yes | - | `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` | Voice to use |
| `response_format` | string | No | `mp3` | `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm` | Audio format |
| `speed` | float | No | `1.0` | `0.25` - `4.0` | Playback speed |

**Response:**

- **Status Code:** `200 OK`
- **Headers:**
  - `Content-Type: audio/mpeg` (for mp3)
  - `Transfer-Encoding: chunked` (optional for streaming)
- **Body:** Raw audio bytes

**Error Response:**

```json
{
  "error": {
    "message": "Invalid voice specified",
    "type": "invalid_request_error",
    "param": "voice",
    "code": null
  }
}
```

**Status Codes:**
- `200` - Success
- `400` - Bad request (invalid parameters)
- `401` - Unauthorized (we won't use this, Runpod handles auth)
- `429` - Rate limit exceeded
- `500` - Server error

### Our Implementation (Phase 1)

**What We'll Support:**

| Feature | Support Level | Notes |
|---------|--------------|-------|
| `model` parameter | Accept but ignore | Both tts-1 and tts-1-hd map to ChatterBox Turbo |
| `input` parameter | ✅ Full support | Max 2000 chars (our limit) |
| `voice` parameter | ✅ Full support | Map to audio_prompt files |
| `response_format` | ⚠️ MP3 only | Other formats in future |
| `speed` parameter | ⚠️ Accept but ignore | Future enhancement |

**Response:**
- Raw audio bytes (not JSON!)
- Content-Type: audio/mpeg
- Audio in MP3 format

---

## Voice Mapping System

### File Location

```
/runpod-volume/chatterbox/
  ├── voices.json          # Voice mapping configuration
  ├── audio_prompts/
  │   ├── voice_alloy.wav  # Reference audio for "alloy"
  │   ├── voice_echo.wav   # Reference audio for "echo"
  │   ├── voice_fable.wav
  │   ├── voice_onyx.wav
  │   ├── voice_nova.wav
  │   ├── voice_shimmer.wav
  │   └── ... (custom voices)
```

### voices.json Schema

```json
{
  "voice_name": {
    "audio_file": "filename.wav",
    "description": "Voice description",
    "enabled": true
  }
}
```

**Fields:**
- `audio_file` (required): Filename relative to `/runpod-volume/chatterbox/audio_prompts/`
- `description` (optional): Human-readable description
- `enabled` (optional, default: true): Whether voice is available

### Loading Logic

```python
def load_voice_mappings():
    """Load voice mappings from JSON file on network volume"""
    voice_map_path = Path(config.CHATTERBOX_DIR) / "voices.json"

    # Try to load from file
    if voice_map_path.exists():
        try:
            with open(voice_map_path, 'r') as f:
                mappings = json.load(f)
            log.info(f"Loaded {len(mappings)} voice mappings from {voice_map_path}")
            return mappings
        except Exception as e:
            log.error(f"Failed to load voices.json: {e}")
            # Fall through to default

    # Default fallback if file doesn't exist or fails to load
    log.warning("Using default voice mappings (voices.json not found)")
    return DEFAULT_VOICE_MAPPINGS

# Default mapping (hardcoded fallback)
DEFAULT_VOICE_MAPPINGS = {
    "alloy": {"audio_file": "voice_alloy.wav", "enabled": True},
    "echo": {"audio_file": "voice_echo.wav", "enabled": True},
    "fable": {"audio_file": "voice_fable.wav", "enabled": True},
    "onyx": {"audio_file": "voice_onyx.wav", "enabled": True},
    "nova": {"audio_file": "voice_nova.wav", "enabled": True},
    "shimmer": {"audio_file": "voice_shimmer.wav", "enabled": True},
}
```

### Voice Resolution

```python
def resolve_voice(voice_name: str) -> str:
    """Resolve OpenAI voice name to audio_prompt path"""
    mappings = load_voice_mappings()

    if voice_name not in mappings:
        raise ValueError(f"Unknown voice: {voice_name}. Available: {list(mappings.keys())}")

    voice_config = mappings[voice_name]

    if not voice_config.get("enabled", True):
        raise ValueError(f"Voice '{voice_name}' is disabled")

    return voice_config["audio_file"]
```

---

## Implementation Details

### New Files to Create

None - we're modifying existing files only.

### Files to Modify

1. **handler.py** - Add request detection and OpenAI handler
2. **config.py** - Add voice mapping configuration
3. **requirements.txt** - Add pydub for MP3 conversion
4. **README.md** - Document OpenAI API usage
5. **CLAUDE.md** - Add development notes

### handler.py Modifications

**Current structure:**
```python
def handler(job):
    # Single handler for custom API
    ...
```

**New structure:**
```python
def handler(job):
    """Main entry point - routes to appropriate handler"""
    job_input = job.get("input", {})

    # Detect request format
    if is_openai_request(job_input):
        return openai_handler(job)
    else:
        return custom_handler(job)

def is_openai_request(job_input: dict) -> bool:
    """Detect if request is OpenAI format"""
    # OpenAI requests have 'model' and 'voice' keys
    # Custom requests have 'audio_prompt' key
    has_openai_keys = 'model' in job_input and 'voice' in job_input
    has_custom_keys = 'audio_prompt' in job_input

    return has_openai_keys and not has_custom_keys

def openai_handler(job):
    """Handle OpenAI TTS API requests"""
    # NEW FUNCTION - See detailed implementation below
    ...

def custom_handler(job):
    """Handle custom API requests"""
    # EXISTING LOGIC - Move current handler code here
    ...
```

### openai_handler() Implementation

```python
def openai_handler(job):
    """Handle OpenAI TTS API requests

    Expected input format:
    {
      "model": "tts-1",
      "input": "text to speak",
      "voice": "alloy",
      "response_format": "mp3",
      "speed": 1.0
    }

    Returns: Raw audio bytes with proper headers
    """
    try:
        job_input = job.get("input", {})

        # Extract and validate parameters
        model = job_input.get("model")  # Required
        text = job_input.get("input")   # Required (confusing name, but that's OpenAI's spec)
        voice = job_input.get("voice")  # Required
        response_format = job_input.get("response_format", "mp3")
        speed = job_input.get("speed", 1.0)

        # Validate required fields
        if not model:
            return openai_error("Missing required parameter: model", "invalid_request_error", "model")
        if not text:
            return openai_error("Missing required parameter: input", "invalid_request_error", "input")
        if not voice:
            return openai_error("Missing required parameter: voice", "invalid_request_error", "voice")

        # Validate text length
        if len(text) > config.MAX_TEXT_LENGTH:
            return openai_error(
                f"Text too long ({len(text)} chars). Maximum is {config.MAX_TEXT_LENGTH}.",
                "invalid_request_error",
                "input"
            )

        # Validate voice and get audio_prompt mapping
        try:
            audio_prompt = resolve_voice(voice)
        except ValueError as e:
            return openai_error(str(e), "invalid_request_error", "voice")

        # Validate response format (only MP3 supported for now)
        if response_format != "mp3":
            return openai_error(
                f"Unsupported response_format: {response_format}. Only 'mp3' is currently supported.",
                "invalid_request_error",
                "response_format"
            )

        # Log speed warning if not 1.0
        if speed != 1.0:
            log.warning(f"Speed parameter ({speed}) is not supported and will be ignored")

        log.info(f"OpenAI TTS request: voice={voice}, text_len={len(text)}, format={response_format}")

        # Generate audio using existing inference engine
        wav = inference_engine.generate(
            text=text,
            audio_prompt=audio_prompt,
            temperature=config.DEFAULT_TEMPERATURE,
            top_k=config.DEFAULT_TOP_K,
            top_p=config.DEFAULT_TOP_P,
            repetition_penalty=config.DEFAULT_REPETITION_PENALTY,
            min_p=config.DEFAULT_MIN_P,
            norm_loudness=config.DEFAULT_NORM_LOUDNESS,
            exaggeration=0.0,
            cfg_weight=0.0,
        )

        # Convert tensor to numpy
        audio_np = wav.squeeze(0).cpu().numpy()

        # Convert to MP3
        mp3_bytes = convert_to_mp3(audio_np, inference_engine.model.sr)

        # Return raw audio bytes
        # Note: Runpod may need special handling for binary responses
        return {
            "audio": base64.b64encode(mp3_bytes).decode('utf-8'),
            "_content_type": "audio/mpeg",  # Hint for Runpod
        }

    except Exception as e:
        log.error(f"OpenAI handler error: {e}", exc_info=True)
        return openai_error(str(e), "server_error", None)

def openai_error(message: str, error_type: str, param: str = None):
    """Format error response in OpenAI format"""
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": None
        }
    }

def resolve_voice(voice_name: str) -> str:
    """Resolve OpenAI voice name to audio_prompt filename"""
    voice_map_path = Path(config.CHATTERBOX_DIR) / "voices.json"

    # Load mappings
    if voice_map_path.exists():
        with open(voice_map_path, 'r') as f:
            mappings = json.load(f)
    else:
        # Default mappings
        mappings = {
            "alloy": {"audio_file": "voice_alloy.wav"},
            "echo": {"audio_file": "voice_echo.wav"},
            "fable": {"audio_file": "voice_fable.wav"},
            "onyx": {"audio_file": "voice_onyx.wav"},
            "nova": {"audio_file": "voice_nova.wav"},
            "shimmer": {"audio_file": "voice_shimmer.wav"},
        }

    # Check voice exists
    if voice_name not in mappings:
        available = ", ".join(mappings.keys())
        raise ValueError(f"Invalid voice '{voice_name}'. Available voices: {available}")

    # Check voice is enabled
    voice_config = mappings[voice_name]
    if not voice_config.get("enabled", True):
        raise ValueError(f"Voice '{voice_name}' is currently disabled")

    return voice_config["audio_file"]

def convert_to_mp3(audio_np: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy audio to MP3 bytes

    Args:
        audio_np: Audio as numpy array (samples,)
        sample_rate: Sample rate in Hz

    Returns:
        MP3 audio as bytes
    """
    from pydub import AudioSegment
    import io

    # Convert to int16 (pydub expects this)
    audio_int16 = (audio_np * 32767).astype(np.int16)

    # Create AudioSegment
    audio_segment = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16-bit = 2 bytes
        channels=1       # Mono
    )

    # Export to MP3
    mp3_buffer = io.BytesIO()
    audio_segment.export(mp3_buffer, format="mp3", bitrate="192k")

    return mp3_buffer.getvalue()
```

### config.py Modifications

```python
# Add at end of file

# Voice mapping configuration
VOICES_JSON_PATH = f"{CHATTERBOX_DIR}/voices.json"

# OpenAI TTS compatibility
OPENAI_TTS_ENABLED = True  # Feature flag
```

### requirements.txt Modifications

Add:
```
pydub>=0.25.1
```

Note: `pydub` requires ffmpeg, which should already be available in the CUDA container.

---

## Testing Plan

### Unit Tests (Manual)

**Test 1: OpenAI Request Detection**
```python
# Should return True
is_openai_request({"model": "tts-1", "voice": "alloy", "input": "Hello"})

# Should return False
is_openai_request({"text": "Hello", "audio_prompt": "voice.wav"})
```

**Test 2: Voice Resolution**
```python
# Should succeed
resolve_voice("alloy")  # Returns "voice_alloy.wav"

# Should raise ValueError
resolve_voice("nonexistent")
```

**Test 3: Error Formatting**
```python
error = openai_error("Test error", "invalid_request_error", "voice")
assert error["error"]["message"] == "Test error"
assert error["error"]["type"] == "invalid_request_error"
assert error["error"]["param"] == "voice"
```

### Integration Tests

**Test 4: OpenAI API Request (Success)**

**Request:**
```bash
curl -X POST https://your-endpoint.runpod.ai/v2/run \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "model": "tts-1",
      "voice": "alloy",
      "input": "Hello, this is a test of the OpenAI TTS API.",
      "response_format": "mp3"
    }
  }'
```

**Expected Response:**
```json
{
  "audio": "base64_encoded_mp3_data...",
  "_content_type": "audio/mpeg"
}
```

**Validation:**
- Decode base64, save as test.mp3
- Play audio, should sound like "alloy" voice
- Should be MP3 format

**Test 5: Custom API Request (Backward Compatibility)**

**Request:**
```bash
curl -X POST https://your-endpoint.runpod.ai/v2/run \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "Hello, this is a test.",
      "audio_prompt": "reference_voice.wav",
      "temperature": 0.8
    }
  }'
```

**Expected Response:**
```json
{
  "status": "success",
  "sample_rate": 24000,
  "duration_sec": 1.5,
  "audio_url": "https://..."
}
```

**Validation:**
- Response format unchanged
- Audio generated correctly
- All existing features work

**Test 6: Error Handling**

**Invalid Voice:**
```json
{
  "input": {
    "model": "tts-1",
    "voice": "invalid_voice",
    "input": "Test"
  }
}
```

**Expected:**
```json
{
  "error": {
    "message": "Invalid voice 'invalid_voice'. Available voices: alloy, echo, fable, onyx, nova, shimmer",
    "type": "invalid_request_error",
    "param": "voice"
  }
}
```

**Missing Required Field:**
```json
{
  "input": {
    "model": "tts-1",
    "voice": "alloy"
  }
}
```

**Expected:**
```json
{
  "error": {
    "message": "Missing required parameter: input",
    "type": "invalid_request_error",
    "param": "input"
  }
}
```

**Test 7: Long Text (Chunking)**

**Request:**
```json
{
  "input": {
    "model": "tts-1",
    "voice": "alloy",
    "input": "Very long text exceeding 300 characters... (total 800 chars)"
  }
}
```

**Expected:**
- Text automatically chunked
- Audio segments concatenated
- Returns single MP3 file
- No errors

**Test 8: Voice Mapping from JSON**

1. Create `/runpod-volume/chatterbox/voices.json`:
```json
{
  "alloy": {"audio_file": "custom_alloy.wav", "enabled": true},
  "echo": {"audio_file": "custom_echo.wav", "enabled": false}
}
```

2. Request with voice="alloy" → Should use custom_alloy.wav
3. Request with voice="echo" → Should return error (disabled)

### Performance Tests

**Test 9: Latency Comparison**

Measure end-to-end latency for:
- OpenAI API request
- Custom API request

Should be similar (OpenAI adds minimal overhead).

**Test 10: Concurrent Requests**

Send 10 concurrent requests (5 OpenAI, 5 custom):
- All should succeed
- No interference between handlers

---

## Deployment

### No Infrastructure Changes Required

- ✅ Same container
- ✅ Same Runpod endpoint
- ✅ Same environment variables
- ✅ Same network volume

### Configuration Steps

1. **Create voices.json** (optional, has defaults):
   ```bash
   # SSH to Runpod volume or create via API
   cat > /runpod-volume/chatterbox/voices.json << 'EOF'
   {
     "alloy": {"audio_file": "voice_alloy.wav"},
     "echo": {"audio_file": "voice_echo.wav"},
     "fable": {"audio_file": "voice_fable.wav"},
     "onyx": {"audio_file": "voice_onyx.wav"},
     "nova": {"audio_file": "voice_nova.wav"},
     "shimmer": {"audio_file": "voice_shimmer.wav"}
   }
   EOF
   ```

2. **Upload reference audio files**:
   ```bash
   # Upload 6 voice samples to:
   /runpod-volume/chatterbox/audio_prompts/voice_alloy.wav
   /runpod-volume/chatterbox/audio_prompts/voice_echo.wav
   # ... etc
   ```

3. **Deploy code**:
   - Commit and push to GitHub
   - Runpod automatically rebuilds container
   - No additional steps

### Rollback Plan

If OpenAI API has issues:

1. **Feature flag**: Set `OPENAI_TTS_ENABLED=false` in environment variables
2. **Code rollback**: Revert to previous commit
3. **Custom API**: Unaffected, continues to work

---

## Future Enhancements

### Phase 2: Additional Features

1. **Speed Control**
   - Implement audio time-stretching using `librosa` or `pyrubberband`
   - Support OpenAI's speed parameter (0.25x - 4.0x)

2. **Additional Audio Formats**
   - opus: Use ffmpeg to convert
   - aac: Use ffmpeg to convert
   - flac: Use ffmpeg to convert
   - wav: Direct output (no conversion)
   - pcm: Raw PCM data

3. **Streaming Response**
   - Stream audio chunks as they're generated
   - Requires chunked transfer encoding
   - May need Runpod support investigation

4. **Model Parameter Support**
   - Map tts-1 → standard quality
   - Map tts-1-hd → high quality (same model, different watermarking?)

5. **Voice Management API**
   - Endpoint to list available voices
   - Endpoint to add/remove voice mappings
   - Upload new voice samples via API

### Phase 3: Advanced Features

1. **Voice Cloning Endpoint**
   - OpenAI-compatible endpoint for custom voices
   - Upload audio sample → get voice ID
   - Use voice ID in synthesis requests

2. **Batch Processing**
   - Process multiple texts in one request
   - Return array of audio files

3. **Webhook Callbacks**
   - Async generation with callback URL
   - For very long texts

---

## Implementation Checklist

Use this checklist when implementing:

### Code Changes

- [ ] Modify handler.py:
  - [ ] Add `is_openai_request()` function
  - [ ] Add `openai_handler()` function
  - [ ] Add `openai_error()` function
  - [ ] Add `resolve_voice()` function
  - [ ] Add `convert_to_mp3()` function
  - [ ] Rename existing `handler()` to `custom_handler()`
  - [ ] Create new routing `handler()` function

- [ ] Modify config.py:
  - [ ] Add `VOICES_JSON_PATH` constant
  - [ ] Add `OPENAI_TTS_ENABLED` flag

- [ ] Modify requirements.txt:
  - [ ] Add `pydub>=0.25.1`

### Documentation

- [ ] Update README.md:
  - [ ] Add OpenAI TTS API to features
  - [ ] Add OpenAI usage examples
  - [ ] Document voice mapping configuration
  - [ ] Add voices.json schema

- [ ] Update CLAUDE.md:
  - [ ] Add dual handler architecture notes
  - [ ] Document voice mapping system
  - [ ] Add testing guidelines

### Testing

- [ ] Unit tests for request detection
- [ ] Unit tests for voice resolution
- [ ] Integration test: OpenAI request
- [ ] Integration test: Custom request (backward compat)
- [ ] Integration test: Error cases
- [ ] Integration test: Long text chunking
- [ ] Integration test: Voice mapping from JSON

### Deployment

- [ ] Create voices.json on network volume
- [ ] Upload 6 reference voice files
- [ ] Commit and push code
- [ ] Test on Runpod
- [ ] Verify both APIs work

---

## Additional Notes

### Runpod Binary Response Handling

Runpod may have specific requirements for returning binary data. If base64 encoding in JSON doesn't work, investigate:

1. Runpod's binary response documentation
2. Alternative response headers/formats
3. Direct streaming if supported

### ffmpeg Availability

The pydub library requires ffmpeg for MP3 encoding. Verify ffmpeg is available in the CUDA container. If not, add to Dockerfile:

```dockerfile
RUN apt-get update && apt-get install -y ffmpeg
```

### Voice Quality Considerations

The 6 default voices (alloy, echo, fable, onyx, nova, shimmer) should be:
- High quality recordings
- 3-10 seconds duration
- Clean speech, no background noise
- Representative of the desired voice character
- Properly named according to their personality

### Authentication Note

This implementation assumes Runpod API key authentication is sufficient. The OpenAI Bearer token is ignored. If additional token validation is needed in the future, it can be added to the `openai_handler()` function.

---

## Conclusion

This implementation plan provides a complete roadmap for adding OpenAI TTS API compatibility while maintaining backward compatibility with the existing custom API. The dual handler approach keeps the code clean and maintainable, and the voice mapping system allows dynamic voice management without container rebuilds.

**Estimated Implementation Time:** 6-8 hours for core functionality, 2-3 hours for testing and documentation.

**Risk Level:** Low - backward compatible, feature flag available, isolated code changes.

**Success Metrics:**
- OpenAI TTS requests work correctly
- Custom API requests continue to work
- Voice mapping is dynamic (no rebuild required)
- Clean error handling and logging
