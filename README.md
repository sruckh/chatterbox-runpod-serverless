# ChatterBox Runpod Serverless

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-downloads)

A production-ready serverless implementation of [Resemble AI's ChatterBox](https://github.com/resemble-ai/chatterbox) text-to-speech model (Turbo variant) deployed on Runpod. Features zero-shot voice cloning and advanced generation controls.

## Features

- 🗣️ **Zero-Shot Voice Cloning** - Clone any voice from audio samples
- ⚡ **Turbo Inference** - Fast inference using the ChatterBox Turbo model
- 🔒 **Built-in Watermarking** - Automatic PerTh watermarking for responsible AI
- 🎭 **Emotion Control** - Adjust expressiveness with the exaggeration parameter (ignored by Turbo if > 0.0)
- 🎚️ **Advanced Controls** - Fine-tune with temperature, top-p, top-k, repetition penalty, and CFG weight (CFG weight ignored by Turbo if > 0.0)
- 🔊 **Loudness Normalization** - Automatic loudness normalization (-27 LUFS)
- 📦 **S3 Integration** - Automatic upload to S3-compatible storage with presigned URLs
- 🔄 **Smart Text Chunking** - Automatically handles long text (splits at sentence boundaries, max 300 chars per chunk)
- 🤖 **OpenAI TTS Compatible** - Optional Cloudflare Worker bridge for drop-in OpenAI API compatibility

## Quick Start

### Prerequisites

- Runpod account with GPU pods
- HuggingFace account and token (for model access)
- S3-compatible storage (optional, falls back to base64)

### Deployment on Runpod

1. **Create a new Serverless Endpoint** on Runpod

2. **Configure the endpoint**:
   - **Container Image**: Select "From GitHub"
   - **Repository**: `https://github.com/sruckh/chatterbox-runpod-serverless`
   - **Branch**: `main`

3. **Set Environment Variables**:
   ```
   HF_TOKEN=your_huggingface_token_here
   S3_ENDPOINT_URL=your_s3_endpoint (optional)
   S3_ACCESS_KEY_ID=your_s3_key (optional)
   S3_SECRET_ACCESS_KEY=your_s3_secret (optional)
   S3_BUCKET_NAME=your_bucket_name (optional)
   ```

4. **Deploy** and wait for the build to complete

### Usage Example

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "Hello! This is a test of the ChatterBox TTS system.",
      "audio_prompt": "reference_voice.wav",
      "exaggeration": 0.0,
      "cfg_weight": 0.0,
      "temperature": 0.8,
      "top_p": 0.95,
      "top_k": 50
    }
  }'
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

### OpenAI TTS API Compatibility

For OpenAI Text-to-Speech API compatibility, deploy the optional Cloudflare Worker bridge (see `bridge/` directory):

```bash
# Using OpenAI SDK (Python)
from openai import OpenAI

client = OpenAI(
    api_key="dummy",
    base_url="https://your-worker.workers.dev"
)

response = client.audio.speech.create(
    model="tts-1",
    voice="Dorota",
    input="Hello! This is a test."
)

response.stream_to_file("output.mp3")
```

Or with curl:

```bash
curl -X POST https://your-worker.workers.dev/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "voice": "Dorota",
    "input": "Hello! This is a test."
  }' \
  --output output.mp3
```

**Features:**
- ✅ True OpenAI TTS API compatibility (works with OpenAI SDKs)
- ✅ Dynamic voice mappings via Cloudflare R2
- ✅ No changes to RunPod deployment required
- ✅ Supports both S3 URLs and base64 responses

See [bridge/README.md](bridge/README.md) for setup and configuration.

## API Reference

### Custom API (Advanced)

### Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | Yes | - | Text to synthesize (max 2000 chars) |
| `audio_prompt` | string | Yes* | - | Path to reference audio for voice cloning (relative to `/runpod-volume/chatterbox/audio_prompts/`) |
| `exaggeration` | float | No | `0.0` | Emotion/expressiveness level (0.0-1.0, ignored by Turbo if > 0.0) |
| `cfg_weight` | float | No | `0.0` | Classifier-free guidance weight (0.0-1.0, ignored by Turbo if > 0.0) |
| `temperature` | float | No | `0.8` | Sampling temperature (0.05-2.0) |
| `top_p` | float | No | `0.95` | Top-p nucleus sampling (0.0-1.0) |
| `top_k` | int | No | `1000` | Top-k sampling (0-1000) |
| `repetition_penalty` | float | No | `1.2` | Penalty for repeating tokens (1.0-2.0) |
| `min_p` | float | No | `0.00` | Minimum probability threshold (0.0-1.0) |
| `norm_loudness` | bool | No | `true` | Normalize loudness to -27 LUFS |

*\*Required unless model has pre-prepared conditionals*

**Note**: Session IDs are auto-generated internally for tracking and file naming - users do not need to provide them.

### Audio Prompt Guidelines

- **Duration**: 3-10 seconds recommended (clean, single speaker)
- **Formats**: `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, `.webm`, `.aac`, `.opus`
- **Quality**: Clear speech, minimal background noise
- **Location**: Upload to `/runpod-volume/chatterbox/audio_prompts/` on the Runpod volume

## Architecture

### Core RunPod Deployment

```
┌─────────────────────────────────────────────────────────────┐
│                      GitHub Repository                       │
│                    (Push code changes)                       │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Runpod Serverless                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Container (CUDA 12.8 + Python 3.12)                   │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │ bootstrap.sh (First Run)                         │  │ │
│  │  │ • Install PyTorch 2.8.0                          │  │ │
│  │  │ • Clone ChatterBox from GitHub                   │  │ │
│  │  │ • Install dependencies                           │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │ handler.py → inference.py                        │  │ │
│  │  │ • Validate input                                 │  │ │
│  │  │ • ChatterboxTurboTTS                             │  │ │
│  │  │ • Generate audio with watermark                  │  │ │
│  │  │ • Upload to S3 or return base64                  │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              Network Volume (/runpod-volume/chatterbox/)     │
│  • HuggingFace cache (models)                               │
│  • Audio prompts (voice references)                          │
│  • Output audio files                                        │
└─────────────────────────────────────────────────────────────┘
```

### Optional: OpenAI TTS Compatibility (Cloudflare Worker Bridge)

```
┌──────────────────┐
│  OpenAI Client   │  (SDK, curl, etc.)
│  (Any tool)      │
└────────┬─────────┘
         │ POST /v1/audio/speech
         │ {"model": "tts-1", "voice": "Dorota", "input": "..."}
         ▼
┌─────────────────────────────────────────┐
│     Cloudflare Worker (Edge)            │
│  ┌───────────────────────────────────┐  │
│  │ • Translate OpenAI → Custom API   │  │
│  │ • Load voice mappings from R2     │  │
│  │ • Call RunPod serverless          │  │
│  │ • Fetch audio (S3 or base64)      │  │
│  │ • Return raw audio/mpeg           │  │
│  └───────────────────────────────────┘  │
└────────┬────────────────────────────────┘
         │
         │ Custom API request
         ▼
┌─────────────────────────────────────────┐
│        RunPod Serverless                │
│  (ChatterBox TTS generation)            │
└────────┬────────────────────────────────┘
         │
         ▼
    Audio Output
```

See [bridge/README.md](bridge/README.md) for bridge setup.

## Performance

- **First Cold Start**: ~2 minutes (PyTorch + dependency installation)
- **Subsequent Starts**: ~30 seconds (model loading only)
- **Generation Speed**: Real-time factor <0.5
- **VRAM Usage**: <8GB
- **Model Size**: ~350M parameters

## Configuration

### Environment Variables

**Required:**
- `HF_TOKEN` - HuggingFace token for model access

**Optional (S3):**
- `S3_ENDPOINT_URL` - S3-compatible endpoint
- `S3_ACCESS_KEY_ID` - S3 access key
- `S3_SECRET_ACCESS_KEY` - S3 secret key
- `S3_BUCKET_NAME` - S3 bucket name
- `S3_REGION` - S3 region (default: "us-east-1")

**Optional (Configuration):**
- `DEFAULT_SAMPLE_RATE` - Default sample rate (default: "24000")
- `MAX_TEXT_LENGTH` - Max text length (default: "2000")

## Development

### Project Structure

```
chatterbox-runpod-serverless/
├── handler.py              # Runpod serverless entry point
├── inference.py            # ChatterBox inference engine
├── config.py               # Configuration management
├── bootstrap.sh            # Runtime setup script
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
├── bridge/                 # Optional Cloudflare Worker for OpenAI TTS compatibility
│   ├── worker.js           # Cloudflare Worker code
│   ├── wrangler.toml.example  # Configuration template
│   ├── voices.json         # Voice mapping template
│   └── README.md           # Bridge documentation
├── .dockerignore          # Docker build exclusions
└── .gitignore             # Git ignore rules
```

### Local Testing

**Note**: This project is designed for Runpod deployment and cannot be run locally. The bootstrap script handles all setup at runtime on Runpod infrastructure.

## Troubleshooting

### Common Issues

**1. Model Download Failures**
- Verify `HF_TOKEN` is valid and has access to ChatterBox models
- Check HuggingFace service status

**2. Voice Cloning Not Working**
- Ensure audio reference is uploaded to `/runpod-volume/chatterbox/audio_prompts/`
- Verify audio format is supported
- Use short, clear audio clips (3-10 seconds)

**3. Slow First Startup**
- Expected behavior on first run
- Subsequent starts are much faster due to caching

## Credits

This project is built on top of [Resemble AI's ChatterBox](https://github.com/resemble-ai/chatterbox). All credit for the underlying model and architecture goes to the Resemble AI team.

## License

MIT License - See [LICENSE](LICENSE) file for details

**Note**: Model weights and audio outputs are subject to ChatterBox's licensing terms. Please review the [ChatterBox repository](https://github.com/resemble-ai/chatterbox) for details.

## Acknowledgments

- [Resemble AI](https://resemble.ai/) for ChatterBox
- [Runpod](https://runpod.io/) for serverless infrastructure
- [HuggingFace](https://huggingface.co/) for model hosting

## Support

For issues and questions:
- Open an issue on [GitHub](https://github.com/sruckh/chatterbox-runpod-serverless/issues)
- Check the [ChatterBox repository](https://github.com/resemble-ai/chatterbox) for model-specific questions

---

**Built with ❤️ for the open-source community**
