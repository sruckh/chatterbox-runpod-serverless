# ChatterBox Runpod Serverless

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-downloads)

A production-ready serverless implementation of [Resemble AI's ChatterBox](https://github.com/resemble-ai/chatterbox) text-to-speech model deployed on Runpod. Features zero-shot voice cloning, 23+ language support, and advanced generation controls.

## Features

- 🗣️ **Zero-Shot Voice Cloning** - Clone any voice from 3-30 second audio samples
- 🌍 **23+ Languages** - Arabic, Danish, German, Greek, English, Spanish, Finnish, French, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Dutch, Norwegian, Polish, Portuguese, Russian, Swedish, Swahili, Turkish, Chinese
- 🎭 **Emotion Control** - Adjust expressiveness with the exaggeration parameter
- ⚡ **Single-Step Generation** - Real-time factor <0.5 for fast inference
- 🔒 **Built-in Watermarking** - Automatic PerTh watermarking for responsible AI
- 🎚️ **Advanced Controls** - Fine-tune with CFG, temperature, and sampling parameters
- 📦 **S3 Integration** - Automatic upload to S3-compatible storage with presigned URLs

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
curl -X POST https://your-endpoint.runpod.ai/v2/runpod \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "Hello! This is a test of the ChatterBox TTS system.",
      "language": "en",
      "audio_prompt": "reference_voice.wav",
      "exaggeration": 0.7
    }
  }'
```

**Response:**
```json
{
  "status": "success",
  "language": "en",
  "sample_rate": 24000,
  "duration_sec": 3.45,
  "audio_url": "https://presigned-s3-url.com/audio.ogg"
}
```

## API Reference

### Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | Yes | - | Text to synthesize (max 2000 chars) |
| `language` | string | No | `"en"` | Language code (see supported languages below) |
| `audio_prompt` | string | Yes* | - | Path to reference audio for voice cloning (relative to `/runpod-volume/chatterbox/audio_prompts/`) |
| `exaggeration` | float | No | `0.5` | Emotion/expressiveness level (0.0-1.0) |
| `cfg_weight` | float | No | `0.5` | Classifier-free guidance weight (0.0-1.0) |
| `temperature` | float | No | `0.8` | Sampling temperature (0.1-2.0) |
| `repetition_penalty` | float | No | `2.0` | Penalty for repeating tokens |
| `min_p` | float | No | `0.05` | Minimum probability threshold |
| `top_p` | float | No | `1.0` | Top-p nucleus sampling (0.0-1.0) |

*\*Required unless model has pre-prepared conditionals*

**Note**: Session IDs are auto-generated internally for tracking and file naming - users do not need to provide them.

### Supported Languages

```
ar (Arabic), da (Danish), de (German), el (Greek), en (English),
es (Spanish), fi (Finnish), fr (French), he (Hebrew), hi (Hindi),
it (Italian), ja (Japanese), ko (Korean), ms (Malay), nl (Dutch),
no (Norwegian), pl (Polish), pt (Portuguese), ru (Russian),
sv (Swedish), sw (Swahili), tr (Turkish), zh (Chinese)
```

### Audio Prompt Guidelines

- **Duration**: 3-30 seconds recommended
- **Formats**: `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, `.webm`, `.aac`, `.opus`
- **Quality**: Clear speech, minimal background noise
- **Location**: Upload to `/runpod-volume/chatterbox/audio_prompts/` on the Runpod volume

## Architecture

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
│  │  │ • ChatterboxMultilingualTTS                      │  │ │
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

## Performance

- **First Cold Start**: ~2 minutes (PyTorch + dependency installation)
- **Subsequent Starts**: ~30 seconds (model loading only)
- **Generation Speed**: Real-time factor <0.5
- **VRAM Usage**: <8GB
- **Model Size**: 350M parameters

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
- `DEFAULT_LANGUAGE` - Default language (default: "en")
- `DEFAULT_SAMPLE_RATE` - Default sample rate (default: "22050")
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
- Check audio duration (3-30 seconds recommended)

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
