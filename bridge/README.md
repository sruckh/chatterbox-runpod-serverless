# ChatterBox OpenAI TTS Bridge

Cloudflare Worker that translates OpenAI Text-to-Speech API requests to RunPod ChatterBox custom format.

## Setup

### 1. Create wrangler.toml

Copy the example configuration and update with your settings:

```bash
cd bridge
cp wrangler.toml.example wrangler.toml
```

Edit `wrangler.toml` and set your RunPod endpoint URL:

```toml
[vars]
RUNPOD_URL = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
```

**Note:** `wrangler.toml` is in `.gitignore` to keep your endpoint URL private.

### 2. Set Secrets

Set the required RunPod API key and optional authentication token:

```bash
cd bridge

# Required: RunPod API key
wrangler secret put RUNPOD_API_KEY
# Paste your RunPod API key when prompted

# Optional but recommended: Worker authentication token
wrangler secret put AUTH_TOKEN
# Paste a secure token (e.g., generate with: openssl rand -hex 32)
```

**Note:** If `AUTH_TOKEN` is not set, the worker will be publicly accessible to anyone with the URL.

### 3. Deploy

```bash
wrangler deploy
```

The worker will be deployed to: `https://chatterbox-openai-bridge.YOUR_SUBDOMAIN.workers.dev`

## Voice Mappings

Voice mappings are stored in the R2 bucket `chatterbox` in a file called `voices.json`.

### Current Format

```json
{
  "Dorota": "Dorota.ogg",
  "Kurt": "Kurt.ogg",
  "Scott": "Scott.ogg"
}
```

### Update Voice Mappings

1. Edit `voices.json` locally
2. Upload to R2:
   ```bash
   wrangler r2 object put chatterbox/voices.json --file voices.json
   ```
3. Wait up to 5 minutes for cache to refresh (or redeploy worker)

### Add New Voice

1. Upload audio file to RunPod volume:
   `/runpod-volume/chatterbox/audio_prompts/NewVoice.ogg`

2. Update `voices.json`:
   ```json
   {
     "Dorota": "Dorota.ogg",
     "Kurt": "Kurt.ogg",
     "Scott": "Scott.ogg",
     "NewVoice": "NewVoice.ogg"
   }
   ```

3. Upload to R2:
   ```bash
   wrangler r2 object put chatterbox/voices.json --file voices.json
   ```

## Usage

### OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-worker-auth-token",  # Your AUTH_TOKEN secret
    base_url="https://chatterbox-openai-bridge.YOUR_SUBDOMAIN.workers.dev"
)

response = client.audio.speech.create(
    model="tts-1",
    voice="Dorota",
    input="Hello, this is a test."
)

response.stream_to_file("output.mp3")
```

**Note:** If you didn't set an `AUTH_TOKEN` secret, you can use any value for `api_key` (e.g., `"dummy"`).

### curl

```bash
curl -X POST https://chatterbox-openai-bridge.YOUR_SUBDOMAIN.workers.dev/v1/audio/speech \
  -H "Authorization: Bearer your-worker-auth-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "voice": "Dorota",
    "input": "Hello, this is a test.",
    "response_format": "mp3"
  }' \
  --output output.mp3
```

**Note:** Omit the `Authorization` header if you didn't set an `AUTH_TOKEN` secret.

### Node.js / JavaScript

```javascript
const response = await fetch('https://chatterbox-openai-bridge.YOUR_SUBDOMAIN.workers.dev/v1/audio/speech', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer your-worker-auth-token',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'tts-1',
    voice: 'Dorota',
    input: 'Hello, this is a test.',
    response_format: 'mp3'
  })
});

const audioBuffer = await response.arrayBuffer();
// Save or play audioBuffer
```

**Note:** Omit the `Authorization` header if you didn't set an `AUTH_TOKEN` secret.

## API Reference

### Request Format (OpenAI TTS Compatible)

**Endpoint:** `POST /v1/audio/speech`

**Request Body:**
```json
{
  "model": "tts-1",
  "voice": "Dorota",
  "input": "Text to synthesize",
  "response_format": "mp3",
  "speed": 1.0
}
```

**Parameters:**
- `model` (required): TTS model (accepted but ignored, always uses ChatterBox Turbo)
- `voice` (required): Voice name (must match a key in voices.json)
- `input` (required): Text to synthesize (max 2000 characters)
- `response_format` (optional): Audio format (only "mp3" supported, default: "mp3")
- `speed` (optional): Playback speed (accepted but ignored, default: 1.0)

**Response:**
- Status: 200 OK
- Content-Type: `audio/mpeg`
- Body: Raw MP3 audio bytes

### Error Response

```json
{
  "error": {
    "message": "Invalid voice 'Unknown'. Available voices: Dorota, Kurt, Scott",
    "type": "invalid_request_error",
    "param": "voice",
    "code": null
  }
}
```

## Monitoring

View logs:
```bash
wrangler tail
```

## Troubleshooting

### "voices.json not found"
Upload voices.json to R2:
```bash
wrangler r2 object put chatterbox/voices.json --file voices.json
```

### "RunPod service error"
- Check `RUNPOD_URL` is correct
- Verify `RUNPOD_API_KEY` secret is set
- Check RunPod endpoint is active

### Voice not found
- Ensure voice name in request matches voices.json key exactly (case-sensitive)
- Verify audio file exists on RunPod volume
- Check voices.json was uploaded to R2
