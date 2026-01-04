# ChatterBox Runpod Serverless 🎙️

A serverless implementation of **Resemble AI's ChatterBox Turbo** TTS model on **Runpod**, featuring:

- 🚀 **Ultra-low latency streaming** (MP3/PCM)
- ⚡ **True OpenAI Streaming** compatibility
- 🧩 **Smart text chunking** for long-form synthesis
- 💾 **LinaCodec** efficiency for internal processing
- 🌉 **Cloudflare Worker Bridge** for OpenAI API compatibility

---

## 🏗️ Architecture

```
User/Client (OpenAI SDK)
      │
      ▼
Tier 2: Middleware (Cloudflare Worker)
      │  - /v1/audio/speech (Batch & Streaming)
      │  - /api/tts/stream (Direct Stream)
      │
      ▼
Tier 3: Backend (RunPod Serverless GPU)
      │  - Generates Audio (ChatterBox)
      │  - Compresses internal stream (LinaCodec)
      │  - Transcodes chunks (MP3/PCM)
      │
      ▼
   Response
```

### Key Features
*   **True Streaming:** Audio chunks are piped to the client immediately as they are generated.
*   **MP3 Support:** On-the-fly transcoding using `ffmpeg` reduces bandwidth usage.
*   **LinaCodec:** Used internally to compress audio between the TTS model and the streaming decoder.
*   **Batch Optimization:** Uses RunPod `/runsync` for reliable, fast batch processing.

---

## 🚀 Deployment

### 1. Backend (RunPod)
Simply push this repository to GitHub. Connect your repo to a RunPod Serverless Template.
*   **Container Image:** `runpod/base:0.4.0-cuda11.8.0` (or similar)
*   **Start Command:** `bash bootstrap.sh`

### 2. Middleware (Cloudflare)
Deploy the bridge worker:
```bash
cd bridge
npm install
npx wrangler deploy
```
Set secrets:
```bash
npx wrangler secret put RUNPOD_API_KEY
```

---

## 🔌 Usage

### 1. OpenAI Compatible Endpoint (Batch & Stream)

**Endpoint:** `POST https://your-worker.workers.dev/v1/audio/speech`

**Streaming (Recommended for Speed):**
```bash
curl -X POST https://your-worker.workers.dev/v1/audio/speech \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello! This audio is streaming in real-time.",
    "voice": "Dorota",
    "stream": true,
    "response_format": "mp3"
  }' > output.mp3
```

**Batch Mode:**
```bash
curl -X POST https://your-worker.workers.dev/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "This is a batch request.",
    "voice": "Dorota"
  }' --output batch.mp3
```

### 2. Direct Streaming Endpoint

**Endpoint:** `POST https://your-worker.workers.dev/api/tts/stream`

Ideal for custom clients needing raw PCM or specialized control.

```json
{
  "text": "Stream this text",
  "voice": "Dorota",
  "service": "chatterbox",
  "output_format": "mp3" 
}
```

---

## 🛠️ Configuration

### RunPod Environment Variables
*   `HF_TOKEN`: HuggingFace Token (for model download)
*   `S3_BUCKET_NAME`: (Optional) For batch storage

### Cloudflare Variables (`wrangler.toml`)
*   `RUNPOD_URL`: Your RunPod endpoint (e.g. `https://api.runpod.ai/v2/xxxx/runsync`)
*   `CHATTERBOX_BUCKET`: R2 Bucket binding

---

## 🧩 Voice Cloning
Upload reference audio files (OGG/WAV) to the R2 bucket or RunPod volume. Map them in `voices.json` in the R2 bucket.

```json
{
  "Dorota": "Dorota.ogg",
  "MyVoice": "custom_sample.wav"
}
```

---

## 📝 License
MIT License. Based on Resemble AI's ChatterBox.