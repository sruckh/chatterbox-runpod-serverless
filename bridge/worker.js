/**
 * Cloudflare Worker: OpenAI TTS API → RunPod ChatterBox Bridge
 *
 * Translates OpenAI Text-to-Speech API requests to RunPod ChatterBox custom format
 * and returns OpenAI-compatible responses (raw audio bytes).
 *
 * Also handles Streaming TTS requests using RunPod Serverless streaming protocol.
 * Follows the EchoTTS middleware pattern: POST /run → GET /stream/{id} → SSE response.
 */

// Cache for voice mappings (5 minute TTL)
let voiceMappingCache = null;
let lastFetch = 0;
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // CORS preflight
    if (request.method === 'OPTIONS') {
      return handleCORS();
    }

    // Health check
    if (url.pathname === '/health') {
      return new Response(JSON.stringify({
        status: 'healthy',
        tier: 'middleware-cloudflare',
        timestamp: Date.now()
      }), {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*'
        }
      });
    }

    // Authenticate request (if AUTH_TOKEN is configured)
    if (env.AUTH_TOKEN) {
      const authHeader = request.headers.get('Authorization');

      if (!authHeader) {
        return new Response(
          JSON.stringify({
            error: {
              message: 'Missing Authorization header',
              type: 'authentication_error',
              param: null,
              code: 'missing_authorization'
            }
          }),
          {
            status: 401,
            headers: {
              'Content-Type': 'application/json',
              'Access-Control-Allow-Origin': '*'
            }
          }
        );
      }

      // Extract token from "Bearer TOKEN" format
      const token = authHeader.replace(/^Bearer\s+/i, '');

      if (token !== env.AUTH_TOKEN) {
        return new Response(
          JSON.stringify({
            error: {
              message: 'Invalid authentication token',
              type: 'authentication_error',
              param: null,
              code: 'invalid_token'
            }
          }),
          {
            status: 401,
            headers: {
              'Content-Type': 'application/json',
              'Access-Control-Allow-Origin': '*'
            }
          }
        );
      }
    }

    // Route: OpenAI TTS Compatible Endpoint
    if (request.method === 'POST' && url.pathname === '/v1/audio/speech') {
      return handleOpenAITTS(request, env);
    }

    // Route: Streaming TTS Endpoint
    if (request.method === 'POST' && url.pathname === '/api/tts/stream') {
      return handleStreamingTTS(request, env);
    }

    return errorResponse('Not found. Available endpoints: POST /v1/audio/speech, POST /api/tts/stream', 404);
  }
};

async function handleOpenAITTS(request, env) {
  try {
    // Parse OpenAI TTS request
    const openaiRequest = await request.json();

    // Validate required fields
    const { model, input, voice, response_format = 'mp3', speed = 1.0 } = openaiRequest;

    if (!model) {
      return openaiError('Missing required parameter: model', 'invalid_request_error', 'model');
    }
    if (!input) {
      return openaiError('Missing required parameter: input', 'invalid_request_error', 'input');
    }
    if (!voice) {
      return openaiError('Missing required parameter: voice', 'invalid_request_error', 'voice');
    }

    // Load voice mappings from R2
    const voiceMappings = await getVoiceMappings(env);

    // Resolve voice to audio file
    const audioFile = voiceMappings[voice];
    if (!audioFile) {
      const available = Object.keys(voiceMappings).join(', ');
      return openaiError(
        `Invalid voice '${voice}'. Available voices: ${available}`,
        'invalid_request_error',
        'voice'
      );
    }

    // Warn about unsupported features (but don't error)
    if (response_format !== 'mp3') {
      console.warn(`Unsupported response_format: ${response_format}. Only 'mp3' is supported. Defaulting to mp3.`);
    }
    if (speed !== 1.0) {
      console.warn(`Speed parameter (${speed}) is not supported and will be ignored.`);
    }

    console.log(`OpenAI TTS request: voice=${voice} (${audioFile}), text_len=${input.length}, format=${response_format}`);

    // Translate to RunPod custom format
    const runpodRequest = {
      input: {
        text: input,
        audio_prompt: audioFile,
        // Use default generation parameters
        temperature: 0.8,
        top_k: 1000,
        top_p: 0.95,
        repetition_penalty: 1.2,
        min_p: 0.0,
        exaggeration: 0.0,
        cfg_weight: 0.0,
        norm_loudness: true
      }
    };

    // Call RunPod serverless
    const runpodResponse = await fetch(env.RUNPOD_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${env.RUNPOD_API_KEY}`
      },
      body: JSON.stringify(runpodRequest)
    });

    if (!runpodResponse.ok) {
      const errorText = await runpodResponse.text();
      console.error('RunPod error:', errorText);
      return openaiError(
        `RunPod service error: ${runpodResponse.status} ${runpodResponse.statusText}`,
        'server_error',
        null
      );
    }

    const runpodResult = await runpodResponse.json();

    // Extract output from RunPod response
    // /runsync endpoint wraps response in: { status: 'COMPLETED', output: {...} }
    const output = runpodResult.output || runpodResult;

    // Check for errors in response
    if (output.error || runpodResult.error) {
      const error = output.error || runpodResult.error;
      console.error('RunPod returned error:', error);
      return openaiError(error, 'server_error', null);
    }

    // Extract audio data (S3 URL or base64)
    let audioBytes;
    let contentType = 'audio/mpeg'; // Default for OpenAI compatibility

    if (output.audio_url) {
      // Fetch audio from S3 URL
      console.log('Fetching audio from S3:', output.audio_url);
      const s3Response = await fetch(output.audio_url);

      if (!s3Response.ok) {
        console.error('Failed to fetch from S3:', s3Response.status);
        return openaiError('Failed to fetch audio from S3', 'server_error', null);
      }

      audioBytes = await s3Response.arrayBuffer();

      // Use actual Content-Type from S3 (OGG format)
      // Note: This breaks strict OpenAI compatibility but ensures audio plays correctly
      contentType = s3Response.headers.get('Content-Type') || 'application/octet-stream';
      console.log('S3 Content-Type:', contentType);

    } else if (output.audio_base64 || output.audio) {
      // Decode base64 to binary
      const audioBase64 = output.audio_base64 || output.audio;
      audioBytes = base64ToArrayBuffer(audioBase64);

    } else {
      console.error('No audio data in RunPod response:', runpodResult);
      return openaiError('No audio data returned from RunPod', 'server_error', null);
    }

    // Return raw audio bytes (OpenAI format)
    return new Response(audioBytes, {
      status: 200,
      headers: {
        'Content-Type': contentType,
        'Access-Control-Allow-Origin': '*',
        'Cache-Control': 'no-cache'
      }
    });

  } catch (error) {
    console.error('Worker error:', error);
    return openaiError(error.message, 'server_error', null);
  }
}

/**
 * Handle streaming TTS requests using RunPod Serverless streaming protocol
 *
 * Pattern (from EchoTTS verify_runpod_backend.py):
 * 1. POST to /run endpoint → get job_id
 * 2. GET /stream/{job_id} → poll for streaming results
 * 3. Return SSE (Server-Sent Events) to client
 */
async function handleStreamingTTS(request, env) {
  const requestId = crypto.randomUUID();
  const startTime = Date.now();

  console.log(`[Tier 2][CF][${requestId}] Streaming TTS request received`);

  try {
    const { text, voice, service } = await request.json();

    // Validation
    if (!text || text.trim().length === 0) {
      return errorResponse('Text is required', 400);
    }

    if (!voice) {
      return errorResponse('Voice is required', 400);
    }

    // Get RunPod endpoint configuration
    const runpodUrl = env.RUNPOD_URL;
    const apiKey = env.RUNPOD_API_KEY;

    if (!runpodUrl) {
      return errorResponse('RunPod endpoint not configured', 500);
    }

    if (!apiKey) {
      return errorResponse('RunPod API key not configured', 500);
    }

    // Derive /run and /stream URLs from RUNPOD_URL
    // RUNPOD_URL is expected to be the /runsync endpoint (from OpenAI TTS batch mode)
    let runUrl, streamBaseUrl;
    if (runpodUrl.endsWith('/runsync')) {
      runUrl = runpodUrl.slice(0, -8) + '/run';
      streamBaseUrl = runpodUrl.slice(0, -8) + '/stream';
    } else if (runpodUrl.endsWith('/run')) {
      runUrl = runpodUrl;
      streamBaseUrl = runpodUrl.slice(0, -4) + '/stream';
    } else {
      // Fallback: assume it's the base URL
      runUrl = runpodUrl.replace(/\/$/, '') + '/run';
      streamBaseUrl = runpodUrl.replace(/\/$/, '') + '/stream';
    }

    console.log(`[Tier 2][CF][${requestId}] RunPod URLs: run=${runUrl}, stream=${streamBaseUrl}`);

    // Map voice to audio prompt
    let audioPrompt = null;
    try {
      const voiceMappings = await getVoiceMappings(env);
      audioPrompt = voiceMappings[voice];
      if (audioPrompt) {
        console.log(`[Tier 2][CF][${requestId}] Mapped voice '${voice}' to '${audioPrompt}'`);
      } else {
        console.warn(`[Tier 2][CF][${requestId}] Voice '${voice}' not found in mappings. Sending raw voice ID.`);
      }
    } catch (e) {
      console.warn(`[Tier 2][CF][${requestId}] Failed to load voice mappings:`, e);
    }

    // Create a readable stream for SSE (Server-Sent Events)
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        try {
          // Step 1: Submit job to RunPod /run endpoint
          console.log(`[Tier 2][CF][${requestId}] Submitting job to /run...`);

          const runResponse = await fetch(runUrl, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({
              input: {
                text,
                audio_prompt: audioPrompt || voice,
                stream: true,
                output_format: 'pcm_16',
                temperature: 0.8,
                top_p: 0.95
              }
            })
          });

          if (!runResponse.ok) {
            const errorText = await runResponse.text();
            throw new Error(`RunPod /run failed: ${runResponse.status} - ${errorText}`);
          }

          const jobData = await runResponse.json();
          const jobId = jobData.id;

          if (!jobId) {
            throw new Error(`No job ID in response: ${JSON.stringify(jobData)}`);
          }

          console.log(`[Tier 2][CF][${requestId}] Job submitted: ${jobId}`);

          // Step 2: Poll /stream/{jobId} for streaming results
          const streamUrl = `${streamBaseUrl}/${jobId}`;
          console.log(`[Tier 2][CF][${requestId}] Polling stream: ${streamUrl}`);

          let lastStreamPosition = 0;
          let totalChunks = 0;
          let isFinished = false;
          const timeout = 300000; // 5 minutes
          const pollStartTime = Date.now();

          while (!isFinished && (Date.now() - pollStartTime) < timeout) {
            const streamResponse = await fetch(streamUrl, {
              method: 'GET',
              headers: {
                'Authorization': `Bearer ${apiKey}`
              }
            });

            if (!streamResponse.ok) {
              const errorText = await streamResponse.text();
              throw new Error(`RunPod /stream failed: ${streamResponse.status} - ${errorText}`);
            }

            const data = await streamResponse.json();
            const status = data.status;
            const streamData = data.stream || [];

            // Process new stream items since last poll
            if (streamData.length > lastStreamPosition) {
              const newItems = streamData.slice(lastStreamPosition);

              for (const item of newItems) {
                totalChunks++;

                if (item.status === 'streaming') {
                  // Extract audio data and send as SSE
                  const audioBase64 = item.audio_chunk;
                  const sampleRate = item.sample_rate || 48000;
                  const chunkNum = item.chunk;

                  // Send SSE event with audio metadata
                  const sseData = {
                    chunk: chunkNum,
                    sample_rate: sampleRate,
                    format: item.format
                  };

                  controller.enqueue(encoder.encode(`event: audio\n`));
                  controller.enqueue(encoder.encode(`data: ${JSON.stringify(sseData)}\n\n`));

                  // Send base64 audio data as separate event
                  controller.enqueue(encoder.encode(`event: audio_data\n`));
                  controller.enqueue(encoder.encode(`data: ${audioBase64}\n\n`));

                  console.log(`[Tier 2][CF][${requestId}] Sent chunk ${chunkNum} @ ${sampleRate}Hz`);

                } else if (item.status === 'complete') {
                  isFinished = true;
                  const totalChunks = item.total_chunks;
                  const elapsed = item.elapsed_time_seconds || 0;

                  console.log(`[Tier 2][CF][${requestId}] Complete: ${totalChunks} chunks, ${elapsed.toFixed(2)}s`);

                  // Send completion event
                  const completeEvent = {
                    status: 'complete',
                    total_chunks: totalChunks,
                    elapsed_time_seconds: elapsed
                  };

                  controller.enqueue(encoder.encode(`event: complete\n`));
                  controller.enqueue(encoder.encode(`data: ${JSON.stringify(completeEvent)}\n\n`));

                } else if (item.error) {
                  throw new Error(item.error);
                }
              }

              lastStreamPosition = streamData.length;
            }

            // Check if job is finished
            if (status === 'COMPLETED' || status === 'FAILED' || status === 'CANCELED') {
              isFinished = true;

              if (status === 'FAILED') {
                throw new Error(data.error || 'RunPod job failed');
              } else if (status === 'CANCELED') {
                throw new Error('RunPod job was canceled');
              }
            }

            // Brief pause before next poll
            if (!isFinished) {
              await new Promise(resolve => setTimeout(resolve, 500));
            }
          }

          if (!isFinished) {
            throw new Error('Timeout waiting for RunPod job to complete');
          }

          const elapsed = Date.now() - startTime;
          console.log(`[Tier 2][CF][${requestId}] Stream complete: ${totalChunks} chunks, ${elapsed}ms`);

        } catch (error) {
          console.error(`[Tier 2][CF][${requestId}] Error:`, error);

          // Send error event to client
          const errorEvent = {
            error: error.message,
            status: 'error'
          };

          controller.enqueue(encoder.encode(`event: error\n`));
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(errorEvent)}\n\n`));
        } finally {
          controller.close();
        }
      }
    });

    // Return SSE stream response
    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization'
      }
    });

  } catch (error) {
    console.error(`[Tier 2][CF] Request error:`, error);
    return errorResponse(error.message, 500);
  }
}

/**
 * Load voice mappings from R2 bucket with caching
 */
async function getVoiceMappings(env) {
  const now = Date.now();

  // Return cached version if still valid
  if (voiceMappingCache && (now - lastFetch) < CACHE_TTL) {
    return voiceMappingCache;
  }

  try {
    // Fetch from R2 bucket
    const object = await env.CHATTERBOX_BUCKET.get('voices.json');

    if (!object) {
      throw new Error('voices.json not found in R2 bucket');
    }

    // Parse JSON
    voiceMappingCache = await object.json();
    lastFetch = now;

    console.log('Loaded voice mappings:', Object.keys(voiceMappingCache));
    return voiceMappingCache;

  } catch (error) {
    console.error('Failed to load voice mappings:', error);

    // Fallback to cached version if available
    if (voiceMappingCache) {
      console.warn('Using stale voice mapping cache due to error');
      return voiceMappingCache;
    }

    throw new Error(`Failed to load voice mappings: ${error.message}`);
  }
}

/**
 * Return OpenAI-formatted error response
 */
function openaiError(message, type, param) {
  return new Response(
    JSON.stringify({
      error: {
        message: message,
        type: type,
        param: param,
        code: null
      }
    }),
    {
      status: 400,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      }
    }
  );
}

/**
 * Generic error response
 */
function errorResponse(message, status = 400) {
  return new Response(
    JSON.stringify({ error: message }),
    {
      status: status,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      }
    }
  );
}

/**
 * Handle CORS preflight
 */
function handleCORS() {
  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      'Access-Control-Max-Age': '86400'
    }
  });
}

/**
 * Convert base64 string to ArrayBuffer
 */
function base64ToArrayBuffer(base64) {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}
