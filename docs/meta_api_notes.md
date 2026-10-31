# Meta Model API notes

Source: the Meta Global AI Developer Hackathon participant guide, the Meta Model API docs (https://dev.meta.ai/docs) and the Meta developer dashboard. Copied 2026-10-31.

Verify against https://dev.meta.ai/docs before relying on any detail.

## 1. Event basics

- Hosts: Meta and Cerebral Valley. Remote, worldwide, free.
- Event window: Oct 26, 1:00 PM ET to Nov 4, 9:00 PM ET, 2026. Building period Oct 27 to Nov 4.
- Prize pool: $1,000,000 across two tracks plus $50K bonus prizes.
- Event page: https://cerebralvalley.ai/e/meta-ai-developer-hackathon
- Participant guide: https://cerebralvalley.ai/e/meta-ai-developer-hackathon/details
- Submission page: https://cerebralvalley.ai/e/meta-ai-developer-hackathon/hackathon/submit

## 2. Problem statements

1. Build for Your World: bring Meta's models into a domain you know firsthand.
2. Push the Frontier: use Meta's models for software, robotics, manufacturing, energy, materials and beyond. Give a machine a new capability, accelerate an experiment, or unlock a better way to understand and control a physical system.

Projects must incorporate Meta's models: Muse Spark series, Muse Wonder, Muse Image, Muse Voice Transcribe, SAM 3.1 via the Meta Model API; open weights Muse Glimmer and Muse Spark Open; the Muse Code coding agent.

## 3. Tracks

- Indie developer: individuals and students. Judged on creativity, technical ambition, and whether it works.
- Startup: teams and founders building a real product.
- Every project enters exactly one track.

## 4. Rules

- Repositories must be public.
- Max team size 4; solo allowed.
- New work only: everything presented must be built during the event.
- The demo must only highlight features and code built during the hackathon. Failing to identify original contributions can mean disqualification.
- Disqualified: projects that violate legal, ethical or platform policies, or use code, data or assets without rights.
- Anti-projects to avoid: medical advice, basic RAG apps, simple image analyzers, simple education chatbots, job application screeners, nutrition coaches, personality analyzers, sports analyzers, anything political or involving war.

## 5. Schedule (ET)

- Mon Oct 26, 12:00 PM: kickoff livestream.
- Tue Oct 27 and Wed Oct 28, 12:00 PM: Agents 101 and Agents 102 sessions.
- Fri Oct 30, 12:00 PM: Fireworks session on Muse open models.
- Weekdays: Discord text office hours 10 to 11 AM ET in #office-hours (Meta team).
- Tue Nov 3: last full day for submission questions and technical troubleshooting.
- Wed Nov 4, 9:00 PM: submissions due. Also the deadline to claim Meta API credits.
- Nov 5 to Nov 8: judging. Nov 9, 12:00 PM: closing ceremony. Nov 16: community vote winner.

## 6. Credits

- Meta Model API: $150 per participant, claimed individually through the hackathon form by Nov 4, 9:00 PM ET. A credit card must be on the account before any API call works, even with credits. Adding one places a temporary pre-authorization hold.
- Baseten: $100 plus a 2M TPM rate limit, by application. Hackathon model id `meta-models/muse-spark-open-1026` (the docs say `meta-models/muse-spark-open`; use the -1026 one). Docs: https://docs.baseten.co/quickstart
- Modal: $100 for the first 1,000 participants, by application. Docs: https://modal.com/docs/guide/intro
- Fireworks: $100 per team, shortlist only. Model string `accounts/fireworks/models/muse-spark-open`, base URL `https://api.fireworks.ai/inference/v1`, env `FIREWORKS_API_KEY`.

Partner credits are not guaranteed.

## 7. Submission requirements

- Submit at the submission page by Nov 4, 9:00 PM ET. Late submissions are not considered.
- One submission per team.
- 3-minute demo video highlighting only what was built during the hackathon.
- Public repo, accessible video link, all team members listed.
- Most Viral prize: post the demo on X and submit the post link.

## 8. Judging

- Stage 1 (pass/fail): fits a theme and reasonably uses the Meta Model API and/or Meta open-weight models.
- Stage 2 (0 to 10 each): Impact 30%, Use of Meta AI tooling 25%, Technical depth and execution 20%, Demo 25%.
- Stage 3: top 5 per track, live final judging with the same rubric.

## 9. Prizes

Indie ($400K): $150K, $100K, $75K, $50K, $25K. Startup ($550K): $250K, $150K, $75K, $50K, $25K.

Bonus ($50K): Most Viral $20K; Best Build on Meta Open Weight Models $10K; Community Vote $10K; Best Agentic Use-Case $5K; Best Creative Build $5K.

## 10. Meta Model API technical reference

### Endpoints and auth

- Base URL (OpenAI-compatible): `https://api.meta.ai/v1`
- Base URL (Anthropic Messages): `https://api.meta.ai` (the client appends `/v1/messages`)
- Auth: `Authorization: Bearer $MODEL_API_KEY`. Keys look like `LLM_<numeric id>_<secret>`.
- Env var convention: `MODEL_API_KEY`. The OpenAI and Anthropic SDKs do not read it automatically; pass it explicitly.
- Dashboard: https://dev.meta.ai/platform (API keys tab: create a key and copy it immediately; the secret is shown once).
- Surfaces: Responses `/v1/responses` (text, image, PDF, video, server-managed state, encrypted reasoning replay); Chat Completions `/v1/chat/completions` (text, image, PDF, tools, streaming; no video); Messages `/v1/messages` and `/v1/messages/count_tokens` (Anthropic format, stateless).
- Images: `/v1/images/generations`, `/v1/images/edits`.
- Speech: `wss://api.meta.ai/v1/asr/realtime`, `POST /v1/asr/transcribe` (multipart).
- `GET /v1/models` lists enabled models.

### Models and prices (standard tier, USD per 1M tokens)

| Model ID | Use | Input | Price | RPM / TPM |
|---|---|---|---|---|
| `muse-wonder-1.0` | Most capable; long-horizon agents, multi-file coding, computer use. Preview. Reasoning minimal to max. | text, image, video, audio, PDF; 1M ctx | in $4.00, cached $0.20, out $20.00 | 3,000 / 4M |
| `muse-spark-1.3` | Default; agentic and coding; supports max effort. Audio degraded. | same; 1M ctx, 131,072 output | in $1.25, cached $0.15, out $4.25 | 3,000 / 4M |
| `muse-spark-1.3-contributor` | Same model, content used for training; no max effort | same | in $0.10, cached $0.002, out $0.20 | 150 / 3M |
| `muse-spark-1.2` / `-1.2-contributor` | Previous version | same | same as 1.3 / contributor | same |
| `muse-spark-1.1` | Original; Standard only | same | same as 1.3 | 3,000 / 4M |
| `muse-image-1.0` | Image generation and editing | text, image | $0.01 per image | 150 RPM |
| `muse-voice-transcribe-1.0` | Speech to text, diarization, 25 languages | audio | $0.18 per hour | 256 concurrent streams |
| `sam-3.1` | Segmentation of images and video from noun phrase or box/point prompts | text, image, video | $2.50 per 1K images; $0.20 per 1K video frames (minimum 50 frames per video request) | 3,000 RPM |
| Muse Spark Open | Open weights, 816B MoE, 42B active; not on the Model API | text, image, video, audio | provider pricing | n/a |
| Muse Glimmer | Open weights, 30B, Apache 2.0, single GPU | text, image | self-host | n/a |

Muse Wonder Full is Muse Code subscription only, not on the API.
Web search grounding: $2.50 per 1,000 queries plus tokens. Rate limits are per team, not per key. Retry 429s with exponential backoff and jitter; honor `Retry-After`. Headers: `x-ratelimit-limit-tokens`, `x-ratelimit-remaining-tokens`, `x-ratelimit-limit-requests`, `x-ratelimit-remaining-requests`. Background responses are capped at 600 submissions per minute.

### Quick calls

```python
from openai import OpenAI
import os
client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])
r = client.responses.create(model="muse-spark-1.3", input="What is the capital of France?")
print(r.output_text)
```

```python
from anthropic import Anthropic
client = Anthropic(base_url="https://api.meta.ai", auth_token=os.environ["MODEL_API_KEY"])
m = client.messages.create(model="muse-spark-1.3", max_tokens=1024, messages=[{"role": "user", "content": "Hi"}])
```

Messages API notes: `max_tokens` required; reasoning always on (`thinking: {type: "disabled"}` returns 400); depth via `output_config.effort` (low, medium, high, xhigh); `stop_sequences`, `top_k`, `container`, `inference_geo` return 400; named `tool_choice` unsupported; built-in tools `web_search` and `tool_search` (`defer_loading: true`); replay `redacted_thinking` to preserve reasoning.

Reasoning continuity: on Responses, request `include: ["reasoning.encrypted_content"]` and replay it. Chat Completions loses reasoning between turns.

### Muse Image

- Responses API with `model="muse-image-1.0"`; output contains `image_generation_call` items with base64 `result`. Iterate with `previous_response_id`.
- Only the `image_generation` tool is allowed; options `enable_image_search`, `enable_web_search`, `enable_shell`, `reasoning_strength` (high or low), `size` ("WxH", aspect only), `output_format` (webp, png, jpeg).
- One-shot: `client.images.generate(model="muse-image-1.0", prompt=..., n=1..10)`; extras via `extra_body`. Edits accept 1 to 10 images.

### SAM 3.1

- `client.responses.stream(model="sam-3.1", input=[{type: "message", role: "user", content: [{type: "input_text", text: "glasses"}, {type: "input_image", image_url: ...}]}])`.
- Output is special-token text encoding boxes and masks; parse with the SAM client libraries (https://dev.meta.ai/docs/sam/client-libraries). One noun phrase per request; use box or point prompts for known locations. Upload clips via the Files API for repeated segmentation.

### Muse Voice Transcribe

- File: `POST https://api.meta.ai/v1/asr/transcribe?sessionId=x` with multipart `request` JSON (`mode`, `model`, `audioEncoding`) and `audio`.
- WAV mono 16-bit PCM at 16 or 24 kHz; max 32 MB, 10 minutes.
- Modes: PUSH_TO_TALK (default), ENDPOINTING, DIARIZATION.
- Realtime: first JSON frame within 10 s includes `authorization.accessToken: "Bearer <key>"`, `audioEncoding: "PCM_24KHZ"`, `model`, `mode`; send `{"type": "endStream"}` to finish; max 60-minute session.

### Muse Spark Open hosted providers

| Provider | Model string | Base URL |
|---|---|---|
| Baseten (hackathon) | `meta-models/muse-spark-open-1026` | see Baseten docs |
| Fireworks | `accounts/fireworks/models/muse-spark-open` | https://api.fireworks.ai/inference/v1 |
| Together AI | `meta-models/muse-spark-open` | https://api.together.xyz/v1 |
| Groq | `meta-models/muse-spark-open` | GroqCloud |
| Nebius Token Factory | `meta-models/muse-spark-open` | Token Factory |

Also via Modal and OpenRouter. Always reasons (reasoning_effort low to xhigh, default xhigh; "none" rejected); reasoning in `reasoning_content`; set a large `max_tokens`; stream long generations. Stop on `<|eot|>`, never `<|eom|>`. Self-host weights: `meta-models/Muse-Spark-Open-816B-A42B` (BF16, 8xB300) and `-NVFP4` (4xB300); GGUF `ggml-org/Muse-Spark-Open-816B-A42B-GGUF`.

### Muse Code

`curl -fsSL https://dev.meta.ai/install.sh | sh` then `muse`.

### Routing Claude Code through Meta's API

The hackathon notes describe environment variables (`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, model overrides) that route Claude Code itself through the Meta API. Twinfit does not do this: it spends Meta credits on every turn and is unrelated to the build. See CLAUDE.md.

## 11. Documentation index

All `/docs/...` paths are under https://dev.meta.ai.

- Overview /docs/overview, Quickstart /docs/quickstart, Auth /docs/authentication, Models /docs/models, Pricing /docs/pricing-rate-limits, SDKs /docs/sdks
- Capabilities: /docs/tool-calling, /docs/tool-search, /docs/search-grounding, /docs/image-understanding, /docs/image-generation, /docs/speech-to-text, /docs/video-understanding, /docs/media-segmentation, /docs/file-handling, /docs/reasoning, /docs/structured-output, /docs/prompt-caching, /docs/token-counting
- Protocols: /docs/protocols, /docs/protocols/responses, /docs/protocols/chat-completions, /docs/protocols/messages
- Agents: /docs/coding-agents, /docs/agent-frameworks, /docs/computer-use
- SAM: /docs/sam/overview, /docs/sam/segmenting, /docs/sam/reading-segmentation, /docs/sam/client-libraries
- Open weights: /docs/muse-spark-open, /docs/muse-glimmer (each with sub-pages)
- Muse Code: /docs/muse-code (and sub-pages)
- Cookbook: /docs/cookbook; GitHub https://github.com/meta-models/meta-model-cookbook and https://github.com/meta-models/meta-oss-cookbook
- API reference: /docs/api-reference, /docs/error-handling
- Model pages: https://dev.meta.ai/models/{muse-spark, muse-wonder, muse-image, muse-voice-transcribe, sam-3-1, muse-glimmer, muse-spark-open}
