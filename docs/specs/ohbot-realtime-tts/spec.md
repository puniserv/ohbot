# Ohbot TTS via the Realtime API, with transcript verification — design

Date: 2026-11-21. Status: draft for review.

## Why
`gpt-4o-mini-tts` (all snapshots), `tts-1` and `tts-1-hd` shut down on 2027-01-06 (deprecation notice 2026-10-01).
OpenAI's replacement is `gpt-realtime-2.1-mini`. It works only over the Realtime API and is not available on
`/v1/audio/speech` ([model page](https://developers.openai.com/api/docs/models/gpt-realtime-2.1-mini)).
The realtime model is conversational, so it can paraphrase or add words. Children hear the audio, so a text that
passed moderation may play only if the model actually said that text.

## Scope
- Only the synthesis step changes. STT, both moderations, chat, `say()`, the fallback cache, fillers, greeting and lip
  sync stay as they are. They keep calling `ChainEngine.synthesize(text, publish, voice) -> bytes` (24 kHz, 16-bit
  mono PCM). The realtime `audio/pcm` output format is the same PCM, at 24000 Hz, the only rate it supports.
- Out of scope: streaming playback, speech-to-speech conversation, custom voices.

## Facts from the docs (checked 2026-11-21)
- Server-side connections use a WebSocket to `wss://api.openai.com/v1/realtime?model=...` with
  `Authorization: Bearer` ([WebSocket guide](https://developers.openai.com/api/docs/guides/realtime-websocket)).
- `response.create` accepts `conversation: "none"` (out of band, nothing goes into the session conversation), its own
  `input` array (which replaces the context; an empty array clears it), per-response `instructions`,
  `output_modalities`, `audio.output.{format,voice}`, `max_output_tokens` and `metadata`
  ([conversations guide](https://developers.openai.com/api/docs/guides/realtime-conversations),
  [client events](https://developers.openai.com/api/reference/resources/realtime/client-events)).
- `output_modalities: ["audio"]` returns audio together with its transcript. Text and audio cannot both be requested.
- Realtime voices: alloy, ash, ballad, coral, echo, sage, shimmer, verse, marin, cedar. The docs recommend marin and
  cedar. A session cannot change voice after it has produced audio. **`fable`, `nova` and `onyx` from our
  `VOICES` allow-list are not realtime voices.**
- Server events: audio arrives base64-encoded in `response.output_audio.delta`. The transcript arrives in
  `response.output_audio_transcript.delta` / `.done`. `response.done` carries `status` (`completed`, `cancelled`,
  `failed`, `incomplete`, `in_progress`) and `status_details` (`reason` such as `content_filter` or
  `max_output_tokens`, plus `error` with `type`/`code`). An `error` event has `error.type` (e.g. `server_error`,
  `invalid_request_error`), `code` and `message`, and leaves the connection open
  ([server events](https://developers.openai.com/api/reference/resources/realtime/server-events)).
- Python SDK (openai 3.57, installed): `async with client.realtime.connect(model=...) as conn`, then
  `conn.session.update(...)`, `conn.response.create(...)`, `async for event in conn`. The SDK does not raise on
  `error` events; the caller has to handle them. `connect()` auto-reconnects by default (`max_retries=5`).
- Price per 1M audio output tokens: gpt-realtime-2.1-mini $20, gpt-4o-mini-tts $12 (model pages). Text input is
  $0.60 on both. Not verified: how many audio tokens one second of speech takes, so the cost per minute is unknown.
  Measure it from `response.done.usage` during the listening test.
- **Not verified:** the exact field names inside the transcript delta/done events (the reference page was too long
  to read in full; the SDK types are authoritative, check them when implementing). Also not verified: whether the
  transcript is the model's own text or a recognition of its audio. The design does not depend on which it is.

## Synthesis flow (realtime path)
1. Under `asyncio.timeout(timeout_s)`: `client.realtime.connect(model=tts_model, max_retries=0)`. Our own retry rule
   stays the only retry.
2. `session.update`: `{"type": "realtime", "output_modalities": ["audio"], "audio": {"output": {"format":
   {"type": "audio/pcm", "rate": 24000}, "voice": voice}}}`.
3. `response.create` with `conversation: "none"`, `input: [one user message containing the text]`, `instructions` (below),
   `tool_choice: "none"`. Each connection is new and the response is out of band, so no state carries over between
   texts.
4. Collect the audio deltas into a bytearray and keep the final transcript. Stop at `response.done`.
5. Status not `completed`, an `error` event, or a closed socket raises an exception. `server_error` (or `failed` with
   `status_details.error.type == "server_error"`) counts as 5xx, so `_stage` retries it once, as it does today.
   A timeout maps to `EngineError("network")`. `content_filter` maps to `EngineError("moderated")`.
6. Verify (next section). Pass: return the PCM. Fail: raise `EngineError("moderated")`. The existing turn handling
   then plays the cached `moderated` fallback phrase, so no new error path is needed.

Instructions, written as the [voice prompting guide](https://developers.openai.com/api/docs/guides/realtime-models-prompting)
suggests (pinned language, explicit "exactly", no fillers or sound effects):
```
You are a text-to-speech reader. Read the user's message aloud in Polish, exactly as written, word for word.
Do not answer it, comment on it, translate it, greet, add or omit any word, or make sounds or effects.
Read numbers and abbreviations the usual Polish way. Speak warmly and clearly, for a child.
```
The text goes in as `{"require_repeat_verbatim": true, "text": "..."}` (the guide's verbatim envelope).
If the listening test shows the model reading the JSON aloud, switch to plain text.

## Transcript verification
`normalise(s)`: Unicode NFC, `casefold()`, every character that is not a letter or digit becomes a space, collapse
whitespace, strip. Polish diacritics stay (`łaska` and `laska` are different words). Stdlib only (`unicodedata`, `re`).

Rule, with `r = difflib.SequenceMatcher(None, normalise(text), normalise(transcript), autojunk=False).ratio()`:
- `r == 1.0`: play.
- `0.90 <= r < 1.0`: run the moderation model on the transcript (stage `moderation_tts`). Play if it is not flagged,
  otherwise reject.
- `r < 0.90` or an empty transcript: reject, and log both strings at WARNING level.

Why 0.90: a typical reply is 1-3 sentences, 60-250 normalised characters. 0.90 tolerates about one changed word per
ten (`3` read as `trzy`, `ok` as `okej`, a dropped `no`) and rejects an added or swapped clause on short texts.
Example: `Oczywiście, ` before a 60-character text gives 0.92 and is moderated; an added sentence of 15+ characters gives under 0.90.
A ratio cannot judge meaning, and on a 250-character text a short inserted phrase still passes 0.90. That is why
every inexact match also goes through moderation. The ratio limits how far the text can drift; moderation decides
whether the words are harmful. The exact case, which should be the common one, adds no latency.
Both 0.90 and the rule are config values (`tts_min_similarity`), tuned after the listening test.

Rejected options: running STT on the output audio (a second recognition, +0.5-1 s and extra cost, while the transcript
already comes in the same response). Word-level ratio: a one-word filler such as `Hmm…` vs `hmmm` scores 0.

## Connection strategy: one WebSocket per call (recommended)
- Per call: a new socket for every `synthesize`. No voice lock (the voice sample plays another voice), no state that
  could leak between texts, no keep-alive or reconnect code. Cost: a TLS + WebSocket handshake for each text.
  Not measured; estimated at 100-300 ms.
- Persistent session: saves the handshake, but needs reconnect handling, a socket per voice (the voice cannot change
  after the first audio), and a session length limit (not verified) to manage.
- Pick per call. Move to a persistent session only if measured latency calls for it.

## Latency and streaming
Today `synthesize` already waits for the whole `/audio/speech` body before playing, so buffering is nothing new.
Verification needs the full transcript before the first sample plays, which rules out streaming: playing early would
play unverified speech. Expected extra latency per reply: the handshake, plus one moderation call (~0.2-0.4 s, not
measured) for inexact matches. The `tts` stage covers the rest; time the whole stage in the listening test.

## Config and switch
- `openai.tts_model` decides the path: names starting with `gpt-realtime` use the Realtime API, anything else uses
  `/audio/speech`. This follows the existing `stt_model` prefix check. The old path stays until 2027-01-06, and changing
  one value flips back. The default becomes `gpt-realtime-2.1-mini` after the listening test passes. A follow-up
  deletes the old path after the shutdown.
- New: `openai.tts_min_similarity = 0.90` (range 0.5..1.0).
- Voice validation depends on the path: realtime allows the 10 realtime voices. The UI voice list is filtered the same
  way. Default `marin` works on both paths.
- The fallback cache folder is already `cache/fallback/<tts_model>-<voice>`, so the realtime phrases are generated
  fresh. `Fallback.ensure` verifies them too. A phrase that fails stays missing and is retried at the next start, as
  it is offline today.

## Doctor
The model-list check already includes `tts_model`. With a realtime `tts_model`, add a "Głos (Realtime)" check:
synthesise `Cześć, jestem Ohbot.` through the real path, including verification, and report OK with the time taken,
or the error (`nie przeszło weryfikacji`, network, auth). Not verified: whether `models.list()` lists realtime
models. If it does not, the model check skips `tts_model` on the realtime path and the new check covers it.

## Dependencies
The `openai[realtime]` extra pins `websockets>=13,<16`, but the project uses `websockets>=17.4` (web UI). The SDK
imports `websockets.asyncio.client`, which 17.x still ships, so it should work. **Not verified:** run one real call
before relying on it. If it fails, use `websockets` directly with the raw JSON events (same flow, about 30 more lines).

## Testing
- Unit (fake connection injected via the client: an async context manager whose `session.update` and
  `response.create` record calls and whose iterator yields scripted events): the PCM is the concatenated deltas; the
  out-of-band request has the voice, format and instructions; exact match plays with no moderation call;
  near match plus clean moderation plays; near match plus flagged moderation leads to `moderated`; low ratio leads to
  `moderated`; empty transcript is rejected; `server_error` is retried once, then `other`; `invalid_request_error` is
  not retried; `content_filter` leads to `moderated`; timeout leads to `network`; a socket closed before
  `response.done` raises; `tts_model` prefix picks the path; voice validation per path.
- `normalise`/ratio table: diacritics kept, punctuation and case ignored, `Hmm…` vs `hmm`, digits vs words.
- Listening test (user, real API, `doctor` plus a "Test głosu" run): about 20 typical replies, all fallback phrases and
  fillers, the greeting, and several numbers, names and questions. Listen for verbatim reading, Polish accent,
  fillers, and the model "answering" a question instead of reading it. Note the stage time and how often texts fall
  to near match or reject (logged), and compare cost with today's.

## Open questions
1. Does the model answer questions in the text ("Wiesz, jak to działa?") instead of reading them? Only the listening
   test can tell.
2. Do digits come back as `3` or as `trzy` in the transcript? If the near-match rate is high, ask the chat persona to
   write numbers as words.
3. Is 0.90 right for fillers and one-word texts? It may need an exact-match rule below about 10 characters.
4. Per-minute cost of realtime audio output vs today (tokens per second not documented).
