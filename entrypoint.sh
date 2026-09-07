#!/usr/bin/env bash
# Serve mistralai/Voxtral-4B-TTS-2603 via vllm-omni (0.22.x) on 0.0.0.0 so the
# host can reach it. In 0.22 the model is registered, so `vllm serve MODEL --omni`
# auto-loads the bundled deploy YAML (vllm_omni/deploy/voxtral_tts.yaml); we copy
# that YAML, patch a few knobs, and hand it back with --deploy-config.
#
# Voxtral is a TWO-STAGE pipeline (see the deploy YAML, top-level `stages:` list):
#   stage_id 0 = AR LLM, text -> audio tokens. Compiled (enforce_eager: false),
#                greedy (temperature 0.0), gpu_memory_utilization ~0.8. It also
#                owns the flow-matching acoustic transformer, so the real quality
#                levers (VOXTRAL_EULER_STEPS, VOXTRAL_CFG_ALPHA) are stage-0 costs.
#   stage_id 1 = audio tokenizer, tokens -> 24 kHz wave. A deterministic codec
#                decoder with no sampling path, so its temperature is very likely
#                a no-op (see the VOXTRAL_ACOUSTIC_TEMP note below).
# The knobs below are DEFAULTS. Since this image builds vllm_omni from our
# patched clone (voxtral/vllm-omni, branch `perso`, see the Dockerfile), a request
# can now override cfg_alpha, the noise seed, the Euler step count and stage 1's
# temperature per call, with no restart and no sweep-by-`up -d`. What is still
# startup-only is everything the deploy YAML fixes for the whole engine:
# max_num_seqs, gpu_memory_utilization, max_model_len, and the engine seed.
# Changing a knob's VALUE needs no rebuild, just `up -d`. Changing THIS FILE does:
# the Dockerfile COPYs it into the image, so a newly added knob is invisible until
# `up -d --build`. The "patched ... -> ..." summary line this script logs at boot
# lists every knob it knows about, which is how you tell the two apart.
#
# NOTE: the 0.22 schema differs from 0.18 (`stages` with flat fields, not
# `stage_args`+`engine_args`+`runtime`). The old `runtime.defaults.max_inflight`
# lever is gone; per the maintainers the acoustic stage is `codec-bs=1`, so audio
# throughput saturates around client concurrency 4-8 regardless.
set -euo pipefail

MODEL="${VOXTRAL_TTS_MODEL:-mistralai/Voxtral-4B-TTS-2603}"
PORT="${VOXTRAL_TTS_PORT:-8003}"

TUNED_YAML=/tmp/voxtral_tts_tuned.yaml

# Locate the bundled deploy YAML AND patch a copy of it in ONE python process, so
# there is no `$(python3 ...)` capture to pollute. (Capturing the path separately
# broke: `import vllm_omni` prints a monkey-patch WARNING to stdout, which got
# concatenated onto the path.) Check both the 0.22 path (deploy/) and the legacy
# 0.18 path so this survives a version bump. Only the knobs an env var asks for
# are touched:
#   VOXTRAL_ACOUSTIC_TEMP -> stage_id 1 sampling temperature. Default 0.7 (lowered
#                            from upstream 0.9 for steadier, less artifacty audio;
#                            raise toward 0.9 for more expressive/varied output).
#                            Suspected no-op: stage 1 is a deterministic codec
#                            decoder with no sampling path. Our fork lets a request
#                            override it (`acoustic_temperature`) so the claim can
#                            be tested without a restart.
#   VOXTRAL_SEED          -> vLLM engine seed on BOTH stages (NOT a sampling param,
#                            see the long note at the patch site: the sampling seed
#                            cannot affect Voxtral's output, the engine seed can
#                            because it seeds the global RNG the flow-matching noise
#                            is drawn from). vLLM already defaults it to 0, so this
#                            selects a different deterministic stream rather than
#                            turning determinism on, and it only governs requests
#                            that carry NO seed of their own: our fork draws a seeded
#                            request's noise from its own generator instead, which is
#                            what makes one specific take reproducible without a
#                            restart (`tts.py --noise-seed N`, or a plain `seed`
#                            field, which upstream would have dropped into the inert
#                            sampling half).
#   VOXTRAL_MAX_NUM_SEQS  -> both stages' max_num_seqs (packaged 32). The acoustic
#                            stage is codec-bs=1, so raising this past the ~4-8
#                            concurrency saturation point buys little.
#   VOXTRAL_CFG_ALPHA     -> stage_id 0 classifier-free guidance scale (default
#                            1.2). Nudge up for tighter text adherence; NEVER 1.0
#                            (disables CFG -> garbled/off-text audio; refused).
#                            Overridable per request (`tts.py --cfg-alpha`), so this
#                            is only the default. KEEP IT SET: stage 0's
#                            default_sampling_params.extra_args must be non-empty or
#                            vllm-omni sets has_sampling_extra_args=False and drops
#                            EVERY per-request extra, the noise seed included.
#   VOXTRAL_GPU_MEM_UTIL  -> stage_id 0 gpu_memory_utilization (packaged 0.8).
#                            Lower it (e.g. 0.7) so stage 0 + stage 1's 0.1 fit on
#                            a 24GB card shared with a desktop; unset = 0.8.
#   VOXTRAL_MAX_TOKENS    -> stage_id 0's default_sampling_params.max_tokens, i.e.
#                            the longest utterance one request can produce (packaged
#                            2048). Stage 0 emits exactly one 12.5 Hz audio frame
#                            per decode step, so 2048 frames = 163.84 s and longer
#                            texts are cut off mid-sentence (finish_reason
#                            "length", no error). Raise it together with
#                            VOXTRAL_MAX_MODEL_LEN below. Note tts.py
#                            --max-new-tokens overrides this per request (up to
#                            vllm-omni's _TTS_MAX_NEW_TOKENS_MAX = 4096).
#   VOXTRAL_MAX_MODEL_LEN -> stage_id 0's max_model_len (packaged 4096). Prompt text
#                            tokens and generated frames SHARE it, so it is the
#                            real ceiling: any max_tokens beyond it is unreachable
#                            because vLLM stops the request at the context limit.
#                            Not a model limit (the checkpoint's params.json says
#                            max_seq_len 65536), but each token of context costs
#                            ~104 KiB of KV cache (2 x 26 layers x 8 kv heads x 128
#                            head_dim x 2 bytes), so raising it trades concurrency
#                            for length within the same gpu_memory_utilization pool.
# (VOXTRAL_EULER_STEPS is handled separately below: it patches the model's
#  params.json, not this deploy YAML. It too is now only a default: a request can
#  carry `euler_steps`, which our fork honours when every request sharing that
#  decode step agrees on the value.)
VOXTRAL_ACOUSTIC_TEMP="${VOXTRAL_ACOUSTIC_TEMP:-0.7}" \
python3 - "$TUNED_YAML" <<'PY'
import os, sys, vllm_omni, yaml

dst = sys.argv[1]
base = os.path.dirname(vllm_omni.__file__)
src = next((p for p in (
    os.path.join(base, "deploy/voxtral_tts.yaml"),
    os.path.join(base, "model_executor/stage_configs/voxtral_tts.yaml"),
) if os.path.exists(p)), None)
if src is None:
    sys.exit("voxtral entrypoint: could not find voxtral_tts.yaml under %s" % base)

with open(src) as f:
    cfg = yaml.safe_load(f)

temp = os.environ.get("VOXTRAL_ACOUSTIC_TEMP")  # always set (default injected above)
seed = os.environ.get("VOXTRAL_SEED")
max_num_seqs = os.environ.get("VOXTRAL_MAX_NUM_SEQS")
gpu_mem_util = os.environ.get("VOXTRAL_GPU_MEM_UTIL")  # stage 0 GPU memory fraction
cfg_alpha = os.environ.get("VOXTRAL_CFG_ALPHA")  # stage 0 classifier-free guidance
max_tokens = os.environ.get("VOXTRAL_MAX_TOKENS")  # stage 0 output-length cap
max_model_len = os.environ.get("VOXTRAL_MAX_MODEL_LEN")  # stage 0 total context


def _bounded_int(name, raw, minimum=1):
    """Parse an integer knob, refusing junk at boot instead of letting pyyaml write a
    string into a field vLLM will later reject with a much less obvious error."""
    try:
        value = int(raw)
    except ValueError:
        sys.exit("voxtral entrypoint: %s must be an integer, got %r" % (name, raw))
    if value < minimum:
        sys.exit("voxtral entrypoint: %s must be >= %d, got %d" % (name, minimum, value))
    return value


# Guardrail: cfg_alpha=1.0 disables CFG, which makes the flow-matching decoder
# hallucinate phonetically-plausible-but-wrong audio (it stops following the
# text). The paper/default is 1.2; refuse the known-broken value outright.
if cfg_alpha is not None and abs(float(cfg_alpha) - 1.0) < 1e-9:
    sys.exit(
        "voxtral entrypoint: VOXTRAL_CFG_ALPHA=1.0 disables classifier-free "
        "guidance and produces garbled/off-text audio. Use ~1.2 (default); "
        "raise slightly for tighter text adherence."
    )

for stage in cfg.get("stages", []):
    sampling = stage.setdefault("default_sampling_params", {})
    # Seed goes on the STAGE (a vLLM EngineArgs field, reaching ModelConfig.seed via
    # StageDeployConfig.engine_extras), NOT in default_sampling_params. The sampling
    # seed is provably inert here: stage 0 hands vLLM's Sampler the output of
    # fake_logits_for_audio_tokens(), a vocab-wide -inf row with exactly ONE finite
    # entry, so the argmax is forced no matter what seed/temperature/top_p do; and
    # stage 1 never calls compute_logits at all. The engine seed is what matters,
    # because the ONE random draw in the whole pipeline (the Gaussian each frame's
    # flow-matching ODE starts from) comes from the global torch RNG, which vLLM
    # re-seeds to ModelConfig.seed after warmup (v1/worker/gpu_worker.py). Default is
    # already 0, so takes are deterministic per process from the first request; this
    # only picks a different stream. Reproducing one specific request still needs a
    # restart, since every generated frame advances that global RNG.
    if seed:
        stage["seed"] = _bounded_int("VOXTRAL_SEED", seed, minimum=0)
    if max_num_seqs:
        stage["max_num_seqs"] = int(max_num_seqs)
    # stage_id 1 = audio_tokenizer (latents -> waveform). SUSPECTED NO-OP: that model
    # has no sampling path and never calls compute_logits, so this temperature very
    # likely changes nothing. Kept because it is harmless and unverified on real
    # hardware; A/B 0.7 vs 0.9 on identical text (byte-identical output = inert).
    if stage.get("stage_id") == 1 and temp:
        sampling["temperature"] = float(temp)
    # stage_id 0 = AR LLM; cfg_alpha lives in ITS sampling extra_args (that is
    # where vllm-omni's voxtral model reads per-request cfg_alpha from too).
    if stage.get("stage_id") == 0 and cfg_alpha:
        sampling.setdefault("extra_args", {})["cfg_alpha"] = float(cfg_alpha)
    # stage_id 0 grabs gpu_memory_utilization: 0.8 by default, which leaves too
    # little for stage 1's 0.1 (+ CUDA-context + desktop usage) on a 24GB card
    # shared with a display. Lower it (e.g. 0.7) to make the two stages fit.
    if stage.get("stage_id") == 0 and gpu_mem_util:
        stage["gpu_memory_utilization"] = float(gpu_mem_util)
    # stage_id 0 also owns audio LENGTH: one decode step = one 12.5 Hz frame, so
    # max_tokens/12.5 is the longest single utterance (packaged 2048 -> 163.84 s),
    # and max_model_len bounds prompt tokens + frames together. Only stage 0: stage
    # 1 is fed 50-frame chunks by generator2tokenizer_async_chunk and never sees the
    # whole utterance, so its own max_tokens does not cap duration.
    if stage.get("stage_id") == 0 and max_tokens:
        sampling["max_tokens"] = _bounded_int("VOXTRAL_MAX_TOKENS", max_tokens)
    if stage.get("stage_id") == 0 and max_model_len:
        stage["max_model_len"] = _bounded_int("VOXTRAL_MAX_MODEL_LEN", max_model_len)

# The prompt always costs tokens, so a max_tokens that reaches max_model_len can
# never be spent in full: vLLM stops the request at the context limit instead
# (finish_reason "length"). Warn rather than refuse, since the packaged 2048/4096
# pair is deliberately in that shape and short texts never notice.
stage0 = next((s for s in cfg.get("stages", []) if s.get("stage_id") == 0), {})
eff_tokens = stage0.get("default_sampling_params", {}).get("max_tokens")
eff_ctx = stage0.get("max_model_len")
if eff_tokens and eff_ctx and eff_tokens >= eff_ctx:
    sys.stderr.write(
        "voxtral entrypoint: WARNING stage 0 max_tokens=%d >= max_model_len=%d, so "
        "audio length is capped by the context (under %.1f s at 12.5 Hz, minus the "
        "prompt), not by max_tokens. Raise VOXTRAL_MAX_MODEL_LEN to use it all.\n"
        % (eff_tokens, eff_ctx, eff_ctx / 12.5)
    )

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

sys.stderr.write(
    "voxtral entrypoint: patched %s -> %s "
    "(acoustic_temp=%s seed=%s max_num_seqs=%s cfg_alpha=%s gpu_mem_util=%s "
    "max_tokens=%s [%.2f s of audio] max_model_len=%s)\n"
    % (src, dst, temp, seed or "default", max_num_seqs or "default",
       cfg_alpha or "default", gpu_mem_util or "default",
       max_tokens or "default", (eff_tokens or 0) / 12.5, max_model_len or "default")
)
PY

# VOXTRAL_EULER_STEPS -> n_decoding_steps: the number of Euler ODE steps the
# flow-matching acoustic transformer runs per frame. That transformer lives in
# STAGE 0 (voxtral_tts_audio_generation.py), not stage 1, so this is a stage-0
# cost even though it shapes the waveform. THIS is the compute<->
# quality lever the user asked for: more steps = steadier/higher-fidelity audio
# for more compute, fewer = faster/rougher. Unlike temp/seed/cfg_alpha it is NOT
# a deploy-YAML field, env var, or per-request knob in vllm-omni. vllm-omni reads
# it ONLY from the model's params.json at
#   multimodal.audio_model_args.acoustic_transformer_args.n_decoding_steps
# and this model does not ship the key, so vllm-omni's mistral parser WARNS and
# falls back to 7 (parsers/voxtral_tts.py). So, exactly as that warning tells you
# to, we inject the key into params.json. We pre-fetch just params.json with
# hf_hub_download (tiny; uses HF_TOKEN if the repo is gated) so this also works on
# the very first boot before the ~9GB weights download, and re-patch every boot
# (idempotent) so it survives HF restoring the original file. Leave the var unset
# to keep vllm-omni's default of 7.
if [ -n "${VOXTRAL_EULER_STEPS:-}" ]; then
  python3 - "$MODEL" "$VOXTRAL_EULER_STEPS" <<'PY'
import sys, os, json
from huggingface_hub import hf_hub_download

model, raw = sys.argv[1], sys.argv[2]
try:
    steps = int(raw)
except ValueError:
    sys.exit("voxtral entrypoint: VOXTRAL_EULER_STEPS must be an integer, got %r" % raw)
if steps < 1:
    sys.exit("voxtral entrypoint: VOXTRAL_EULER_STEPS must be >= 1, got %d" % steps)

path = os.path.realpath(hf_hub_download(model, "params.json"))
with open(path) as f:
    cfg = json.load(f)
ata = (cfg.setdefault("multimodal", {})
          .setdefault("audio_model_args", {})
          .setdefault("acoustic_transformer_args", {}))
old = ata.get("n_decoding_steps")
ata["n_decoding_steps"] = steps
os.chmod(path, 0o644)  # HF cache blobs can be read-only; make writable
with open(path, "w") as f:
    json.dump(cfg, f)
sys.stderr.write(
    "voxtral entrypoint: set acoustic_transformer_args.n_decoding_steps "
    "%s -> %d in %s\n" % (old if old is not None else "unset(default 7)", steps, path)
)
PY
fi

# Per-request seeding used to be bolted on here, by putting a sitecustomize.py
# monkeypatch on PYTHONPATH when VOXTRAL_PER_REQUEST_SEED was set. Both the var and
# the patch are gone: the image now installs our patched vllm_omni (see the
# Dockerfile), where a request's seed reaches the flow-matching noise draw
# directly, per request, with nothing to enable and no way to half-enable it.
# Report what the running vllm_omni actually supports, since "the env var was set
# and nothing happened" was exactly the failure mode that cost the most time.
python3 - <<'PY' || true
import sys, sysconfig, os
sp = sysconfig.get_paths()["purelib"]
target = os.path.join(sp, "vllm_omni", "model_executor", "models", "voxtral_tts", "voxtral_tts.py")
try:
    with open(target) as f:
        forked = "_make_frame_noise" in f.read()
except OSError as exc:
    sys.stderr.write("voxtral entrypoint: could not inspect %s (%s)\n" % (target, exc))
    raise SystemExit(0)
sys.stderr.write(
    "voxtral entrypoint: vllm_omni at %s is %s -> per-request cfg_alpha/seed/euler_steps %s\n"
    % (sp, "PATCHED (perso fork)" if forked else "STOCK", "available" if forked else "NOT available")
)
PY

# 0.22 serve form: `vllm serve MODEL --omni` (the vllm-omni plugin registers the
# omni pipeline into vllm). --deploy-config feeds our patched YAML. VLLM_EXTRA_ARGS
# is an opt-in escape hatch, left unquoted so an empty value expands to nothing.
exec vllm serve "$MODEL" \
  --omni \
  --deploy-config "$TUNED_YAML" \
  --host 0.0.0.0 \
  --port "$PORT" \
  ${VLLM_EXTRA_ARGS:-}
