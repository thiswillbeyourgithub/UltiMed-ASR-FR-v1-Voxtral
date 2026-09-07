# Voxtral TTS patches (tracked backup of the `perso` fork)

Written with Claude Code.

The `voxtral-tts` image is built from a local clone of
[vllm-omni](https://github.com/vllm-project/vllm-omni) at `vllm-omni`, checked out at the tag matching the
base image (`v0.22.0`) with our changes on a branch named `perso`. That clone is **gitignored**: it is a full upstream
tree with its own history, and vendoring it would bury this repo's own diffs.

The `.patch` files here are the tracked backup of that branch, so the clone can be rebuilt from nothing. They are the
only copy of the work that survives losing the machine, so **regenerate them whenever `perso` gains a commit**:

```bash
cd vllm-omni
git format-patch --no-signature v0.22.0..perso -o ../patches/
```

Recreate the clone (this is also what a fresh checkout of this repo needs before its first `--build`):

```bash
git clone --branch v0.22.0 --depth 1 https://github.com/vllm-project/vllm-omni.git vllm-omni
cd vllm-omni
git checkout -b perso
git am ../patches/*.patch
```

## What the patches do

Upstream has no request-level route to Voxtral's quality levers, and worse, silently pretends it does:
`OpenAICreateSpeechRequest` declares no `model_config`, so pydantic v2's default `extra='ignore'` **drops** any
unknown flat key and answers 200. A client sending `{"cfg_alpha": 1.5}` looks like it worked while every request
actually runs at the server's startup default.

1. **`0001` request fields.** Declares `cfg_alpha`, `euler_steps` and `acoustic_temperature` as real fields and routes
   them, plus the pre-existing `seed`, in `serving_speech.py::_apply_voxtral_request_overrides`: the first two and the
   seed into stage 0's `extra_args`, `acoustic_temperature` into stage 1's sampling temperature. `extra_params` keeps
   working and still wins on conflict.
2. **`0002` model side.** Draws each frame's flow-matching start noise per request from its own `torch.Generator`
   (seeded `SplitMix64(seed, frame index)`) instead of from the global torch RNG, and reads a batch-uniform
   `euler_steps`. This is what makes a take reproducible **without restarting the container**, and unlike re-pinning
   the global RNG it survives concurrency, because each row gets its own stream rather than a slice of one sequence
   whose split depends on batch composition.
3. **`0003` tests.** `tests/model_executor/models/voxtral_tts/test_per_request_noise_and_steps.py`. Run them inside
   the built image (the host has no matching vLLM):
   ```bash
   sudo docker compose run --rm --entrypoint bash voxtral-tts -lc \
     "cd /src/vllm-omni && python -m pytest tests/model_executor/models/voxtral_tts/test_per_request_noise_and_steps.py -q"
   ```

## History: the monkeypatch this replaces

Until 2026-07-31 per-request seeding lived here as `sitecustomize.py`, a host-mounted monkeypatch enabled by
`VOXTRAL_PER_REQUEST_SEED=1` plus `PYTHONPATH=/opt/voxtral-patches`. It hooked `make_omni_output` and re-pinned the
**global** torch RNG before each frame's draw, which worked but could only serve one stream at a time (its own
docstring warned that reproducibility held per batch composition). Both the env var and the mount are gone: the fork
does the same job by construction, per request, with no toggle to forget.
