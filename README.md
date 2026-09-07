# UltiMed-ASR-FR-v1-Voxtral

The Docker container that spoke **[UltiMed-ASR-FR-v1](https://huggingface.co/datasets/Olicorne/UltiMed-ASR-FR-v1)**: `mistralai/Voxtral-4B-TTS-2603` served by vLLM + [vllm-omni](https://github.com/vllm-project/vllm-omni), on one consumer GPU, behind an OpenAI-compatible `POST /v1/audio/speech` on port 8003.

Written with Claude Code.

> **This is a snapshot, not a product.** It is a copy of the `voxtral/` directory of a private machine's TTS setup, flattened into a standalone repository so the audio side of the dataset is reproducible instead of being a black box. There is no installer and no orchestration: you get a Dockerfile, an entrypoint, a compose file, the upstream patches, and a client script. If the paragraphs below do not answer your question, read the comments in `entrypoint.sh` and `docker-compose.yml`, which are where the real documentation lives.

## Contents

- [What is in this repository](#what-is-in-this-repository)
- [Running it](#running-it)
- [How UltiMed-ASR-FR-v1 was generated with it](#how-ultimed-asr-fr-v1-was-generated-with-it)
- [Seeding and reproducibility](#seeding-and-reproducibility)
- [Related repositories](#related-repositories)
- [Licensing](#licensing)

## What is in this repository

| File | What it is |
| --- | --- |
| `Dockerfile` | The image: official `vllm/vllm-omni:v0.22.0-x86_64`, plus a prebuilt flash-attn wheel, plus `vllm_omni` reinstalled from a **local patched clone** (see `patches/`). |
| `entrypoint.sh` | Copies the bundled `voxtral_tts` deploy YAML, patches the tuning knobs from `VOXTRAL_*` env vars into it, and runs `vllm serve MODEL --omni --deploy-config <patched>`. Heavily commented; it is the reference for what each knob actually reaches. |
| `docker-compose.yml` | The `voxtral-tts` service: GPU reservation, `./cache` mount for the ~9 GB of weights, and the tuning env vars with their rationale. |
| `patches/` | The four `git format-patch` files that turn upstream vllm-omni `v0.22.0` into the `perso` branch the image builds from: per-request `cfg_alpha`, `seed` (flow-matching noise) and `euler_steps`. See `patches/README.md`. |
| `tts.py` | The interactive client. A verbatim copy of the origin repository's `tts.py`, so it also carries flags for a TADA backend that is **not** shipped here; ignore those (its docstring says which). |
| `test_seed_determinism.sh` | The experiment that established what is and is not reproducible in this pipeline, and that the patches actually work. Owns the container lifecycle. |
| `install.sh` | The bare-metal predecessor of the container (`uv pip install vllm vllm-omni`, then `vllm-omni serve`). Kept because it is the shortest possible statement of what the image does. |
| `env.example` | The three optional compose overrides (uid/gid/GPU count). Copy to `.env` if the defaults do not fit. |
| `env.reference` | A `pip list` taken inside an earlier build of this image. Historical record of an exact working dependency set, not a lockfile: it predates the `v0.22.0` upgrade, so it lists vllm 0.18.1. |

## Running it

The `vllm-omni` clone is gitignored, so a fresh checkout must recreate it **before** the first build:

```bash
git clone --branch v0.22.0 --depth 1 https://github.com/vllm-project/vllm-omni.git vllm-omni
cd vllm-omni
git checkout -b perso
git am ../patches/*.patch
cd ..
```

Then:

```bash
sudo docker compose up -d --build voxtral-tts   # first boot downloads ~9GB of weights
./tts.py --check                                # is it alive?
./tts.py -t "Bonjour, ceci est un test." --voice fr_female -o out.wav
```

The build fails loudly if the patched `vllm_omni` did not end up shadowing the one bundled in the base image, because a silently-stock server looks exactly like "my patch does nothing".

Requirements: an NVIDIA GPU with enough VRAM for both stages (developed on a 24 GB card, sharing it with a desktop, hence `VOXTRAL_GPU_MEM_UTIL=0.7`), Docker with the NVIDIA runtime, and about 9 GB of disk for the weights.

## How UltiMed-ASR-FR-v1 was generated with it

The 601,338 clips of the dataset were synthesized against this server between 2026-07-05 and 2026-08-19, in a single `fr_female` voice, 24 kHz mono, no post-processing.

The client that drove the bulk generation is **not** `tts.py` but [`05_generate_audio/01_generate_audio.py`](https://github.com/thiswillbeyourgithub/UltiMed-ASR-FR-v1-scripts/blob/main/05_generate_audio/01_generate_audio.py) in the scripts repository: a simplified, resumable batch client that reads the pipeline's JSONL and writes one file per row. `tts.py` here is the interactive tool used to tune the server and listen to single takes.

Two things about the text side are worth knowing, and both live in the scripts repository rather than here: the text handed to this server is a **deterministically normalized spoken form** of the written label (`utils/voxtral_normalize.py`), and every generated clip is transcribed back and scored, with bad clips regenerated (stage `06_hotfixes`). The model-specific pronunciation failures that motivated the normalizer are catalogued in [`01_dictionnary/VOXTRAL_QUIRKS.md`](https://github.com/thiswillbeyourgithub/UltiMed-ASR-FR-v1-scripts/blob/main/01_dictionnary/VOXTRAL_QUIRKS.md).

**Caveat on exactness.** This is the container at the end of the project, i.e. the lineage that produced the dataset, not a byte-exact snapshot of any single generation day. The `v0.22.0` base landed on 2026-07-05, at the very start of generation, but the per-request seeding patches in `patches/` landed on 2026-07-27 to 07-31, partway through: clips generated before that date came off the same model and the same tuning, with the noise seeding done by a host-mounted monkeypatch instead (the history section of `patches/README.md` describes it). The tuning values in `docker-compose.yml` are the ones the run ended on.

## Seeding and reproducibility

This is the non-obvious part, and the reason the fork exists.

Voxtral's stage 0 (the autoregressive LLM, text to audio tokens) hands vLLM's sampler a logits row with exactly **one** finite entry. So the usual sampling parameters, `seed` included, cannot change the output: the argmax is forced. The only random draw in the pipeline is the Gaussian each frame's flow-matching ODE starts from, and upstream takes it from the **global** torch RNG. The practical consequences, all four confirmed by `test_seed_determinism.sh`:

- a request's `seed` field does nothing;
- takes are deterministic per process, from the first request onward;
- repeated calls inside one process differ, because each generated frame advances one global stream;
- so the only way to repeat a take upstream is to restart the container and send the same request first.

Upstream also has no request-level route to the quality levers at all, and silently pretends it does: `OpenAICreateSpeechRequest` declares no `model_config`, so pydantic's default `extra='ignore'` **drops** unknown keys and still answers 200. A client sending `{"cfg_alpha": 1.5}` looks like it worked while every request runs at the server's startup default.

The patches fix both. `cfg_alpha`, `euler_steps` and `acoustic_temperature` become real request fields and are routed to the right stage, and each frame's start noise is drawn from its own `torch.Generator` seeded `SplitMix64(seed, frame_index)`. That makes a take reproducible **without restarting the container**, and unlike re-pinning the global RNG it survives concurrency, because each row gets its own stream rather than a slice of one sequence whose split depends on batch composition. In practice: `./tts.py --noise-seed 7 ...` returns identical audio at any point in the container's life.

Against a stock (unpatched) server those flags are merged and ignored rather than rejected, so sending them is always safe; it just silently changes nothing. `test_seed_determinism.sh` skips its own last two tests when it detects a stock image.

## Related repositories

This container is one piece of a French medical ASR stack; every other piece is public.

| Repository | What it is |
| --- | --- |
| [Olicorne/UltiMed-ASR-FR-v1](https://huggingface.co/datasets/Olicorne/UltiMed-ASR-FR-v1) | The dataset the fine-tune was trained on: 601,338 clips / 3,105 h of synthesized French medical speech, plus an eval-only PARROT subset. |
| [Olicorne/parakeet-tdt-0.6b-v3-UltiMed-onnx](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-UltiMed-onnx) | The French medical fine-tune trained on that dataset, exported to ONNX (fp32 / fp16 / int8 / w4a8). |
| [Olicorne/parakeet-tdt-0.6b-v3-optimized-onnx](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-optimized-onnx) | The multilingual baseline the fine-tune builds on: the upstream ONNX re-quantized for int8 accuracy on long audio and graph-optimized for browser speed. |
| [UltiMed-ASR-FR-v1-scripts](https://github.com/thiswillbeyourgithub/UltiMed-ASR-FR-v1-scripts) | The open recipe that built that dataset end to end: text sources, spoken-form normalization, the batch synthesis client, and the transcribe-and-rescore QC pass. |
| [UltiMed-ASR-FR-v1-NeMo_training_scripts](https://github.com/thiswillbeyourgithub/UltiMed-ASR-FR-v1-NeMo_training_scripts) | The NeMo fork and training configs used to run the fine-tune itself. |
| **This repository** | The Voxtral TTS container documented above. |
| [Parakeet Web](https://github.com/thiswillbeyourgithub/parakeet_web) | The in-browser ASR app that loads either ONNX model, live at [parakeetweb.olicorne.org](https://parakeetweb.olicorne.org/). Everything runs client-side. |

Upstream projects this repository is a thin layer over: [vllm-omni](https://github.com/vllm-project/vllm-omni), [vLLM](https://github.com/vllm-project/vllm), and the model [mistralai/Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603).

## Licensing

This repository is AGPLv3 (see `LICENSE`). The files under `patches/` are diffs against [vllm-omni](https://github.com/vllm-project/vllm-omni), which is Apache 2.0; they are redistributed here as patches, under that project's terms, purely so the build is reproducible.

Repository: <https://github.com/thiswillbeyourgithub/UltiMed-ASR-FR-v1-Voxtral>
