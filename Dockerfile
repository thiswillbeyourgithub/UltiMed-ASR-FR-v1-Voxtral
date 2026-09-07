# Voxtral text->audio (TTS) served via vLLM's "omni" stack.
#
# This is NOT CrispASR: mistralai/Voxtral-4B-TTS-2603 is a vLLM model. As of
# vllm-omni 0.22 there is an OFFICIAL prebuilt image (vllm/vllm-omni) that
# already bundles the matching vLLM + vllm-omni + the deploy YAMLs, so we build
# a layer on top of it instead of the old base-image + `pip install vllm-omni` +
# hand-built flash-attn wheel. -x86_64 pins the arch (there are also -aarch64 and
# multi-arch tags); bump the tag to move vllm-omni versions.
#
# The layer is no longer thin: it reinstalls vllm_omni from ./vllm-omni, a local
# clone of vllm-project/vllm-omni at this same tag whose `perso` branch adds the
# per-request cfg_alpha / seed / euler_steps routing that upstream has no way to
# express. Keep the clone's tag and the FROM tag in step, or the patched Python
# will run against a vLLM it was never written for.
FROM vllm/vllm-omni:v0.22.0-x86_64

# compose runs this container as a non-root uid (default 1000) so ./cache stays
# host-owned, but the base image has no /etc/passwd entry for that uid. vLLM
# startup calls getpass.getuser() -> pwd.getpwuid(), which then dies with
# "getpwuid(): uid not found: 1000". Bake a matching user so the lookup resolves.
# APP_UID/APP_GID are fed from VOXTRAL_UID/GID by docker-compose build.args, so
# this tracks whatever uid compose actually runs as.
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd -g "${APP_GID}" voxtral 2>/dev/null || true \
 && useradd -u "${APP_UID}" -g "${APP_GID}" -M -d /cache -s /usr/sbin/nologin voxtral 2>/dev/null || true

# The prebuilt image does NOT ship the standalone flash-attn package, so without
# this the stage-1 acoustic decoder logs "flash_attn is not installed. Falling
# back to PyTorch SDPA". Restore it with a matching prebuilt wheel (no slow source
# build). This image is cp312 / cu13 / cxx11abi TRUE with torch 2.11.0; flash-attn
# has no torch2.11 wheel yet (2.8.3.post1 tops out at torch2.9, 2.8.1 at torch2.10),
# but the torch2.10 cu13 wheel is ABI-compatible here (only the torch minor differs;
# Python/CUDA-major/ABI all match) and was verified in-image: install + import + a
# GPU flash_attn_func forward pass all pass with finite output. Bump to a real
# torch2.11 wheel once Dao-AILab ships one. %2B is the URL-encoded '+' in the tag.
RUN pip install --no-cache-dir \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1%2Bcu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

# Build vllm_omni from the LOCAL clone instead of using the copy bundled in the
# base image, so our Voxtral patches (per-request cfg_alpha, seed and
# euler_steps) are what actually serves. The clone lives at ./vllm-omni, is
# gitignored by the parent repo, and carries the patches on its `perso` branch;
# patches/*.patch is the tracked backup, see that directory's README.
#
# Kept AFTER the flash-attn layer on purpose: the wheel install is the slow,
# stable layer, so editing the clone rebuilds only from here down.
#
# --no-deps because the base image already resolved the whole dependency tree at
# the matching versions; letting pip re-resolve would risk pulling a different
# vLLM under our feet. VLLM_OMNI_VERSION_OVERRIDE skips setuptools_scm's git
# archaeology (the clone is shallow and its HEAD is an untagged branch, which
# would otherwise version the package as 0.0.dev*).
COPY vllm-omni /src/vllm-omni
RUN VLLM_OMNI_VERSION_OVERRIDE=0.22.0 pip install --no-cache-dir --no-deps /src/vllm-omni

# Fail the build loudly if the local copy did NOT end up shadowing the bundled
# one: a silently-stock vllm_omni would look exactly like "my patch does
# nothing", which is expensive to debug from the outside. Greps the installed
# files rather than importing (importing vllm_omni pulls in vLLM and wants a GPU).
RUN set -eux; \
    SP="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    grep -q "_apply_voxtral_request_overrides" "$SP/vllm_omni/entrypoints/openai/serving_speech.py"; \
    grep -q "_make_frame_noise" "$SP/vllm_omni/model_executor/models/voxtral_tts/voxtral_tts.py"; \
    echo "voxtral: forked vllm_omni installed at $SP"

# The entrypoint copies the bundled voxtral_tts deploy config, patches the tuning
# knobs into it, and runs `vllm serve MODEL --omni --deploy-config <patched>`.
COPY entrypoint.sh /usr/local/bin/voxtral-tts-entrypoint.sh
RUN chmod +x /usr/local/bin/voxtral-tts-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/voxtral-tts-entrypoint.sh"]
