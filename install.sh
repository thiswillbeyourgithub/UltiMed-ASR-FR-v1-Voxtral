# source: https://huggingface.co/mistralai/Voxtral-4B-TTS-2603/discussions/29

# # to retry in a fresh env
# deactivate
# trash env_voxtral
# uv venv env_voxtral --python 3.10
# source env_voxtral/bin/activate

# to stop if we crash
set -eu

# installing stuff
uv pip install vllm==0.18.1 vllm-omni==0.18.0


# vllm serve mistralai/Voxtral-4B-TTS-2603 --omni

VOXTRAL_YAML=$(python -c 'import vllm_omni; import os; print(os.path.join(os.path.dirname(vllm_omni.__file__), "model_executor/stage_configs/voxtral_tts.yaml"))')
cp "$VOXTRAL_YAML" /tmp/voxtral_tts_tuned.yaml
# sed -i 's/gpu_memory_utilization: 0.8/gpu_memory_utilization: 0.35/g' /tmp/voxtral_tts_tuned.yaml
vllm-omni serve mistralai/Voxtral-4B-TTS-2603 \
 --omni \
 --stage-configs-path /tmp/voxtral_tts_tuned.yaml \
 --port 8003 --host 127.0.0.1 --enforce-eager

set +eu
