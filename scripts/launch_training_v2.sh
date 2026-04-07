#!/bin/bash
# Launch LoRA v2 training on srvlc01-dev with 2× L4 GPUs
#
# Key differences from v1:
#   - cond_enc FROZEN (speaker identity preserved)
#   - LoRA r=8 (was 64)
#   - 2× L4 GPUs via accelerate DDP
#   - Zero speaker conditioning during training
#
# Usage:
#   scp scripts/train_lora_v2.py srvlc01-dev:/opt/chatterbox-training/scripts/
#   scp scripts/launch_training_v2.sh srvlc01-dev:/opt/chatterbox-training/scripts/
#   ssh srvlc01-dev "bash /opt/chatterbox-training/scripts/launch_training_v2.sh"

set -e

TRAINING_DIR="/opt/chatterbox-training"
CONTAINER_NAME="chatterbox-training-v2"
IMAGE="chatterbox-training:latest"

echo "=== LoRA v2 Training (speaker-safe, 2× L4) ==="
echo "  Container: ${CONTAINER_NAME}"
echo "  Image: ${IMAGE}"
echo "  GPUs: 2× NVIDIA L4"
echo ""

# Stop existing training container if running
docker rm -f ${CONTAINER_NAME} 2>/dev/null || true

# Create accelerate config for 2-GPU DDP
mkdir -p ${TRAINING_DIR}/config
cat > ${TRAINING_DIR}/config/accelerate_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: '0,1'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

echo "Accelerate config created."

# Launch training container with both GPUs
docker run -d \
    --name ${CONTAINER_NAME} \
    --gpus all \
    --shm-size=16g \
    --restart unless-stopped \
    -v ${TRAINING_DIR}/data:/workspace/data:ro \
    -v ${TRAINING_DIR}/checkpoints_v2:/workspace/checkpoints_v2 \
    -v ${TRAINING_DIR}/scripts:/workspace/scripts:ro \
    -v ${TRAINING_DIR}/config:/workspace/config:ro \
    -v ${TRAINING_DIR}/models:/workspace/models \
    -e NCCL_P2P_DISABLE=0 \
    -e NCCL_IB_DISABLE=1 \
    -e HF_HOME=/workspace/models \
    ${IMAGE} \
    bash -c '
        # Set accelerate config
        mkdir -p ~/.cache/huggingface/accelerate
        cp /workspace/config/accelerate_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml

        # Launch multi-GPU training
        accelerate launch \
            --config_file /workspace/config/accelerate_config.yaml \
            /workspace/scripts/train_lora_v2.py \
            --lora-r 8 \
            --lr 1e-4 \
            --max-steps 20000 \
            --batch-size 4 \
            --save-every 2000
    '

echo ""
echo "Container started: ${CONTAINER_NAME}"
echo ""
echo "Monitor training:"
echo "  docker logs -f ${CONTAINER_NAME}"
echo "  tail -f ${TRAINING_DIR}/checkpoints_v2/train.log"
echo ""
echo "Check GPU usage:"
echo "  nvidia-smi"
echo ""
echo "Stop training:"
echo "  docker stop ${CONTAINER_NAME}"
