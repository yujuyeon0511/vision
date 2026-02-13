#!/bin/bash
# ============================================================================
# TBI-MLLM Multi-Node Environment Setup Script
# Run this from the master node (ai-server-18, 192.168.0.28)
# Replicates conda env 'tbi-mllm' to all worker nodes
# ============================================================================

set -e

MASTER_NODE="192.168.0.28"
WORKER_NODES=("192.168.0.229" "192.168.0.180" "192.168.0.177")
ALL_NODES=("$MASTER_NODE" "${WORKER_NODES[@]}")
CONDA_ENV="tbi-mllm"
CONDA_BASE="$HOME/miniconda3"
USER=$(whoami)

echo "=== TBI-MLLM Multi-Node Setup ==="
echo "Master: $MASTER_NODE (ai-server-18)"
echo "Workers: ${WORKER_NODES[*]}"
echo "Conda env: $CONDA_ENV"
echo ""

# ============================================================================
# Step 1: Export conda environment from master
# ============================================================================
echo "[Step 1] Exporting conda environment..."
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV
pip freeze > /tmp/tbi-mllm-requirements.txt
conda list --export > /tmp/tbi-mllm-conda-list.txt
echo "  Exported $(wc -l < /tmp/tbi-mllm-requirements.txt) pip packages"

# ============================================================================
# Step 2: SSH connectivity check
# ============================================================================
echo ""
echo "[Step 2] Checking SSH connectivity..."
for node in "${WORKER_NODES[@]}"; do
    if ssh -o ConnectTimeout=5 -o BatchMode=yes "$USER@$node" "hostname" &>/dev/null; then
        echo "  $node: OK ($(ssh $USER@$node hostname))"
    else
        echo "  $node: FAILED - SSH key setup required!"
        echo ""
        echo "  Fix: Run on master node:"
        echo "    ssh-keygen -t rsa -N '' -f ~/.ssh/id_rsa  # (skip if key exists)"
        echo "    ssh-copy-id $USER@$node"
        echo ""
        exit 1
    fi
done

# ============================================================================
# Step 3: Setup conda env on each worker
# ============================================================================
echo ""
echo "[Step 3] Setting up conda environment on worker nodes..."
for node in "${WORKER_NODES[@]}"; do
    echo ""
    echo "--- Setting up $node ---"

    # Check if conda exists
    ssh "$USER@$node" "test -f $CONDA_BASE/bin/conda" || {
        echo "  ERROR: conda not found on $node at $CONDA_BASE"
        echo "  Install miniconda first: https://docs.conda.io/en/latest/miniconda.html"
        continue
    }

    # Create env if not exists
    ssh "$USER@$node" "
        eval \"\$($CONDA_BASE/bin/conda shell.bash hook)\"
        if conda env list | grep -q $CONDA_ENV; then
            echo '  Conda env $CONDA_ENV already exists, updating...'
        else
            echo '  Creating conda env $CONDA_ENV...'
            conda create -n $CONDA_ENV python=3.11 -y
        fi
    "

    # Copy requirements and install
    scp /tmp/tbi-mllm-requirements.txt "$USER@$node:/tmp/"
    ssh "$USER@$node" "
        eval \"\$($CONDA_BASE/bin/conda shell.bash hook)\"
        conda activate $CONDA_ENV
        pip install -r /tmp/tbi-mllm-requirements.txt 2>&1 | tail -3
        echo '  Done: \$(python --version), torch=\$(python -c \"import torch; print(torch.__version__)\")'
    "
done

# ============================================================================
# Step 4: Verify GPU access on all nodes
# ============================================================================
echo ""
echo "[Step 4] Verifying GPU access on all nodes..."
for node in "${ALL_NODES[@]}"; do
    echo -n "  $node: "
    if [ "$node" == "$MASTER_NODE" ]; then
        eval "$(conda shell.bash hook)"
        conda activate $CONDA_ENV
        python -c "import torch; print(f'{torch.cuda.device_count()} GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}')"
    else
        ssh "$USER@$node" "
            eval \"\$($CONDA_BASE/bin/conda shell.bash hook)\"
            conda activate $CONDA_ENV
            python -c \"import torch; print(f'{torch.cuda.device_count()} GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}')\"
        "
    fi
done

# ============================================================================
# Step 5: NCCL connectivity test
# ============================================================================
echo ""
echo "[Step 5] Testing NCCL multi-node communication..."
echo "  Run manually on master node:"
echo ""
echo "  deepspeed --hostfile experiments/configs/multi-node/hostfile \\"
echo "    --master_addr $MASTER_NODE --master_port 29500 \\"
echo "    experiments/scripts/test_multinode.py"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "=== Setup Complete ==="
echo ""
echo "Multi-node configuration:"
echo "  Master: $MASTER_NODE (ai-server-18)"
for node in "${WORKER_NODES[@]}"; do
    echo "  Worker: $node"
done
echo ""
echo "Total GPUs: 8x A100-PCIE-40GB"
echo "Conda env: $CONDA_ENV"
echo ""
echo "To launch multi-node training:"
echo "  deepspeed --hostfile experiments/configs/multi-node/hostfile \\"
echo "    --master_addr $MASTER_NODE --master_port 29500 \\"
echo "    experiments/scripts/train.py \\"
echo "    --deepspeed experiments/configs/multi-node/ds_config_zero2_multinode.json \\"
echo "    --config experiments/configs/EXP-20260211-001-config.yaml \\"
echo "    --variant F4"
