#!/bin/bash
# JAXpi 環境重建腳本 (修正版 - 支援 SOAP)
# 此腳本會完全移除並重新安裝所有必要的套件，並確保SOAP optimizer可用

set -e  # Exit on error

echo "========================================="
echo "JAXpi 環境重建腳本 (SOAP 支援版)"
echo "========================================="
echo ""

# 1. 移除所有相關套件
echo "步驟 1/6: 移除現有套件..."
pip uninstall -y jax jaxlib optax chex flax ml-dtypes numpy scipy torch torchvision torchaudio soap-jax 2>/dev/null || true

# 2. 安裝核心依賴（固定版本）
echo ""
echo "步驟 2/6: 安裝核心依賴..."
pip install 'numpy==1.26.4'
pip install 'scipy==1.11.4'
pip install 'ml-dtypes==0.2.0'

# 3. 安裝 JAX with CUDA 11 (不使用 --upgrade，避免依賴升級)
echo ""
echo "步驟 3/6: 安裝 JAX 0.4.23 with CUDA 11..."
pip install --no-deps 'jax==0.4.23'
pip install --no-deps 'jaxlib==0.4.23+cuda11.cudnn86' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 4. 安裝 JAXpi 相關套件 (不使用 --no-deps，讓套件自行處理依賴)
echo ""
echo "步驟 4/6: 安裝 Optax, Chex, Flax..."
pip install --no-deps 'optax==0.1.9'
pip install --no-deps 'chex==0.1.85'
pip install 'flax==0.7.5'
pip install 'wandb'
pip install 'matplotlib'
pip install 'ml_collections'
pip install 'absl-py'
pip install 'orbax-checkpoint'

# 5. 安裝 SOAP (從 GitHub)
echo ""
echo "步驟 5/6: 安裝 SOAP optimizer..."
pip install git+https://github.com/haydn-jones/SOAP_JAX.git

# 6. 安裝 PyTorch CPU only (避免 CUDA 庫衝突)
echo ""
echo "步驟 6/6: 安裝 PyTorch (CPU only)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "========================================="
echo "環境重建完成！"
echo "========================================="
echo ""
echo "驗證安裝..."

export LD_LIBRARY_PATH="${HOME}/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:${HOME}/.local/lib/python3.10/site-packages/nvidia/cublas/lib:${HOME}/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"

cd /tmp  # 切換到 /tmp 避免 jaxpi/logging.py 干擾
python3 -c "import jax; print('JAX version:', jax.__version__)" 2>&1 | grep -v libcudnn || true
python3 -c "import jax; print('Device count:', jax.device_count())" 2>&1 | grep -v libcudnn || true
python3 -c "import optax; print('Optax version:', optax.__version__)"
python3 -c "import flax; print('Flax version:', flax.__version__)"
python3 -c "import wandb; print('Wandb version:', wandb.__version__)"

echo ""
echo "測試 SOAP optimizer..."
python3 ~/jaxpi/jaxpi/optax_soap_patch.py 2>&1 | tail -3

echo ""
echo "========================================="
echo "安裝完成！可以開始訓練了。"
echo "========================================="
