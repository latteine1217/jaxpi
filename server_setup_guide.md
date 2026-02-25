# 伺服器訓練指南 - Kolmogorov Flow

## 📋 伺服器資訊
- **伺服器位址**: `junyi@140.114.120.128`
- **專案路徑**: `~/jaxpi/`
- **範例路徑**: `~/jaxpi/examples/kolmogorov_flow/`

## 🚀 快速開始

### 1. 登入伺服器
```bash
ssh junyi@140.114.120.128
```

### 2. 進入專案目錄
```bash
cd ~/jaxpi
```

### 3. 檢查環境與依賴
```bash
# 檢查 Python 版本（建議 3.8+）
python --version

# 安裝專案依賴
pip install -e .

# 檢查 JAX 是否已安裝
python -c "import jax; print(jax.__version__); print(jax.devices())"
```

### 4. 訓練 Kolmogorov Flow

#### 使用 PIRATE 配置訓練
```bash
cd examples/kolmogorov_flow

# 基礎訓練指令
python main.py --config=configs/pirate.py --workdir=./results_pirate
```

#### 使用 SOAP 配置訓練
```bash
python main.py --config=configs/soap.py --workdir=./results_soap
```

### 5. 背景執行（建議）
如果訓練時間較長，建議使用 `nohup` 或 `screen`：

#### 使用 nohup
```bash
nohup python main.py --config=configs/pirate.py --workdir=./results_pirate > training.log 2>&1 &
```

#### 使用 screen（推薦）
```bash
# 創建新 session
screen -S kf_train

# 執行訓練
python main.py --config=configs/pirate.py --workdir=./results_pirate

# 按 Ctrl+A, 然後按 D 來 detach
# 重新連接: screen -r kf_train
# 查看所有 sessions: screen -ls
```

## 📊 監控訓練

### 查看訓練日誌
```bash
# 即時查看日誌
tail -f training.log

# 或使用 watch 指令查看最新輸出
watch -n 10 tail -50 training.log
```

### 檢查 GPU 使用情況
```bash
# 監控 GPU
watch -n 1 nvidia-smi
```

## 📁 重要檔案位置

- **訓練配置**: `configs/pirate.py`, `configs/soap.py`
- **模型定義**: `models.py`
- **訓練腳本**: `train.py`
- **評估腳本**: `eval.py`
- **資料檔案**: `data/kolmogorov_dns/kolmogorov_dns_10000.npy`

## ⚙️ 主要訓練參數

### PIRATE 配置重點
- **架構**: PirateNet (3層, 256維)
- **訓練步數**: 20,000 steps
- **Batch size**: 8,192 per device
- **時間窗口**: 10 個
- **學習率**: 1e-3 (指數衰減)
- **權重方案**: Gradient normalization with causal weighting

### 可調整參數
修改 `configs/pirate.py`:
```python
# 訓練步數
training.max_steps = 20000

# Batch size
training.batch_size_per_device = 8192

# 學習率
optim.learning_rate = 1e-3

# 網路深度
arch.num_layers = 3
arch.hidden_dim = 256
```

## 🔍 檢查訓練結果

### 評估模型
```bash
# 訓練完成後進行評估
python main.py --config=configs/pirate.py --workdir=./results_pirate --config.mode=eval
```

### 結果位置
- **Checkpoints**: `results_pirate/checkpoints/`
- **日誌**: `training.log`
- **Weights & Biases**: 如有配置，可至 wandb.ai 查看

## 🛠️ 疑難排解

### 記憶體不足
```python
# 減少 batch size
training.batch_size_per_device = 4096
```

### CUDA 錯誤
```bash
# 檢查 JAX 是否正確識別 GPU
python -c "import jax; print(jax.devices())"

# 重新安裝 JAX GPU 版本
pip install --upgrade "jax[cuda12]"  # 根據 CUDA 版本調整
```

## 📤 下載結果回本地

```bash
# 從本地機器執行
rsync -avz --progress junyi@140.114.120.128:~/jaxpi/examples/kolmogorov_flow/results_pirate ./
```

## 📊 Weights & Biases 整合

配置檔已設定 W&B：
- **Project**: PINN-Kolmogorov_flow
- **Run name**: pirate

如需關閉 W&B，在訓練前執行：
```bash
export WANDB_MODE=disabled
```

## 🔄 更新程式碼

從本地更新到伺服器：
```bash
# 在本地機器執行
rsync -avz --progress \
  --exclude='.git' \
  --exclude='.DS_Store' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  . junyi@140.114.120.128:~/jaxpi/
```
