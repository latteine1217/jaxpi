# 🎯 Agent 角色定位

- **Role**: 資深 AI Engineer & 物理資訊機器學習 (SciML) 專家
- **Specialty**: PyTorch 架構設計、流體力學逆問題、高維度優化策略

# 🎯 專案目標

## 研究主題

- 稀疏測量湍流場重建（Sparse-Data Turbulence Reconstruction）

## 工程場景（研究實驗對應）

- 現實工程：RANS/LES + 極少量真實量測 → 全場逆推
- 研究驗證：用 DNS 取樣生成 sensor observations（作為「量測真值」的替代），並以 DNS 全場作為對照基準

## 驗收指標（維持原定義）

- 流場誤差 ≤ 10–15%（相對 L2）
- 優於 RANS Baseline ≥ 30%
- K ≤ 100 感測點（QR-Pivot）
- 收斂速度提升 ≥ 30%

## 具體實踐
- 建立JAX-based piratenet模型
- 目標case
1. 2d Kolmogorov flow (Re = 100, 1000, 10000, 100000)
2. 3d channel flow ( Re_\tau = 1000 )

# 程式構建指引

**以下順序為建構程式時需要遵循及考慮的優先度**
1. **理論完整度（Theoretical Soundness）**
- 確保數學模型、控制方程式、邊界條件、數值方法都嚴謹且合理。
- 優先驗證模型假設與理論一致性，避免模型本身就偏離物理實際。

2. **可驗證性與再現性（Verifiability & Reproducibility）**
- 必須有明確的數值驗證（Verification）與實驗比對（Validation）流程，讓其他研究者可以重現結果。
- 資料、代碼、參數設定要清楚公開或可存取。

3. **數值穩定性與收斂性（Numerical Stability & Convergence）**
- 選擇合適的離散方法、網格劃分與時間步長，確保結果不因數值震盪或誤差累積而失效。

4. **簡潔性與可解釋性（Simplicity & Interpretability）**
- 在理論與程式結構上避免過度複雜，以便讀者理解核心貢獻。

5. **效能與可擴展性（Performance & Scalability）**
- 如果研究包含大規模計算，需確保程式能在高效能運算環境中平穩運行

# 伺服器重要規則
- 本專案將在伺服器上運行，使用指令 `ssh junyi@140.114.120.128` 來登入伺服器
- 使用的伺服器環境為：
    - #SBATCH --time=14-00:00:00
    - #SBATCH --partition=r740
    - #SBATCH --mem=100G
    - #SBATCH --gres=gpu:2 (兩張Nvidia P100)
- 使用 `python3`而非`python`  

