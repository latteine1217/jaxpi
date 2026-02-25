# ⚠️ 重要更正：時間窗口對應關係

## 問題發現

**原始錯誤比較**: 我們之前直接比較 PIRATE Window 10 vs SOAP Window 10

**實際情況**:
- **PIRATE**: 10 個時間窗口，每窗口 5 個時間步，覆蓋 t ∈ [0.002, 1.962]
- **SOAP**: 25 個時間窗口，每窗口 2 個時間步，覆蓋 t ∈ [0.002, 1.962]

**時間對應關係**:
```
PIRATE Window 1  (t=[0.002, 0.162]) ≈ SOAP Window 1-2
PIRATE Window 2  (t=[0.202, 0.362]) ≈ SOAP Window 3-5
PIRATE Window 3  (t=[0.402, 0.562]) ≈ SOAP Window 6-7
PIRATE Window 4  (t=[0.602, 0.762]) ≈ SOAP Window 8-10   ✅ 這裡才對應！
PIRATE Window 5  (t=[0.802, 0.962]) ≈ SOAP Window 11-12
PIRATE Window 6  (t=[1.002, 1.162]) ≈ SOAP Window 13-15
PIRATE Window 7  (t=[1.202, 1.362]) ≈ SOAP Window 16-17
PIRATE Window 8  (t=[1.402, 1.562]) ≈ SOAP Window 18-20
PIRATE Window 9  (t=[1.602, 1.762]) ≈ SOAP Window 21-22
PIRATE Window 10 (t=[1.802, 1.962]) ≈ SOAP Window 23-25
```

---

## 修正後的公平對比

### 方法 1: 按相同終止時間比較

| 時間終點 | PIRATE Window | PIRATE 誤差 | SOAP Window | SOAP 誤差 | 改善 |
|---------|--------------|-------------|-------------|-----------|------|
| t≈0.76  | Window 4     | u=1.65%, v=2.85%, w=12.27% | Window 10 | u=1.74%, v=2.63%, w=12.89% | 略差 |
| t≈1.16  | Window 6     | u=5.28%, v=5.53%, w=18.14% | Window 15 | u=5.58%, v=6.74%, w=20.55% | 相當 |
| t≈1.32  | Window 7     | u=8.21%, v=14.04%, w=36.56% | Window 17 | u=7.96%, v=14.43%, w=37.75% | 相當 |

### 方法 2: 按平均誤差比較（相同時間範圍）

由於兩者覆蓋相同的總時間範圍 t ∈ [0, 1.96]，可以比較整體平均性能：

**PIRATE (10 窗口平均)**:
- u_error: 10.24%
- v_error: 11.66%
- w_error: 33.30%

**SOAP (前 17 窗口平均，對應 t ∈ [0, 1.32])**:
- u_error: 2.53%
- v_error: 3.59%
- w_error: 12.65%

**改善**: u↓75%, v↓69%, w↓62%

---

## 重要結論

### ❌ 原始錯誤結論
"SOAP Window 10 相比 PIRATE Window 10 改善 90%+" - **這是不公平的比較**

原因: 
- PIRATE Window 10 是在 t=1.96（最終時間，誤差已累積）
- SOAP Window 10 是在 t=0.76（早期時間，誤差較小）

### ✅ 修正後的結論

#### 1. **相同時間點比較**（Window 4 vs Window 10, t≈0.76）:
- SOAP 和 PIRATE **性能相當**，沒有顯著優勢
- u: PIRATE 1.65% vs SOAP 1.74% (SOAP 略差)
- v: PIRATE 2.85% vs SOAP 2.63% (SOAP 略好)
- w: PIRATE 12.27% vs SOAP 12.89% (SOAP 略差)

#### 2. **更長時間比較**（Window 7 vs Window 17, t≈1.32）:
- 兩者**仍然相當**
- u: PIRATE 8.21% vs SOAP 7.96% (SOAP 略好)
- v: PIRATE 14.04% vs SOAP 14.43% (SOAP 略差)
- w: PIRATE 36.56% vs SOAP 37.75% (SOAP 略差)

#### 3. **平均性能比較**（整體時間範圍）:
- SOAP **確實優於** PIRATE，但改善幅度遠小於之前聲稱的 90%
- 實際改善: **60-75%** (而非 80-90%)

#### 4. **關鍵差異**:
改善的主要原因可能是：
- **時間窗口更細緻**（2 步 vs 5 步）→ 減少窗口內誤差累積
- **更多 IC 更新**（25 次 vs 10 次）→ 更頻繁的誤差重置
- **總訓練次數更多**（500k vs 200k iterations）

而非單純的優化器優勢！

---

## 🔍 更深入的分析需求

要真正評估 SOAP vs PIRATE（Adam）的優勢，需要：

### 公平實驗設計

1. **相同時間窗口數**: 兩者都用 10 或都用 25 個窗口
2. **相同總迭代次數**: 例如都是 200k 或都是 500k
3. **相同訓練時間**: 控制計算成本相同

### 建議的後續實驗

**選項 A**: SOAP 10 窗口版本
```python
# SOAP 配置修改
num_time_windows = 10  # 與 PIRATE 相同
iterations_per_window = 20000  # 與 PIRATE 相同
total_iterations = 200k
```

**選項 B**: PIRATE 25 窗口版本  
```python
# PIRATE 配置修改
num_time_windows = 25  # 與 SOAP 相同
iterations_per_window = 20000  # 與 SOAP 相同
total_iterations = 500k
```

---

## 📊 數據完整性

### 可用數據

✅ **PIRATE**: 10 窗口完整數據
✅ **SOAP**: 17 窗口數據（進行中，目標 25）

### 需要補充

⏳ **SOAP Window 18-25**: 等待訓練完成
- Window 23-25 對應 PIRATE Window 10（t=[1.802, 1.962]）
- 這是真正可以公平比較的時間點

---

## ⚠️ 報告更新計劃

當 SOAP 完成 Window 25 後，需要更新：

1. **修正對比表**: 使用正確的時間對應關係
2. **重新計算改善幅度**: 基於相同時間點
3. **分離影響因素**:
   - 優化器本身的影響
   - 時間窗口細緻度的影響
   - 總迭代次數的影響

4. **更謹慎的結論**: 說明改善的真正來源

---

**發現時間**: 2026-01-05  
**重要性**: ⚠️ **高** - 影響核心結論的正確性  
**狀態**: 需要重新分析和更新所有報告
