# HW2：學生生產力分數多元線性回歸

本專案使用 `StudentsPerformance_train.csv` 預測 Kaggle competition 的官方目標 `productivity_score`，完整遵循 **CRISP-DM + Feature Selection** 流程。內容包含資料理解、One-Hot Encoding、特徵選擇、線性模型比較、交叉驗證、殘差診斷、95% 預測區間、部署 Pipeline 與一頁式互動網站。

> 資料來源：[Student Performance Prediction Machine Learning Challenge — train.csv](https://www.kaggle.com/competitions/student-performance-prediction-machine-learning-challenge/data?select=train.csv)

## 1. 分析目標

- 以 `productivity_score` 為連續型目標欄位。
- 排除識別欄 `id`、`student_id`。
- 使用其餘 16 個原始 predictors，符合題目要求的 10–20 個特徵。
- 比較 Dummy、LinearRegression、Ridge 與 Lasso。
- 執行 SelectKBest 特徵選擇，主要模型使用 `k=12`。
- 使用 MAE、RMSE、R²、5-fold Cross-Validation、殘差與預測區間評估模型。

## 2. 資料規格

資料共有 **16,000 筆、19 欄**，沒有缺失值與重複列。

| 欄位類型 | 欄位 |
|---|---|
| 識別欄，不進入模型 | `id`, `student_id` |
| 類別 predictor | `gender` |
| 數值 predictors | `age`, `study_hours_per_day`, `sleep_hours`, `phone_usage_hours`, `social_media_hours`, `youtube_hours`, `gaming_hours`, `breaks_per_day`, `coffee_intake_mg`, `exercise_minutes`, `assignments_completed`, `attendance_percentage`, `stress_level`, `focus_score`, `final_grade` |
| 預測目標 | `productivity_score` |

`gender` 使用 `OneHotEncoder(drop="first", handle_unknown="ignore")`，因此 16 個原始 predictors 經前處理後形成 **17 個模型候選特徵**。

## 3. 專案架構

```text
AIoT_HW2/
├─ StudentsPerformance_train.csv        # 目前使用的 Kaggle train.csv
├─ main.py                              # CRISP-DM 主程式
├─ requirements.txt
├─ README.md
├─ outputs/
│  ├─ student_productivity_model.joblib
│  ├─ deployment_schema.json
│  ├─ analysis_summary.txt
│  ├─ *.csv                             # 統計、特徵、評估與預測結果
│  └─ *.png                             # 研究與模型診斷圖
└─ site-demo/
   ├─ generate_experiments.py           # 產生網站的 255 組模型實驗
   ├─ test-custom-analysis.cjs          # 瀏覽器回歸引擎測試
   └─ dist/
      ├─ index.html
      ├─ styles.css
      ├─ tech-theme.css
      ├─ app.js
      ├─ custom-analysis.js             # CSV 上傳與瀏覽器端線性回歸
      └─ assets/
```

## 4. CRISP-DM 流程

```mermaid
flowchart LR
    A[Business Understanding] --> B[Data Understanding]
    B --> C[Data Preparation]
    C --> D[Feature Selection]
    D --> E[Modeling]
    E --> F[Evaluation]
    F --> G[Deployment]
```

### 4.1 Business Understanding

使用學生行為、生活、出席與學習資料預測官方 `productivity_score`。主要比較指標為 RMSE，並以 Dummy 平均值模型作為最低基準。

### 4.2 Data Understanding

程式執行以下檢查：

1. 確認 19 個必要欄位存在。
2. 確認 `productivity_score` 可轉為數值且沒有缺漏。
3. 檢查資料筆數、缺失值、重複列與描述統計。
4. 排除識別欄後計算 Pearson 相關矩陣。
5. 分析目標分布、平均數、中位數與主要線性關係。

主要發現：

| 特徵 | 與 productivity_score 的 Pearson r |
|---|---:|
| `study_hours_per_day` | +0.7323 |
| `focus_score` | +0.4151 |
| `sleep_hours` | +0.3386 |
| `phone_usage_hours` | −0.3269 |
| `stress_level` | −0.1956 |
| `attendance_percentage` | +0.1748 |

`productivity_score` 的平均數為 50.15、中位數為 50.22，分布近似對稱。Predictors 彼此的最高絕對相關係數約為 0.020，沒有明顯多重共線性。

### 4.3 Data Preparation

- `X`：16 個非識別 predictors。
- `y`：`productivity_score`。
- 80% training、20% holdout validation，`random_state=42`。
- 數值欄：Median Imputation → StandardScaler。
- 類別欄：Most-frequent Imputation → One-Hot Encoding。
- 前處理、特徵選擇與模型包在同一個 sklearn Pipeline 中，避免資料洩漏。

### 4.4 Feature Selection

主要方法為：

```python
SelectKBest(score_func=f_regression, k=12)
```

排名最高的六個核心特徵為：

1. `study_hours_per_day`
2. `focus_score`
3. `sleep_hours`
4. `phone_usage_hours`
5. `stress_level`
6. `attendance_percentage`

網站模型實驗室另外提供 RFE、Forward SFS、Lasso ranking 與 Permutation Importance，可選擇 `k=1–17`。

### 4.5 Modeling

| 模型 | 目的 |
|---|---|
| Dummy Baseline | 固定預測 training target 平均值 |
| Full Multiple Linear Regression | 使用全部 17 個轉換後特徵 |
| Selected Linear Regression | 使用 SelectKBest 選取的 12 個特徵 |
| Selected Ridge | Selected features + L2 正則化 |
| Selected Lasso | Selected features + L1 正則化 |

### 4.6 Evaluation

固定 holdout 結果：

| 模型 | MAE | RMSE | R² |
|---|---:|---:|---:|
| Dummy Baseline | 13.3125 | 16.2398 | −0.0034 |
| Full Multiple Linear Regression | 0.0025 | 0.0029 | 1.0000 |
| Selected Linear Regression | 0.0025 | 0.0029 | 1.0000 |
| Selected Ridge | 0.0027 | 0.0032 | 1.0000 |
| Selected Lasso | 0.1003 | 0.1246 | 0.9999 |

Selected Linear 的 5-fold CV：

- CV RMSE：`0.0029 ± 0.0000`
- CV R²：接近 `1.0000`
- Split-conformal 95% interval half-width：`0.0047`
- Holdout interval coverage：`94.31%`

這個近乎完美的結果顯示 `productivity_score` 高度接近資料中數個特徵的線性生成公式。這不是把 target 放進 X 所造成的洩漏；程式已明確將 `productivity_score` 從 predictors 排除。但此結果也不代表其他來源的真實學生資料必然能達到相同表現。

### 4.7 Deployment

程式使用全部 16,000 筆資料重新訓練 Selected Linear Pipeline，輸出：

- `outputs/student_productivity_model.joblib`
- `outputs/deployment_schema.json`
- 靜態互動網站與瀏覽器端試算器

部署模型包含 imputation、scaling、One-Hot、SelectKBest 與 LinearRegression，可直接接收原始 16 欄資料。

## 5. 執行方式

### 建立環境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 執行完整分析

```powershell
python main.py
```

或明確指定資料與輸出位置：

```powershell
python main.py --data-file StudentsPerformance_train.csv --output-dir outputs
```

### 重建網站模型實驗資料

```powershell
python site-demo\generate_experiments.py
```

### 啟動網站

```powershell
python -m http.server 4173 --directory site-demo\dist
```

開啟：<http://localhost:4173>

### 網頁自訂 CSV 回歸

網站預設使用 `StudentsPerformance_train.csv` 與已產生的研究結果。使用者可在「上傳 CSV」區塊拖放自己的資料；只有選擇 Target、Features 並按下「開始回歸分析」後，才會啟用自訂資料模式。

- CSV 第一列必須是欄位名稱，檔案上限 50 MB。
- Target 必須至少 85% 可轉為數值，且至少有 20 筆有效資料。
- 數值特徵以訓練集的中位數補值並標準化。
- 類別特徵在瀏覽器中執行 One-Hot Encoding；高基數欄位會停用。
- 固定亂數種子進行 80/20、75/25 或 70/30 holdout 切分。
- 顯示 MAE、RMSE、R²、實際值 vs. 預測值及主要標準化係數。
- 所有運算都在使用者的瀏覽器完成，CSV 不會上傳伺服器，也不會覆寫預設資料與模型。

驗證瀏覽器端回歸引擎：

```powershell
node site-demo\test-custom-analysis.cjs
```

## 6. 使用部署模型

```python
import joblib
import pandas as pd

model = joblib.load("outputs/student_productivity_model.joblib")

sample = pd.DataFrame([{
    "age": 23,
    "gender": "Male",
    "study_hours_per_day": 5.25,
    "sleep_hours": 6.52,
    "phone_usage_hours": 6.23,
    "social_media_hours": 3.99,
    "youtube_hours": 2.97,
    "gaming_hours": 2.95,
    "breaks_per_day": 8,
    "coffee_intake_mg": 248,
    "exercise_minutes": 60,
    "assignments_completed": 9,
    "attendance_percentage": 69.87,
    "stress_level": 5,
    "focus_score": 65,
    "final_grade": 70.55,
}])

prediction = model.predict(sample)[0]
print(f"預測 productivity_score：{prediction:.2f}")
```

## 7. 輸出檔案

| 檔案 | 內容 | CRISP-DM |
|---|---|---|
| `missing_values.csv` | 缺失數與比例 | Data Understanding |
| `descriptive_statistics.csv` | 描述統計 | Data Understanding |
| `feature_correlations.csv` | predictors 與目標的 Pearson r | Data Understanding |
| `pearson_correlation_matrix.csv` | 完整 Pearson 相關矩陣 | Data Understanding |
| `selected_features.csv` | F-score、p-value、是否入選 | Modeling |
| `regression_coefficients.csv` | Selected Linear 係數 | Modeling / Evaluation |
| `holdout_model_results.csv` | Holdout MAE、RMSE、R² | Evaluation |
| `cross_validation_results.csv` | 5-fold CV 平均與標準差 | Evaluation |
| `validation_predictions_with_interval.csv` | 實際值、預測值、區間上下界 | Evaluation |
| `01`～`08` PNG | 分布、相關、選模與診斷圖 | Understanding / Evaluation |
| `student_productivity_model.joblib` | 完整部署 Pipeline | Deployment |
| `deployment_schema.json` | 部署欄位與任務說明 | Deployment |
| `analysis_summary.txt` | 關鍵結果摘要 | Evaluation / Deployment |

## 8. 實作與檢查規則

- 固定 `random_state=42`。
- 先切分資料，再於 training data fit 前處理與特徵選擇。
- Holdout 不參與模型訓練或特徵選擇。
- `id`、`student_id` 不進入模型。
- `productivity_score` 只作為 y，絕不進入 X。
- 類別資料使用 One-Hot Encoding，不使用任意整數代碼。
- 使用 Dummy Baseline 判斷模型是否真正提供價值。
- Pearson correlation 與迴歸係數只能解讀為關聯，不可直接宣稱因果。
- Kaggle 最終名次仍需使用 competition test.csv 產生 submission 並上傳評分；本專案目前呈現的是本地 holdout 與 CV 結果。

## 9. GPT 輔助內容

GPT 協助範圍包含：

- 檢查資料欄位、目標與識別欄。
- 設計 CRISP-DM 程式架構。
- 建立 sklearn Pipeline、One-Hot Encoding 與 Feature Selection。
- 建立模型評估、研究圖表、預測區間與互動網站。
- Debug、驗證輸出一致性與更新文件。

完整對話與繳交附件：

- [ChatGPT 分享頁：HW2 回歸作業規劃與實作對話](https://chatgpt.com/share/6ad1865f-b544-83e8-bfec-2fef26136d76)
- [下載完整 GPT 對話 PDF（115 頁）](site-demo/dist/assets/ChatGPT-HW2-regression-planning-conversation.pdf)

上述 PDF 為本次作業與 ChatGPT 商討、規劃、程式實作及除錯的完整匯出紀錄，可作為提交報告的「GPT 輔助內容」附件。模型判讀與最終報告內容仍應由作業提交者確認。

## 10. NotebookLM 摘要

此段需由作業提交者實際使用 NotebookLM 研究同主題資料後填寫，至少 100 字，並在報告中清楚標示「NotebookLM 摘要」。建議比較：多元線性回歸假設、特徵選擇方法、Ridge/Lasso、交叉驗證、資料洩漏與預測區間。

## 11. 主流或更進階方法比較

| 方法 | 優點 | 本專案中的角色 |
|---|---|---|
| Linear Regression | 可解釋、速度快 | 主要模型，且資料高度線性 |
| Ridge | 降低係數不穩定 | 作為 L2 正則化比較 |
| Lasso | 可將弱特徵係數壓到零 | 作為稀疏模型與特徵選擇比較 |
| Random Forest / Gradient Boosting | 能處理非線性與交互作用 | 若未來資料不是線性生成，可作進一步比較 |
| Nested Cross-Validation | 避免大量調參後高估表現 | 正式選模時比單次 holdout 更嚴謹 |

目前資料的線性模型已接近零誤差，增加複雜非線性模型沒有明顯必要；研究重點應放在結果合理性、資料生成特性與外部泛化限制。
