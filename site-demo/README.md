# 一頁式互動網站

網站使用 `StudentsPerformance_train.csv` 與官方目標 `productivity_score`，呈現 CRISP-DM、資料圖表、Feature Selection、模型評估、診斷與即時預測。

## 重建實驗資料

```powershell
python site-demo\generate_experiments.py
```

此命令會建立 `dist/assets/model_experiments.json`，內容包含 5 種特徵選擇方法、`k=1–17` 與 3 種回歸器，共 255 組實際 holdout 結果。

## 本機預覽

```powershell
python -m http.server 4173 --directory site-demo\dist
```

開啟 <http://localhost:4173>。請透過 HTTP server 瀏覽，不要直接雙擊 `index.html`，否則瀏覽器可能阻擋 JSON 載入。
