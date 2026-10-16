(() => {
  "use strict";

  const $c = (selector) => document.querySelector(selector);
  const MAX_FILE_BYTES = 50 * 1024 * 1024;
  const MAX_ANALYSIS_ROWS = 30000;
  const MAX_RAW_FEATURES = 30;
  const MAX_ENCODED_FEATURES = 100;
  const DEFAULT_FILE = "StudentsPerformance_train.csv";
  let customData = null;

  const elements = {
    input: $c("#csv-file-input"),
    drop: $c("#csv-drop-zone"),
    empty: $c("#custom-empty"),
    config: $c("#custom-config"),
    results: $c("#custom-results"),
    target: $c("#custom-target"),
    featureList: $c("#custom-feature-list"),
    featureCount: $c("#custom-feature-count"),
    status: $c("#custom-status"),
    reset: $c("#custom-reset"),
    run: $c("#run-custom-regression")
  };

  function setStatus(message, isError = false) {
    elements.status.textContent = message;
    elements.status.classList.toggle("error", isError);
  }

  function setStep(step) {
    document.querySelectorAll(".upload-steps span").forEach((item, index) => {
      item.classList.toggle("active", index <= step);
    });
  }

  function parseCSV(text) {
    const rows = [];
    let row = [];
    let field = "";
    let quoted = false;
    for (let index = 0; index < text.length; index += 1) {
      const char = text[index];
      if (quoted) {
        if (char === '"' && text[index + 1] === '"') {
          field += '"';
          index += 1;
        } else if (char === '"') {
          quoted = false;
        } else {
          field += char;
        }
      } else if (char === '"') {
        quoted = true;
      } else if (char === ",") {
        row.push(field.trim());
        field = "";
      } else if (char === "\n") {
        row.push(field.trim());
        if (row.some((value) => value !== "")) rows.push(row);
        row = [];
        field = "";
      } else if (char !== "\r") {
        field += char;
      }
    }
    row.push(field.trim());
    if (row.some((value) => value !== "")) rows.push(row);
    if (quoted) throw new Error("CSV 中有未關閉的雙引號。");
    if (rows.length < 3) throw new Error("CSV 至少需要標題列與 2 筆資料。");
    return rows;
  }

  function uniqueHeaders(rawHeaders) {
    const counts = new Map();
    return rawHeaders.map((value, index) => {
      const base = (index === 0 ? value.replace(/^\uFEFF/, "") : value) || `column_${index + 1}`;
      const count = (counts.get(base) || 0) + 1;
      counts.set(base, count);
      return count === 1 ? base : `${base}_${count}`;
    });
  }

  function numericValue(value) {
    if (value === null || value === undefined || String(value).trim() === "") return NaN;
    const number = Number(String(value).replace(/,/g, ""));
    return Number.isFinite(number) ? number : NaN;
  }

  function profileColumn(rows, index, name) {
    let nonEmpty = 0;
    let numeric = 0;
    const unique = new Set();
    rows.forEach((row) => {
      const value = row[index] ?? "";
      if (value !== "") {
        nonEmpty += 1;
        if (Number.isFinite(numericValue(value))) numeric += 1;
        if (unique.size <= 101) unique.add(value);
      }
    });
    const numericRatio = nonEmpty ? numeric / nonEmpty : 0;
    const type = numericRatio >= 0.85 ? "numeric" : "categorical";
    const idLike = /(^id$|_id$|^index$|^unnamed)/i.test(name);
    const usable = nonEmpty > 1 && (type === "numeric" || (unique.size > 1 && unique.size <= 20));
    return { name, index, nonEmpty, numeric, numericRatio, uniqueCount: unique.size, type, idLike, usable };
  }

  function renderPreview(headers, rows) {
    const table = $c("#custom-preview-table");
    table.replaceChildren();
    const head = document.createElement("thead");
    const headRow = document.createElement("tr");
    headers.forEach((header) => {
      const th = document.createElement("th");
      th.textContent = header;
      headRow.append(th);
    });
    head.append(headRow);
    const body = document.createElement("tbody");
    rows.slice(0, 5).forEach((row) => {
      const tr = document.createElement("tr");
      headers.forEach((_, index) => {
        const td = document.createElement("td");
        td.textContent = row[index] || "—";
        tr.append(td);
      });
      body.append(tr);
    });
    table.append(head, body);
  }

  function renderFeatures() {
    if (!customData) return;
    const targetName = elements.target.value;
    elements.featureList.replaceChildren();
    let selectedByDefault = 0;
    customData.profiles.forEach((profile) => {
      if (profile.name === targetName) return;
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.name = "custom-feature";
      input.value = profile.name;
      input.disabled = !profile.usable;
      if (!profile.usable) label.classList.add("unavailable");
      if (profile.usable && !profile.idLike && selectedByDefault < 20) {
        input.checked = true;
        selectedByDefault += 1;
      }
      const name = document.createElement("b");
      name.textContent = profile.name;
      const type = document.createElement("small");
      type.textContent = !profile.usable ? `略過 · ${profile.uniqueCount} 類` : profile.type === "numeric" ? "數值" : `${profile.uniqueCount} 類`;
      label.append(input, name, type);
      elements.featureList.append(label);
    });
    updateFeatureCount();
  }

  function updateFeatureCount() {
    const count = document.querySelectorAll('input[name="custom-feature"]:checked').length;
    elements.featureCount.textContent = `已選 ${count} 個`;
  }

  async function loadFile(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setStatus("請選擇副檔名為 .csv 的檔案。", true);
      return;
    }
    if (file.size > MAX_FILE_BYTES) {
      setStatus("檔案超過 50 MB，請先縮小或抽樣資料。", true);
      return;
    }
    try {
      setStatus("正在讀取與檢查 CSV…");
      const parsed = parseCSV(await file.text());
      const headers = uniqueHeaders(parsed[0]);
      const allRows = parsed.slice(1).map((row) => headers.map((_, index) => row[index] ?? ""));
      const rows = allRows.slice(0, MAX_ANALYSIS_ROWS);
      const profiles = headers.map((name, index) => profileColumn(rows, index, name));
      const targets = profiles.filter((profile) => profile.numericRatio >= 0.85 && profile.numeric >= 10);
      if (!targets.length) throw new Error("找不到可作為 Target 的數值欄位。");
      customData = { fileName: file.name, headers, rows, totalRows: allRows.length, profiles };
      elements.target.replaceChildren(...targets.map((profile) => {
        const option = document.createElement("option");
        option.value = profile.name;
        option.textContent = `${profile.name}（${Math.round(profile.numericRatio * 100)}% 數值）`;
        return option;
      }));
      const likelyTarget = targets.find((profile) => /(^target$|productivity_score|target_|_target$|^label$|^y$)/i.test(profile.name))
        || targets.find((profile) => /target|price|label|output|grade|sales|score/i.test(profile.name));
      if (likelyTarget) elements.target.value = likelyTarget.name;
      $c("#custom-row-count").textContent = allRows.length.toLocaleString();
      $c("#custom-column-count").textContent = headers.length.toLocaleString();
      $c("#custom-missing-count").textContent = rows.reduce((sum, row) => sum + row.filter((value) => value === "").length, 0).toLocaleString();
      $c("#custom-usable-count").textContent = profiles.filter((profile) => profile.usable).length;
      $c("#current-file-name").textContent = file.name;
      $c("#current-file-meta").textContent = `${allRows.length.toLocaleString()} 筆 × ${headers.length} 欄${allRows.length > rows.length ? ` · 分析前 ${rows.length.toLocaleString()} 筆` : ""}`;
      $c("#custom-mode-label").textContent = `預設結果仍啟用 · ${file.name} 尚未訓練`;
      elements.empty.classList.add("hidden");
      elements.config.classList.remove("hidden");
      elements.results.classList.add("hidden");
      elements.reset.disabled = false;
      renderFeatures();
      renderPreview(headers, rows);
      setStep(1);
      setStatus("欄位辨識完成。確認 Target 與 Features 後即可訓練。");
    } catch (error) {
      setStatus(error.message || "無法解析這份 CSV。", true);
    }
  }

  function median(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }

  function seededShuffle(values, seed = 42) {
    const result = [...values];
    let state = seed >>> 0;
    const random = () => {
      state = (1664525 * state + 1013904223) >>> 0;
      return state / 4294967296;
    };
    for (let index = result.length - 1; index > 0; index -= 1) {
      const swap = Math.floor(random() * (index + 1));
      [result[index], result[swap]] = [result[swap], result[index]];
    }
    return result;
  }

  function solveLinearSystem(matrix, vector) {
    const size = vector.length;
    const work = matrix.map((row, index) => [...row, vector[index]]);
    for (let column = 0; column < size; column += 1) {
      let pivot = column;
      for (let row = column + 1; row < size; row += 1) {
        if (Math.abs(work[row][column]) > Math.abs(work[pivot][column])) pivot = row;
      }
      if (Math.abs(work[pivot][column]) < 1e-12) throw new Error("特徵之間完全共線，無法穩定求解；請減少重複欄位。");
      [work[column], work[pivot]] = [work[pivot], work[column]];
      const divisor = work[column][column];
      for (let item = column; item <= size; item += 1) work[column][item] /= divisor;
      for (let row = 0; row < size; row += 1) {
        if (row === column) continue;
        const factor = work[row][column];
        for (let item = column; item <= size; item += 1) work[row][item] -= factor * work[column][item];
      }
    }
    return work.map((row) => row[size]);
  }

  function fitEncoders(trainRows, profiles) {
    const encoders = [];
    const encodedNames = [];
    profiles.forEach((profile) => {
      if (profile.type === "numeric") {
        const valid = trainRows.map((row) => numericValue(row[profile.index])).filter(Number.isFinite);
        if (!valid.length) return;
        const fill = median(valid);
        const completed = trainRows.map((row) => {
          const value = numericValue(row[profile.index]);
          return Number.isFinite(value) ? value : fill;
        });
        const mean = completed.reduce((sum, value) => sum + value, 0) / completed.length;
        const variance = completed.reduce((sum, value) => sum + (value - mean) ** 2, 0) / completed.length;
        const scale = Math.sqrt(variance) || 1;
        encoders.push({ type: "numeric", profile, fill, mean, scale });
        encodedNames.push(profile.name);
      } else {
        const counts = new Map();
        trainRows.forEach((row) => {
          const value = row[profile.index] || "(missing)";
          counts.set(value, (counts.get(value) || 0) + 1);
        });
        const categories = [...counts].sort((a, b) => b[1] - a[1]).map(([name]) => name).slice(0, 20);
        encoders.push({ type: "categorical", profile, categories });
        categories.slice(1).forEach((category) => encodedNames.push(`${profile.name}=${category}`));
      }
    });
    return { encoders, encodedNames };
  }

  function transformRow(row, encoders) {
    const output = [1];
    encoders.forEach((encoder) => {
      if (encoder.type === "numeric") {
        const raw = numericValue(row[encoder.profile.index]);
        const value = Number.isFinite(raw) ? raw : encoder.fill;
        output.push((value - encoder.mean) / encoder.scale);
      } else {
        const value = row[encoder.profile.index] || "(missing)";
        encoder.categories.slice(1).forEach((category) => output.push(value === category ? 1 : 0));
      }
    });
    return output;
  }

  function fitLinearRegression(matrix, target) {
    const width = matrix[0].length;
    const xtx = Array.from({ length: width }, () => Array(width).fill(0));
    const xty = Array(width).fill(0);
    matrix.forEach((row, rowIndex) => {
      for (let left = 0; left < width; left += 1) {
        xty[left] += row[left] * target[rowIndex];
        for (let right = left; right < width; right += 1) xtx[left][right] += row[left] * row[right];
      }
    });
    for (let left = 0; left < width; left += 1) {
      for (let right = 0; right < left; right += 1) xtx[left][right] = xtx[right][left];
      if (left > 0) xtx[left][left] += 1e-8;
    }
    return solveLinearSystem(xtx, xty);
  }

  function metrics(actual, predicted, baselineMean) {
    const count = actual.length;
    const errors = actual.map((value, index) => value - predicted[index]);
    const mae = errors.reduce((sum, value) => sum + Math.abs(value), 0) / count;
    const rmse = Math.sqrt(errors.reduce((sum, value) => sum + value ** 2, 0) / count);
    const mean = actual.reduce((sum, value) => sum + value, 0) / count;
    const total = actual.reduce((sum, value) => sum + (value - mean) ** 2, 0);
    const residual = errors.reduce((sum, value) => sum + value ** 2, 0);
    const baselineRmse = Math.sqrt(actual.reduce((sum, value) => sum + (value - baselineMean) ** 2, 0) / count);
    return { mae, rmse, r2: total > 0 ? 1 - residual / total : NaN, baselineRmse };
  }

  function formatMetric(value, digits = 4) {
    if (!Number.isFinite(value)) return "N/A";
    if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
    return value.toFixed(digits);
  }

  function drawPredictionChart(actual, predicted) {
    const canvas = $c("#custom-prediction-chart");
    const cssWidth = Math.max(320, canvas.clientWidth || 760);
    const cssHeight = Math.round(cssWidth * 0.56);
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(cssWidth * ratio);
    canvas.height = Math.round(cssHeight * ratio);
    canvas.style.height = `${cssHeight}px`;
    const ctx = canvas.getContext("2d");
    ctx.scale(ratio, ratio);
    ctx.fillStyle = "#06151f";
    ctx.fillRect(0, 0, cssWidth, cssHeight);
    const padding = { left: 58, right: 24, top: 22, bottom: 48 };
    const values = [...actual, ...predicted];
    let minimum = Math.min(...values);
    let maximum = Math.max(...values);
    const margin = (maximum - minimum || 1) * 0.06;
    minimum -= margin;
    maximum += margin;
    const x = (value) => padding.left + ((value - minimum) / (maximum - minimum)) * (cssWidth - padding.left - padding.right);
    const y = (value) => cssHeight - padding.bottom - ((value - minimum) / (maximum - minimum)) * (cssHeight - padding.top - padding.bottom);
    ctx.strokeStyle = "#294755";
    ctx.fillStyle = "#91aab5";
    ctx.font = "11px system-ui";
    for (let tick = 0; tick <= 4; tick += 1) {
      const value = minimum + ((maximum - minimum) * tick) / 4;
      const px = x(value);
      const py = y(value);
      ctx.beginPath(); ctx.moveTo(px, padding.top); ctx.lineTo(px, cssHeight - padding.bottom); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(padding.left, py); ctx.lineTo(cssWidth - padding.right, py); ctx.stroke();
      ctx.fillText(formatMetric(value, 1), px - 13, cssHeight - padding.bottom + 18);
      ctx.fillText(formatMetric(value, 1), 5, py + 4);
    }
    ctx.strokeStyle = "#ffae69";
    ctx.setLineDash([7, 6]);
    ctx.beginPath(); ctx.moveTo(x(minimum), y(minimum)); ctx.lineTo(x(maximum), y(maximum)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(84, 231, 212, .64)";
    const stride = Math.max(1, Math.ceil(actual.length / 1200));
    for (let index = 0; index < actual.length; index += stride) {
      ctx.beginPath(); ctx.arc(x(actual[index]), y(predicted[index]), 2.4, 0, Math.PI * 2); ctx.fill();
    }
    ctx.fillStyle = "#eaf7f7";
    ctx.font = "12px system-ui";
    ctx.fillText("Actual", cssWidth / 2 - 16, cssHeight - 12);
    ctx.save(); ctx.translate(15, cssHeight / 2 + 20); ctx.rotate(-Math.PI / 2); ctx.fillText("Predicted", 0, 0); ctx.restore();
  }

  function renderCoefficients(names, coefficients) {
    const list = $c("#custom-coefficient-list");
    list.replaceChildren();
    names.map((name, index) => ({ name, value: coefficients[index + 1] }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, 10)
      .forEach((item) => {
        const li = document.createElement("li");
        const name = document.createElement("span");
        const value = document.createElement("b");
        name.textContent = item.name;
        value.textContent = `${item.value >= 0 ? "+" : ""}${formatMetric(item.value, 4)}`;
        li.append(name, value);
        list.append(li);
      });
  }

  function runRegression() {
    const targetName = elements.target.value;
    const selectedNames = [...document.querySelectorAll('input[name="custom-feature"]:checked')].map((input) => input.value);
    if (!selectedNames.length) throw new Error("請至少選擇 1 個 Feature。");
    if (selectedNames.length > MAX_RAW_FEATURES) throw new Error(`一次最多選擇 ${MAX_RAW_FEATURES} 個原始 Features。`);
    const targetProfile = customData.profiles.find((profile) => profile.name === targetName);
    const featureProfiles = selectedNames.map((name) => customData.profiles.find((profile) => profile.name === name));
    const validRows = customData.rows.filter((row) => Number.isFinite(numericValue(row[targetProfile.index])));
    if (validRows.length < 20) throw new Error("Target 有效資料少於 20 筆，無法可靠切分訓練與測試集。");
    const shuffled = seededShuffle(validRows);
    const testCount = Math.max(4, Math.floor(shuffled.length * Number($c("#custom-test-size").value)));
    const testRows = shuffled.slice(0, testCount);
    const trainRows = shuffled.slice(testCount);
    const { encoders, encodedNames } = fitEncoders(trainRows, featureProfiles);
    if (!encodedNames.length) throw new Error("選取的 Features 無法產生有效模型欄位。");
    if (encodedNames.length > MAX_ENCODED_FEATURES) throw new Error(`One-Hot 後共有 ${encodedNames.length} 欄，超過瀏覽器版上限 ${MAX_ENCODED_FEATURES}；請減少類別特徵。`);
    if (trainRows.length <= encodedNames.length + 2) throw new Error("訓練資料列不足以估計目前的特徵數，請減少 Features 或增加資料。");
    const xTrain = trainRows.map((row) => transformRow(row, encoders));
    const yTrain = trainRows.map((row) => numericValue(row[targetProfile.index]));
    const coefficients = fitLinearRegression(xTrain, yTrain);
    const xTest = testRows.map((row) => transformRow(row, encoders));
    const actual = testRows.map((row) => numericValue(row[targetProfile.index]));
    const predicted = xTest.map((row) => row.reduce((sum, value, index) => sum + value * coefficients[index], 0));
    const trainMean = yTrain.reduce((sum, value) => sum + value, 0) / yTrain.length;
    const result = metrics(actual, predicted, trainMean);
    return { targetName, selectedNames, encodedNames, coefficients, actual, predicted, trainRows, testRows, result };
  }

  function presentResult(output) {
    elements.results.classList.remove("hidden");
    $c("#custom-result-title").textContent = `${customData.fileName} · 預測 ${output.targetName}`;
    $c("#custom-mae").textContent = formatMetric(output.result.mae);
    $c("#custom-rmse").textContent = formatMetric(output.result.rmse);
    $c("#custom-r2").textContent = formatMetric(output.result.r2);
    $c("#custom-encoded-count").textContent = output.encodedNames.length;
    const improvement = 1 - output.result.rmse / output.result.baselineRmse;
    $c("#custom-result-note").textContent = `80/20 holdout：${output.trainRows.length.toLocaleString()} 筆訓練、${output.testRows.length.toLocaleString()} 筆測試。相較於預測訓練平均值，RMSE ${improvement > 0 ? `降低 ${formatMetric(improvement * 100, 1)}%` : `增加 ${formatMetric(Math.abs(improvement) * 100, 1)}%`}。係數代表預測關係，不等於因果。`;
    renderCoefficients(output.encodedNames, output.coefficients);
    requestAnimationFrame(() => drawPredictionChart(output.actual, output.predicted));
    $c("#custom-mode-label").textContent = `自訂分析完成 · ${customData.fileName}`;
    elements.reset.disabled = false;
    setStep(3);
    setStatus(`分析完成：${output.selectedNames.length} 個原始 Features，One-Hot 後 ${output.encodedNames.length} 欄。`);
    elements.results.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  }

  function resetCustomAnalysis() {
    customData = null;
    elements.input.value = "";
    elements.config.classList.add("hidden");
    elements.results.classList.add("hidden");
    elements.empty.classList.remove("hidden");
    elements.reset.disabled = true;
    $c("#current-file-name").textContent = DEFAULT_FILE;
    $c("#current-file-meta").textContent = "預設分析 · 16,000 筆 × 19 欄";
    $c("#custom-mode-label").textContent = `預設資料集 · ${DEFAULT_FILE}`;
    setStep(0);
  }

  elements.input.addEventListener("change", () => loadFile(elements.input.files[0]));
  ["dragenter", "dragover"].forEach((type) => elements.drop.addEventListener(type, (event) => {
    event.preventDefault();
    elements.drop.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach((type) => elements.drop.addEventListener(type, (event) => {
    event.preventDefault();
    elements.drop.classList.remove("dragging");
  }));
  elements.drop.addEventListener("drop", (event) => loadFile(event.dataTransfer.files[0]));
  elements.drop.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      elements.input.click();
    }
  });
  elements.target.addEventListener("change", () => {
    renderFeatures();
    elements.results.classList.add("hidden");
    $c("#custom-mode-label").textContent = `預設結果仍啟用 · ${customData.fileName} 尚未訓練`;
    setStep(1);
  });
  elements.featureList.addEventListener("change", updateFeatureCount);
  $c("#select-all-features").addEventListener("click", () => {
    document.querySelectorAll('input[name="custom-feature"]:not(:disabled)').forEach((input, index) => { input.checked = index < MAX_RAW_FEATURES; });
    updateFeatureCount();
  });
  $c("#clear-all-features").addEventListener("click", () => {
    document.querySelectorAll('input[name="custom-feature"]').forEach((input) => { input.checked = false; });
    updateFeatureCount();
  });
  elements.config.addEventListener("submit", (event) => {
    event.preventDefault();
    elements.run.disabled = true;
    setStep(2);
    setStatus("正在建立 One-Hot 特徵、訓練模型與計算測試集指標…");
    window.setTimeout(() => {
      try {
        presentResult(runRegression());
      } catch (error) {
        setStatus(error.message || "模型訓練失敗。", true);
        setStep(1);
      } finally {
        elements.run.disabled = false;
      }
    }, 30);
  });
  elements.reset.addEventListener("click", resetCustomAnalysis);

  // Public, read-only engine surface for automated regression checks.
  window.CustomRegressionEngine = Object.freeze({
    parseCSV,
    uniqueHeaders,
    numericValue,
    profileColumn,
    seededShuffle,
    fitEncoders,
    transformRow,
    fitLinearRegression,
    metrics
  });
})();
