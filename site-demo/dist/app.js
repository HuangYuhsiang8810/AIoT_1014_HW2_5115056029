const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

let labData = null;
let activeLabConfig = null;

const FIELD_LABELS = {
  age: "年齡",
  gender: "性別",
  study_hours_per_day: "每日讀書時數",
  sleep_hours: "睡眠時數",
  phone_usage_hours: "手機使用時數",
  social_media_hours: "社群媒體時數",
  youtube_hours: "YouTube 時數",
  gaming_hours: "遊戲時數",
  breaks_per_day: "每日休息次數",
  coffee_intake_mg: "咖啡因攝取量（mg）",
  exercise_minutes: "運動時間（分鐘）",
  assignments_completed: "完成作業數",
  attendance_percentage: "出席率（%）",
  stress_level: "壓力程度",
  focus_score: "專注分數",
  final_grade: "期末成績"
};

const TOUR_STEPS = [
  {
    target: "intro",
    title: "先看清楚官方預測目標",
    copy: "本作業使用 competition train.csv 的 16 個非識別欄位，預測官方目標 productivity_score。",
    tip: "id 與 student_id 僅用來識別資料列，不會進入模型。",
    next: "下一步：模型結果 →"
  },
  {
    target: "models",
    title: "線性模型明顯優於基準",
    copy: "Dummy、完整線性回歸與 SelectKBest 模型使用相同切分。Selected Linear 的 RMSE 約 0.0029，遠低於 Dummy 的 16.24。",
    tip: "R² 幾乎為 1，表示 productivity_score 幾乎可由數個欄位的線性組合重建。",
    next: "下一步：認識資料 →"
  },
  {
    target: "data",
    title: "資料中存在清楚線性訊號",
    copy: "16,000 筆資料沒有缺失或重複。讀書時數與 productivity_score 的 Pearson r 約為 0.732。",
    tip: "相關矩陣同時呈現正向與負向關係，例如手機使用時數約為 −0.327。",
    next: "下一步：特徵選擇 →"
  },
  {
    target: "features",
    title: "六個核心特徵已解釋主要訊號",
    copy: "SelectKBest 將讀書時數、專注、睡眠、手機使用、壓力與出席率排在前六名；k=12 保留核心訊號與額外欄位。",
    tip: "特徵排名仍須搭配 holdout、交叉驗證與係數解讀，不能只看 F-score。",
    next: "下一步：實驗模型 →"
  },
  {
    target: "lab",
    title: "用 255 組設定比較特徵組合",
    copy: "切換 5 種特徵選擇、k=1–17 和 3 種線性回歸器。每一組都由目前 train.csv 實際訓練。",
    tip: "觀察加入前六個核心特徵時，RMSE 如何快速下降、R² 如何接近 1。",
    next: "下一步：檢查誤差 →"
  },
  {
    target: "diagnostics",
    title: "診斷圖確認預測貼近實際值",
    copy: "實際值與預測值幾乎落在 45 度線上，殘差非常接近零。",
    tip: "95% conformal 區間半寬約 0.0047，holdout 覆蓋率為 94.31%。",
    next: "下一步：動手預測 →"
  },
  {
    target: "predictor",
    title: "改變輸入，觀察模型反應",
    copy: "填入 16 個欄位後，試算器使用模型實驗室目前的特徵與係數預測 productivity_score。",
    tip: "調整讀書時數、專注、睡眠、手機使用、壓力與出席率，可以觀察主要線性效果。",
    next: "下一步：完整流程 →"
  },
  {
    target: "workflow",
    title: "用 CRISP-DM 收束正式訓練結果",
    copy: "六階段都有可追溯輸出，並以 holdout、五折交叉驗證、殘差與預測區間共同評估。",
    tip: "近乎完美的結果應解讀為資料生成規則高度線性；仍不能把係數直接宣稱為因果效果。",
    next: "完成導讀 ✓"
  }
];

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let tourIndex = 0;
const guidePanel = $("#guide-panel");

function showTourStep(index, shouldScroll = true) {
  tourIndex = clamp(index, 0, TOUR_STEPS.length - 1);
  const step = TOUR_STEPS[tourIndex];
  const target = $(`[data-tour-id="${step.target}"]`);
  guidePanel.classList.remove("hidden");
  guidePanel.classList.toggle("side-left", step.target === "predictor");
  $$("[data-tour-id]").forEach((element) => element.classList.remove("tour-focus"));
  if (target) {
    target.classList.add("tour-focus");
    if (shouldScroll) target.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
  }
  $("#guide-step-label").textContent = `導讀 ${tourIndex + 1} / ${TOUR_STEPS.length}`;
  $("#guide-title").textContent = step.title;
  $("#guide-copy").textContent = step.copy;
  $("#guide-tip span").textContent = step.tip;
  $("#guide-progress-fill").style.width = `${((tourIndex + 1) / TOUR_STEPS.length) * 100}%`;
  $("#guide-prev").disabled = tourIndex === 0;
  $("#guide-next").textContent = step.next;
  $$("[data-tour-jump]").forEach((button) => button.classList.toggle("active", Number(button.dataset.tourJump) === tourIndex));
}

function closeTour() {
  guidePanel.classList.add("hidden");
  guidePanel.classList.remove("side-left");
  $$("[data-tour-id]").forEach((element) => element.classList.remove("tour-focus"));
  $("#start-guide").focus({ preventScroll: true });
}

$("#start-guide").addEventListener("click", () => showTourStep(0));
$("#guide-close").addEventListener("click", closeTour);
$("#guide-prev").addEventListener("click", () => showTourStep(tourIndex - 1));
$("#guide-next").addEventListener("click", () => tourIndex === TOUR_STEPS.length - 1 ? closeTour() : showTourStep(tourIndex + 1));
$$('[data-tour-jump]').forEach((button) => button.addEventListener('click', () => showTourStep(Number(button.dataset.tourJump))));
document.addEventListener("keydown", (event) => {
  if (guidePanel.classList.contains("hidden")) return;
  if (event.key === "Escape") closeTour();
  if (event.key === "ArrowRight" && tourIndex < TOUR_STEPS.length - 1) showTourStep(tourIndex + 1);
  if (event.key === "ArrowLeft" && tourIndex > 0) showTourStep(tourIndex - 1);
});

function friendlyFeatureName(name) {
  if (name.startsWith("num__")) return FIELD_LABELS[name.slice(5)] || name.slice(5);
  if (name.startsWith("cat__gender_")) return `性別 = ${name.slice(12)}`;
  return name;
}

function buildInputForm() {
  const container = $("#dynamic-inputs");
  const fields = labData.raw_features.map((feature) => {
    const schema = labData.input_schema[feature];
    const label = document.createElement("label");
    label.htmlFor = `model-input-${feature}`;
    label.append(document.createTextNode(FIELD_LABELS[feature] || feature));
    if (schema.type === "category") {
      const select = document.createElement("select");
      select.id = `model-input-${feature}`;
      select.dataset.feature = feature;
      schema.values.forEach((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        option.selected = value === schema.default;
        select.append(option);
      });
      label.append(select);
    } else {
      const input = document.createElement("input");
      input.type = "number";
      input.id = `model-input-${feature}`;
      input.dataset.feature = feature;
      input.min = schema.min;
      input.max = schema.max;
      input.step = schema.step;
      input.value = schema.default;
      label.append(input);
    }
    return label;
  });
  container.replaceChildren(...fields);
  container.addEventListener("input", predict);
  container.addEventListener("change", predict);
}

function getRawInputValues() {
  const values = {};
  labData.raw_features.forEach((feature) => {
    const element = $(`#model-input-${feature}`);
    values[feature] = labData.input_schema[feature].type === "number" ? Number(element.value) : element.value;
  });
  return values;
}

function buildFeatureValues(rawValues) {
  const result = {};
  labData.feature_names.forEach((name) => {
    if (name.startsWith("num__")) {
      const feature = name.slice(5);
      const stats = labData.numeric_stats[feature];
      result[name] = (rawValues[feature] - stats.mean) / stats.scale;
    } else if (name.startsWith("cat__gender_")) {
      result[name] = Number(rawValues.gender === name.slice(12));
    }
  });
  return result;
}

function predict() {
  if (!labData || !activeLabConfig || !$("#model-input-age")) return;
  const featureValues = buildFeatureValues(getRawInputValues());
  let raw = activeLabConfig.intercept;
  Object.entries(activeLabConfig.coefficients).forEach(([feature, coefficient]) => {
    raw += (featureValues[feature] || 0) * coefficient;
  });
  const prediction = clamp(raw, 0, 100);
  const band = activeLabConfig.metrics.rmse;
  $("#prediction-value").textContent = prediction.toFixed(2);
  $("#prediction-range").textContent = `${clamp(raw - band, 0, 100).toFixed(2)} – ${clamp(raw + band, 0, 100).toFixed(2)}`;
  $("#score-meter-fill").style.width = `${prediction}%`;
}

function currentLabSelection() {
  return {
    method: $('input[name="feature-method"]:checked').value,
    k: Number($("#feature-k").value),
    model: $("#regression-model").value
  };
}

function grillConfig(config) {
  const baseline = labData.baseline_metrics.rmse;
  const delta = config.metrics.rmse - baseline;
  if (config.k === labData.meta.candidate_count) return [`k=${config.k} 等於沒有刪除特徵`, "你執行了排序，但最後全部保留。若報告要求 Feature Selection，這組不能當主要成果。"];
  if (config.method === "lasso" && config.model === "lasso") return ["你用了兩次 L1 稀疏化", "Lasso ranking 先排序，Lasso regression 又將部分係數壓到零。請同時檢查 k 與有效係數數量。"];
  if (delta >= 0) return ["這組仍輸給 Dummy", `RMSE 比只猜平均值高 ${delta.toFixed(3)}，請檢查特徵排序與模型設定。`];
  if (config.k < 6) return ["核心特徵尚未完整納入", `目前只保留 ${config.k} 個特徵。前六個主要線性因子完整加入後，誤差才會大幅下降。`];
  if (config.metrics.r2 > 0.9999) return ["結果近乎完美，需要合理解讀", "這表示目標高度接近資料中的線性生成公式，不代表在不同來源的真實學生資料也會有相同表現。"];
  return ["模型已明顯優於 Dummy", `相較平均值基準，RMSE 改善 ${Math.abs(delta).toFixed(3)}。仍應搭配五折交叉驗證與殘差圖判讀。`];
}

function updateLab() {
  if (!labData) return;
  const { method, k, model } = currentLabSelection();
  const config = labData.configs[`${method}|${k}|${model}`];
  const methodInfo = labData.methods[method];
  const modelInfo = labData.models[model];
  activeLabConfig = config;
  $("#feature-k-output").value = k;
  $("#lab-combination").textContent = `${methodInfo.label} · k=${k} · ${modelInfo.label}`;
  $("#active-config-label").textContent = `${methodInfo.label} · k=${k} · ${modelInfo.label}`;
  $("#lab-mae").textContent = config.metrics.mae.toFixed(3);
  $("#lab-rmse").textContent = config.metrics.rmse.toFixed(3);
  $("#lab-r2").textContent = config.metrics.r2.toFixed(4);
  $("#lab-nonzero").textContent = config.nonzero_count;
  ["#lab-mae", "#lab-rmse", "#lab-r2", "#lab-nonzero"].forEach((selector) => {
    const metric = $(selector);
    metric.classList.remove("metric-updated");
    void metric.offsetWidth;
    metric.classList.add("metric-updated");
  });
  $("#lab-selected-count").textContent = `選取 ${k} 個`;
  $("#feature-count-badge").textContent = `${k} / ${labData.meta.candidate_count}`;
  $("#method-note-title").textContent = methodInfo.label;
  $("#method-note-copy").textContent = `${methodInfo.detail} ${modelInfo.detail}`;
  $("#feature-chips").replaceChildren(...config.selected.map((feature, index) => {
    const chip = document.createElement("span");
    chip.textContent = friendlyFeatureName(feature);
    if (index < 2) chip.classList.add("top-feature");
    return chip;
  }));
  const [title, copy] = grillConfig(config);
  $("#grill-title").textContent = title;
  $("#grill-copy").textContent = copy;
  predict();
}

$$('#method-options input, #feature-k, #regression-model').forEach((element) => {
  element.addEventListener('input', updateLab);
  element.addEventListener('change', updateLab);
});

fetch("assets/model_experiments.json")
  .then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  })
  .then((data) => {
    labData = data;
    $("#feature-k").max = data.meta.candidate_count;
    buildInputForm();
    updateLab();
  })
  .catch((error) => {
    $("#grill-title").textContent = "實驗資料載入失敗";
    $("#grill-copy").textContent = `請透過 HTTP server 開啟網站。錯誤：${error.message}`;
  });

$$('.tab').forEach((button) => {
  button.addEventListener('click', () => {
    $$('.tab').forEach((item) => item.classList.remove('active'));
    $$('.diagnostic-panel').forEach((panel) => panel.classList.remove('active'));
    button.classList.add('active');
    $(`#panel-${button.dataset.panel}`).classList.add('active');
  });
});

if (reduceMotion) {
  $$('.reveal').forEach((element) => element.classList.add('visible'));
} else {
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        revealObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });
  $$('.reveal').forEach((element) => revealObserver.observe(element));
}

const sections = $$('main section[id]');
const navLinks = $$('.top-nav a');
const navObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (!entry.isIntersecting) return;
    navLinks.forEach((link) => link.classList.toggle('active', link.getAttribute('href') === `#${entry.target.id}`));
  });
}, { rootMargin: '-35% 0px -55% 0px' });
sections.forEach((section) => navObserver.observe(section));

showTourStep(0, false);

// A restrained pointer glow makes the dashboard feel responsive without
// competing with the charts. Motion-sensitive users keep a static interface.
if (!reduceMotion && window.matchMedia("(pointer: fine)").matches) {
  window.addEventListener("pointermove", (event) => {
    document.documentElement.style.setProperty("--pointer-x", `${event.clientX}px`);
    document.documentElement.style.setProperty("--pointer-y", `${event.clientY}px`);
  }, { passive: true });
}
