"""Interactive HTML report of category-level typicality probes.

The page shows, for a selected train dataset, probe type, layer and metric:

1. A heatmap of train category x eval category, for one eval dataset.
2. A heatmap of category x eval dataset (the probe category is the eval
   category).
3. The mean metric at each layer for each relation (in-sample,
   cross-category, cross-dataset, both).
4. The detail of one probe on one eval category: the projection against the
   typicality score, and a slope chart of the Rosch order against the
   probe order.

A click on a heatmap cell selects the detail.
"""

import base64
import html

import numpy as np
import plotly.offline

from spar_typicality.pca_plots import (
    GRID,
    MUTED,
    SEQUENTIAL_BLUE,
    SERIES_COLOURS,
    SURFACE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    script_json,
)

METRICS = ["spearman", "pearson", "kendall"]
RELATIONS = ["in_sample", "cross_category", "cross_dataset", "cross_both"]
RELATION_LABELS = {
    "in_sample": "In-sample (probe category, train dataset)",
    "cross_category": "Other categories, train dataset",
    "cross_dataset": "Probe category, other datasets",
    "cross_both": "Other categories, other datasets",
}
# Red (negative) to grey to blue (positive). Fixed range -1 to 1.
DIVERGING = [
    [0.0, "#a3302e"],
    [0.2, "#e34948"],
    [0.4, "#f2b8b0"],
    [0.5, "#f0efec"],
    [0.6, "#b7d3f6"],
    [0.8, "#3987e5"],
    [1.0, "#184f95"],
]
QUANTISE_LEVELS = 65535


def metric_array(metric_rows, metric, indexes):
    """Return the metric as an array with the axes
    (train dataset, probe, layer, train category, eval dataset, eval category).
    """
    shape = (
        len(indexes["train_dataset"]),
        len(indexes["probe"]),
        len(indexes["layer"]),
        len(indexes["category"]),
        len(indexes["eval_dataset"]),
        len(indexes["category"]),
    )
    result = np.full(shape, np.nan)
    for row in metric_rows:
        position = (
            indexes["train_dataset"][row["train_dataset"]],
            indexes["probe"][row["probe"]],
            indexes["layer"][row["layer"]],
            indexes["category"][row["train_category"]],
            indexes["eval_dataset"][row["eval_dataset"]],
            indexes["category"][row["eval_category"]],
        )
        result[position] = row[metric]
    return result


def nullable_list(values, digits=4):
    """Return a flat list of rounded values. NaN becomes None."""
    result = []
    for value in np.asarray(values).ravel():
        if np.isnan(value):
            result.append(None)
        else:
            result.append(round(float(value), digits))
    return result


def quantise_vectors(vectors):
    """Encode vectors (n_vectors, length) as base64 little-endian uint16.

    Each vector has its own low and high value. Order is kept to
    1 / 65535 of the range of the vector.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    low = vectors.min(axis=1)
    high = vectors.max(axis=1)
    span = high - low
    span[span == 0] = 1
    levels = np.rint((vectors - low[:, None]) / span[:, None] * QUANTISE_LEVELS)
    data = base64.b64encode(levels.astype("<u2").tobytes()).decode("ascii")
    return {"data": data, "low": nullable_list(low, 6), "high": nullable_list(high, 6)}


def probe_payload(
    title,
    layers,
    train_datasets,
    probe_types,
    categories,
    eval_datasets,
    rows,
    metric_rows,
    projections,
):
    """Return the data that the HTML page needs.

    `projections` has the axes (train dataset, probe, layer, category,
    eval dataset, row). `rows` are the reference rows for each row index.
    """
    indexes = {
        "train_dataset": {name: i for i, name in enumerate(train_datasets)},
        "probe": {name: i for i, name in enumerate(probe_types)},
        "layer": {layer: i for i, layer in enumerate(layers)},
        "category": {name: i for i, name in enumerate(categories)},
        "eval_dataset": {name: i for i, name in enumerate(eval_datasets)},
    }
    metrics = {}
    for metric in METRICS:
        metrics[metric] = nullable_list(metric_array(metric_rows, metric, indexes))

    payload_rows = []
    for row in rows:
        payload_rows.append(
            {
                "item": row["item"],
                "category": row["category"],
                "z": row["typicality_rating_normalised"],
                "member": row["is_member"],
            }
        )

    n_rows = projections.shape[-1]
    if n_rows != len(rows):
        raise ValueError(
            "There are " + str(len(rows)) + " rows but " + str(n_rows) + " projections."
        )
    return {
        "title": title,
        "layers": [int(layer) for layer in layers],
        "trainDatasets": list(train_datasets),
        "probeTypes": list(probe_types),
        "categories": list(categories),
        "evalDatasets": list(eval_datasets),
        "metrics": metrics,
        "metricNames": METRICS,
        "relations": RELATIONS,
        "relationLabels": RELATION_LABELS,
        "rows": payload_rows,
        "projections": quantise_vectors(projections.reshape(-1, n_rows)),
        "style": {
            "seriesColours": SERIES_COLOURS,
            "sequential": SEQUENTIAL_BLUE,
            "diverging": DIVERGING,
            "muted": MUTED,
            "surface": SURFACE,
            "text": TEXT_PRIMARY,
            "textSecondary": TEXT_SECONDARY,
            "grid": GRID,
        },
    }


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="https://cdn.plot.ly/plotly-__PLOTLY_VERSION__.min.js"></script>
<style>
  body {
    margin: 0;
    background: __SURFACE__;
    color: __TEXT__;
    font-family: system-ui, -apple-system, sans-serif;
  }
  main { padding: 16px; max-width: 1500px; margin: 0 auto; }
  h1 { font-size: 18px; font-weight: 600; margin: 0 0 8px; }
  h2 { font-size: 15px; font-weight: 600; margin: 0 0 4px; }
  p.note { font-size: 13px; color: __TEXT_SECONDARY__; margin: 0 0 8px; max-width: 900px; }
  .controls { display: flex; flex-wrap: wrap; gap: 12px 20px; margin: 8px 0; }
  .sticky {
    position: sticky; top: 0; z-index: 10; background: __SURFACE__;
    border-bottom: 1px solid __GRID__; padding: 4px 0 8px;
  }
  label {
    display: flex; flex-direction: column; gap: 4px;
    font-size: 12px; color: __TEXT_SECONDARY__;
  }
  select { font: inherit; font-size: 14px; padding: 4px 6px; min-width: 130px; }
  section { margin: 20px 0 28px; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  @media (max-width: 1000px) { .grid2 { grid-template-columns: 1fr; } }
  .plot { width: 100%; }
  #stats { font-size: 13px; margin: 4px 0 8px; }
</style>
</head>
<body>
<main>
<h1>__TITLE__</h1>
<p class="note">
  Each probe is fitted on the member rows of one category in the train dataset
  (non-members are not used). Typicality score = minus the Rosch z-score, so a
  higher score is more typical. Probe directions point to more typical items, so a
  positive correlation means the typicality order is kept. Cells with a dark
  outline are in-sample (the fit data) and are not a test of generalisation.
</p>
<div class="sticky">
  <div class="controls">
    <label>Train dataset <select id="train"></select></label>
    <label>Probe <select id="probe"></select></label>
    <label>Layer <select id="layer"></select></label>
    <label>Metric <select id="metric"></select></label>
  </div>
</div>

<section>
  <h2>Mean over cells, by layer</h2>
  <p class="note">Mean of the metric over all cells of each relation, for the selected train dataset and probe.</p>
  <div id="layers" class="plot" style="height:320px"></div>
</section>

<section class="grid2">
  <div>
    <h2>Cross-category: probe category x eval category</h2>
    <div class="controls">
      <label>Eval dataset <select id="heatDataset"></select></label>
    </div>
    <div id="heatCategory" class="plot" style="height:520px"></div>
  </div>
  <div>
    <h2>Cross-dataset: probe category = eval category</h2>
    <div class="controls" style="visibility:hidden"><label>-<select></select></label></div>
    <div id="heatDataset2" class="plot" style="height:520px"></div>
  </div>
</section>

<section>
  <h2>Ordering detail</h2>
  <p class="note">Click a heatmap cell to select it here. Left: projection against
  typicality score; non-member pairs of the eval category are on the right, for
  reference only. Right: Rosch order (left side, most typical at top) against the
  order of the projections (right side).</p>
  <div class="controls">
    <label>Probe category <select id="detailTrainCategory"></select></label>
    <label>Eval dataset <select id="detailDataset"></select></label>
    <label>Eval category <select id="detailEvalCategory"></select></label>
  </div>
  <div id="stats"></div>
  <div class="grid2">
    <div id="scatter" class="plot" style="height:520px"></div>
    <div id="slope" class="plot"></div>
  </div>
</section>
</main>
<script>
const DATA = __DATA__;
const STYLE = DATA.style;
const T = DATA.trainDatasets.length;
const P = DATA.probeTypes.length;
const L = DATA.layers.length;
const C = DATA.categories.length;
const E = DATA.evalDatasets.length;
const N = DATA.rows.length;

const projectionLevels = (() => {
  const bytes = Uint8Array.from(atob(DATA.projections.data), (char) => char.charCodeAt(0));
  return new Uint16Array(bytes.buffer);
})();

function select(id) { return document.getElementById(id); }
function addOption(element, value, text) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = text;
  element.appendChild(option);
}
DATA.trainDatasets.forEach((name, i) => addOption(select("train"), i, name));
DATA.probeTypes.forEach((name, i) => addOption(select("probe"), i, name));
DATA.layers.forEach((layer, i) => addOption(select("layer"), i, "Layer " + layer));
DATA.metricNames.forEach((name) => addOption(select("metric"), name, name));
["heatDataset", "detailDataset"].forEach((id) =>
  DATA.evalDatasets.forEach((name, i) => addOption(select(id), i, name)));
["detailTrainCategory", "detailEvalCategory"].forEach((id) =>
  DATA.categories.forEach((name, i) => addOption(select(id), i, name)));

function state() {
  return {
    t: Number(select("train").value),
    p: Number(select("probe").value),
    l: Number(select("layer").value),
    metric: select("metric").value,
  };
}

function metricValue(metric, t, p, l, tc, e, ec) {
  return DATA.metrics[metric][((((t * P + p) * L + l) * C + tc) * E + e) * C + ec];
}

function projectionVector(t, p, l, tc, e) {
  const v = (((t * P + p) * L + l) * C + tc) * E + e;
  const low = DATA.projections.low[v];
  const span = DATA.projections.high[v] - low;
  const result = new Array(N);
  for (let i = 0; i < N; i++) {
    result[i] = low + (projectionLevels[v * N + i] / 65535) * span;
  }
  return result;
}

function trainEvalIndex(t) {
  return DATA.evalDatasets.indexOf(DATA.trainDatasets[t]);
}

function relation(t, tc, e, ec) {
  const sameDataset = e === trainEvalIndex(t);
  const sameCategory = tc === ec;
  if (sameDataset && sameCategory) return "in_sample";
  if (sameCategory) return "cross_dataset";
  if (sameDataset) return "cross_category";
  return "cross_both";
}

function baseLayout() {
  return {
    paper_bgcolor: STYLE.surface,
    plot_bgcolor: STYLE.surface,
    font: { family: "system-ui, -apple-system, sans-serif", color: STYLE.text, size: 12 },
    hoverlabel: { bgcolor: STYLE.surface, bordercolor: STYLE.grid, font: { color: STYLE.text } },
  };
}

function outlineShape(x, y, xref, yref) {
  return {
    type: "rect", xref: xref || "x", yref: yref || "y",
    x0: x - 0.5, x1: x + 0.5, y0: y - 0.5, y1: y + 0.5,
    line: { color: STYLE.text, width: 2 },
  };
}

function heatmapTrace(z, x, y, hover) {
  return {
    type: "heatmap", z: z, x: x, y: y,
    zmin: -1, zmax: 1, colorscale: STYLE.diverging,
    texttemplate: "%{z:.2f}", textfont: { size: 10 },
    xgap: 2, ygap: 2,
    customdata: hover, hovertemplate: "%{customdata}<extra></extra>",
    colorbar: { thickness: 12, len: 0.9 },
  };
}

function renderCategoryHeatmap() {
  const s = state();
  const e = Number(select("heatDataset").value);
  const z = [];
  const hover = [];
  const shapes = [];
  for (let tc = 0; tc < C; tc++) {
    const zRow = [];
    const hoverRow = [];
    for (let ec = 0; ec < C; ec++) {
      const value = metricValue(s.metric, s.t, s.p, s.l, tc, e, ec);
      zRow.push(value);
      hoverRow.push("probe category: " + DATA.categories[tc] + "<br>eval category: "
        + DATA.categories[ec] + "<br>" + s.metric + ": "
        + (value === null ? "none" : value.toFixed(3))
        + "<br>" + relation(s.t, tc, e, ec));
      if (relation(s.t, tc, e, ec) === "in_sample") shapes.push(outlineShape(ec, tc));
    }
    z.push(zRow);
    hover.push(hoverRow);
  }
  const indices = DATA.categories.map((_, i) => i);
  const layout = Object.assign(baseLayout(), {
    margin: { l: 90, r: 10, t: 10, b: 90 },
    xaxis: { title: { text: "Eval category" }, tickvals: indices, ticktext: DATA.categories, tickangle: -45 },
    yaxis: { title: { text: "Probe category" }, tickvals: indices, ticktext: DATA.categories, autorange: "reversed" },
    shapes: shapes,
  });
  Plotly.react("heatCategory", [heatmapTrace(z, indices, indices, hover)], layout, { responsive: true });
}

function renderDatasetHeatmap() {
  const s = state();
  const z = [];
  const hover = [];
  const shapes = [];
  for (let c = 0; c < C; c++) {
    const zRow = [];
    const hoverRow = [];
    for (let e = 0; e < E; e++) {
      const value = metricValue(s.metric, s.t, s.p, s.l, c, e, c);
      zRow.push(value);
      hoverRow.push("category: " + DATA.categories[c] + "<br>eval dataset: "
        + DATA.evalDatasets[e] + "<br>" + s.metric + ": "
        + (value === null ? "none" : value.toFixed(3))
        + "<br>" + relation(s.t, c, e, c));
      if (e === trainEvalIndex(s.t)) shapes.push(outlineShape(e, c));
    }
    z.push(zRow);
    hover.push(hoverRow);
  }
  const xs = DATA.evalDatasets.map((_, i) => i);
  const ys = DATA.categories.map((_, i) => i);
  const layout = Object.assign(baseLayout(), {
    margin: { l: 90, r: 10, t: 10, b: 130 },
    xaxis: { title: { text: "Eval dataset" }, tickvals: xs, ticktext: DATA.evalDatasets, tickangle: -45 },
    yaxis: { title: { text: "Category" }, tickvals: ys, ticktext: DATA.categories, autorange: "reversed" },
    shapes: shapes,
  });
  Plotly.react("heatDataset2", [heatmapTrace(z, xs, ys, hover)], layout, { responsive: true });
}

function renderLayers() {
  const s = state();
  const traces = DATA.relations.map((name, position) => {
    const means = [];
    for (let l = 0; l < L; l++) {
      let total = 0;
      let count = 0;
      for (let tc = 0; tc < C; tc++) {
        for (let e = 0; e < E; e++) {
          for (let ec = 0; ec < C; ec++) {
            if (relation(s.t, tc, e, ec) !== name) continue;
            const value = metricValue(s.metric, s.t, s.p, l, tc, e, ec);
            if (value === null) continue;
            total += value;
            count += 1;
          }
        }
      }
      means.push(count > 0 ? total / count : null);
    }
    return {
      type: "scatter", mode: "lines+markers", name: DATA.relationLabels[name],
      x: DATA.layers, y: means,
      line: { width: 2, color: STYLE.seriesColours[position] },
      marker: { size: 8, color: STYLE.seriesColours[position], line: { width: 2, color: STYLE.surface } },
      hovertemplate: DATA.relationLabels[name] + "<br>layer %{x}: %{y:.3f}<extra></extra>",
    };
  });
  const layout = Object.assign(baseLayout(), {
    margin: { l: 60, r: 10, t: 10, b: 50 },
    xaxis: { title: { text: "Layer" }, tickvals: DATA.layers, gridcolor: STYLE.grid, zeroline: false },
    yaxis: { title: { text: "Mean " + s.metric }, range: [-1, 1], gridcolor: STYLE.grid,
             zeroline: true, zerolinecolor: STYLE.muted },
    legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom" },
    hovermode: "x unified",
  });
  Plotly.react("layers", traces, layout, { responsive: true });
}

function rankDescending(values) {
  // Rank 1 is the largest value.
  const order = values.map((value, i) => i).sort((a, b) => values[b] - values[a]);
  const ranks = new Array(values.length);
  order.forEach((index, position) => { ranks[index] = position + 1; });
  return ranks;
}

function interpolateColour(scale, fraction) {
  for (let i = 1; i < scale.length; i++) {
    if (fraction <= scale[i][0]) {
      const [x0, c0] = scale[i - 1];
      const [x1, c1] = scale[i];
      const f = (fraction - x0) / (x1 - x0 || 1);
      const mix = (k) => Math.round(parseInt(c0.substr(k, 2), 16) * (1 - f)
        + parseInt(c1.substr(k, 2), 16) * f);
      return "rgb(" + mix(1) + "," + mix(3) + "," + mix(5) + ")";
    }
  }
  return scale[scale.length - 1][1];
}

function renderDetail() {
  const s = state();
  const tc = Number(select("detailTrainCategory").value);
  const e = Number(select("detailDataset").value);
  const ec = Number(select("detailEvalCategory").value);
  const category = DATA.categories[ec];
  const projections = projectionVector(s.t, s.p, s.l, tc, e);

  const members = [];
  const others = [];
  DATA.rows.forEach((row, i) => {
    if (row.category !== category) return;
    if (row.member === 1) members.push(i);
    else others.push(i);
  });
  const scores = members.map((i) => -DATA.rows[i].z);
  const memberProjections = members.map((i) => projections[i]);

  const values = DATA.metricNames.map((name) => {
    const value = metricValue(name, s.t, s.p, s.l, tc, e, ec);
    return name + " " + (value === null ? "none" : value.toFixed(3));
  });
  select("stats").textContent = "Probe: " + DATA.probeTypes[s.p] + " on "
    + DATA.trainDatasets[s.t] + " / " + DATA.categories[tc] + ", layer "
    + DATA.layers[s.l] + ". Eval: " + DATA.evalDatasets[e] + " / " + category
    + " (" + members.length + " members, " + relation(s.t, tc, e, ec) + "). "
    + values.join(", ") + ".";

  const scatterTraces = [{
    type: "scatter", mode: "markers", name: "members",
    x: scores, y: memberProjections,
    text: members.map((i) => DATA.rows[i].item),
    hovertemplate: "%{text}<br>typicality score %{x:.2f}<br>projection %{y:.3f}<extra></extra>",
    marker: { size: 8, color: STYLE.seriesColours[0], line: { width: 1, color: STYLE.surface } },
  }, {
    type: "scatter", mode: "markers", name: "non-member pairs",
    xaxis: "x2", yaxis: "y",
    x: others.map((_, k) => (k % 5) - 2), y: others.map((i) => projections[i]),
    text: others.map((i) => DATA.rows[i].item),
    hovertemplate: "%{text} (non-member)<br>projection %{y:.3f}<extra></extra>",
    marker: { size: 8, color: STYLE.muted, line: { width: 1, color: STYLE.surface } },
  }];
  const scatterLayout = Object.assign(baseLayout(), {
    margin: { l: 60, r: 10, t: 30, b: 50 },
    xaxis: { domain: [0, 0.8], title: { text: "Typicality score (higher = more typical)" },
             gridcolor: STYLE.grid, zeroline: false },
    xaxis2: { domain: [0.85, 1], range: [-3, 3], showticklabels: false, showgrid: false,
              zeroline: false, title: { text: "non-members" } },
    yaxis: { title: { text: "Projection on probe direction" }, gridcolor: STYLE.grid, zeroline: false },
    legend: { orientation: "h", x: 0, y: 1.0, yanchor: "bottom" },
  });
  Plotly.react("scatter", scatterTraces, scatterLayout, { responsive: true });

  const roschRanks = rankDescending(scores);
  const probeRanks = rankDescending(memberProjections);
  const count = members.length;
  const slopeTraces = [];
  members.forEach((rowIndex, k) => {
    const colour = interpolateColour(STYLE.sequential, 1 - (roschRanks[k] - 1) / Math.max(count - 1, 1));
    slopeTraces.push({
      type: "scatter", mode: "lines+markers", showlegend: false,
      x: [0, 1], y: [roschRanks[k], probeRanks[k]],
      line: { width: 2, color: colour }, marker: { size: 6, color: colour },
      hovertemplate: DATA.rows[rowIndex].item + "<br>Rosch rank " + roschRanks[k]
        + "<br>probe rank " + probeRanks[k] + "<extra></extra>",
    });
  });
  slopeTraces.push({
    type: "scatter", mode: "text", showlegend: false, hoverinfo: "skip",
    x: members.map(() => 0), y: roschRanks,
    text: members.map((i, k) => roschRanks[k] + ". " + DATA.rows[i].item),
    textposition: "middle left", textfont: { size: 11, color: STYLE.textSecondary },
  }, {
    type: "scatter", mode: "text", showlegend: false, hoverinfo: "skip",
    x: members.map(() => 1), y: probeRanks,
    text: members.map((i, k) => DATA.rows[i].item + " (" + roschRanks[k] + ")"),
    textposition: "middle right", textfont: { size: 11, color: STYLE.textSecondary },
  });
  const height = Math.max(400, count * 15 + 80);
  select("slope").style.height = height + "px";
  const slopeLayout = Object.assign(baseLayout(), {
    margin: { l: 10, r: 10, t: 30, b: 10 },
    xaxis: { range: [-0.9, 1.9], tickvals: [0, 1], ticktext: ["Rosch order", "Probe order"],
             side: "top", showgrid: false, zeroline: false },
    yaxis: { autorange: "reversed", showticklabels: false, showgrid: false, zeroline: false },
  });
  Plotly.react("slope", slopeTraces, slopeLayout, { responsive: true });
}

function renderAll() {
  renderLayers();
  renderCategoryHeatmap();
  renderDatasetHeatmap();
  renderDetail();
}

select("heatDataset").value = trainEvalIndex(0);
select("detailDataset").value = trainEvalIndex(0);
select("detailEvalCategory").value = 1 % C;

select("train").addEventListener("change", () => {
  select("heatDataset").value = trainEvalIndex(state().t);
  renderAll();
});
["probe", "layer", "metric"].forEach((id) => select(id).addEventListener("change", renderAll));
select("heatDataset").addEventListener("change", renderCategoryHeatmap);
["detailTrainCategory", "detailDataset", "detailEvalCategory"].forEach((id) =>
  select(id).addEventListener("change", renderDetail));

renderAll();

select("heatCategory").on("plotly_click", (event) => {
  const point = event.points[0];
  select("detailTrainCategory").value = point.y;
  select("detailEvalCategory").value = point.x;
  select("detailDataset").value = select("heatDataset").value;
  renderDetail();
});
select("heatDataset2").on("plotly_click", (event) => {
  const point = event.points[0];
  select("detailTrainCategory").value = point.y;
  select("detailEvalCategory").value = point.y;
  select("detailDataset").value = point.x;
  renderDetail();
});
</script>
</body>
</html>
"""


def probe_html(payload):
    """Return the HTML page for a payload from `probe_payload`."""
    replacements = {
        "__TITLE__": html.escape(payload["title"]),
        "__PLOTLY_VERSION__": plotly.offline.get_plotlyjs_version(),
        "__SURFACE__": SURFACE,
        "__TEXT_SECONDARY__": TEXT_SECONDARY,
        "__TEXT__": TEXT_PRIMARY,
        "__GRID__": GRID,
        "__DATA__": script_json(payload),
    }
    page = PAGE_TEMPLATE
    for key, value in replacements.items():
        page = page.replace(key, value)
    return page


def write_html(payload, path):
    """Write the report as a standalone HTML file. Plotly is loaded from a CDN."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(probe_html(payload))
