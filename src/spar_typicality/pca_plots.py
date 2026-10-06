"""Interactive 3D HTML plot of PCA projections for all prompt sets of a suite.

The plot is one HTML file. Menus select the prompt set, the Rosch category,
the layer, the colour mode and whether PC1-3 (3D) or PC1-2 (2D) is shown. A category selection only filters the points;
the PCA is the one fitted on the full prompt set.
"""

import html
import json

import plotly.offline

# Categorical colours, in fixed order. Light mode reference palette.
SERIES_COLOURS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
# Each category also gets a marker symbol, so identity is not colour alone.
SERIES_SYMBOLS = ["circle", "square", "diamond", "cross", "x"]
SEQUENTIAL_BLUE = [
    [0.0, "#cde2fb"],
    [0.25, "#86b6ef"],
    [0.5, "#3987e5"],
    [0.75, "#1c5cab"],
    [1.0, "#0d366b"],
]
MUTED = "#898781"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#55534e"
GRID = "#e1e0d9"

COLOUR_MODES = ["category", "typicality"]


def round_list(values, digits=4):
    return [round(float(value), digits) for value in values]


def dataset_payload(name, rows, results_by_layer):
    """Return the plot data of one prompt set.

    `results_by_layer` maps a layer to a PcaResult with 3 components.
    """
    payload_rows = []
    for row in rows:
        payload_rows.append(
            {
                "prompt": row["prompt"],
                "item": row["item"],
                "category": row["category"],
                "z": row["typicality_rating_normalised"],
                "member": row["is_member"],
            }
        )
    layers = {}
    for layer, result in results_by_layer.items():
        if len(result.projections) != len(rows):
            raise ValueError(
                "Prompt set "
                + name
                + " has "
                + str(len(rows))
                + " rows but "
                + str(len(result.projections))
                + " projections at layer "
                + str(layer)
            )
        layers[str(layer)] = {
            "projections": [round_list(point) for point in result.projections],
            "explained": round_list(result.explained_variance_ratio, 6),
        }
    return {"name": name, "rows": payload_rows, "layers": layers}


def plot_payload(title, layers, datasets):
    """Return the data that the HTML page needs.

    `datasets` is a list of (name, rows, results_by_layer).
    """
    categories = set()
    dataset_payloads = []
    for name, rows, results_by_layer in datasets:
        for row in rows:
            if row["category"] is not None:
                categories.add(row["category"])
        dataset_payloads.append(dataset_payload(name, rows, results_by_layer))
    return {
        "title": title,
        "layers": [int(layer) for layer in layers],
        "categories": sorted(categories),
        "colourModes": COLOUR_MODES,
        "datasets": dataset_payloads,
        "style": {
            "seriesColours": SERIES_COLOURS,
            "seriesSymbols": SERIES_SYMBOLS,
            "sequential": SEQUENTIAL_BLUE,
            "muted": MUTED,
            "surface": SURFACE,
            "text": TEXT_PRIMARY,
            "grid": GRID,
        },
    }


def script_json(value):
    """Encode a value as JSON that is safe inside a <script> element."""
    return json.dumps(value, separators=(",", ":")).replace("</", "<\\/")


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
  header { padding: 16px 16px 0; }
  h1 { font-size: 18px; font-weight: 600; margin: 0 0 12px; }
  .controls { display: flex; flex-wrap: wrap; gap: 12px 20px; }
  label {
    display: flex; flex-direction: column; gap: 4px;
    font-size: 12px; color: __TEXT_SECONDARY__;
  }
  select { font: inherit; font-size: 14px; padding: 4px 6px; min-width: 140px; }
  #status { font-size: 13px; color: __TEXT_SECONDARY__; margin: 10px 0 0; }
  #plot { width: 100%; height: calc(100vh - 130px); min-height: 480px; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="controls">
    <label>Prompt set <select id="dataset"></select></label>
    <label>Rosch category <select id="category"></select></label>
    <label>Layer <select id="layer"></select></label>
    <label>Colour <select id="mode"></select></label>
    <label>Components <select id="dims"></select></label>
  </div>
  <p id="status"></p>
</header>
<div id="plot"></div>
<script>
const DATA = __DATA__;
const STYLE = DATA.style;
const ALL = "all";

const datasetSelect = document.getElementById("dataset");
const categorySelect = document.getElementById("category");
const layerSelect = document.getElementById("layer");
const modeSelect = document.getElementById("mode");
const dimsSelect = document.getElementById("dims");

function addOption(select, value, text) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = text;
  select.appendChild(option);
}

DATA.datasets.forEach((dataset, index) => addOption(datasetSelect, index, dataset.name));
addOption(categorySelect, ALL, "All");
DATA.categories.forEach((category) => addOption(categorySelect, category, category));
DATA.layers.forEach((layer) => addOption(layerSelect, layer, "Layer " + layer));
DATA.colourModes.forEach((mode) => addOption(modeSelect, mode, mode));
addOption(dimsSelect, 3, "PC1-3 (3D)");
addOption(dimsSelect, 2, "PC1-2 (2D)");

function hoverText(row) {
  const rating = row.z === null ? "none" : row.z.toFixed(2);
  return [
    row.prompt,
    "item: " + row.item,
    "category: " + row.category,
    "typicality z-score: " + rating,
    "member: " + row.member,
  ].join("<br>");
}

function is3d() {
  return dimsSelect.value === "3";
}

function scatter(points, rows, indices, name, marker) {
  const trace = {
    type: is3d() ? "scatter3d" : "scatter",
    mode: "markers",
    name: name,
    x: indices.map((index) => points[index][0]),
    y: indices.map((index) => points[index][1]),
    text: indices.map((index) => hoverText(rows[index])),
    hoverinfo: "text",
    marker: marker,
  };
  if (is3d()) {
    trace.z = indices.map((index) => points[index][2]);
  } else {
    // Markers in 2D look smaller than in 3D at the same size.
    trace.marker = Object.assign({}, marker, { size: marker.size + 2 });
  }
  return trace;
}

function categoryTraces(points, rows, indices) {
  const traces = [];
  DATA.categories.forEach((category, position) => {
    const members = indices.filter(
      (index) => rows[index].category === category && rows[index].member === 1
    );
    if (members.length === 0) return;
    traces.push(scatter(points, rows, members, category, {
      size: 4,
      color: STYLE.seriesColours[position % STYLE.seriesColours.length],
      symbol: STYLE.seriesSymbols[position % STYLE.seriesSymbols.length],
      line: { width: 0 },
    }));
  });
  const others = indices.filter(
    (index) => rows[index].category === null || rows[index].member !== 1
  );
  if (others.length > 0) {
    const name = DATA.categories.length > 0 && rows[others[0]].category !== null
      ? "non-member" : "no category";
    traces.push(scatter(points, rows, others, name, {
      size: 3, color: STYLE.muted, symbol: "circle", opacity: 0.6,
    }));
  }
  return traces;
}

function typicalityTraces(points, rows, indices) {
  // The colour range comes from the full prompt set, so that it does not
  // change when a category is selected.
  const allRatings = rows.filter((row) => row.z !== null).map((row) => row.z);
  const rated = indices.filter((index) => rows[index].z !== null);
  const unrated = indices.filter((index) => rows[index].z === null);
  const traces = [];
  if (rated.length > 0) {
    traces.push(scatter(points, rows, rated, "rated", {
      size: 4,
      color: rated.map((index) => rows[index].z),
      cmin: Math.min(...allRatings),
      cmax: Math.max(...allRatings),
      colorscale: STYLE.sequential,
      colorbar: {
        title: { text: "Typicality z-score (low = more typical)", side: "right" },
        thickness: 12,
        len: 0.8,
      },
    }));
  }
  if (unrated.length > 0) {
    traces.push(scatter(points, rows, unrated, "no rating", {
      size: 3, color: STYLE.muted, opacity: 0.6,
    }));
  }
  return traces;
}

function axisRange(points, axis) {
  // The range comes from the full prompt set, so that the axes do not
  // change when a category is selected.
  const values = points.map((point) => point[axis]);
  const low = Math.min(...values);
  const high = Math.max(...values);
  const pad = (high - low) * 0.05 || 1;
  return [low - pad, high + pad];
}

function axis(title, range) {
  return {
    title: { text: title }, range: range,
    backgroundcolor: STYLE.surface, gridcolor: STYLE.grid, zeroline: false,
  };
}

function axis2d(title, range) {
  return {
    title: { text: title }, range: range,
    gridcolor: STYLE.grid, zeroline: false, linecolor: STYLE.grid,
  };
}

function render() {
  const dataset = DATA.datasets[Number(datasetSelect.value)];
  const hasCategories = dataset.rows.some((row) => row.category !== null);
  categorySelect.disabled = !hasCategories;
  if (!hasCategories) categorySelect.value = ALL;

  const category = categorySelect.value;
  const layer = dataset.layers[layerSelect.value];
  const rows = dataset.rows;
  const points = layer.projections;
  const indices = [];
  rows.forEach((row, index) => {
    if (category === ALL || row.category === category) indices.push(index);
  });

  const traces = modeSelect.value === "category"
    ? categoryTraces(points, rows, indices)
    : typicalityTraces(points, rows, indices);
  const titles = layer.explained.map(
    (ratio, index) => "PC" + (index + 1) + " (" + (ratio * 100).toFixed(1) + "%)"
  );

  let status = indices.length + " of " + rows.length + " prompts shown.";
  if (!hasCategories) status += " This prompt set has no categories.";
  document.getElementById("status").textContent = status;

  const layout = {
    paper_bgcolor: STYLE.surface,
    plot_bgcolor: STYLE.surface,
    font: { family: "system-ui, -apple-system, sans-serif", color: STYLE.text },
    // The legend is above the plot, so it does not overlap the colour bar
    // on the right.
    legend: {
      itemsizing: "constant", orientation: "h",
      x: 0, xanchor: "left", y: 1, yanchor: "bottom",
    },
    margin: { l: 0, r: 0, t: 40, b: 0 },
    uirevision: "keep-view-" + dimsSelect.value,
  };
  if (is3d()) {
    layout.scene = {
      xaxis: axis(titles[0], axisRange(points, 0)),
      yaxis: axis(titles[1], axisRange(points, 1)),
      zaxis: axis(titles[2], axisRange(points, 2)),
      aspectmode: "cube",
    };
  } else {
    layout.xaxis = axis2d(titles[0], axisRange(points, 0));
    layout.yaxis = axis2d(titles[1], axisRange(points, 1));
    layout.margin = { l: 60, r: 0, t: 40, b: 50 };
  }
  Plotly.react("plot", traces, layout, { responsive: true });
}

[datasetSelect, categorySelect, layerSelect, modeSelect, dimsSelect].forEach(
  (select) => select.addEventListener("change", render)
);
render();
</script>
</body>
</html>
"""


def plot_html(payload):
    """Return the HTML page for a payload from `plot_payload`."""
    replacements = {
        "__TITLE__": html.escape(payload["title"]),
        "__PLOTLY_VERSION__": plotly.offline.get_plotlyjs_version(),
        "__SURFACE__": SURFACE,
        "__TEXT_SECONDARY__": TEXT_SECONDARY,
        "__TEXT__": TEXT_PRIMARY,
        "__DATA__": script_json(payload),
    }
    page = PAGE_TEMPLATE
    for key, value in replacements.items():
        page = page.replace(key, value)
    return page


def write_html(payload, path):
    """Write the plot as a standalone HTML file. Plotly is loaded from a CDN."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plot_html(payload))
