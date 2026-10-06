"""Interactive 3D HTML plots of PCA projections."""

import plotly.graph_objects as go

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
GRID = "#e1e0d9"

COLOUR_MODES = ["category", "typicality"]


def hover_text(row):
    rating = row["typicality_rating_normalised"]
    if rating is None:
        rating_text = "none"
    else:
        rating_text = format(rating, ".2f")
    lines = [
        row["prompt"],
        "item: " + row["item"],
        "category: " + str(row["category"]),
        "typicality z-score: " + rating_text,
        "member: " + str(row["is_member"]),
    ]
    return "<br>".join(lines)


def scatter(projections, indices, rows, name, marker):
    return go.Scatter3d(
        x=projections[indices, 0],
        y=projections[indices, 1],
        z=projections[indices, 2],
        mode="markers",
        name=name,
        marker=marker,
        text=[hover_text(rows[index]) for index in indices],
        hoverinfo="text",
        visible=False,
    )


def category_traces(projections, rows):
    """One trace per category for members, and one grey trace for the rest."""
    categories = sorted(
        {row["category"] for row in rows if row["category"] is not None}
    )
    traces = []
    for position, category in enumerate(categories):
        indices = []
        for index, row in enumerate(rows):
            if row["category"] == category and row["is_member"] == 1:
                indices.append(index)
        marker = {
            "size": 4,
            "color": SERIES_COLOURS[position % len(SERIES_COLOURS)],
            "symbol": SERIES_SYMBOLS[position % len(SERIES_SYMBOLS)],
            "line": {"width": 0},
        }
        traces.append(scatter(projections, indices, rows, category, marker))

    other_indices = []
    for index, row in enumerate(rows):
        if row["category"] is None or row["is_member"] != 1:
            other_indices.append(index)
    if other_indices:
        if categories:
            name = "non-member"
        else:
            name = "no category"
        marker = {"size": 3, "color": MUTED, "symbol": "circle", "opacity": 0.6}
        traces.append(scatter(projections, other_indices, rows, name, marker))
    return traces


def typicality_traces(projections, rows):
    """One trace coloured by typicality z-score, and one grey trace for no rating."""
    rated = []
    unrated = []
    for index, row in enumerate(rows):
        if row["typicality_rating_normalised"] is None:
            unrated.append(index)
        else:
            rated.append(index)
    traces = []
    if rated:
        marker = {
            "size": 4,
            "color": [rows[index]["typicality_rating_normalised"] for index in rated],
            "colorscale": SEQUENTIAL_BLUE,
            "colorbar": {
                "title": {"text": "Typicality z-score<br>(low = more typical)"},
                "thickness": 12,
            },
        }
        traces.append(scatter(projections, rated, rows, "rated", marker))
    if unrated:
        marker = {"size": 3, "color": MUTED, "opacity": 0.6}
        traces.append(scatter(projections, unrated, rows, "no rating", marker))
    return traces


def axis_titles(explained_variance_ratio):
    titles = []
    for index, ratio in enumerate(explained_variance_ratio):
        titles.append("PC" + str(index + 1) + " (" + format(ratio * 100, ".1f") + "%)")
    return titles


def pca_figure(title, rows, results_by_layer):
    """Return a 3D figure with a menu to select the layer and colour mode.

    `results_by_layer` maps a layer to a PcaResult with 3 components.
    """
    figure = go.Figure()
    views = []
    for layer, result in results_by_layer.items():
        for mode in COLOUR_MODES:
            if mode == "category":
                traces = category_traces(result.projections, rows)
            else:
                traces = typicality_traces(result.projections, rows)
            first_trace = len(figure.data)
            for trace in traces:
                figure.add_trace(trace)
            trace_range = range(first_trace, len(figure.data))
            views.append((layer, mode, trace_range, result))

    buttons = []
    for layer, mode, trace_range, result in views:
        visible = [index in trace_range for index in range(len(figure.data))]
        titles = axis_titles(result.explained_variance_ratio)
        layout_update = {
            "title.text": title + " - layer " + str(layer),
            "scene.xaxis.title.text": titles[0],
            "scene.yaxis.title.text": titles[1],
            "scene.zaxis.title.text": titles[2],
        }
        buttons.append(
            {
                "label": "Layer " + str(layer) + " - " + mode,
                "method": "update",
                "args": [{"visible": visible}, layout_update],
            }
        )

    axis_style = {"backgroundcolor": SURFACE, "gridcolor": GRID, "zeroline": False}
    figure.update_layout(
        paper_bgcolor=SURFACE,
        font={"family": "system-ui, -apple-system, sans-serif", "color": TEXT_PRIMARY},
        scene={"xaxis": axis_style, "yaxis": axis_style, "zaxis": axis_style},
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0,
                "y": 1.08,
                "xanchor": "left",
            }
        ],
    )

    # Show the first view at load time.
    first_args = buttons[0]["args"]
    for index, trace in enumerate(figure.data):
        trace.visible = first_args[0]["visible"][index]
    titles = axis_titles(views[0][3].explained_variance_ratio)
    figure.update_layout(
        title_text=first_args[1]["title.text"],
        scene_xaxis_title_text=titles[0],
        scene_yaxis_title_text=titles[1],
        scene_zaxis_title_text=titles[2],
    )
    return figure


def write_html(figure, path):
    """Write a figure as a standalone HTML file. Plotly is loaded from a CDN."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs="cdn")
