"""Evaluation chart and metric-table HTML fragments for the Methods page.

These helpers build SVG/HTML fragments from model metrics and ROC curves. In the
hybrid template architecture they stay in Python (SVG coordinate math is clearer
here than in Jinja) and are injected into templates with the ``|safe`` filter.
"""

from html import escape


def _ranked_metrics(metrics: dict) -> list[tuple[str, dict]]:
    import math

    def sort_key(item):
        m = item[1]
        roc_auc = m.get("roc_auc", 0)
        f1 = m.get("f1", 0)
        epr = m.get("extreme_probability_rate", 0)
        brier = m.get("brier_score", 1)

        # Handle NaN values
        roc_auc = 0.0 if math.isnan(roc_auc) else roc_auc
        f1 = 0.0 if math.isnan(f1) else f1
        epr = 0.0 if math.isnan(epr) else epr
        brier = 1.0 if math.isnan(brier) else brier

        return (roc_auc, f1, -epr, -brier)

    return sorted(metrics.items(), key=sort_key, reverse=True)


def best_model_reason(result: dict) -> str:
    metrics = result.get("metrics", {})
    best_name = result.get("base_model_name", "the selected model")
    if not metrics or best_name not in metrics:
        return "The app will explain the winning algorithm after the model is trained and metrics are available."

    ranked = _ranked_metrics(metrics)
    best_metrics = metrics[best_name]
    runner_name = ranked[1][0] if len(ranked) > 1 else None
    import math
    brier = best_metrics.get("brier_score")
    brier_text = (
        f" with a Brier score of {brier:.4f}"
        if isinstance(brier, (int, float)) and not math.isnan(brier)
        else ""
    )
    if runner_name:
        return (
            f"{best_name} is selected for the current training data because the tournament sorts models by "
            f"cross-validated ROC-AUC first, F1 score second, fewer extreme probabilities third, and Brier score fourth. "
            f"It stayed at the top of that rule{brier_text}. The nearest comparison model in this run is {runner_name}."
        )
    return f"{best_name} is selected for the current training data under the current tournament rule."


def metric_bars_html(metrics: dict) -> str:
    if not metrics:
        return "<p>No metric bars yet. Train the model first.</p>"
    import math
    rows = []
    for name, values in _ranked_metrics(metrics):
        score = values.get("roc_auc", values.get("f1", 0))
        if isinstance(score, (int, float)) and not math.isnan(score):
            percent = max(0, min(100, round(float(score) * 100)))
        else:
            percent = 0
        rows.append(
            f'<div class="bar-row"><span>{name}</span><div class="bar-track"><div class="bar-fill" style="--score:{percent}%"></div></div><strong>{percent}%</strong></div>'
        )
    return '<div class="bars">' + ''.join(rows) + '</div>'


def metric_table_html(metrics: dict) -> str:
    if not metrics:
        return ""
    rows = []
    for name, values in _ranked_metrics(metrics):
        rows.append(
            "<tr>"
            f"<td>{escape(name)}</td>"
            f"<td>{values.get('accuracy', 0):.2f}</td>"
            f"<td>{values.get('kappa', 0):.2f}</td>"
            f"<td>{values.get('sensitivity', 0):.2f}</td>"
            f"<td>{values.get('specificity', 0):.2f}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrapper">'
        '<table class="metric-table">'
        "<colgroup><col><col><col><col><col></colgroup>"
        "<thead><tr><th>Model</th><th>Accuracy</th><th>Kappa</th>"
        '<th><abbr title="Sensitivity">Sens.</abbr></th>'
        '<th><abbr title="Specificity">Spec.</abbr></th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


EVALUATION_METRICS = (
    ("roc_auc", "ROC-AUC", "#347849"),
    ("f1", "F1 score", "#397f9a"),
    ("accuracy", "Accuracy", "#e29a70"),
    ("kappa", "Kappa", "#b89038"),
    ("sensitivity", "Sensitivity", "#7658a6"),
    ("specificity", "Specificity", "#bd5d64"),
)


def _finite_metric(value) -> float | None:
    import math

    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _friendly_model_name(name: str) -> str:
    return str(name).replace("_", " ").title().replace("Rbf", "RBF").replace("Mlp", "MLP")


def selected_model_banner_html(best_model: str, metrics: dict) -> str:
    values = metrics.get(best_model, {})
    roc_auc = _finite_metric(values.get("roc_auc"))
    f1 = _finite_metric(values.get("f1"))
    if not values:
        return """
        <section class="selected-model-banner selected-model-empty">
          <div class="selected-model-icon" aria-hidden="true">?</div>
          <div><span>Current evaluation</span><h2>Train the model to see the selected result</h2></div>
        </section>
        """
    roc_text = f"ROC-AUC {roc_auc:.2f}" if roc_auc is not None else "ROC-AUC not available"
    f1_text = f"F1 score {f1:.2f}" if f1 is not None else "F1 score not available"
    return f"""
    <section class="selected-model-banner">
      <div class="selected-model-icon" aria-hidden="true">&#10003;</div>
      <div>
        <span>Selected for the current training data</span>
        <h2>{escape(_friendly_model_name(best_model))}</h2>
        <p>{roc_text} <b aria-hidden="true">&middot;</b> {f1_text}</p>
      </div>
    </section>
    """


def metric_comparison_chart_html(metrics: dict) -> str:
    legend = "".join(
        f'<span><i style="--metric-color:{color}"></i>{label}</span>'
        for _, label, color in EVALUATION_METRICS
    )
    groups = []
    for model_name, values in _ranked_metrics(metrics):
        bars = []
        for key, label, color in EVALUATION_METRICS:
            value = _finite_metric(values.get(key))
            if value is None:
                bars.append(
                    f'<span class="eval-bar unavailable" style="--metric-color:{color}" '
                    f'title="{escape(_friendly_model_name(model_name))} - {label}: Not available">'
                    '<i></i><small>Not available</small></span>'
                )
                continue
            percent = max(0, min(100, value * 100))
            bars.append(
                f'<span class="eval-bar" style="--metric-color:{color};--metric-value:{percent:.2f}%" '
                f'title="{escape(_friendly_model_name(model_name))} - {label}: {value:.2f}">'
                f'<i></i><small>{value:.2f}</small></span>'
            )
        groups.append(
            f'<div class="eval-model-group"><strong>{escape(_friendly_model_name(model_name))}</strong>'
            f'<div class="eval-bars">{"".join(bars)}</div></div>'
        )
    if not groups:
        groups.append('<div class="evaluation-empty">Train the models to compare their evaluation scores.</div>')
    return f"""
    <section class="evaluation-panel metric-comparison">
      <div class="evaluation-panel-head">
        <div><span>Performance comparison</span><h2>How the models compare</h2></div>
        <p>Higher scores are generally better for these measures.</p>
      </div>
      <div class="evaluation-legend">{legend}</div>
      <div class="eval-models">{"".join(groups)}</div>
    </section>
    """


def roc_curves_html(roc_curves: dict, metrics: dict) -> str:
    import math

    colors = ("#347849", "#7658a6", "#397f9a", "#e29a70", "#bd5d64", "#b89038")
    curves = []
    legends = []
    for index, (model_name, curve) in enumerate(roc_curves.items()):
        false_positive_rate = curve.get("false_positive_rate", [])
        true_positive_rate = curve.get("true_positive_rate", [])
        if len(false_positive_rate) != len(true_positive_rate) or len(false_positive_rate) < 2:
            continue
        pairs = []
        for false_positive, true_positive in zip(false_positive_rate, true_positive_rate):
            if not isinstance(false_positive, (int, float)) or not isinstance(true_positive, (int, float)):
                pairs = []
                break
            if not math.isfinite(false_positive) or not math.isfinite(true_positive):
                pairs = []
                break
            if not 0 <= false_positive <= 1 or not 0 <= true_positive <= 1:
                pairs = []
                break
            x = 52 + (float(false_positive) * 350)
            y = 270 - (float(true_positive) * 220)
            pairs.append(f"{x:.1f},{y:.1f}")
        if not pairs:
            continue
        color = colors[index % len(colors)]
        auc = _finite_metric(metrics.get(model_name, {}).get("roc_auc"))
        auc_text = f"AUC {auc:.2f}" if auc is not None else "AUC not available"
        curves.append(
            f'<polyline class="roc-model-curve" points="{" ".join(pairs)}" '
            f'style="--curve-color:{color}"><title>{escape(_friendly_model_name(model_name))} - {auc_text}</title></polyline>'
        )
        legends.append(
            f'<span><i style="--curve-color:{color}"></i>{escape(_friendly_model_name(model_name))} ({auc_text})</span>'
        )
    if not curves:
        return """
        <section class="evaluation-panel roc-panel">
          <div class="evaluation-panel-head">
            <div><span>ROC curves</span><h2>True-positive and false-positive trade-off</h2></div>
          </div>
          <div class="roc-unavailable">ROC curves are available after retraining the models.</div>
        </section>
        """
    return f"""
    <section class="evaluation-panel roc-panel">
      <div class="evaluation-panel-head">
        <div><span>ROC curves</span><h2>True-positive and false-positive trade-off</h2></div>
        <p>Curves closer to the upper-left corner show stronger separation.</p>
      </div>
      <div class="roc-legend">{"".join(legends)}<span><i class="roc-baseline-key"></i>Random guess</span></div>
      <svg class="roc-chart" viewBox="0 0 440 320" role="img" aria-label="Cross-validated ROC curves">
        <line class="roc-grid" x1="52" y1="50" x2="52" y2="270"></line>
        <line class="roc-grid" x1="52" y1="270" x2="402" y2="270"></line>
        <line class="roc-baseline" x1="52" y1="270" x2="402" y2="50"></line>
        {"".join(curves)}
        <text x="227" y="307" text-anchor="middle">False positive rate</text>
        <text x="15" y="160" text-anchor="middle" transform="rotate(-90 15 160)">True positive rate</text>
        <text x="48" y="287" text-anchor="end">0</text><text x="402" y="287" text-anchor="middle">1</text>
        <text x="42" y="274" text-anchor="end">0</text><text x="42" y="55" text-anchor="end">1</text>
      </svg>
    </section>
    """


def evaluation_dashboard_html(best_model: str, metrics: dict, roc_curves: dict) -> str:
    return (
        '<div class="evaluation-dashboard">'
        + selected_model_banner_html(best_model, metrics)
        + '<div class="evaluation-panels">'
        + metric_comparison_chart_html(metrics)
        + roc_curves_html(roc_curves, metrics)
        + "</div></div>"
    )


