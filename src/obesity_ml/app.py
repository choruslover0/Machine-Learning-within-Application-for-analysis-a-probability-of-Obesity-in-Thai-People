from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from obesity_ml.predict import load_artifact, predict_probability


app = FastAPI(title="Obesity Probability ML App")


PRODUCERS = [
    "Phawich Pilathong",
    "Paphawin Kraipakdeekul",
    "Watcharawee Kengkarunkij",
]

METHODS = [
    ("SMOTENC", "Balances minority/majority classes while respecting categorical survey answers."),
    ("Logistic Regression", "Baseline medical-risk model; easy to explain."),
    ("Support Vector Machine", "Finds a strong boundary between risk groups."),
    ("Random Forest", "Many decision trees vote together; robust for survey data."),
    ("Neural Network", "Learns nonlinear patterns across lifestyle and body features."),
    ("XGBoost", "Gradient-boosted trees; often strong on tabular health data."),
]


STYLE = """
<style>
  :root {
    --ink: #15151a;
    --muted: #71717a;
    --line: rgba(24, 24, 27, 0.12);
    --surface: rgba(255, 255, 255, 0.84);
    --hot: #e1306c;
    --sun: #f77737;
    --gold: #fcaf45;
    --violet: #833ab4;
    --blue: #405de6;
  }

  * { box-sizing: border-box; }

  html { scroll-behavior: smooth; }

  body {
    margin: 0;
    min-height: 100vh;
    color: var(--ink);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background:
      radial-gradient(circle at 16% 8%, rgba(252, 175, 69, 0.34), transparent 28%),
      radial-gradient(circle at 92% 18%, rgba(225, 48, 108, 0.30), transparent 25%),
      linear-gradient(135deg, #fff7ed 0%, #fff 38%, #f4f7fb 100%);
    animation: pageIn 520ms ease-out both;
  }

  a { color: inherit; }

  main {
    width: min(1160px, calc(100vw - 32px));
    margin: 0 auto;
    padding: 28px 0 46px;
  }

  .nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    margin-bottom: 18px;
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 12px 14px;
    background: rgba(255, 255, 255, 0.70);
    backdrop-filter: blur(18px);
  }

  .brand { font-weight: 1000; }

  .nav-links {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .pill, .nav-links a {
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 8px 12px;
    color: var(--muted);
    font-size: 13px;
    font-weight: 900;
    background: rgba(255, 255, 255, 0.72);
    text-decoration: none;
    transition: transform 180ms ease, box-shadow 180ms ease, background 180ms ease;
  }

  .nav-links a:hover, .pill:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(21, 21, 26, 0.10);
    background: rgba(255, 255, 255, 0.94);
  }

  .hero, .card, form {
    border: 1px solid var(--line);
    border-radius: 30px;
    background: var(--surface);
    box-shadow: 0 24px 70px rgba(21, 21, 26, 0.13);
    backdrop-filter: blur(18px);
    animation: cardIn 620ms ease-out both;
  }

  .hero {
    display: grid;
    grid-template-columns: minmax(0, 1.05fr) minmax(300px, 0.95fr);
    gap: 18px;
    padding: 22px;
    align-items: center;
  }

  .hero-copy { padding: 14px; }

  .kicker {
    width: fit-content;
    border-radius: 999px;
    padding: 8px 12px;
    color: white;
    background: linear-gradient(135deg, var(--violet), var(--hot), var(--sun));
    font-size: 13px;
    font-weight: 1000;
  }

  h1 {
    margin: 16px 0 12px;
    font-size: clamp(42px, 7vw, 86px);
    line-height: 0.92;
    letter-spacing: 0;
  }

  h2 {
    margin: 0 0 12px;
    font-size: 27px;
    line-height: 1.05;
  }

  .lead {
    max-width: 620px;
    color: var(--muted);
    font-size: 18px;
  }

  .actions {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 22px;
  }

  .button {
    min-height: 48px;
    display: inline-grid;
    place-items: center;
    border-radius: 17px;
    padding: 0 18px;
    color: white;
    background: linear-gradient(135deg, var(--violet), var(--hot), var(--sun));
    box-shadow: 0 15px 35px rgba(225, 48, 108, 0.25);
    font-weight: 1000;
    text-decoration: none;
    transition: transform 180ms ease, box-shadow 180ms ease, filter 180ms ease;
  }

  .button:hover, button:hover {
    transform: translateY(-2px) scale(1.01);
    filter: saturate(1.1);
  }

  .button:active, button:active, .nav-links a:active {
    transform: translateY(1px) scale(0.98);
  }

  .button.secondary {
    color: var(--ink);
    background: rgba(255, 255, 255, 0.76);
    box-shadow: none;
    border: 1px solid var(--line);
  }

  .phone-preview {
    border-radius: 34px;
    padding: 18px;
    background: #141417;
    color: white;
    min-height: 430px;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,.1);
  }

  .avatar {
    width: 112px;
    height: 112px;
    border-radius: 50%;
    padding: 5px;
    margin: 10px auto 16px;
    background: conic-gradient(from 220deg, var(--gold), var(--sun), var(--hot), var(--violet), var(--gold));
  }

  .avatar-inner {
    width: 100%;
    height: 100%;
    border-radius: 50%;
    display: grid;
    place-items: center;
    background: white;
    color: var(--ink);
    font-size: 36px;
    font-weight: 1000;
  }

  .mini-stat {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin: 18px 0;
  }

  .mini-stat div {
    border-radius: 16px;
    padding: 12px 6px;
    background: rgba(255,255,255,.1);
    text-align: center;
  }

  .mini-stat strong { display: block; font-size: 21px; }
  .mini-stat span { color: rgba(255,255,255,.68); font-size: 12px; }

  .story {
    display: grid;
    grid-template-columns: 42px 1fr;
    gap: 10px;
    align-items: center;
    border-radius: 18px;
    padding: 10px;
    background: rgba(255,255,255,.09);
    margin-top: 9px;
    transition: transform 180ms ease, background 180ms ease;
  }

  .story:hover { transform: translateX(4px); background: rgba(255,255,255,.14); }

  .story-icon {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    font-weight: 1000;
    background: linear-gradient(135deg, var(--hot), var(--sun));
  }

  .section-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin-top: 18px;
  }

  .card {
    padding: 18px;
    transition: transform 200ms ease, box-shadow 200ms ease, border-color 200ms ease;
  }

  .card:hover {
    transform: translateY(-3px);
    box-shadow: 0 30px 80px rgba(21, 21, 26, 0.16);
    border-color: rgba(225, 48, 108, 0.24);
  }

  .card p, .note {
    color: var(--muted);
    line-height: 1.5;
  }

  .method-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
  }

  .method {
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 13px;
    background: rgba(255, 255, 255, 0.70);
    transition: transform 180ms ease, background 180ms ease, border-color 180ms ease;
  }

  .method:hover {
    transform: translateY(-2px);
    background: rgba(255, 255, 255, 0.94);
    border-color: rgba(225, 48, 108, 0.22);
  }

  .method strong { display: block; margin-bottom: 5px; }
  .method span { color: var(--muted); font-size: 13px; line-height: 1.35; }

  .predict-layout {
    display: grid;
    grid-template-columns: minmax(300px, 0.72fr) minmax(0, 1.28fr);
    gap: 22px;
    align-items: start;
  }

  .profile {
    position: sticky;
    top: 24px;
    padding: 22px;
  }

  form { padding: 22px; }

  .form-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 18px;
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
  }

  label {
    display: grid;
    gap: 8px;
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 13px;
    background: rgba(255, 255, 255, 0.72);
    color: #3f3f46;
    font-size: 13px;
    font-weight: 900;
  }

  input, select {
    width: 100%;
    min-height: 42px;
    border: 0;
    border-radius: 12px;
    padding: 8px 10px;
    color: var(--ink);
    background: #f4f4f5;
    font: inherit;
    font-size: 16px;
    outline: none;
  }

  input:focus, select:focus {
    box-shadow: 0 0 0 3px rgba(225, 48, 108, 0.18);
    background: #fff;
  }

  button {
    width: 100%;
    min-height: 52px;
    margin-top: 18px;
    border: 0;
    border-radius: 18px;
    color: white;
    background: linear-gradient(135deg, var(--violet), var(--hot), var(--sun));
    box-shadow: 0 15px 35px rgba(225, 48, 108, 0.25);
    font: inherit;
    font-size: 16px;
    font-weight: 1000;
    cursor: pointer;
    transition: transform 180ms ease, box-shadow 180ms ease, filter 180ms ease;
  }

  .ring {
    width: 220px;
    height: 220px;
    margin: 0 auto 22px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    background:
      radial-gradient(circle closest-side, #fff 72%, transparent 73%),
      conic-gradient(#e1306c var(--percent), #ede9e4 0);
    box-shadow: inset 0 0 0 1px rgba(24, 24, 27, 0.08);
  }

  .prob {
    font-size: 48px;
    font-weight: 1000;
  }

  .feed-output {
    margin-top: 16px;
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 16px;
    background: rgba(255, 255, 255, 0.72);
  }

  .feed-output[hidden] {
    display: none;
  }

  .feed-output .ring {
    width: 154px;
    height: 154px;
    margin-bottom: 12px;
  }

  .feed-output .prob {
    font-size: 34px;
  }

  .feed-meta {
    display: grid;
    gap: 7px;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.35;
  }

  .result-card {
    width: min(560px, 100%);
    margin: 34px auto;
    padding: 24px;
    text-align: center;
  }

  .band {
    width: fit-content;
    margin: 14px auto;
    border-radius: 999px;
    padding: 9px 14px;
    color: white;
    background: linear-gradient(135deg, var(--violet), var(--hot), var(--sun));
    font-weight: 1000;
  }



  .algorithm-lab {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin-top: 18px;
  }

  .algo-card {
    min-height: 210px;
    overflow: hidden;
  }

  .algo-visual {
    height: 96px;
    border: 1px solid var(--line);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.70);
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
  }

  .dot {
    position: absolute;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--hot);
    box-shadow: 0 0 0 4px rgba(225, 48, 108, 0.14);
    animation: dotPulse 1.9s ease-in-out infinite;
  }

  .dot.alt {
    background: var(--blue);
    box-shadow: 0 0 0 4px rgba(64, 93, 230, 0.14);
    animation-delay: 260ms;
  }

  .line-viz {
    position: absolute;
    left: 12%;
    right: 12%;
    top: 50%;
    height: 4px;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--violet), var(--hot), var(--sun));
    transform: rotate(-14deg);
    animation: scanLine 2.4s ease-in-out infinite alternate;
  }

  .tree-viz {
    position: absolute;
    inset: 16px;
    background:
      linear-gradient(var(--hot), var(--hot)) 50% 16px / 3px 32px no-repeat,
      linear-gradient(135deg, transparent 47%, var(--sun) 48%, var(--sun) 52%, transparent 53%) 50% 42px / 110px 34px no-repeat,
      linear-gradient(45deg, transparent 47%, var(--blue) 48%, var(--blue) 52%, transparent 53%) 50% 42px / 110px 34px no-repeat;
  }

  .node {
    position: absolute;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    color: white;
    font-size: 11px;
    font-weight: 1000;
    background: linear-gradient(135deg, var(--violet), var(--hot));
    animation: nodePop 2.2s ease-in-out infinite;
  }

  .bars {
    display: grid;
    gap: 8px;
    margin-top: 12px;
  }

  .bar-row {
    display: grid;
    grid-template-columns: 150px 1fr 52px;
    gap: 10px;
    align-items: center;
    color: var(--muted);
    font-size: 13px;
    font-weight: 800;
  }

  .bar-track {
    height: 12px;
    border-radius: 999px;
    background: #eceff3;
    overflow: hidden;
  }

  .bar-fill {
    width: var(--score);
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, var(--violet), var(--hot), var(--sun));
    animation: growBar 900ms ease-out both;
  }

  .reason-box {
    border: 1px solid rgba(225, 48, 108, 0.22);
    border-radius: 22px;
    padding: 16px;
    background: rgba(255, 255, 255, 0.80);
    text-align: left;
  }

  @keyframes pageIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
  }

  @keyframes cardIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
  }

  @keyframes dotPulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.24); }
  }

  @keyframes scanLine {
    from { transform: rotate(-18deg) translateX(-8px); }
    to { transform: rotate(-8deg) translateX(8px); }
  }

  @keyframes nodePop {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.12); }
  }

  @keyframes growBar {
    from { width: 0; }
    to { width: var(--score); }
  }

  @media (max-width: 920px) {
    .hero, .predict-layout, .section-grid, .algorithm-lab { grid-template-columns: 1fr; }
    .profile { position: static; }
  }

  @media (max-width: 640px) {
    main { width: min(100vw - 18px, 1160px); padding-top: 12px; }
    .grid, .method-grid { grid-template-columns: 1fr; }
    .hero, .card, form { border-radius: 22px; padding: 16px; }
    h1 { font-size: 42px; }
    .bar-row { grid-template-columns: 1fr; gap: 5px; }
  }

  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
      animation-duration: 1ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 1ms !important;
      scroll-behavior: auto !important;
    }
  }
</style>
"""


class ObesityInput(BaseModel):
    age: int = Field(..., ge=5, le=100)
    sex: str = Field(..., examples=["M", "F"])
    height_cm: float = Field(..., ge=80, le=230)
    weight_kg: float = Field(..., ge=20, le=250)
    physical_activity_hours_per_week: float = Field(..., ge=0, le=40)
    screen_time_hours_per_day: float = Field(..., ge=0, le=24)
    sleep_hours: float = Field(..., ge=0, le=16)
    fast_food_meals_per_week: int = Field(..., ge=0, le=30)
    sugary_drinks_per_day: float = Field(..., ge=0, le=20)
    family_history_obesity: int = Field(..., ge=0, le=1)


def page_shell(title: str, body: str) -> str:
    return f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>{title}</title>
      {STYLE}
    </head>
    <body>
      <main>
        <nav class="nav">
          <div class="brand">sk_research.ml</div>
          <div class="nav-links">
            <a href="/">Home</a>
            <a href="/predictor">Predictor</a>
            <a href="/methods">Methods</a>
          </div>
        </nav>
        {body}
      </main>
    </body>
    </html>
    """


def methods_html() -> str:
    methods = "".join(
        f'<div class="method"><strong>{name}</strong><span>{body}</span></div>'
        for name, body in METHODS
    )
    return f'<div class="method-grid">{methods}</div>'



def algorithm_visual_html() -> str:
    cards = [
        (
            "Logistic Regression",
            "Draws one best line that separates lower-risk and higher-risk examples.",
            """
            <div class="algo-visual">
              <span class="dot" style="left:18%; top:62%"></span><span class="dot" style="left:32%; top:54%"></span>
              <span class="dot alt" style="left:62%; top:28%"></span><span class="dot alt" style="left:76%; top:34%"></span>
              <span class="line-viz"></span>
            </div>
            """,
        ),
        (
            "Support Vector Machine",
            "Finds the widest safe gap between groups so new answers land on one side.",
            """
            <div class="algo-visual">
              <span class="dot" style="left:16%; top:32%"></span><span class="dot" style="left:28%; top:54%"></span>
              <span class="dot alt" style="left:68%; top:36%"></span><span class="dot alt" style="left:78%; top:58%"></span>
              <span class="line-viz" style="top:48%; transform:rotate(18deg)"></span>
            </div>
            """,
        ),
        (
            "Random Forest",
            "Uses many small decision trees, then lets them vote on the final risk.",
            """
            <div class="algo-visual">
              <span class="tree-viz"></span>
              <span class="node" style="left:45%; top:8%">Q</span><span class="node" style="left:21%; top:60%">A</span><span class="node" style="left:69%; top:60%">B</span>
            </div>
            """,
        ),
        (
            "Neural Network",
            "Passes information through connected layers and adjusts the connections after mistakes.",
            """
            <div class="algo-visual">
              <span class="node" style="left:12%; top:18%">1</span><span class="node" style="left:12%; top:58%">2</span>
              <span class="node" style="left:45%; top:38%">H</span><span class="node" style="left:78%; top:38%">%</span>
              <span class="line-viz" style="top:34%; transform:rotate(8deg)"></span><span class="line-viz" style="top:61%; transform:rotate(-8deg)"></span>
            </div>
            """,
        ),
        (
            "XGBoost",
            "Builds trees one after another, where each new tree focuses on fixing previous errors.",
            """
            <div class="algo-visual">
              <span class="node" style="left:10%; top:36%">1</span><span class="node" style="left:38%; top:36%">2</span><span class="node" style="left:66%; top:36%">3</span>
              <span class="line-viz" style="top:50%; transform:rotate(0deg)"></span>
            </div>
            """,
        ),
        (
            "SMOTE",
            "Creates realistic synthetic minority examples so training is less biased by uneven data.",
            """
            <div class="algo-visual">
              <span class="dot" style="left:22%; top:35%"></span><span class="dot" style="left:42%; top:40%"></span>
              <span class="dot alt" style="left:32%; top:38%"></span><span class="dot alt" style="left:62%; top:58%"></span>
              <span class="line-viz" style="top:44%; transform:rotate(9deg)"></span>
            </div>
            """,
        ),
    ]
    return '<div class="algorithm-lab">' + ''.join(
        f'<div class="card algo-card">{visual}<h2>{name}</h2><p>{body}</p></div>'
        for name, body, visual in cards
    ) + '</div>'


def _ranked_metrics(metrics: dict) -> list[tuple[str, dict]]:
    return sorted(
        metrics.items(),
        key=lambda item: (
            item[1].get("roc_auc", 0),
            item[1].get("f1", 0),
            -item[1].get("brier_score", 1),
        ),
        reverse=True,
    )


def best_model_reason(result: dict) -> str:
    metrics = result.get("metrics", {})
    best_name = result.get("base_model_name", "the selected model")
    if not metrics or best_name not in metrics:
        return "The app will explain the winning algorithm after the model is trained and metrics are available."

    ranked = _ranked_metrics(metrics)
    best_metrics = metrics[best_name]
    runner_name = ranked[1][0] if len(ranked) > 1 else None
    brier = best_metrics.get("brier_score")
    brier_text = f" and the lowest Brier score ({brier:.4f}), meaning its probability estimates had lower error on validation data" if isinstance(brier, (int, float)) else ""
    if runner_name:
        return (
            f"{best_name} is selected for the current training data because the tournament sorts models by "
            f"cross-validated ROC-AUC first, F1 score second, and Brier score third. "
            f"It stayed at the top of that rule{brier_text}. The nearest comparison model in this run is {runner_name}."
        )
    return f"{best_name} is selected for the current training data under the current tournament rule."


def metric_bars_html(metrics: dict) -> str:
    if not metrics:
        return "<p>No metric bars yet. Train the model first.</p>"
    rows = []
    for name, values in _ranked_metrics(metrics):
        score = values.get("roc_auc", values.get("f1", 0))
        percent = max(0, min(100, round(float(score) * 100))) if isinstance(score, (int, float)) else 0
        rows.append(
            f'<div class="bar-row"><span>{name}</span><div class="bar-track"><div class="bar-fill" style="--score:{percent}%"></div></div><strong>{percent}%</strong></div>'
        )
    return '<div class="bars">' + ''.join(rows) + '</div>'


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    producer_cards = "".join(f"<div class='method'><strong>{name}</strong><span>Producer</span></div>" for name in PRODUCERS)
    algorithm_cards = algorithm_visual_html()
    body = f"""
    <section class="hero">
      <div class="hero-copy">
        <div class="kicker">The SK Research Project</div>
        <h1>Obesity Probability ML</h1>
        <p class="lead">
          Welcome to our obesity probability research app. Explore the predictor, learn how the model works,
          and see how machine learning can support a student research project in a clear and interactive way.
        </p>
        <div class="actions">
          <a class="button" href="/predictor">Start prediction</a>
          <a class="button secondary" href="/methods">View ML methods</a>
        </div>
      </div>
      <div class="phone-preview">
        <div class="pill">Prototype model</div>
        <div class="avatar"><div class="avatar-inner">SK</div></div>
        <h2 style="text-align:center">Risk feed</h2>
        <div class="mini-stat">
          <div><strong>10</strong><span>inputs</span></div>
          <div><strong>6</strong><span>methods</span></div>
          <div><strong>%</strong><span>output</span></div>
        </div>
        <div class="story"><div class="story-icon">A</div><div><strong>Activity</strong><span>exercise and screen habits</span></div></div>
        <div class="story"><div class="story-icon">F</div><div><strong>Food</strong><span>fast food and sugary drinks</span></div></div>
        <div class="story"><div class="story-icon">M</div><div><strong>Model</strong><span>best algorithm chosen by metrics</span></div></div>
      </div>
    </section>

    <section class="card" style="margin-top:18px">
      <div class="kicker">Algorithm visualization</div>
      <h1 style="font-size:44px">How Each Algorithm Learns</h1>
      <p class="lead">Each method studies the same student profile answers, makes a probability guess, checks training mistakes, and improves in its own style.</p>
      {algorithm_cards}
    </section>

    <section class="section-grid">
      <div class="card">
        <h2>Producer Credits</h2>
        <div class="method-grid">{producer_cards}</div>
      </div>
      <div class="card">
        <h2>Research Goal</h2>
        <p>Build a school research tool that can learn from your survey data and estimate obesity-risk probability responsibly.</p>
      </div>
    </section>
    """
    return page_shell("SK Obesity ML Research", body)


@app.get("/methods", response_class=HTMLResponse)
def methods() -> str:
    try:
        artifact = load_artifact()
        best = artifact.get("base_model_name", "not trained yet")
        used_smote = "Yes" if artifact.get("used_smote") else "No or not enough minority samples"
        selection = artifact.get("selection_rule", "Train first to see selection rule.")
        xgboost_status = artifact.get("xgboost_status", "unknown")
        validation_strategy = artifact.get("validation_strategy", "Not recorded yet")
        resampling_strategy = artifact.get("resampling_strategy", "Not recorded yet")
        dataset_warning = artifact.get("dataset_warning", "")
    except FileNotFoundError:
        best = "not trained yet"
        used_smote = "not trained yet"
        selection = "Train the model first."
        xgboost_status = "not trained yet"
        validation_strategy = "not trained yet"
        resampling_strategy = "not trained yet"
        dataset_warning = ""

    body = f"""
    <section class="card">
      <div class="kicker">Model strategy</div>
      <h1>Methods We Use</h1>
      <p class="lead">
        The pipeline follows the direction of your Notion references: compare logistic regression with
        modern machine-learning models, balance data when classes are uneven, and keep the winning
        method visible for research transparency. Model selection now uses cross-validation on the training split.
      </p>
      {methods_html()}
    </section>
    <section class="section-grid">
      <div class="card"><h2>Current best model</h2><p>{best}</p></div>
      <div class="card"><h2>SMOTENC used?</h2><p>{used_smote}</p></div>
      <div class="card"><h2>Selection rule</h2><p>{selection}</p></div>
      <div class="card"><h2>XGBoost status</h2><p>{xgboost_status}</p></div>
      <div class="card"><h2>Validation strategy</h2><p>{validation_strategy}</p></div>
      <div class="card"><h2>Balancing strategy</h2><p>{resampling_strategy}</p></div>
      <div class="card"><h2>Dataset note</h2><p>{dataset_warning or "Ready for real dataset."}</p></div>
    </section>
    """
    return page_shell("Methods - SK Obesity ML", body)


@app.get("/predictor", response_class=HTMLResponse)
def predictor() -> str:
    body = f"""
    <section class="predict-layout">
      <aside class="card profile">
        <div class="pill">Prediction screen</div>
        <div class="avatar"><div class="avatar-inner">SK</div></div>
        <h2 style="text-align:center">New risk check</h2>
        <p class="note" style="text-align:center">
          Fill the profile. The trained model calculates BMI and combines lifestyle clues into a probability.
        </p>
        {methods_html()}
      </aside>

      <form id="predictionForm" action="/predict-form" method="post">
        <div class="form-head">
          <h2>Input profile</h2>
          <div class="pill">Research form</div>
        </div>
        <div class="grid">
          <label>Age <input name="age" type="number" value="16" required></label>
          <label>Sex <select name="sex"><option>M</option><option>F</option></select></label>
          <label>Height (cm) <input name="height_cm" type="number" step="0.1" value="170" required></label>
          <label>Weight (kg) <input name="weight_kg" type="number" step="0.1" value="65" required></label>
          <label>Physical activity hours/week <input name="physical_activity_hours_per_week" type="number" step="0.1" value="3" required></label>
          <label>Screen time hours/day <input name="screen_time_hours_per_day" type="number" step="0.1" value="5" required></label>
          <label>Sleep hours/day <input name="sleep_hours" type="number" step="0.1" value="7" required></label>
          <label>Fast food meals/week <input name="fast_food_meals_per_week" type="number" value="2" required></label>
          <label>Sugary drinks/day <input name="sugary_drinks_per_day" type="number" step="0.1" value="1" required></label>
          <label>Family history <select name="family_history_obesity"><option value="0">No</option><option value="1">Yes</option></select></label>
        </div>
        <button type="submit">Estimate Probability</button>
        <p class="note" style="text-align:center">After you submit, the app switches to a result-only screen.</p>
      </form>
    </section>
    """
    return page_shell("Predictor - SK Obesity ML", body)


@app.post("/predict")
def predict_api(payload: ObesityInput) -> dict:
    return predict_probability(payload.model_dump())


@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(
    age: int = Form(...),
    sex: str = Form(...),
    height_cm: float = Form(...),
    weight_kg: float = Form(...),
    physical_activity_hours_per_week: float = Form(...),
    screen_time_hours_per_day: float = Form(...),
    sleep_hours: float = Form(...),
    fast_food_meals_per_week: int = Form(...),
    sugary_drinks_per_day: float = Form(...),
    family_history_obesity: int = Form(...),
) -> str:
    result = predict_probability(
        {
            "age": age,
            "sex": sex,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "physical_activity_hours_per_week": physical_activity_hours_per_week,
            "screen_time_hours_per_day": screen_time_hours_per_day,
            "sleep_hours": sleep_hours,
            "fast_food_meals_per_week": fast_food_meals_per_week,
            "sugary_drinks_per_day": sugary_drinks_per_day,
            "family_history_obesity": family_history_obesity,
        }
    )
    percent = round(result["obesity_probability"] * 100, 1)
    reason = best_model_reason(result)
    metric_bars = metric_bars_html(result.get("metrics", {}))
    dataset_warning_html = f"<p>{result['dataset_warning']}</p>" if result.get("dataset_warning") else ""
    body = f"""
    <section class="card result-card">
      <div class="pill">Prediction result</div>
      <div class="ring" style="--percent:{percent}%"><div class="prob">{percent}%</div></div>
      <h1 style="font-size:38px">Estimated Probability</h1>
      <div class="band">{result["risk_band"].title()} risk band</div>
      <div class="reason-box">
        <h2>Why this algorithm is best</h2>
        <p>Winning model: <strong>{result["base_model_name"]}</strong></p>
        <p>{reason}</p>
        <p>SMOTENC used during training: <strong>{"Yes" if result["used_smote"] else "No"}</strong></p>
      </div>
      <div class="reason-box" style="margin-top:14px">
        <h2>Model tournament scores</h2>
        <p>These bars use cross-validation on the current training data. They are not universal medical accuracy claims.</p>
        {metric_bars}
      </div>
      {dataset_warning_html}
      <p>{result["disclaimer"]}</p>
      <div class="actions" style="justify-content:center">
        <a class="button" href="/predictor">Try another input</a>
        <a class="button secondary" href="/methods">See methods</a>
      </div>
    </section>
    """
    return page_shell("Result - SK Obesity ML", body)
