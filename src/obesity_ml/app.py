from html import escape
from pathlib import Path
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from obesity_ml.advice import generate_advice
from obesity_ml.features import validate_prediction_frame
from obesity_ml.predict import load_artifact, predict_probability
from obesity_ml.templating import render
from obesity_ml.charts import (
    best_model_reason,
    evaluation_dashboard_html,
    metric_bars_html,
    metric_comparison_chart_html,
    metric_table_html,
    roc_curves_html,
    selected_model_banner_html,
)


app = FastAPI(title="Obesity Probability ML App")
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


PRODUCERS = [
    "Phawich Pilathong",
    "Paphawin Kraipakdeekul",
    "Watcharawee Kengkarunkij",
]

PRODUCER_PROFILES = [
    {
        "name": "Phawich Pilathong",
        "image": "/static/producer-phawich-pilathong.jpg",
        "photo_class": "photo-phawich",
        "contacts": [
            ("LinkedIn", "https://www.linkedin.com/in/phawich-pilathong-6ba8a1404"),
            ("Instagram", "https://www.instagram.com/baipad_phawich?igsh=MWNzdnc0ejgzN2c0eA%3D%3D&utm_source=qr"),
        ],
    },
    {
        "name": "Watcharawee Kengkarunkij",
        "image": "/static/producer-watcharawee-kengkarunkij.jpg",
        "photo_class": "photo-watcharawee",
        "contacts": [
            ("Instagram", "https://www.instagram.com/padiertart?utm_source=ig_web_button_share_sheet&igsh=ZDNlZDc0MzIxNw=="),
        ],
    },
    {
        "name": "Paphawin Kraipakdeekul",
        "image": "/static/producer-paphawin-kraipakdeekul.jpg",
        "photo_class": "photo-paphawin",
        "contacts": [
            ("Instagram", "https://www.instagram.com/ppw._.ccopter?utm_source=ig_web_button_share_sheet&igsh=ZDNlZDc0MzIxNw=="),
        ],
    },
]

METHODS = [
    ("Logistic Regression", "Baseline medical-risk model; easy to explain."),
    ("Support Vector Machine", "Finds a strong boundary between risk groups."),
    ("Random Forest", "Many decision trees vote together; robust for survey data."),
    ("Naive Bayes", "A simple probability baseline: each answer gives evidence, then the model combines the evidence."),
    ("Neural Network", "Learns nonlinear patterns across lifestyle features."),
    ("XGBoost", "Gradient-boosted trees; often strong on tabular health data."),
]



CHAT_WIDGET_STYLE = """
<style>
  /* Neutralise the global `button {}` rule (min-height:52px, margin-top:18px,
     box-shadow) that would otherwise stretch the chat widget buttons into tall
     ovals. Width is left to each button's own class. */
  .beast-fab, .beast-chat button, .beast-notify button {
    min-height: 0; margin-top: 0; box-shadow: none;
  }
  .beast-fab {
    position: fixed; bottom: 20px; right: 20px; z-index: 1000;
    width: 72px; height: 72px; border-radius: 22px;
    background: linear-gradient(135deg, #7c3aed, #db2777);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 3px; padding: 6px 6px 4px;
    box-shadow: 0 8px 28px rgba(124,58,237,.40);
    border: 0; cursor: pointer; color: white;
    transition: box-shadow 180ms ease;
  }
  .beast-fab img { width: 46px; height: 46px; object-fit: contain; border-radius: 8px; }
  .beast-fab-label { font-size: 9px; font-weight: 900; letter-spacing: .04em; font-family: inherit; }
  .beast-fab-badge {
    position: absolute; top: -4px; right: -4px;
    width: 14px; height: 14px; border-radius: 50%;
    background: var(--hot); border: 2px solid white; display: none;
  }
  @keyframes beastShake {
    0%,100% { transform: rotate(0deg) scale(1); }
    15%      { transform: rotate(-10deg) scale(1.08); }
    30%      { transform: rotate(10deg)  scale(1.08); }
    45%      { transform: rotate(-7deg)  scale(1.04); }
    60%      { transform: rotate(7deg)   scale(1.04); }
    75%      { transform: rotate(-3deg)  scale(1.02); }
    90%      { transform: rotate(3deg)   scale(1.02); }
  }
  .beast-fab.shake { animation: beastShake 0.9s ease-in-out; }
  .beast-close {
    margin-left: auto; background: rgba(255,255,255,.22); border: 0;
    border-radius: 50%; width: 26px; height: 26px; color: white;
    font-size: 14px; cursor: pointer; display: grid; place-items: center;
    flex-shrink: 0; font-family: inherit; line-height: 1;
    transition: background 150ms ease;
  }
  .beast-close:hover { background: rgba(255,255,255,.38); }
  .beast-chat {
    display: none;
    position: fixed; bottom: 104px; right: 16px; z-index: 1000;
    width: 310px; max-height: calc(100vh - 140px); border-radius: 26px;
    background: rgba(255,255,255,0.97);
    border: 1px solid var(--line);
    box-shadow: 0 28px 72px rgba(21,21,26,.20);
    overflow: hidden; font-size: 13px;
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    flex-direction: column;
  }
  .beast-chat.open { display: flex; }
  .beast-head {
    padding: 12px 14px;
    background: linear-gradient(135deg, var(--violet), var(--hot), var(--sun));
    color: white; display: flex; align-items: center; gap: 9px;
  }
  .beast-head-img { width: 36px; height: 36px; object-fit: contain; border-radius: 10px; background: rgba(255,255,255,.18); flex-shrink: 0; }
  .beast-head-info strong { display: block; font-size: 14px; }
  .beast-head-info small { opacity: .82; font-size: 11px; }
  .beast-lang-toggle { margin-left: auto; display: flex; gap: 3px; background: rgba(255,255,255,.18); border-radius: 999px; padding: 2px; flex-shrink: 0; }
  .beast-lang-btn { width: auto; height: 22px; padding: 0 9px; border-radius: 999px; font-size: 10px; font-weight: 900; color: rgba(255,255,255,.78); border: 0; background: transparent; cursor: pointer; font-family: inherit; line-height: 1; display: inline-flex; align-items: center; }
  .beast-lang-btn.active { background: rgba(255,255,255,.32); color: white; }
  .beast-ctx { margin: 8px 12px 0; border-radius: 14px; padding: 8px 12px; background: rgba(124,58,237,.08); border: 1px solid rgba(124,58,237,.22); font-size: 11px; color: #7c3aed; font-weight: 700; line-height: 1.4; }
  .beast-msgs { padding: 12px; display: flex; flex-direction: column; gap: 8px; flex: 1; overflow-y: auto; min-height: 80px; }
  .beast-msg { line-height: 1.4; max-width: 88%; }
  .beast-bot { background: #f4f1fa; border-radius: 16px 16px 16px 4px; padding: 9px 12px; color: #1c1c2e; }
  .beast-user { background: linear-gradient(135deg, #7c3aed, #db2777); color: white; border-radius: 16px 16px 4px 16px; padding: 9px 12px; align-self: flex-end; }
  .beast-src { display: block; margin-top: 4px; font-size: 10px; color: #6b6880; font-weight: 700; }
  .beast-chips { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 5px; padding: 4px 12px 10px; }
  .beast-chip { flex: 0 0 auto; width: auto; display: inline-flex; align-items: center; border: 1px solid rgba(124,58,237,.28); border-radius: 999px; padding: 5px 12px; font-size: 11px; font-weight: 700; color: #7c3aed; background: rgba(124,58,237,.07); cursor: pointer; font-family: inherit; white-space: nowrap; transition: padding 150ms ease, font-size 150ms ease, opacity 150ms ease; }
  .beast-chips.compact { gap: 4px; padding: 2px 12px 6px; }
  .beast-chips.compact .beast-chip { padding: 3px 9px; font-size: 10px; opacity: 0.75; }
  .beast-form { display: flex; gap: 8px; padding: 10px 12px; border-top: 1px solid rgba(0,0,0,0.08); }
  .beast-input { flex: 1; border: 1px solid rgba(0,0,0,0.10); border-radius: 999px; padding: 8px 13px; font-size: 12px; outline: none; background: #faf9ff; font-family: inherit; color: #1c1c2e; }
  .beast-input:focus { box-shadow: 0 0 0 3px rgba(124,58,237,.18); background: #fff; }
  .beast-send { width: 34px; height: 34px; border-radius: 50%; flex-shrink: 0; background: linear-gradient(135deg, #7c3aed, #db2777); border: 0; color: white; font-size: 13px; cursor: pointer; display: grid; place-items: center; }
  .beast-notify {
    display: none;
    position: fixed; bottom: 104px; right: 20px; z-index: 999;
    background: rgba(255,255,255,0.98);
    border: 1px solid rgba(124,58,237,0.18);
    border-radius: 16px 16px 4px 16px;
    padding: 10px 12px 10px 14px; font-size: 12px; font-weight: 600; line-height: 1.5;
    color: #1c1c2e;
    box-shadow: 0 8px 32px rgba(0,0,0,0.12), 0 2px 8px rgba(124,58,237,0.10);
    max-width: 210px; align-items: flex-start; gap: 8px;
  }
  .beast-notify.show { display: flex; }
  .beast-notify.pop { animation: notifyIn 380ms cubic-bezier(.34,1.56,.64,1) both; }
  .beast-notify-text { flex: 1; }
  .beast-notify-close {
    width: auto; flex-shrink: 0; background: none; border: 0; font-size: 14px;
    color: #b0adc0; cursor: pointer; padding: 0; line-height: 1;
    margin-top: -1px;
  }
  .beast-notify-close:hover { color: #1c1c2e; }
  .beast-notify::after {
    content: ""; position: absolute; bottom: -8px; right: 16px;
    width: 13px; height: 13px; background: rgba(255,255,255,0.98);
    border-right: 1px solid rgba(124,58,237,0.18);
    border-bottom: 1px solid rgba(124,58,237,0.18);
    transform: rotate(45deg); clip-path: polygon(0 0, 100% 100%, 0 100%);
  }
  @keyframes notifyIn {
    from {{ opacity: 0; transform: scale(.82) translateY(10px); }}
    to   {{ opacity: 1; transform: scale(1)   translateY(0);    }}
  }
</style>
"""




PREMIUM_HEALTH_CHAT_STYLE = """
<style>
  .beast-fab { background: var(--forest); box-shadow: 0 12px 30px rgba(21,60,41,.24); }
  .beast-fab-badge { background: var(--peach); }
  .beast-chat { border-color: rgba(21,60,41,.12); font-family: "DM Sans",sans-serif; box-shadow: 0 26px 70px rgba(21,60,41,.18); }
  .beast-head { background: var(--forest); }
  .beast-bot { color: var(--ink); background: var(--sage); }
  .beast-user, .beast-send { background: var(--green); }
  .beast-chip { color: var(--green); border-color: rgba(52,120,73,.25); background: rgba(233,240,232,.8); }
  .beast-ctx { color: var(--green); border-color: rgba(52,120,73,.22); background: var(--sage); }
  @media (max-width: 640px) {
    .beast-notify { display: none !important; }
    .beast-fab {
      width: 52px; height: 52px; border-radius: 50%; padding: 5px;
      right: 12px; bottom: 12px;
    }
    .beast-fab img { width: 40px; height: 40px; border-radius: 50%; }
    .beast-fab-label { display: none; }
  }
</style>
"""


def render_styles() -> str:
    """Return the inline <style> blocks extracted to templates/partials/styles.html."""
    return render("partials/styles.html", {})


def page_shell(title: str, body: str, chat_context: dict | None = None) -> str:
    risk_tier = chat_context.get("risk_tier", "") if chat_context else ""
    probability = str(chat_context.get("probability", "")) if chat_context else ""
    notify = bool(chat_context.get("notify", False)) if chat_context else False
    widget = chat_widget_html(risk_tier, probability, notify)
    return render(
        "base.html",
        {"title": title, "styles": render_styles(), "body": body, "widget": widget},
    )


def prediction_error_html(message: str) -> str:
    body = f"""
    <section class="card result-card">
      <div class="pill">Check your inputs</div>
      <h1 style="font-size:38px">Input Error</h1>
      <p class="note">{escape(message)}</p>
      <div class="actions" style="justify-content:center">
        <a class="button" href="/predictor">Back to predictor</a>
      </div>
    </section>
    """
    return page_shell("Input Error - O-Beast", body)


def validate_form_input(input_data: dict) -> None:
    import pandas as pd

    validate_prediction_frame(pd.DataFrame([input_data]))


def build_prediction_input(
    *,
    age: int,
    sex: str,
    height_cm: float,
    weight_kg: float,
    family_history_obesity: int,
    high_calorie_food_frequency: int,
    vegetable_frequency: float,
    main_meals_per_day: float,
    food_between_meals_frequency: int,
    smoke: int,
    water_daily: float,
    calorie_monitoring: int,
    physical_activity_freq: float,
    screen_time_band: int,
    alcohol_frequency: int,
    transportation: str,
) -> dict:
    return {
        "age": age,
        "sex": sex,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "family_history_obesity": family_history_obesity,
        "high_calorie_food_frequency": high_calorie_food_frequency,
        "vegetable_frequency": vegetable_frequency,
        "main_meals_per_day": main_meals_per_day,
        "food_between_meals_frequency": food_between_meals_frequency,
        "smoke": smoke,
        "water_daily": water_daily,
        "calorie_monitoring": calorie_monitoring,
        "physical_activity_freq": physical_activity_freq,
        "screen_time_band": screen_time_band,
        "alcohol_frequency": alcohol_frequency,
        "transportation": transportation,
    }


def methods_html() -> str:
    methods = "".join(
        f'<div class="method"><strong>{name}</strong><span>{body}</span></div>'
        for name, body in METHODS
    )
    return f'<div class="method-grid">{methods}</div>'


def data_preparation_html() -> str:
    return """
    <div class="preparation-block">
      <div class="preparation-heading">
        <h2>Before learning: balance training data</h2>
        <p><strong>SMOTENC is not a prediction algorithm.</strong> It prepares uneven training data so smaller groups are not overlooked.</p>
      </div>
      <div class="preparation-flow" aria-label="SMOTENC data preparation flow">
        <div class="preparation-step">
          <div class="preparation-icon">1</div>
          <strong>Uneven groups</strong>
          <span>Some outcomes have many examples. Others have few.</span>
        </div>
        <div class="preparation-arrow" aria-hidden="true">&rarr;</div>
        <div class="preparation-step">
          <div class="preparation-icon">2</div>
          <strong>SMOTENC balances</strong>
          <span>Creates realistic training examples while respecting answer types.</span>
        </div>
        <div class="preparation-arrow" aria-hidden="true">&rarr;</div>
        <div class="preparation-step">
          <div class="preparation-icon">3</div>
          <strong>Balanced training data</strong>
          <span>Prediction algorithms learn from a fairer set of examples.</span>
        </div>
      </div>
    </div>
    """


def research_visual_html() -> str:
    return """
    <div class="research-map">
      <div class="pipeline-stack">
        <div class="pipeline-step">
          <span>1</span>
          <div>
            <strong>Your answers become dots</strong>
            <small>Body, food, activity, sleep, and family history are placed in a simple learning space.</small>
          </div>
        </div>
        <div class="pipeline-step">
          <span>2</span>
          <div>
            <strong>The app looks for groups</strong>
            <small>Similar dots gather together, helping O-Beast notice lower-risk and higher-risk patterns.</small>
          </div>
        </div>
        <div class="pipeline-step">
          <span>3</span>
          <div>
            <strong>Algorithms try a dividing wall</strong>
            <small>Each method tries its own way to separate the groups and make a better guess.</small>
          </div>
        </div>
        <div class="pipeline-step">
          <span>4</span>
          <div>
            <strong>You get a clear result</strong>
            <small>O-Beast turns the pattern into a probability, a risk band, and practical wellness advice.</small>
          </div>
        </div>
      </div>
      <div class="cube-scene" aria-label="3D pattern view showing example groups and a dividing wall">
        <div class="cube-title">3D pattern view</div>
        <div class="cube-box">
          <span class="cube-face cube-floor"></span>
          <span class="cube-face cube-back"></span>
          <span class="cube-face cube-side"></span>
          <span class="decision-sheet"></span>
          <span class="cube-point" style="left:48px; top:154px"></span>
          <span class="cube-point" style="left:72px; top:130px"></span>
          <span class="cube-point" style="left:98px; top:168px"></span>
          <span class="cube-point" style="left:122px; top:138px"></span>
          <span class="cube-point" style="left:136px; top:116px"></span>
          <span class="cube-point hot" style="left:176px; top:86px"></span>
          <span class="cube-point hot" style="left:204px; top:62px"></span>
          <span class="cube-point hot" style="left:226px; top:102px"></span>
        </div>
        <div class="cube-legend">
          <span><i></i>lower pattern</span>
          <span><i></i>higher pattern</span>
        </div>
      </div>
    </div>
    """



def algorithm_visual_html() -> str:
    cards = [
        (
            "Logistic Regression",
            "Makes an S-curve, so the predicted probability stays between 0 and 1.",
            """
            <div class="algo-visual">
              <svg class="logistic-svg" viewBox="0 0 300 140" aria-label="Logistic regression S-curve">
                <line x1="42" y1="16" x2="42" y2="116" stroke="#18181b" stroke-width="3" stroke-linecap="round"/>
                <line x1="42" y1="116" x2="280" y2="116" stroke="#18181b" stroke-width="3" stroke-linecap="round"/>
                <line x1="42" y1="30" x2="270" y2="30" stroke="#8a8a95" stroke-width="2" stroke-dasharray="6 6"/>
                <text x="10" y="34" font-size="14">y=1</text>
                <text x="10" y="120" font-size="14">y=0</text>
                <text x="112" y="135" font-size="13">input score</text>
                <text x="88" y="22" font-size="14" fill="#c2410c">S-curve</text>
                <path d="M55 112 C105 112 127 105 141 74 C154 42 172 30 255 28" fill="none" stroke="#e1306c" stroke-width="4" stroke-linecap="round"/>
                <circle cx="54" cy="116" r="5" fill="#12a8c7"/><circle cx="72" cy="116" r="5" fill="#12a8c7"/>
                <circle cx="90" cy="116" r="5" fill="#12a8c7"/><circle cx="108" cy="116" r="5" fill="#12a8c7"/>
                <circle cx="186" cy="30" r="5" fill="#12a8c7"/><circle cx="204" cy="30" r="5" fill="#12a8c7"/>
                <circle cx="222" cy="30" r="5" fill="#12a8c7"/><circle cx="240" cy="30" r="5" fill="#12a8c7"/>
                <circle cx="260" cy="28" r="6" fill="#ef4444"/>
                <text x="176" y="72" font-size="12">probability stays</text>
                <text x="176" y="87" font-size="12">inside 0 to 1</text>
              </svg>
            </div>
            """,
            ["Combines answers into one score", "S-curve converts score to probability", "Output cannot go below 0 or above 1"],
        ),
        (
            "Support Vector Machine",
            "Looks for the widest gap between lower-risk and higher-risk examples.",
            """
            <div class="algo-visual">
              <span class="viz-chip cool" style="left:16px; top:18px">group A</span>
              <span class="viz-chip hot" style="right:16px; bottom:18px">group B</span>
              <span class="dot" style="left:21%; top:35%"></span><span class="dot" style="left:32%; top:57%"></span>
              <span class="dot alt" style="left:69%; top:38%"></span><span class="dot alt" style="left:79%; top:61%"></span>
              <span class="line-viz vertical"></span>
              <span class="viz-chip" style="left:44%; top:72%">wide gap</span>
            </div>
            """,
            ["Compares groups by distance", "Uses support examples near the boundary", "Good for clean separation"],
        ),
        (
            "Random Forest",
            "Many small trees vote together, so one strange answer does not control everything.",
            """
            <div class="algo-visual">
              <span class="tree-viz"></span>
              <span class="node" style="left:45%; top:8%">Q</span><span class="node" style="left:18%; top:58%">L</span><span class="node" style="left:44%; top:58%">M</span><span class="node" style="left:70%; top:58%">H</span>
              <span class="viz-chip hot" style="right:12px; top:18px">vote</span>
            </div>
            """,
            ["Tree 1 asks about activity", "Tree 2 asks about habits", "Final answer is the vote"],
        ),
        (
            "Naive Bayes",
            "Each answer is a clue, then the clues are multiplied into a probability.",
            """
            <div class="algo-visual">
              <span class="flow-step" style="left:14%; top:24%">activity</span>
              <span class="flow-step" style="left:14%; top:66%">food</span>
              <span class="viz-arrow" style="left:38%; top:44%; width:36%; transform:rotate(-8deg)"></span>
              <span class="viz-arrow" style="left:38%; top:74%; width:34%; transform:rotate(-24deg)"></span>
              <span class="flow-step" style="right:12%; top:42%">%</span>
            </div>
            """,
            ["Counts how common each clue is", "Assumes clues are independent", "Simple baseline for comparison"],
        ),
        (
            "Neural Network",
            "Input answers pass through hidden layers that learn nonlinear patterns.",
            """
            <div class="algo-visual">
              <span class="viz-chip" style="left:12px; top:12px">inputs</span>
              <span class="flow-step" style="left:13%; top:38%">10</span>
              <span class="flow-step" style="left:42%; top:24%">H1</span>
              <span class="flow-step" style="left:42%; top:64%">H2</span>
              <span class="flow-step" style="right:12%; top:42%">%</span>
              <span class="viz-arrow" style="left:29%; top:46%; width:26%; transform:rotate(-17deg)"></span>
              <span class="viz-arrow" style="left:29%; top:65%; width:26%; transform:rotate(15deg)"></span>
              <span class="viz-arrow" style="left:58%; top:43%; width:24%; transform:rotate(18deg)"></span>
            </div>
            """,
            ["Learns combinations of answers", "Updates weights after mistakes", "Useful but needs enough data"],
        ),
        (
            "XGBoost",
            "Boosted trees learn in rounds, where each round fixes mistakes from the last one.",
            """
            <div class="algo-visual">
              <span class="flow-step" style="left:10%; top:42%">T1</span>
              <span class="flow-step" style="left:38%; top:42%">T2</span>
              <span class="flow-step" style="left:66%; top:42%">T3</span>
              <span class="viz-arrow" style="left:25%; top:56%; width:20%"></span>
              <span class="viz-arrow" style="left:53%; top:56%; width:20%"></span>
              <span class="viz-chip hot" style="right:12px; top:14px">fix errors</span>
            </div>
            """,
            ["Tree after tree", "Next tree focuses on errors", "Strong for tabular survey data"],
        ),
        (
            "SMOTENC",
            "Balances uneven classes while keeping category answers like sex and family history valid.",
            """
            <div class="algo-visual">
              <span class="viz-chip cool" style="left:14px; top:12px">minority class</span>
              <span class="dot alt" style="left:23%; top:42%"></span><span class="dot alt" style="left:40%; top:48%"></span>
              <span class="dot ghost" style="left:31%; top:45%"></span><span class="dot ghost" style="left:49%; top:52%"></span>
              <span class="viz-chip hot" style="right:12px; top:12px">numeric + category</span>
              <span class="dot" style="left:68%; top:66%"></span><span class="dot" style="left:78%; top:58%"></span>
              <span class="viz-arrow" style="left:43%; top:51%; width:24%; transform:rotate(14deg)"></span>
              <span class="viz-chip" style="left:22%; bottom:14px">new safe samples</span>
            </div>
            """,
            ["Creates extra minority examples", "Does not make fake half-categories", "Runs before one-hot encoding"],
        ),
    ]
    return '<div class="algorithm-lab">' + ''.join(
        f"""
        <div class="card algo-card">
          {visual}
          <h2>{name}</h2>
          <p>{body}</p>
          <div class="algo-points">
            {''.join(f'<div class="algo-point"><strong>{index}</strong><span>{point}</span></div>' for index, point in enumerate(points, start=1))}
          </div>
        </div>
        """
        for name, body, visual, points in cards
    ) + '</div>'


def advice_cards_html(advice: dict) -> str:
    cards = "".join(
        f"""
        <div class="card advice-card">
          <div class="pill">{item["priority"]}</div>
          <h2>{item["title"]}</h2>
          <p><strong>Reason:</strong> {item["why"]}</p>
          <p><strong>Action:</strong> {item["action"]}</p>
          <p class="note">Source idea: {item["source"]}</p>
        </div>
        """
        for item in advice["cards"]
    )
    return f'<div class="advice-grid">{cards}</div>'


def source_list_html(sources: list[dict]) -> str:
    links = "".join(
        f'<div class="method"><strong><a href="{source["url"]}" target="_blank" rel="noreferrer">{source["name"]}</a></strong><span>{source["supports"]}</span></div>'
        for source in sources
    )
    return f'<div class="source-list">{links}</div>'


def hidden_inputs_html(input_data: dict) -> str:
    return "".join(
        f'<input type="hidden" name="{escape(str(key), quote=True)}" value="{escape(str(value), quote=True)}">'
        for key, value in input_data.items()
    )


def producer_section_html() -> str:
    cards = []
    for producer in PRODUCER_PROFILES:
        contact_links = "".join(
            f'<a class="contact-link" href="{escape(url, quote=True)}" target="_blank" rel="noreferrer">{escape(label)}</a>'
            for label, url in producer["contacts"]
        )
        if not contact_links:
            contact_links = '<span class="contact-slot">Contact link coming soon</span>'
        cards.append(
            f"""
            <article class="producer-card">
              <div class="producer-photo-wrap">
                <img class="producer-photo {escape(producer["photo_class"], quote=True)}" src="{escape(producer["image"], quote=True)}" alt="Portrait of {escape(producer["name"], quote=True)}">
              </div>
              <div class="producer-body">
                <h3>{escape(producer["name"])}</h3>
                <p>Producer</p>
                <div class="contact-row" aria-label="Contact links for {escape(producer["name"], quote=True)}">
                  {contact_links}
                </div>
              </div>
            </article>
            """
        )
    return f"""
    <section class="producers-section" aria-labelledby="producer-heading">
      <div class="producer-heading">
        <div>
          <div class="kicker">Know About Producers</div>
          <h2 id="producer-heading">Meet the O-Beast Team</h2>
        </div>
        <p>The students behind this research app.</p>
      </div>
      <div class="producer-grid">{''.join(cards)}</div>
    </section>
    """


def chat_widget_html(risk_tier: str = "", probability: str = "", notify: bool = False) -> str:
    tier_attr = f'data-risk-tier="{escape(risk_tier, quote=True)}"'
    prob_attr = f'data-probability="{escape(probability, quote=True)}"'
    notify_attr = 'data-notify="1"' if notify else ""
    badge_style = 'style="display:block"' if (risk_tier or notify) else ""
    return f"""
{CHAT_WIDGET_STYLE}
{PREMIUM_HEALTH_CHAT_STYLE}
<button id="beast-fab" class="beast-fab" aria-label="Open Beast 1.0 chat" {tier_attr} {prob_attr} {notify_attr}>
  <img src="/static/beast1-logo.png" alt="Beast 1.0 logo">
  <span class="beast-fab-label">Beast 1.0</span>
  <span class="beast-fab-badge" {badge_style}></span>
</button>
<div id="beast-chat" class="beast-chat">
  <div class="beast-head">
    <img class="beast-head-img" src="/static/beast1-logo.png" alt="Beast 1.0">
    <div class="beast-head-info"><strong>Beast 1.0</strong><small>Obesity Q&amp;A &middot; Online</small></div>
    <div class="beast-lang-toggle">
      <button class="beast-lang-btn active" data-lang="en">EN</button>
      <button class="beast-lang-btn" data-lang="th">TH</button>
    </div>
    <button class="beast-close" id="beast-close-btn" aria-label="Close chat">&#x2715;</button>
  </div>
  <div id="beast-ctx" class="beast-ctx" hidden></div>
  <div id="beast-msgs" class="beast-msgs">
    <div class="beast-msg beast-bot">Hi! I&apos;m Beast 1.0 &#x1F981; Ask me about obesity &mdash; or tap a chip below.</div>
  </div>
  <div class="beast-chips">
    <button class="beast-chip" data-query="What causes obesity?">Causes</button>
    <button class="beast-chip" data-query="How to prevent obesity?">Prevention</button>
    <button class="beast-chip" data-query="What should I eat?">Diet</button>
    <button class="beast-chip" data-query="How much should I exercise?">Exercise</button>
  </div>
  <form id="beast-form" class="beast-form" onsubmit="return false">
    <input id="beast-input" class="beast-input" placeholder="Ask Beast 1.0&hellip;" autocomplete="off">
    <button type="submit" class="beast-send" aria-label="Send">&#x27A4;</button>
  </form>
</div>
<div id="beast-notify" class="beast-notify">
  <span class="beast-notify-text" id="beast-notify-text"></span>
  <button class="beast-notify-close" id="beast-notify-close" aria-label="Dismiss">&#x2715;</button>
</div>
<script>
(function(){{
  var fab=document.getElementById('beast-fab');
  var chat=document.getElementById('beast-chat');
  var msgs=document.getElementById('beast-msgs');
  var form=document.getElementById('beast-form');
  var inp=document.getElementById('beast-input');
  var ctx=document.getElementById('beast-ctx');
  var ntf=document.getElementById('beast-notify');
  var ntfText=document.getElementById('beast-notify-text');
  var ntfClose=document.getElementById('beast-notify-close');
  var closeBtn=document.getElementById('beast-close-btn');
  var lang='auto';
  var tier=fab.dataset.riskTier||'';
  var prob=fab.dataset.probability||'';
  var shouldNotify=fab.dataset.notify==='1';

  function closeChat(){{ chat.classList.remove('open'); }}
  function openChat(){{ chat.classList.add('open'); dismissNotify(); inp.focus(); }}
  function dismissNotify(){{ ntf.classList.remove('show'); }}
  function isChatOpen(){{ return chat.classList.contains('open'); }}

  /* Context banner */
  if(tier){{ctx.hidden=false;ctx.textContent='📊 Your result: '+tier+' ('+Math.round(parseFloat(prob)*100)+'%) — I will tailor my answers to your score.';}}

  /* Notification + shake on Result / Advice pages (only if chat not already open) */
  if(shouldNotify){{
    setTimeout(function(){{
      if(isChatOpen())return;
      var l=(lang==='auto')?'en':lang;
      var msgs_en='If you want any extra answers, I am here 👋';
      var msgs_th='ถ้าอยากรู้เพิ่มเติม ถามฉันได้เลย 👋';
      ntfText.textContent=(l==='th')?msgs_th:msgs_en;
      ntf.classList.add('show','pop');
      fab.classList.add('shake');
      fab.addEventListener('animationend',function(){{fab.classList.remove('shake');}},{{once:true}});
    }},1500);
  }}

  /* Close notification */
  ntfClose.addEventListener('click',function(e){{e.stopPropagation();dismissNotify();}});

  /* FAB: toggle chat open/close */
  fab.addEventListener('click',function(){{
    if(isChatOpen()){{closeChat();}}else{{openChat();}}
  }});

  /* × button in header closes chat */
  closeBtn.addEventListener('click',function(){{closeChat();}});

  /* Click outside chat closes it */
  document.addEventListener('click',function(e){{
    if(isChatOpen()&&!chat.contains(e.target)&&!fab.contains(e.target)){{closeChat();}}
  }});

  /* Lang toggle */
  document.querySelectorAll('.beast-lang-btn').forEach(function(b){{
    b.addEventListener('click',function(e){{
      e.stopPropagation();
      document.querySelectorAll('.beast-lang-btn').forEach(function(x){{x.classList.remove('active');}});
      b.classList.add('active');lang=b.dataset.lang;updateChips(lang);
      inp.placeholder=lang==='th'?'พิมพ์คำถาม…':'Ask Beast 1.0…';
    }});
  }});

  /* Chips */
  document.querySelectorAll('.beast-chip').forEach(function(c){{c.addEventListener('click',function(){{send(c.dataset.query);}});}});

  /* Send */
  form.addEventListener('submit',function(e){{e.preventDefault();var t=inp.value.trim();if(!t)return;inp.value='';send(t);}});
  function send(text){{
    addBubble(text,'user');
    document.querySelector('.beast-chips').classList.add('compact');
    var body={{message:text,lang:lang}};
    if(tier)body.context={{risk_tier:tier,probability:parseFloat(prob)}};
    fetch('/chat',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}})
      .then(function(r){{return r.json();}})
      .then(function(d){{if(lang==='auto'&&d.detected_lang)lang=d.detected_lang;addBubble(d.answer,'bot',d.source);}})
      .catch(function(){{addBubble('Sorry, something went wrong.','bot');}});
  }}
  function addBubble(text,role,src){{
    var d=document.createElement('div');
    d.className='beast-msg beast-'+role;d.textContent=text;
    if(src){{var s=document.createElement('span');s.className='beast-src';s.textContent='Source: '+src;d.appendChild(s);}}
    msgs.appendChild(d);msgs.scrollTop=msgs.scrollHeight;
  }}
  function updateChips(l){{
    var labels={{en:['Causes','Prevention','Diet','Exercise'],th:['สาเหตุ','การป้องกัน','อาหาร','ออกกำลังกาย']}};
    var queries={{en:['What causes obesity?','How to prevent obesity?','What should I eat?','How much should I exercise?'],th:['สาเหตุของโรคอ้วนคืออะไร','ป้องกันโรคอ้วนอย่างไร','ควรกินอะไร','ควรออกกำลังกายเท่าไหร่']}};
    var lk=(l==='auto')?'en':l;
    document.querySelectorAll('.beast-chip').forEach(function(c,i){{c.textContent=labels[lk][i];c.dataset.query=queries[lk][i];}});
  }}
}})();
</script>
"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    research_visual = research_visual_html()
    algorithm_cards = algorithm_visual_html()
    producers = producer_section_html()
    body = f"""
    <section class="hero premium-hero">
      <div class="hero-copy">
        <div class="kicker">Understand your habits</div>
        <h1>Know your risk.<br><span>Shape tomorrow.</span></h1>
        <p class="lead">
          O-Beast turns body and lifestyle answers into an easy obesity-risk probability,
          then offers practical wellness guidance for your next step.
        </p>
        <div class="actions">
          <a class="button" href="/predictor">Begin risk check</a>
          <a class="button secondary" href="#learn">See how it works</a>
        </div>
      </div>
      <div class="premium-health-visual">
        <img src="/static/premium-health-pattern.png" alt="Lifestyle signals flowing into an organized wellness pattern">
        <div class="visual-caption">
          <strong>Daily signals become a clearer pattern</strong>
          <span>Body profile, movement, food, sleep, and family history work together.</span>
        </div>
      </div>
    </section>

    <section class="trust-strip" aria-label="O-Beast summary">
      <div class="trust-item"><strong>10</strong><span>simple body and lifestyle answers</span></div>
      <div class="trust-item"><strong>7</strong><span>learning methods compared fairly</span></div>
      <div class="trust-item"><strong>1</strong><span>clear probability with useful guidance</span></div>
    </section>

    <section class="section-intro" id="learn">
      <div class="kicker">See how learning works</div>
      <h1>Answers become patterns.</h1>
      <p class="lead">O-Beast lets several learning methods study the same answers, checks their performance, and uses the strongest current pattern to estimate probability.</p>
    </section>
    <section>
      {research_visual}
      {algorithm_cards}
    </section>

    <section class="section-grid support-grid">
      <div class="card">
        <div class="kicker">Built for understanding</div>
        <h2>A result you can read</h2>
        <p>Probability, risk band, and next actions appear first. Model details remain available when you want to explore them.</p>
        <a class="button secondary" href="/methods">Explore methods</a>
      </div>
      <div class="card">
        <div class="kicker">Practical wellness</div>
        <h2>Small steps matched to your answers</h2>
        <p>Advice focuses on movement, screen habits, sleep, food, and awareness using trusted public-health guidance.</p>
        <a class="button secondary" href="/advice">Preview guidance</a>
      </div>
    </section>
    {producers}
    """
    return page_shell("SK Obesity ML Research", body)


@app.get("/advice", response_class=HTMLResponse)
def advice_intro() -> str:
    example_advice = generate_advice(
        {
            "age": 16,
            "sex": "M",
            "height_cm": 170,
            "weight_kg": 65,
            "family_history_obesity": 0,
            "high_calorie_food_frequency": 1,
            "vegetable_frequency": 2,
            "main_meals_per_day": 3,
            "food_between_meals_frequency": 1,
            "smoke": 0,
            "water_daily": 2,
            "calorie_monitoring": 0,
            "physical_activity_freq": 2,
            "screen_time_band": 1,
            "alcohol_frequency": 0,
            "transportation": "Public_Transportation",
        },
        {"obesity_probability": 0.25},
    )
    body = f"""
    <section class="card">
      <div class="kicker">Wellness advice model</div>
      <h1>Advice Based on Your Answers</h1>
      <p class="lead">
        This page uses a transparent rule-based model. It does not diagnose disease; it turns lifestyle answers into
        practical wellness suggestions using trustworthy public-health references.
      </p>
      <div class="actions">
        <a class="button" href="/predictor">Start with your answers</a>
        <a class="button secondary" href="/methods">Review ML methods</a>
      </div>
    </section>
    {advice_cards_html(example_advice)}
    <section class="card" style="margin-top:18px">
      <h2>References used</h2>
      <p class="note">These are the evidence sources behind the rule-based advice engine.</p>
      {source_list_html(example_advice["sources"])}
    </section>
    """
    return page_shell("Advice - O-Beast", body)


@app.get("/methods", response_class=HTMLResponse)
def methods() -> str:
    try:
        artifact = load_artifact()
        best = artifact.get("base_model_name", "not trained yet")
        used_smote = "Yes" if artifact.get("used_smote") else "No or not enough minority samples"
        selection = artifact.get("selection_rule", "Train first to see selection rule.")
        xgboost_status = artifact.get("xgboost_status", "unknown")
        xgboost_display = (
            "Unavailable in this app environment. Other models remain available for comparison."
            if str(xgboost_status).startswith("disabled:")
            else xgboost_status
        )
        validation_strategy = artifact.get("validation_strategy", "Not recorded yet")
        resampling_strategy = artifact.get("resampling_strategy", "Not recorded yet")
        dataset_warning = artifact.get("dataset_warning", "")
        metrics = artifact.get("metrics", {})
        roc_curves = artifact.get("roc_curves", {})
    except FileNotFoundError:
        best = "not trained yet"
        used_smote = "not trained yet"
        selection = "Train the model first."
        xgboost_status = "not trained yet"
        xgboost_display = xgboost_status
        validation_strategy = "not trained yet"
        resampling_strategy = "not trained yet"
        dataset_warning = ""
        metrics = {}
        roc_curves = {}

    metric_bars = metric_bars_html(metrics)
    metric_table = metric_table_html(metrics)
    evaluation_dashboard = evaluation_dashboard_html(best, metrics, roc_curves)

    body = f"""
    <section class="card">
      <div class="kicker">Model check</div>
      <h1>How O-Beast Checks Results</h1>
      <p class="lead">
        O-Beast compares several learning methods, checks how well each one performs, and shows
        which method produced the current result. The scores are included to make the prediction
        easier to understand, not to claim medical certainty.
      </p>
      {data_preparation_html()}
      <div class="algorithm-heading">
        <h2>Prediction algorithms</h2>
        <p>These models learn patterns and produce obesity-risk probabilities.</p>
      </div>
      {methods_html()}
    </section>
    <section class="section-grid">
      <div class="card"><h2>Current best model</h2><p>{best}</p></div>
      <div class="card"><h2>Training data balanced?</h2><p>{used_smote}. SMOTENC prepares data; it does not make predictions.</p></div>
      <div class="card"><h2>Selection rule</h2><p>{selection}</p></div>
      <div class="card"><h2>XGBoost status</h2><p>{xgboost_display}</p></div>
      <div class="card"><h2>Validation strategy</h2><p>{validation_strategy}</p></div>
      <div class="card"><h2>Balancing strategy</h2><p>{resampling_strategy}</p></div>
      <div class="card"><h2>Dataset note</h2><p>{dataset_warning or "Ready for real dataset."}</p></div>
      <div class="card"><h2>Risk tiers</h2><p>Output now uses five probability tiers: very low, low, moderate, high, and very high.</p></div>
      <div class="card"><h2>Survey data support</h2><p>O-Beast can prepare answers from more than one survey format before later training.</p></div>
    </section>
    <section class="card" style="margin-top:18px">
      <div class="kicker">Research details</div>
      <h1 style="font-size:44px">Model tournament scores</h1>
      <p class="lead">
        These scores compare the learning methods during validation. They are kept on this Methods page
        so the prediction result can stay simple for users.
      </p>
      {metric_bars}
      {metric_table}
      {evaluation_dashboard}
    </section>
    """
    return page_shell("Methods - SK Obesity ML", body)


@app.get("/predictor", response_class=HTMLResponse)
def predictor() -> str:
    body = f"""
    <section class="predict-layout">
      <aside class="card profile">
        <div class="pill">Guided risk check</div>
        <img class="beast-mark" src="/static/obeast-logo.png" alt="O-Beast aggressive beast mascot logo">
        <h2>One clear step at a time</h2>
        <p class="note">
          Answer UCI-style body and lifestyle questions. O-Beast combines them into an educational probability estimate.
        </p>
        <div class="profile-guide">
          <div><strong>1</strong><span>Share your body profile.</span></div>
          <div><strong>2</strong><span>Describe activity and food patterns.</span></div>
          <div><strong>3</strong><span>Review answers before prediction.</span></div>
        </div>
      </aside>

      <form class="predict-form" id="predictionForm" action="/predict-form" method="post">
        <div class="form-head">
          <div>
            <div class="kicker">Your profile</div>
            <h2 style="margin-top:12px">Start your risk check</h2>
          </div>
          <div class="pill" id="stepCounter">Step 1 of 5</div>
        </div>
        <div class="form-progress" aria-label="Form progress"><span id="formProgress"></span></div>
        <div class="step-meta"><span id="stepName">Body profile</span><span>Educational estimate</span></div>

        <section class="form-step active" data-step="1">
          <h2>First, your body profile.</h2>
          <p>These answers help calculate BMI and provide basic context for the model.</p>
          <div class="field-grid">
            <label>Age <input name="age" type="number" min="14" max="100" value="16" required></label>
            <label>Sex <select name="sex"><option value="M">Male</option><option value="F">Female</option></select></label>
            <label>Height in centimetres <input name="height_cm" type="number" min="80" max="230" step="0.1" value="170" required></label>
            <label>Weight in kilograms <input name="weight_kg" type="number" min="20" max="250" step="0.1" value="65" required></label>
          </div>
          <div class="step-actions"><span></span><button type="button" data-next>Continue</button></div>
        </section>

        <section class="form-step" data-step="2">
          <h2>How does a usual day feel?</h2>
          <p>Movement, sitting time, water intake, sleep, and transport create important lifestyle signals.</p>
          <div class="field-grid">
            <label>Physical activity (days per week)
              <select name="physical_activity_freq">
                <option value="0">I do not exercise</option>
                <option value="1">1-2 days</option>
                <option value="2" selected>2-4 days</option>
                <option value="3">4-5 days</option>
              </select>
            </label>
            <label>Daily device / screen time
              <select name="screen_time_band">
                <option value="0">0-2 hours</option>
                <option value="1" selected>3-5 hours</option>
                <option value="2">More than 5 hours</option>
              </select>
            </label>
            <label>Daily water intake
              <select name="water_daily">
                <option value="1">Less than 1 litre</option>
                <option value="2" selected>1-2 litres</option>
                <option value="3">More than 2 litres</option>
              </select>
            </label>
            <label>Main transportation
              <select name="transportation">
                <option value="Public_Transportation">Public transportation</option>
                <option value="Walking">Walking</option>
                <option value="Bike">Bike</option>
                <option value="Motorbike">Motorbike</option>
                <option value="Automobile">Automobile</option>
              </select>
            </label>
          </div>
          <div class="step-actions"><button class="step-back" type="button" data-back>Previous</button><button type="button" data-next>Continue</button></div>
        </section>

        <section class="form-step" data-step="3">
          <h2>Now, your food pattern.</h2>
          <p>These questions follow the UCI-style habit format used by the prototype model.</p>
          <div class="field-grid">
            <label>Frequent high-calorie food <select name="high_calorie_food_frequency"><option value="0">No</option><option value="1">Yes</option></select></label>
            <label>Vegetable frequency, 1-3 <input name="vegetable_frequency" type="number" min="1" max="3" step="0.1" value="2" required></label>
            <label>Main meals per day, 1-4 <input name="main_meals_per_day" type="number" min="1" max="4" step="0.1" value="3" required></label>
            <label>Food between meals
              <select name="food_between_meals_frequency">
                <option value="0">No</option>
                <option value="1" selected>Sometimes</option>
                <option value="2">Frequently</option>
                <option value="3">Always</option>
              </select>
            </label>
          </div>
          <div class="step-actions"><button class="step-back" type="button" data-back>Previous</button><button type="button" data-next>Continue</button></div>
        </section>

        <section class="form-step" data-step="4">
          <h2>Finally, health context.</h2>
          <p>Family history and a few UCI-style context answers help the prototype match the training format.</p>
          <div class="field-grid">
            <label>Family history of obesity <select name="family_history_obesity"><option value="0">No</option><option value="1">Yes</option></select></label>
            <label>Do you smoke? (your own smoking, not people around you) <select name="smoke"><option value="0">No</option><option value="1">Yes</option></select></label>
            <label>Calorie monitoring <select name="calorie_monitoring"><option value="0">No</option><option value="1">Yes</option></select></label>
            <label>Alcohol frequency
              <select name="alcohol_frequency">
                <option value="0">No</option>
                <option value="1">Sometimes</option>
                <option value="2">Frequently</option>
                <option value="3">Always</option>
              </select>
            </label>
          </div>
          <div class="step-actions"><button class="step-back" type="button" data-back>Previous</button><button type="button" data-next>Review answers</button></div>
        </section>

        <section class="form-step" data-step="5">
          <h2>Ready for your estimate.</h2>
          <p>Review your answers, then let O-Beast find the strongest current pattern.</p>
          <div class="review-grid" id="reviewGrid" aria-live="polite"></div>
          <p class="note">This result supports health education. It is not a medical diagnosis.</p>
          <div class="step-actions"><button class="step-back" type="button" data-back>Previous</button><button type="submit">Estimate probability</button></div>
        </section>
        <noscript>
          <p class="note">JavaScript is off. All questions remain available; complete every field and submit the form.</p>
          <button type="submit">Estimate probability</button>
        </noscript>
      </form>
    </section>
    <script>
    (function(){{
      var form=document.getElementById('predictionForm');
      var steps=Array.from(form.querySelectorAll('.form-step'));
      var counter=document.getElementById('stepCounter');
      var progress=document.getElementById('formProgress');
      var name=document.getElementById('stepName');
      var review=document.getElementById('reviewGrid');
      var current=0;
      var names=['Body profile','Daily routine','Food pattern','Health context','Review'];

      function show(index,shouldScroll){{
        current=Math.max(0,Math.min(index,steps.length-1));
        steps.forEach(function(step,i){{step.classList.toggle('active',i===current);}});
        counter.textContent='Step '+(current+1)+' of '+steps.length;
        name.textContent=names[current];
        progress.style.transform='scaleX('+((current+1)/steps.length)+')';
        if(current===steps.length-1)buildReview();
        if(shouldScroll!==false){{
          form.scrollIntoView({{behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth',block:'start'}});
        }}
      }}

      function validStep(){{
        var inputs=Array.from(steps[current].querySelectorAll('input,select'));
        return inputs.every(function(input){{return input.reportValidity();}});
      }}

      function buildReview(){{
          var labels={{
            age:'Age',sex:'Sex',height_cm:'Height',weight_kg:'Weight',
            physical_activity_freq:'Activity (days/week)',screen_time_band:'Device time',
            water_daily:'Daily water',transportation:'Transportation',
            high_calorie_food_frequency:'High-calorie food',vegetable_frequency:'Vegetables',
            main_meals_per_day:'Main meals',food_between_meals_frequency:'Between-meal food',
            family_history_obesity:'Family history',smoke:'Smoke (self)',
            calorie_monitoring:'Calorie monitoring',alcohol_frequency:'Alcohol frequency'
          }};
        review.innerHTML='';
        Object.keys(labels).forEach(function(key){{
          var input=form.elements[key];
          var value=input.options?input.options[input.selectedIndex].text:input.value;
          var item=document.createElement('div');
          item.className='review-item';
          item.innerHTML='<span>'+labels[key]+'</span><strong>'+value+'</strong>';
          review.appendChild(item);
        }});
      }}

      form.querySelectorAll('[data-next]').forEach(function(button){{
        button.addEventListener('click',function(){{if(validStep())show(current+1);}});
      }});
      form.querySelectorAll('[data-back]').forEach(function(button){{
        button.addEventListener('click',function(){{show(current-1);}});
      }});
      show(0,false);
    }})();
    </script>
    """
    return page_shell("Predictor - SK Obesity ML", body)


@app.post("/advice", response_class=HTMLResponse)
def advice_from_form(
    age: int = Form(...),
    sex: str = Form(...),
    height_cm: float = Form(...),
    weight_kg: float = Form(...),
    family_history_obesity: int = Form(...),
    high_calorie_food_frequency: int = Form(...),
    vegetable_frequency: float = Form(...),
    main_meals_per_day: float = Form(...),
    food_between_meals_frequency: int = Form(...),
    smoke: int = Form(...),
    water_daily: float = Form(...),
    calorie_monitoring: int = Form(...),
    physical_activity_freq: float = Form(...),
    screen_time_band: int = Form(...),
    alcohol_frequency: int = Form(...),
    transportation: str = Form(...),
) -> str:
    input_data = build_prediction_input(
        age=age,
        sex=sex,
        height_cm=height_cm,
        weight_kg=weight_kg,
        family_history_obesity=family_history_obesity,
        high_calorie_food_frequency=high_calorie_food_frequency,
        vegetable_frequency=vegetable_frequency,
        main_meals_per_day=main_meals_per_day,
        food_between_meals_frequency=food_between_meals_frequency,
        smoke=smoke,
        water_daily=water_daily,
        calorie_monitoring=calorie_monitoring,
        alcohol_frequency=alcohol_frequency,
        transportation=transportation,
        physical_activity_freq=physical_activity_freq,
        screen_time_band=screen_time_band,
    )
    try:
        validate_form_input(input_data)
        result = predict_probability(input_data)
    except ValueError as exc:
        return HTMLResponse(prediction_error_html(str(exc)), status_code=400)
    advice = generate_advice(input_data, result)
    percent = round(result["obesity_probability"] * 100, 1)
    body = f"""
    <section class="card">
      <div class="kicker">{advice["focus"]}</div>
      <h1>Your Wellness Advice</h1>
      <p class="lead">
        Prediction estimate: <strong>{percent}%</strong> obesity-risk probability, with calculated BMI <strong>{advice["bmi"]}</strong>.
        The advice below is based on your answers and public-health guidance.
      </p>
      <p class="note">{advice["disclaimer"]}</p>
    </section>
    {advice_cards_html(advice)}
    <section class="card" style="margin-top:18px">
      <h2>Why these sources?</h2>
      <p class="note">The advice engine uses official international and public-health references rather than random internet tips.</p>
      {source_list_html(advice["sources"])}
      <div class="actions">
        <a class="button" href="/predictor">Try another input</a>
        <a class="button secondary" href="/methods">See methods</a>
      </div>
    </section>
    """
    return page_shell("Advice - O-Beast", body, chat_context={"notify": True})


@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(
    age: int = Form(...),
    sex: str = Form(...),
    height_cm: float = Form(...),
    weight_kg: float = Form(...),
    family_history_obesity: int = Form(...),
    high_calorie_food_frequency: int = Form(...),
    vegetable_frequency: float = Form(...),
    main_meals_per_day: float = Form(...),
    food_between_meals_frequency: int = Form(...),
    smoke: int = Form(...),
    water_daily: float = Form(...),
    calorie_monitoring: int = Form(...),
    physical_activity_freq: float = Form(...),
    screen_time_band: int = Form(...),
    alcohol_frequency: int = Form(...),
    transportation: str = Form(...),
) -> str:
    input_data = build_prediction_input(
        age=age,
        sex=sex,
        height_cm=height_cm,
        weight_kg=weight_kg,
        family_history_obesity=family_history_obesity,
        high_calorie_food_frequency=high_calorie_food_frequency,
        vegetable_frequency=vegetable_frequency,
        main_meals_per_day=main_meals_per_day,
        food_between_meals_frequency=food_between_meals_frequency,
        smoke=smoke,
        water_daily=water_daily,
        calorie_monitoring=calorie_monitoring,
        alcohol_frequency=alcohol_frequency,
        transportation=transportation,
        physical_activity_freq=physical_activity_freq,
        screen_time_band=screen_time_band,
    )
    try:
        validate_form_input(input_data)
        result = predict_probability(input_data)
    except ValueError as exc:
        return HTMLResponse(prediction_error_html(str(exc)), status_code=400)
    percent = round(result["obesity_probability"] * 100, 1)
    reason = best_model_reason(result)
    advice = generate_advice(input_data, result)
    next_actions = advice["cards"][1:4]
    next_actions_html = "".join(
        f'<div class="next-action"><strong>{index}</strong><span>{escape(item["action"])}</span></div>'
        for index, item in enumerate(next_actions, start=1)
    )
    dataset_warning_html = f"<p>{result['dataset_warning']}</p>" if result.get("dataset_warning") else ""
    body = f"""
    <section class="card result-card">
      <div class="kicker">Your O-Beast result</div>
      <div class="ring-wrap" id="result-ring-wrap" data-target="{percent}">
        <svg class="ring-svg" viewBox="0 0 220 220" aria-hidden="true">
          <circle class="ring-bg-circle" cx="110" cy="110" r="90"/>
          <circle class="ring-arc" id="result-arc" cx="110" cy="110" r="90"
            stroke-dasharray="565.49" stroke-dashoffset="565.49"
            transform="rotate(-90 110 110)"/>
        </svg>
        <div class="ring-inner">
          <span id="result-prob" class="prob">0%</span>
        </div>
      </div>
      <h1 style="font-size:38px">Estimated Probability</h1>
      <div class="band">{result["risk_tier_label"]}</div>
      <p class="note">Tier range: {result["risk_tier_range"]}. Coarse band: {result["risk_band"].title()}.</p>

      <div class="result-meaning">
        <h2>What this result means</h2>
        <p>This probability shows how closely your answers match patterns in the current training data. It can support awareness and healthier choices, but it does not diagnose obesity or predict your future with certainty.</p>
      </div>

      <div class="next-actions">
        <h2>Three useful next steps</h2>
        {next_actions_html}
      </div>

      {dataset_warning_html}
      <p>{result["disclaimer"]}</p>
      <div class="actions" style="justify-content:center">
        <form class="action-form" action="/advice" method="post">
          {hidden_inputs_html(input_data)}
          <button type="submit">View full guidance</button>
        </form>
        <a class="button secondary" href="/predictor">Try another input</a>
      </div>

      <details class="reason-box">
        <summary><strong>How O-Beast selected this model</strong></summary>
        <p>Current model: <strong>{result["base_model_name"]}</strong></p>
        <p>{reason}</p>
        <p>SMOTENC used during training: <strong>{"Yes" if result["used_smote"] else "No"}</strong></p>
        <a class="button secondary" href="/methods">See methods</a>
      </details>
    </section>
    <script>
    (function(){{
      var wrap=document.getElementById('result-ring-wrap');
      var arc=document.getElementById('result-arc');
      var probEl=document.getElementById('result-prob');
      if(!wrap||!arc||!probEl)return;
      var target=parseFloat(wrap.dataset.target);
      var C=2*Math.PI*90; /* 565.49 */
      var reduced=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      if(reduced){{
        arc.style.strokeDashoffset=C*(1-target/100);
        probEl.textContent=target.toFixed(1)+'%';
        arc.style.animation='neonPulse 2s ease-in-out infinite';
        return;
      }}
      var dur=1400,t0=null;
      function ease(t){{return 1-Math.pow(1-t,3);}}
      function tick(ts){{
        if(!t0)t0=ts;
        var p=Math.min((ts-t0)/dur,1),e=ease(p),v=e*target;
        arc.style.strokeDashoffset=C*(1-v/100);
        probEl.textContent=v.toFixed(1)+'%';
        if(p<1){{
          requestAnimationFrame(tick);
        }}else{{
          probEl.textContent=target.toFixed(1)+'%';
          arc.style.animation='neonPulse 2s ease-in-out infinite';
        }}
      }}
      requestAnimationFrame(tick);
    }})();
    </script>
    """
    return page_shell(
        "Result - SK Obesity ML",
        body,
        chat_context={
            "risk_tier": result["risk_tier_label"],
            "probability": result["obesity_probability"],
            "notify": True,
        },
    )


from obesity_ml.routes import api as _api_routes  # noqa: E402  (routes import app-level helpers)
app.include_router(_api_routes.router)
