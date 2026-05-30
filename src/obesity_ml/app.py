from html import escape
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from obesity_ml.advice import generate_advice
from obesity_ml.features import validate_prediction_frame
from obesity_ml.predict import load_artifact, predict_probability


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
    ("SMOTENC", "Balances minority/majority classes while respecting categorical survey answers."),
    ("Logistic Regression", "Baseline medical-risk model; easy to explain."),
    ("Support Vector Machine", "Finds a strong boundary between risk groups."),
    ("Random Forest", "Many decision trees vote together; robust for survey data."),
    ("Naive Bayes", "A simple probability baseline: each answer gives evidence, then the model combines the evidence."),
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
  }

  a { color: inherit; }

  main {
    width: min(1160px, calc(100vw - 32px));
    margin: 0 auto;
    padding: 28px 0 46px;
    animation: pageIn 520ms ease-out both;
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

  .brand {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    font-weight: 1000;
  }

  .brand-logo {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    box-shadow: 0 8px 20px rgba(225, 48, 108, 0.22);
  }

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
    align-items: start;
  }

  .hero-copy {
    padding: 14px;
    align-self: start;
    margin-top: 28px;
  }

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

  .action-form {
    display: inline;
    padding: 0;
    border: 0;
    border-radius: 0;
    background: transparent;
    box-shadow: none;
    backdrop-filter: none;
    animation: none;
  }

  .action-form button {
    width: auto;
    min-height: 48px;
    margin-top: 0;
    padding: 0 18px;
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

  .beast-mark {
    width: min(190px, 68vw);
    aspect-ratio: 1;
    display: block;
    margin: 8px auto 16px;
    border-radius: 34px;
    box-shadow: 0 22px 52px rgba(225, 48, 108, 0.28);
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
    gap: 14px;
    align-items: center;
    border-radius: 18px;
    padding: 10px;
    background: rgba(255,255,255,.09);
    margin-top: 9px;
    transition: transform 180ms ease, background 180ms ease;
  }

  .story:hover { transform: translateX(4px); background: rgba(255,255,255,.14); }

  .story > div:last-child {
    display: flex;
    align-items: baseline;
    gap: 9px;
    min-width: 0;
    flex-wrap: wrap;
  }

  .story strong {
    flex: 0 0 auto;
  }

  .story span {
    color: rgba(255,255,255,.78);
    line-height: 1.35;
  }

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

  .support-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .advice-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
    margin-top: 18px;
  }

  .advice-card {
    border: 1px solid rgba(225, 48, 108, 0.18);
  }

  .advice-card .pill {
    display: inline-block;
    margin-bottom: 12px;
  }

  .source-list {
    display: grid;
    gap: 10px;
    margin-top: 12px;
  }

  .source-list a {
    color: var(--blue);
    font-weight: 900;
  }

  .producers-section {
    margin-top: 22px;
    padding: 4px 0 0;
  }

  .producer-heading {
    display: flex;
    justify-content: space-between;
    gap: 18px;
    align-items: end;
    margin-bottom: 14px;
  }

  .producer-heading h2 {
    margin: 0;
    font-size: clamp(30px, 5vw, 52px);
    line-height: 0.96;
    letter-spacing: 0;
  }

  .producer-heading p {
    max-width: 430px;
    margin: 0;
    color: var(--muted);
    font-weight: 750;
    line-height: 1.45;
  }

  .producer-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
  }

  .producer-card {
    position: relative;
    overflow: hidden;
    min-height: 448px;
    border: 1px solid rgba(225, 48, 108, 0.18);
    border-radius: 26px;
    background: rgba(255, 255, 255, 0.74);
    box-shadow: 0 24px 70px rgba(21, 21, 26, 0.12);
    transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
  }

  .producer-card:hover {
    transform: translateY(-4px);
    border-color: rgba(225, 48, 108, 0.30);
    box-shadow: 0 34px 90px rgba(21, 21, 26, 0.18);
  }

  .producer-photo-wrap {
    height: 292px;
    overflow: hidden;
    background: #f4f4f5;
  }

  .producer-photo {
    display: block;
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 420ms ease;
  }

  .producer-card:hover .producer-photo {
    transform: scale(1.04);
  }

  .photo-phawich {
    object-position: center 28%;
  }

  .photo-watcharawee {
    object-position: 72% 48%;
  }

  .photo-paphawin {
    object-position: 53% 34%;
  }

  .producer-body {
    padding: 16px;
  }

  .producer-body h3 {
    margin: 0;
    font-size: clamp(20px, 2.5vw, 28px);
    line-height: 1.05;
  }

  .producer-body p {
    margin: 8px 0 14px;
    color: var(--muted);
    font-weight: 850;
  }

  .contact-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .contact-link, .contact-slot {
    min-height: 40px;
    border-radius: 999px;
    padding: 10px 12px;
    font-size: 13px;
    font-weight: 1000;
    text-decoration: none;
    line-height: 1;
  }

  .contact-link {
    color: white;
    background: linear-gradient(135deg, var(--violet), var(--hot), var(--sun));
    box-shadow: 0 14px 28px rgba(225, 48, 108, 0.20);
  }

  .contact-slot {
    display: inline-flex;
    align-items: center;
    color: var(--muted);
    border: 1px dashed rgba(113, 113, 122, 0.44);
    background: rgba(255, 255, 255, 0.62);
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
    margin: 88px auto 34px;
    padding: 24px;
    text-align: center;
  }

  .result-card .ring {
    margin: 28px auto 24px;
  }

  .band {
    width: fit-content;
    margin: 24px auto 26px;
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

  .research-map {
    display: grid;
    grid-template-columns: minmax(0, 1.05fr) minmax(280px, 0.95fr);
    gap: 18px;
    align-items: center;
    margin-top: 18px;
    padding: 18px;
    border: 1px solid rgba(225, 48, 108, 0.18);
    border-radius: 26px;
    background:
      linear-gradient(135deg, rgba(255, 255, 255, 0.82), rgba(255, 255, 255, 0.54)),
      radial-gradient(circle at 88% 12%, rgba(64, 93, 230, 0.16), transparent 30%);
    overflow: hidden;
  }

  .pipeline-stack {
    display: grid;
    gap: 12px;
  }

  .pipeline-step {
    display: grid;
    grid-template-columns: 42px 1fr;
    gap: 12px;
    align-items: center;
    min-height: 74px;
    padding: 13px;
    border: 1px solid var(--line);
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.74);
    box-shadow: 0 18px 34px rgba(21, 21, 26, 0.09);
    transform: perspective(900px) rotateX(3deg) rotateY(-5deg);
  }

  .pipeline-step span {
    display: grid;
    place-items: center;
    width: 42px;
    height: 42px;
    border-radius: 14px;
    color: white;
    background: linear-gradient(135deg, var(--violet), var(--hot), var(--sun));
    font-weight: 1000;
    box-shadow: 0 12px 24px rgba(225, 48, 108, 0.24);
  }

  .pipeline-step strong {
    display: block;
    font-size: 16px;
    line-height: 1.15;
  }

  .pipeline-step small {
    display: block;
    margin-top: 4px;
    color: var(--muted);
    font-weight: 750;
    line-height: 1.35;
  }

  .cube-scene {
    min-height: 318px;
    display: grid;
    place-items: center;
    position: relative;
    perspective: 900px;
  }

  .cube-title {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    text-align: center;
    color: var(--muted);
    font-size: 13px;
    font-weight: 1000;
  }

  .cube-legend {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 4px;
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
  }

  .cube-legend span {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 7px 9px;
    color: var(--muted);
    background: rgba(255, 255, 255, 0.88);
    font-size: 11px;
    font-weight: 1000;
    box-shadow: 0 12px 24px rgba(21, 21, 26, 0.08);
  }

  .cube-legend i {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: var(--blue);
  }

  .cube-legend span:last-child i {
    background: var(--hot);
  }

  .cube-box {
    position: relative;
    width: min(280px, 84vw);
    height: 230px;
    transform-style: preserve-3d;
    transform: rotateX(60deg) rotateZ(-38deg);
    animation: cubeDrift 4.5s ease-in-out infinite alternate;
  }

  .cube-face {
    position: absolute;
    border: 1px solid rgba(24, 24, 27, 0.12);
    background: rgba(255, 255, 255, 0.36);
    box-shadow: inset 0 0 32px rgba(64, 93, 230, 0.08);
  }

  .cube-floor {
    inset: 38px 20px 20px;
    border-radius: 20px;
  }

  .cube-back {
    left: 20px;
    top: 18px;
    width: calc(100% - 40px);
    height: 112px;
    transform: rotateX(72deg) translateY(-86px);
    transform-origin: top;
    border-radius: 18px;
  }

  .cube-side {
    right: 20px;
    top: 38px;
    width: 112px;
    height: calc(100% - 58px);
    transform: rotateY(72deg) translateX(86px);
    transform-origin: right;
    border-radius: 18px;
  }

  .decision-sheet {
    position: absolute;
    left: 92px;
    top: 44px;
    width: 106px;
    height: 150px;
    border: 2px solid rgba(225, 48, 108, 0.56);
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(225, 48, 108, 0.18), rgba(252, 175, 69, 0.08));
    transform: rotateY(65deg) rotateZ(6deg);
    box-shadow: 0 18px 38px rgba(225, 48, 108, 0.12);
  }

  .cube-point {
    position: absolute;
    z-index: 3;
    width: 13px;
    height: 13px;
    border-radius: 50%;
    background: var(--blue);
    box-shadow: 0 0 0 5px rgba(64, 93, 230, 0.14), 0 16px 18px rgba(21, 21, 26, 0.18);
  }

  .cube-point.hot {
    background: var(--hot);
    box-shadow: 0 0 0 5px rgba(225, 48, 108, 0.14), 0 16px 18px rgba(21, 21, 26, 0.18);
  }

  .cube-point.synthetic {
    background: #22c55e;
    box-shadow: 0 0 0 5px rgba(34, 197, 94, 0.16), 0 16px 18px rgba(21, 21, 26, 0.18);
  }

  .algo-card {
    min-height: 316px;
    overflow: hidden;
  }

  .algo-visual {
    height: 154px;
    border: 1px solid var(--line);
    border-radius: 18px;
    background:
      linear-gradient(90deg, rgba(24,24,27,.06) 1px, transparent 1px) 0 0 / 28px 28px,
      linear-gradient(0deg, rgba(24,24,27,.06) 1px, transparent 1px) 0 0 / 28px 28px,
      rgba(255, 255, 255, 0.76);
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
  }

  .algo-visual::before {
    content: "";
    position: absolute;
    inset: 14px;
    border-left: 2px solid rgba(24,24,27,.16);
    border-bottom: 2px solid rgba(24,24,27,.16);
    pointer-events: none;
  }

  .logistic-svg {
    position: absolute;
    inset: 8px;
    width: calc(100% - 16px);
    height: calc(100% - 16px);
    z-index: 1;
  }

  .logistic-svg text {
    font-family: inherit;
    font-weight: 900;
    fill: var(--muted);
  }

  .viz-chip {
    position: absolute;
    z-index: 2;
    border-radius: 999px;
    padding: 5px 8px;
    color: var(--ink);
    background: rgba(255, 255, 255, 0.90);
    border: 1px solid var(--line);
    font-size: 11px;
    font-weight: 1000;
    box-shadow: 0 8px 18px rgba(21, 21, 26, 0.08);
  }

  .viz-chip.hot {
    color: white;
    border: 0;
    background: linear-gradient(135deg, var(--hot), var(--sun));
  }

  .viz-chip.cool {
    color: white;
    border: 0;
    background: linear-gradient(135deg, var(--blue), var(--violet));
  }

  .viz-arrow {
    position: absolute;
    z-index: 1;
    height: 3px;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--violet), var(--hot), var(--sun));
    transform-origin: left center;
  }

  .viz-arrow::after {
    content: "";
    position: absolute;
    right: -2px;
    top: 50%;
    width: 10px;
    height: 10px;
    border-top: 3px solid var(--sun);
    border-right: 3px solid var(--sun);
    transform: translateY(-50%) rotate(45deg);
  }

  .flow-step {
    position: absolute;
    display: grid;
    place-items: center;
    width: 42px;
    height: 42px;
    border-radius: 13px;
    color: white;
    background: linear-gradient(135deg, var(--violet), var(--hot));
    font-size: 12px;
    font-weight: 1000;
    box-shadow: 0 12px 28px rgba(225, 48, 108, 0.20);
    animation: nodePop 2.2s ease-in-out infinite;
  }

  .algo-points {
    display: grid;
    gap: 7px;
    margin-top: 12px;
  }

  .algo-point {
    display: grid;
    grid-template-columns: 22px 1fr;
    gap: 8px;
    align-items: start;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.35;
  }

  .algo-point strong {
    display: grid;
    place-items: center;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    color: white;
    background: linear-gradient(135deg, var(--hot), var(--sun));
    font-size: 11px;
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

  .dot.ghost {
    width: 10px;
    height: 10px;
    background: #22c55e;
    box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.16);
    opacity: 0.92;
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

  .line-viz.vertical {
    width: 4px;
    height: 112px;
    left: 50%;
    right: auto;
    top: 18px;
    transform: rotate(12deg);
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
    margin-top: 24px;
  }

  .bar-row {
    display: grid;
    grid-template-columns: minmax(210px, 0.42fr) minmax(150px, 1fr) 52px;
    gap: 12px;
    align-items: center;
    color: var(--muted);
    font-size: 13px;
    font-weight: 800;
  }

  .bar-row span {
    min-width: 0;
    overflow-wrap: anywhere;
    line-height: 1.15;
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

  .reason-box p {
    margin: 0 0 16px;
  }

  .reason-box p:last-child {
    margin-bottom: 0;
  }

  .table-wrapper {
    margin-top: 12px;
    width: 100%;
    max-width: 100%;
    overflow: hidden;
    border: 1px solid var(--line);
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.72);
  }

  .metric-table {
    width: 100%;
    table-layout: fixed;
    border-collapse: collapse;
    color: var(--muted);
    font-size: clamp(10px, 1.7vw, 13px);
    text-align: left;
  }

  .metric-table th, .metric-table td {
    padding: 12px 10px;
    border-bottom: 1px solid var(--line);
    vertical-align: middle;
  }

  .metric-table tr:last-child td {
    border-bottom: none;
  }

  .metric-table th {
    background: rgba(255, 255, 255, 0.78);
    font-weight: 900;
    color: var(--ink);
    padding: 14px 10px;
    white-space: nowrap;
  }

  .metric-table abbr {
    text-decoration: none;
    cursor: help;
  }

  .metric-table td:first-child {
    font-weight: 600;
    color: var(--ink);
    overflow-wrap: anywhere;
    word-break: break-word;
  }

  .metric-table th:not(:first-child),
  .metric-table td:not(:first-child) {
    text-align: center;
  }

  .metric-table col:first-child {
    width: 42%;
  }

  .metric-table col:not(:first-child) {
    width: 14.5%;
  }

  @media (max-width: 640px) {
    .metric-table th, .metric-table td {
      padding: 10px 6px;
    }

    .metric-table {
      font-size: 10px;
    }
  }

  @keyframes pageIn {
    from { opacity: 0; transform: translateY(10px); }
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

  @keyframes cubeDrift {
    from { transform: rotateX(60deg) rotateZ(-38deg) translate3d(-2px, 0, 0); }
    to { transform: rotateX(57deg) rotateZ(-34deg) translate3d(8px, -4px, 0); }
  }

  @media (max-width: 920px) {
    .hero, .predict-layout, .section-grid, .algorithm-lab, .advice-grid, .research-map, .producer-grid { grid-template-columns: 1fr; }
    .producer-heading { display: block; }
    .producer-heading p { margin-top: 10px; }
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

CHAT_WIDGET_STYLE = """
<style>
  .beast-fab {
    position: fixed; bottom: 20px; right: 20px; z-index: 1000;
    width: 72px; height: 72px; border-radius: 22px;
    background: linear-gradient(135deg, var(--violet), var(--hot), var(--sun));
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 3px; padding: 6px 6px 4px;
    box-shadow: 0 16px 36px rgba(225,48,108,.34);
    border: 0; cursor: pointer; color: white;
    transition: transform 180ms ease, box-shadow 180ms ease;
  }
  .beast-fab:hover { transform: translateY(-2px) scale(1.04); }
  .beast-fab img { width: 46px; height: 46px; object-fit: contain; border-radius: 8px; }
  .beast-fab-label { font-size: 9px; font-weight: 900; letter-spacing: .04em; font-family: inherit; }
  .beast-fab-badge {
    position: absolute; top: -4px; right: -4px;
    width: 16px; height: 16px; border-radius: 50%;
    background: #22c55e; border: 2px solid white; display: none;
  }
  .beast-chat {
    position: fixed; bottom: 104px; right: 16px; z-index: 1000;
    width: 310px; max-height: calc(100vh - 140px); border-radius: 26px;
    background: rgba(255,255,255,0.97);
    border: 1px solid var(--line);
    box-shadow: 0 28px 72px rgba(21,21,26,.20);
    overflow: hidden; font-size: 13px;
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    display: flex; flex-direction: column;
  }
  .beast-head {
    padding: 12px 14px;
    background: linear-gradient(135deg, var(--violet), var(--hot), var(--sun));
    color: white; display: flex; align-items: center; gap: 9px;
  }
  .beast-head-img { width: 36px; height: 36px; object-fit: contain; border-radius: 10px; background: rgba(255,255,255,.18); flex-shrink: 0; }
  .beast-head-info strong { display: block; font-size: 14px; }
  .beast-head-info small { opacity: .82; font-size: 11px; }
  .beast-lang-toggle { margin-left: auto; display: flex; gap: 4px; background: rgba(255,255,255,.20); border-radius: 999px; padding: 3px; }
  .beast-lang-btn { padding: 3px 8px; border-radius: 999px; font-size: 10px; font-weight: 900; color: rgba(255,255,255,.75); border: 0; background: transparent; cursor: pointer; font-family: inherit; }
  .beast-lang-btn.active { background: rgba(255,255,255,.30); color: white; }
  .beast-ctx { margin: 8px 12px 0; border-radius: 14px; padding: 8px 12px; background: rgba(225,48,108,.08); border: 1px solid rgba(225,48,108,.22); font-size: 11px; color: var(--hot); font-weight: 800; line-height: 1.4; }
  .beast-msgs { padding: 12px; display: flex; flex-direction: column; gap: 8px; flex: 1; overflow-y: auto; min-height: 80px; }
  .beast-msg { line-height: 1.4; max-width: 88%; }
  .beast-bot { background: #f4f4f5; border-radius: 16px 16px 16px 4px; padding: 9px 12px; }
  .beast-user { background: linear-gradient(135deg, var(--violet), var(--hot)); color: white; border-radius: 16px 16px 4px 16px; padding: 9px 12px; align-self: flex-end; }
  .beast-src { display: block; margin-top: 4px; font-size: 10px; color: var(--muted); font-weight: 800; }
  .beast-chips { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 5px; padding: 4px 12px 10px; }
  .beast-chip { flex: 0 0 auto; width: auto; display: inline-flex; align-items: center; border: 1px solid rgba(225,48,108,.28); border-radius: 999px; padding: 5px 12px; font-size: 11px; font-weight: 900; color: var(--hot); background: rgba(225,48,108,.06); cursor: pointer; font-family: inherit; white-space: nowrap; transition: padding 150ms ease, font-size 150ms ease, opacity 150ms ease; }
  .beast-chips.compact { gap: 4px; padding: 2px 12px 6px; }
  .beast-chips.compact .beast-chip { padding: 3px 9px; font-size: 10px; opacity: 0.75; }
  .beast-form { display: flex; gap: 8px; padding: 10px 12px; border-top: 1px solid var(--line); }
  .beast-input { flex: 1; border: 1px solid var(--line); border-radius: 999px; padding: 8px 13px; font-size: 12px; outline: none; background: #fafafa; font-family: inherit; }
  .beast-input:focus { box-shadow: 0 0 0 3px rgba(225,48,108,.16); background: #fff; }
  .beast-send { width: 34px; height: 34px; border-radius: 50%; flex-shrink: 0; background: linear-gradient(135deg, var(--hot), var(--sun)); border: 0; color: white; font-size: 13px; cursor: pointer; display: grid; place-items: center; }
</style>
"""


class ObesityInput(BaseModel):
    age: int = Field(..., ge=5, le=100)
    sex: Literal["M", "F"] = Field(..., examples=["M", "F"])
    height_cm: float = Field(..., ge=80, le=230)
    weight_kg: float = Field(..., ge=20, le=250)
    physical_activity_hours_per_week: float = Field(..., ge=0, le=40)
    screen_time_hours_per_day: float = Field(..., ge=0, le=24)
    sleep_hours: float = Field(..., ge=0, le=16)
    fast_food_meals_per_week: int = Field(..., ge=0, le=30)
    sugary_drinks_per_day: float = Field(..., ge=0, le=20)
    family_history_obesity: int = Field(..., ge=0, le=1)


class ChatRequest(BaseModel):
    message: str
    lang: str = "auto"
    context: Optional[dict] = None


def page_shell(title: str, body: str, chat_context: dict | None = None) -> str:
    risk_tier = chat_context.get("risk_tier", "") if chat_context else ""
    probability = str(chat_context.get("probability", "")) if chat_context else ""
    widget = chat_widget_html(risk_tier, probability)
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
          <div class="brand"><img class="brand-logo" src="/static/obeast-logo.svg" alt="O-Beast logo">O-Beast</div>
          <div class="nav-links">
            <a href="/">Home</a>
            <a href="/predictor">Predictor</a>
            <a href="/advice">Advice</a>
            <a href="/methods">Methods</a>
          </div>
        </nav>
        {body}
      </main>
      {widget}
    </body>
    </html>
    """


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


def methods_html() -> str:
    methods = "".join(
        f'<div class="method"><strong>{name}</strong><span>{body}</span></div>'
        for name, body in METHODS
    )
    return f'<div class="method-grid">{methods}</div>'


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
            ["Tree 1 asks about BMI", "Tree 2 asks about habits", "Final answer is the vote"],
        ),
        (
            "Naive Bayes",
            "Each answer is a clue, then the clues are multiplied into a probability.",
            """
            <div class="algo-visual">
              <span class="flow-step" style="left:14%; top:24%">BMI</span>
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


def chat_widget_html(risk_tier: str = "", probability: str = "") -> str:
    tier_attr = f'data-risk-tier="{escape(risk_tier, quote=True)}"'
    prob_attr = f'data-probability="{escape(probability, quote=True)}"'
    badge_style = 'style="display:block"' if risk_tier else ""
    return f"""
{CHAT_WIDGET_STYLE}
<button id="beast-fab" class="beast-fab" aria-label="Open Beast 1.0 chat" {tier_attr} {prob_attr}>
  <img src="/static/beast1-logo.png" alt="Beast 1.0 logo">
  <span class="beast-fab-label">Beast 1.0</span>
  <span class="beast-fab-badge" {badge_style}></span>
</button>
<div id="beast-chat" class="beast-chat" hidden>
  <div class="beast-head">
    <img class="beast-head-img" src="/static/beast1-logo.png" alt="Beast 1.0">
    <div class="beast-head-info"><strong>Beast 1.0</strong><small>Obesity Q&amp;A &middot; Online</small></div>
    <div class="beast-lang-toggle">
      <button class="beast-lang-btn active" data-lang="en">EN</button>
      <button class="beast-lang-btn" data-lang="th">TH</button>
    </div>
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
<script>
(function(){{
  var fab=document.getElementById('beast-fab');
  var chat=document.getElementById('beast-chat');
  var msgs=document.getElementById('beast-msgs');
  var form=document.getElementById('beast-form');
  var inp=document.getElementById('beast-input');
  var ctx=document.getElementById('beast-ctx');
  var lang='auto';
  var tier=fab.dataset.riskTier||'';
  var prob=fab.dataset.probability||'';
  if(tier){{ctx.hidden=false;ctx.textContent='📊 Your result: '+tier+' ('+Math.round(parseFloat(prob)*100)+'%) — I’ll tailor my answers to your score.';}}
  fab.addEventListener('click',function(){{chat.hidden=!chat.hidden;if(!chat.hidden)inp.focus();}});
  document.querySelectorAll('.beast-lang-btn').forEach(function(b){{
    b.addEventListener('click',function(){{
      document.querySelectorAll('.beast-lang-btn').forEach(function(x){{x.classList.remove('active');}});
      b.classList.add('active');lang=b.dataset.lang;updateChips(lang);
      inp.placeholder=lang==='th'?'พิมพ์คำถาม…':'Ask Beast 1.0…';
    }});
  }});
  document.querySelectorAll('.beast-chip').forEach(function(c){{c.addEventListener('click',function(){{send(c.dataset.query);}});}});
  form.addEventListener('submit',function(){{var t=inp.value.trim();if(!t)return;inp.value='';send(t);}});
  function send(text){{
    bubble(text,'user');
    document.querySelector('.beast-chips').classList.add('compact');
    var body={{message:text,lang:lang}};
    if(tier)body.context={{risk_tier:tier,probability:parseFloat(prob)}};
    fetch('/chat',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}})
      .then(function(r){{return r.json();}})
      .then(function(d){{if(lang==='auto'&&d.detected_lang)lang=d.detected_lang;bubble(d.answer,'bot',d.source);}})
      .catch(function(){{bubble('Sorry, something went wrong.','bot');}});
  }}
  function bubble(text,role,src){{
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
    <section class="hero">
      <div class="hero-copy">
        <div class="kicker">The SK Research Project</div>
        <h1>O-Beast</h1>
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
        <img class="beast-mark" src="/static/obeast-logo.svg" alt="O-Beast aggressive beast mascot logo">
        <h2 style="text-align:center">Risk feed</h2>
        <div class="mini-stat">
          <div><strong>10</strong><span>inputs</span></div>
          <div><strong>7</strong><span>methods</span></div>
          <div><strong>%</strong><span>output</span></div>
        </div>
        <div class="story"><div class="story-icon">A</div><div><strong>Activity</strong><span>exercise and screen habits</span></div></div>
        <div class="story"><div class="story-icon">F</div><div><strong>Food</strong><span>fast food and sugary drinks</span></div></div>
        <div class="story"><div class="story-icon">P</div><div><strong>Pattern</strong><span>answers become an easy result</span></div></div>
      </div>
    </section>

    <section class="card" style="margin-top:18px">
      <div class="kicker">3D learning visual</div>
      <h1 style="font-size:44px">How O-Beast Finds Patterns</h1>
      <p class="lead">O-Beast turns answers into dots, lets several algorithms look for patterns, and changes the final pattern into an easy probability result.</p>
      {research_visual}
      {algorithm_cards}
    </section>

    <section class="section-grid support-grid">
      <div class="card">
        <h2>Research Goal</h2>
        <p>Build a school research tool that learns from student survey answers and estimates obesity-risk probability responsibly.</p>
      </div>
      <div class="card">
        <h2>Wellness Advice</h2>
        <p>After prediction, O-Beast can translate answers into clear habit suggestions using WHO, CDC, and Thai dietary guidance.</p>
        <a class="button secondary" href="/advice">See advice model</a>
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
            "physical_activity_hours_per_week": 3,
            "screen_time_hours_per_day": 5,
            "sleep_hours": 7,
            "fast_food_meals_per_week": 2,
            "sugary_drinks_per_day": 1,
            "family_history_obesity": 0,
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
      <div class="kicker">Model check</div>
      <h1>How O-Beast Checks Results</h1>
      <p class="lead">
        O-Beast compares several learning methods, checks how well each one performs, and shows
        which method produced the current result. The scores are included to make the prediction
        easier to understand, not to claim medical certainty.
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
      <div class="card"><h2>Risk tiers</h2><p>Output now uses five probability tiers: very low, low, moderate, high, and very high.</p></div>
      <div class="card"><h2>Survey data support</h2><p>O-Beast can prepare answers from more than one survey format before later training.</p></div>
    </section>
    """
    return page_shell("Methods - SK Obesity ML", body)


@app.get("/predictor", response_class=HTMLResponse)
def predictor() -> str:
    body = f"""
    <section class="predict-layout">
      <aside class="card profile">
        <div class="pill">Prediction screen</div>
        <img class="beast-mark" src="/static/obeast-logo.svg" alt="O-Beast aggressive beast mascot logo">
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


@app.post("/chat")
def chat_endpoint(payload: ChatRequest) -> dict:
    from obesity_ml.chatbot import chat
    return chat(payload.message, payload.lang, payload.context)


@app.post("/advice", response_class=HTMLResponse)
def advice_from_form(
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
    input_data = {
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
    return page_shell("Advice - O-Beast", body)


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
    input_data = {
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
    try:
        validate_form_input(input_data)
        result = predict_probability(input_data)
    except ValueError as exc:
        return HTMLResponse(prediction_error_html(str(exc)), status_code=400)
    percent = round(result["obesity_probability"] * 100, 1)
    reason = best_model_reason(result)
    metric_bars = metric_bars_html(result.get("metrics", {}))
    metric_table = metric_table_html(result.get("metrics", {}))
    dataset_warning_html = f"<p>{result['dataset_warning']}</p>" if result.get("dataset_warning") else ""
    body = f"""
    <section class="card result-card">
      <div class="pill">Prediction result</div>
      <div class="ring" style="--percent:{percent}%"><div class="prob">{percent}%</div></div>
      <h1 style="font-size:38px">Estimated Probability</h1>
      <div class="band">{result["risk_tier_label"]}</div>
      <p class="note">Tier range: {result["risk_tier_range"]}. Coarse band: {result["risk_band"].title()}.</p>
      <div class="reason-box">
        <h2>Why this algorithm is best</h2>
        <p>Winning model: <strong>{result["base_model_name"]}</strong></p>
        <p>{reason}</p>
        <p>SMOTENC used during training: <strong>{"Yes" if result["used_smote"] else "No"}</strong></p>
      </div>
      <div class="reason-box" style="margin-top:14px">
        <h2>Model tournament scores</h2>
        <p>These scores show how the learning methods compared during testing. They help explain the prototype result and should not be read as medical proof.</p>
        {metric_bars}
        {metric_table}
      </div>
      {dataset_warning_html}
      <p>{result["disclaimer"]}</p>
      <div class="actions" style="justify-content:center">
        <a class="button" href="/predictor">Try another input</a>
        <form class="action-form" action="/advice" method="post">
          {hidden_inputs_html(input_data)}
          <button type="submit">Get wellness advice</button>
        </form>
        <a class="button secondary" href="/methods">See methods</a>
      </div>
    </section>
    """
    return page_shell(
        "Result - SK Obesity ML",
        body,
        chat_context={
            "risk_tier": result["risk_tier_label"],
            "probability": result["obesity_probability"],
        },
    )
