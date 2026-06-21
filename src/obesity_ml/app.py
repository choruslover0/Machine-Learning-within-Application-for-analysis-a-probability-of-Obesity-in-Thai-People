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
    ("Logistic Regression", "Baseline medical-risk model; easy to explain."),
    ("Support Vector Machine", "Finds a strong boundary between risk groups."),
    ("Random Forest", "Many decision trees vote together; robust for survey data."),
    ("Naive Bayes", "A simple probability baseline: each answer gives evidence, then the model combines the evidence."),
    ("Neural Network", "Learns nonlinear patterns across lifestyle features."),
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

  .preparation-block {
    margin: 26px 0;
    padding: 20px 0 24px;
    border-top: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
  }

  .preparation-heading, .algorithm-heading {
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: 18px;
    margin-bottom: 16px;
  }

  .preparation-heading h2, .algorithm-heading h2 { margin: 0; }

  .preparation-heading p, .algorithm-heading p {
    max-width: 520px;
    margin: 0;
    color: var(--muted);
    line-height: 1.5;
  }

  .preparation-flow {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 52px minmax(0, 1fr) 52px minmax(0, 1fr);
    gap: 10px;
    align-items: center;
  }

  .preparation-step {
    min-height: 124px;
    display: grid;
    align-content: center;
    gap: 7px;
    padding: 16px;
    border: 1px solid var(--line);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.72);
  }

  .preparation-step strong { font-size: 18px; }

  .preparation-step span {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.4;
  }

  .preparation-icon {
    width: 38px;
    height: 38px;
    display: grid;
    place-items: center;
    border-radius: 12px;
    color: white;
    background: var(--forest, var(--violet));
    font-weight: 1000;
  }

  .preparation-arrow {
    color: var(--green, var(--violet));
    font-size: 25px;
    font-weight: 1000;
    text-align: center;
  }

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
    height: 2px;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--violet), var(--hot), var(--sun));
    transform-origin: left center;
    opacity: 0.58;
    box-shadow: none;
  }

  .viz-arrow::after {
    content: "";
    position: absolute;
    right: -1px;
    top: 50%;
    width: 7px;
    height: 7px;
    border-top: 2px solid var(--sun);
    border-right: 2px solid var(--sun);
    transform: translateY(-50%) rotate(45deg);
    display: none;
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

  @property --percent {
    syntax: '<percentage>';
    inherits: false;
    initial-value: 0%;
  }

  .ring {
    transition: --percent 1100ms cubic-bezier(0.22, 1, 0.36, 1);
  }

  @keyframes bandIn {
    from { opacity: 0; transform: scale(0.72) translateY(6px); }
    to   { opacity: 1; transform: scale(1)    translateY(0);   }
  }

  .result-card .band {
    animation: bandIn 480ms cubic-bezier(0.34, 1.56, 0.64, 1) 860ms both;
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
    .preparation-heading, .algorithm-heading { display: grid; }
    .preparation-flow { grid-template-columns: 1fr; }
    .preparation-arrow { transform: rotate(90deg); }
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


CYBERPUNK_OVERRIDE = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,200..800;1,200..800&family=DM+Sans:ital,opsz,wght@0,9..40,100..700;1,9..40,100..700&display=swap" rel="stylesheet">
<style>
  /* ── REFINED SOFT MODERN THEME ─────────────────────────────── */
  :root {
    --ink:     #1c1c2e;
    --muted:   #6b6880;
    --line:    rgba(0,0,0,0.08);
    --surface: rgba(255,255,255,0.97);
    --hot:     #db2777;
    --sun:     #f59e0b;
    --gold:    #f59e0b;
    --violet:  #7c3aed;
    --blue:    #3b82f6;
    --grad-primary: linear-gradient(135deg, #7c3aed 0%, #db2777 100%);
  }

  /* ── BACKGROUND: brand-tinted neutral, no cream ─────────────── */
  body {
    color: var(--ink);
    font-family: 'DM Sans', ui-sans-serif, system-ui, sans-serif;
    background: oklch(97.5% 0.005 280);
    min-height: 100vh;
  }

  body::before {
    display: block !important;
    content: "";
    position: fixed;
    inset: 0;
    z-index: -1;
    background:
      radial-gradient(ellipse 80% 60% at 8% 0%,    oklch(82% 0.10 300 / 0.18), transparent 55%),
      radial-gradient(ellipse 55% 45% at 95% 8%,   oklch(76% 0.13 350 / 0.14), transparent 45%),
      radial-gradient(ellipse 45% 30% at 55% 100%, oklch(88% 0.07 255 / 0.10), transparent 50%),
      oklch(97.5% 0.005 280);
    pointer-events: none;
  }

  h1, h2, h3, .brand { font-family: 'Plus Jakarta Sans', ui-sans-serif, sans-serif; }

  h1 { color: #1c1c2e; letter-spacing: -0.03em; }
  h2 { color: #1c1c2e; letter-spacing: -0.01em; }

  /* ── NAV ───────────────────────────────────────────────────── */
  .nav {
    background: oklch(100% 0 0 / 0.86);
    border-color: rgba(0,0,0,0.06);
    box-shadow: 0 1px 0 rgba(0,0,0,0.04), 0 4px 24px rgba(0,0,0,0.05);
    backdrop-filter: blur(24px) saturate(180%);
  }

  .brand { color: #1c1c2e; letter-spacing: -0.02em; }
  .brand-logo { border-radius: 12px; box-shadow: 0 4px 14px oklch(52% 0.22 300 / 0.22); }

  .pill, .nav-links a {
    background: rgba(255,255,255,0.90);
    border-color: rgba(0,0,0,0.08);
    color: var(--muted);
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    letter-spacing: 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }

  .nav-links a:hover, .pill:hover {
    background: #fff;
    color: var(--ink);
    box-shadow: 0 4px 16px rgba(0,0,0,0.09);
    transform: translateY(-2px);
  }

  /* ── CARDS ─────────────────────────────────────────────────── */
  .hero, .card, form {
    background: oklch(100% 0 0 / 0.97);
    border-color: rgba(0,0,0,0.06);
    box-shadow: 0 1px 0 rgba(255,255,255,0.8), 0 4px 32px rgba(0,0,0,0.06), 0 12px 48px rgba(0,0,0,0.04);
    backdrop-filter: none;
    border-radius: 28px;
  }

  .card:hover {
    border-color: oklch(52% 0.22 300 / 0.16);
    box-shadow: 0 1px 0 rgba(255,255,255,0.9), 0 8px 48px rgba(0,0,0,0.09), 0 0 0 1px oklch(52% 0.22 300 / 0.06);
    transform: translateY(-3px);
  }

  /* Kicker: solid violet tint, not gradient fill */
  .kicker {
    background: oklch(94% 0.04 280);
    border: 1px solid oklch(87% 0.07 280);
    color: #4c1d95;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.04em;
  }

  .lead { color: var(--muted); }
  .card p, .note { color: var(--muted); }

  /* ── BUTTONS ───────────────────────────────────────────────── */
  .button {
    background: var(--grad-primary);
    border: none;
    color: #fff;
    box-shadow: 0 4px 20px oklch(52% 0.22 300 / 0.36);
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    letter-spacing: 0.01em;
    border-radius: 16px;
    transition: box-shadow 200ms ease, filter 200ms ease, transform 160ms ease;
  }

  .button:hover { box-shadow: 0 6px 32px oklch(52% 0.22 300 / 0.52); filter: brightness(1.06); }

  .button.secondary {
    background: rgba(255,255,255,0.95);
    border: 1px solid rgba(0,0,0,0.10);
    color: #1c1c2e;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
  .button.secondary:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.10); filter: none; }

  button {
    background: var(--grad-primary);
    border: none;
    color: #fff;
    box-shadow: 0 4px 20px oklch(52% 0.22 300 / 0.33);
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    letter-spacing: 0.01em;
    border-radius: 16px;
  }

  button:hover { box-shadow: 0 6px 28px oklch(52% 0.22 300 / 0.48); filter: brightness(1.06); }

  .action-form button {
    background: var(--grad-primary);
    border: none;
    color: #fff;
    box-shadow: 0 4px 18px oklch(52% 0.22 300 / 0.30);
    border-radius: 16px;
  }

  /* ── FORM ──────────────────────────────────────────────────── */
  label {
    background: oklch(98% 0.005 280);
    border-color: rgba(0,0,0,0.07);
    color: #1c1c2e;
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 600;
    border-radius: 18px;
  }

  input, select {
    background: oklch(96.5% 0.005 280);
    color: #1c1c2e;
    border: 1px solid rgba(0,0,0,0.08);
    font-family: 'DM Sans', sans-serif;
    border-radius: 12px;
  }

  input::placeholder { color: oklch(68% 0.02 280); }

  input:focus, select:focus {
    background: #fff;
    border-color: oklch(52% 0.22 300 / 0.40);
    box-shadow: 0 0 0 3px oklch(52% 0.22 300 / 0.15), 0 2px 8px rgba(0,0,0,0.05);
  }

  /* ── SVG RING ──────────────────────────────────────────────── */
  .ring-wrap {
    position: relative;
    width: 220px;
    height: 220px;
    margin: 28px auto 24px;
  }

  .ring-svg {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    overflow: visible;
  }

  .ring-bg-circle {
    fill: none;
    stroke: oklch(92% 0.03 280);
    stroke-width: 18;
  }

  .ring-arc {
    fill: none;
    stroke: url(#arcGradient);
    stroke-width: 18;
    stroke-linecap: round;
    filter: drop-shadow(0 4px 16px oklch(52% 0.22 300 / 0.45));
  }

  .ring-inner {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
  }

  .prob {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 46px;
    font-weight: 800;
    color: #1c1c2e;
    text-shadow: none;
    letter-spacing: -0.04em;
  }

  /* ── TIER BAND ─────────────────────────────────────────────── */
  .band {
    background: var(--grad-primary);
    border: none;
    color: #fff;
    box-shadow: 0 4px 20px oklch(52% 0.22 300 / 0.36);
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    letter-spacing: 0.02em;
    font-size: 14px;
    border-radius: 999px;
  }

  /* ── DATA VISUALS ───────────────────────────────────────────── */
  .bar-track { background: oklch(93% 0.02 280); border-radius: 999px; }
  .bar-fill  { background: var(--grad-primary); box-shadow: none; }
  .bar-row   { color: var(--muted); }

  .reason-box {
    background: oklch(98% 0.008 280);
    border-color: oklch(52% 0.22 300 / 0.12);
    border-radius: 22px;
  }

  .method {
    background: rgba(255,255,255,0.92);
    border-color: rgba(0,0,0,0.07);
    border-radius: 18px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
  }
  .method:hover { background: #fff; border-color: oklch(52% 0.22 300 / 0.20); box-shadow: 0 4px 18px rgba(0,0,0,0.08); }
  .method strong { color: var(--violet); font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 13px; }

  .table-wrapper { background: rgba(255,255,255,0.96); border-color: rgba(0,0,0,0.07); border-radius: 20px; }
  .metric-table th { background: oklch(97% 0.01 280); color: var(--violet); font-family: 'Plus Jakarta Sans', sans-serif; font-size: 11px; font-weight: 700; letter-spacing: 0.04em; }
  .metric-table th, .metric-table td { border-bottom-color: rgba(0,0,0,0.06); }
  .metric-table td:first-child { color: #1c1c2e; }

  .pipeline-step { background: rgba(255,255,255,0.95); border-color: rgba(0,0,0,0.07); box-shadow: 0 2px 16px rgba(0,0,0,0.06); border-radius: 20px; }
  .pipeline-step strong { color: #1c1c2e; }
  .pipeline-step small  { color: var(--muted); }
  .pipeline-step span {
    background: var(--violet);
    color: white;
    box-shadow: 0 4px 12px oklch(52% 0.22 300 / 0.30);
    border: none;
  }

  .cube-face { background: rgba(255,255,255,0.55); border-color: rgba(0,0,0,0.08); box-shadow: none; }
  .decision-sheet { border-color: rgba(124,58,237,0.40); background: linear-gradient(135deg, rgba(124,58,237,0.08), rgba(219,39,119,0.05)); box-shadow: 0 4px 16px rgba(124,58,237,0.10); }
  .cube-legend span { background: rgba(255,255,255,0.92); border-color: rgba(0,0,0,0.08); color: var(--muted); }
  .cube-legend i { background: var(--blue); }
  .cube-legend span:last-child i { background: var(--hot); }
  .cube-point { background: var(--blue); box-shadow: 0 0 0 5px rgba(59,130,246,0.14), 0 3px 10px rgba(59,130,246,0.30); }
  .cube-point.hot { background: var(--hot); box-shadow: 0 0 0 5px rgba(219,39,119,0.14), 0 3px 10px rgba(219,39,119,0.30); }
  .cube-point.synthetic { background: #10b981; box-shadow: 0 0 0 5px rgba(16,185,129,0.14), 0 3px 10px rgba(16,185,129,0.30); }

  .dot { background: var(--hot); box-shadow: 0 0 0 4px rgba(219,39,119,0.12), 0 2px 8px rgba(219,39,119,0.25); }
  .dot.alt { background: var(--blue); box-shadow: 0 0 0 4px rgba(59,130,246,0.12), 0 2px 8px rgba(59,130,246,0.25); }
  .dot.ghost { background: #10b981; box-shadow: 0 0 0 4px rgba(16,185,129,0.14); }

  .algo-visual {
    background:
      linear-gradient(90deg, oklch(52% 0.22 300 / 0.05) 1px, transparent 1px) 0 0 / 28px 28px,
      linear-gradient(0deg, oklch(52% 0.22 300 / 0.05) 1px, transparent 1px) 0 0 / 28px 28px,
      oklch(98% 0.005 280);
    border-color: rgba(0,0,0,0.07);
    border-radius: 18px;
  }
  .algo-visual::before { border-left-color: oklch(52% 0.22 300 / 0.22); border-bottom-color: oklch(52% 0.22 300 / 0.22); }

  .flow-step { background: var(--violet); color: white; border: none; box-shadow: 0 2px 12px oklch(52% 0.22 300 / 0.25); }
  .node { background: var(--violet); color: white; border: none; }
  .algo-point strong { background: var(--violet); color: white; }

  .viz-chip { background: rgba(255,255,255,0.92); border-color: rgba(0,0,0,0.10); color: #1c1c2e; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
  .viz-chip.hot { background: rgba(219,39,119,0.08); border: 1px solid rgba(219,39,119,0.30); color: var(--hot); }
  .viz-chip.cool { background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.30); color: var(--blue); }
  .viz-arrow { background: linear-gradient(90deg, var(--violet), var(--hot)); opacity: 0.58; box-shadow: none; }
  .viz-arrow::after { border-top-color: var(--hot); border-right-color: var(--hot); }
  .line-viz { background: linear-gradient(90deg, #7c3aed, var(--hot), var(--sun)); }
  .tree-viz {
    background:
      linear-gradient(var(--hot),var(--hot)) 50% 16px/3px 32px no-repeat,
      linear-gradient(135deg,transparent 47%,var(--blue) 48%,var(--blue) 52%,transparent 53%) 50% 42px/110px 34px no-repeat,
      linear-gradient(45deg,transparent 47%,#10b981 48%,#10b981 52%,transparent 53%) 50% 42px/110px 34px no-repeat;
  }

  /* ── PRODUCERS ──────────────────────────────────────────────── */
  .producer-card { background: rgba(255,255,255,0.98); border-color: rgba(0,0,0,0.06); box-shadow: 0 2px 24px rgba(0,0,0,0.06), 0 1px 0 rgba(255,255,255,0.9); border-radius: 26px; }
  .producer-card:hover { border-color: oklch(52% 0.22 300 / 0.18); box-shadow: 0 8px 48px rgba(0,0,0,0.11), 0 0 0 1px oklch(52% 0.22 300 / 0.06); }
  .producer-photo-wrap { background: oklch(95% 0.02 280); }
  .contact-link {
    background: oklch(95% 0.05 280);
    border: 1px solid oklch(87% 0.08 280);
    color: #4c1d95;
    box-shadow: none;
    transition: background 200ms ease, color 200ms ease, box-shadow 200ms ease;
  }
  .contact-link:hover {
    background: var(--violet);
    color: white;
    box-shadow: 0 4px 14px oklch(52% 0.22 300 / 0.32);
    border-color: transparent;
  }
  .contact-slot { background: rgba(255,255,255,0.80); border-color: rgba(0,0,0,0.08); }
  .producer-heading h2 { text-shadow: none; }

  /* ── MISC ───────────────────────────────────────────────────── */
  .advice-card { border-color: oklch(52% 0.22 300 / 0.12); }
  .source-list a { color: var(--violet); }

  /* Hero dark card — deeper, richer */
  .phone-preview {
    background: oklch(13% 0.04 285);
    box-shadow:
      inset 0 0 0 1px rgba(255,255,255,0.08),
      inset 0 1px 0 rgba(255,255,255,0.14),
      0 8px 48px rgba(0,0,0,0.28),
      0 2px 8px rgba(0,0,0,0.18);
  }
  .avatar { background: conic-gradient(from 220deg, var(--gold), var(--blue), var(--hot), #7c3aed, var(--gold)); }
  .avatar-inner { background: oklch(13% 0.04 285); color: #fff; }
  .beast-mark { box-shadow: 0 10px 40px oklch(52% 0.22 300 / 0.32); border-radius: 34px; }
  .mini-stat div { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10); }
  .mini-stat span { color: rgba(255,255,255,0.58); }
  .story { background: rgba(255,255,255,0.07); border-radius: 16px; }
  .story:hover { background: rgba(255,255,255,0.12); }
  .story-icon { background: var(--violet); color: white; border: none; box-shadow: none; }
  .research-map { border-color: rgba(0,0,0,0.07); background: rgba(255,255,255,0.72); border-radius: 26px; }
  .algo-card { border-color: rgba(0,0,0,0.07); }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: oklch(95% 0.01 280); }
  ::-webkit-scrollbar-thumb { background: oklch(52% 0.22 300 / 0.28); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: oklch(52% 0.22 300 / 0.48); }

  /* Ring arc glow */
  @keyframes neonPulse {
    0%,100% { filter: drop-shadow(0 4px 12px oklch(52% 0.22 300 / 0.42)); }
    50%      { filter: drop-shadow(0 6px 22px oklch(52% 0.22 300 / 0.62)); }
  }

  @media (prefers-reduced-motion: reduce) {
    .ring-arc { animation: none !important; filter: drop-shadow(0 4px 12px oklch(52% 0.22 300 / 0.42)); }
  }

  /* ── ENHANCE: contact-link needs display:inline-flex for hover transition ── */
  .contact-link { display: inline-flex; align-items: center; }
</style>
"""

PREMIUM_HEALTH_OVERRIDE = """
<style>
  :root {
    --ink: #14241b;
    --muted: #66746c;
    --line: rgba(21, 60, 41, 0.12);
    --surface: #ffffff;
    --hot: #eaa377;
    --sun: #f0c092;
    --gold: #d9b879;
    --violet: #347849;
    --blue: #4f7f65;
    --forest: #153c29;
    --green: #347849;
    --sage: #e9f0e8;
    --paper: #f7faf6;
    --peach: #eaa377;
    --health-shadow: 0 24px 70px rgba(21, 60, 41, 0.10);
    --health-ease: cubic-bezier(0.32, 0.72, 0, 1);
  }

  body {
    color: var(--ink);
    font-family: "DM Sans", ui-sans-serif, system-ui, sans-serif;
    background: #edf2ed;
  }

  body::before {
    display: block !important;
    content: "";
    position: fixed;
    inset: 0;
    z-index: -1;
    pointer-events: none;
    background:
      linear-gradient(180deg, rgba(255,255,255,.72), rgba(237,242,237,.88)),
      url("/static/premium-health-pattern.png") center top / cover fixed no-repeat;
    opacity: .34;
  }

  main {
    width: min(1240px, calc(100vw - 36px));
    padding: 20px 0 72px;
  }

  h1, h2, h3, .brand {
    color: var(--ink);
    font-family: "Manrope", "DM Sans", ui-sans-serif, sans-serif;
    letter-spacing: 0;
  }

  h1 {
    font-size: clamp(42px, 6.4vw, 82px);
    line-height: .96;
  }

  h2 { line-height: 1.1; }
  p { line-height: 1.65; }

  a:focus-visible, button:focus-visible, input:focus-visible, select:focus-visible {
    outline: 3px solid rgba(52, 120, 73, .38);
    outline-offset: 3px;
  }

  .nav {
    position: sticky;
    top: 14px;
    z-index: 20;
    max-width: 860px;
    margin: 0 auto 22px;
    padding: 8px 9px 8px 12px;
    border-color: rgba(21,60,41,.10);
    border-radius: 999px;
    background: rgba(250,252,249,.88);
    box-shadow: 0 14px 42px rgba(21,60,41,.11);
    backdrop-filter: blur(24px) saturate(150%);
  }

  .brand {
    color: var(--forest);
    font-weight: 800;
    flex-shrink: 0;
    white-space: nowrap;
  }

  .brand-logo {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    box-shadow: none;
  }

  .nav-links { gap: 2px; }

  .nav-links a, .pill {
    min-height: 38px;
    display: inline-flex;
    align-items: center;
    border-color: transparent;
    color: var(--muted);
    background: transparent;
    box-shadow: none;
    font-weight: 700;
    transition: color 420ms var(--health-ease), background 420ms var(--health-ease), transform 420ms var(--health-ease);
  }

  .nav-links a:hover, .pill:hover {
    color: var(--forest);
    background: var(--sage);
    box-shadow: none;
    transform: translateY(-1px);
  }

  .hero, .card, form {
    border-color: rgba(21,60,41,.09);
    border-radius: 26px;
    background: rgba(255,255,255,.94);
    box-shadow: var(--health-shadow);
    backdrop-filter: none;
  }

  .hero {
    padding: 8px;
    gap: 8px;
    background: rgba(255,255,255,.48);
    border-radius: 32px;
  }

  .hero-copy {
    min-height: 540px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    margin: 0;
    padding: clamp(30px, 6vw, 72px);
    border-radius: 25px;
    background: linear-gradient(145deg, #fbfdf9, #e4eee1);
  }

  .hero-copy h1 span { color: var(--green); }

  .lead {
    color: var(--muted);
    font-size: clamp(16px, 2vw, 19px);
    line-height: 1.65;
  }

  .kicker {
    padding: 7px 11px;
    border: 1px solid rgba(52,120,73,.16);
    border-radius: 999px;
    color: var(--green);
    background: rgba(233,240,232,.9);
    font-family: "DM Sans", sans-serif;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .12em;
    text-transform: uppercase;
  }

  .button, button {
    border: 0;
    border-radius: 999px;
    color: #fff;
    background: var(--forest);
    box-shadow: 0 12px 28px rgba(21,60,41,.20);
    font-family: "DM Sans", sans-serif;
    font-weight: 800;
    transition: transform 520ms var(--health-ease), background 520ms var(--health-ease), box-shadow 520ms var(--health-ease);
  }

  .button {
    min-height: 52px;
    display: inline-flex;
    gap: 14px;
    align-items: center;
    justify-content: center;
    padding: 7px 8px 7px 20px;
  }

  .button::after {
    content: "↗";
    width: 36px;
    height: 36px;
    display: grid;
    place-items: center;
    border-radius: 50%;
    color: var(--forest);
    background: #dff3d5;
    transition: transform 520ms var(--health-ease);
  }

  .button:hover, button:hover {
    color: #fff;
    background: #1d5036;
    box-shadow: 0 18px 38px rgba(21,60,41,.26);
    filter: none;
    transform: translateY(-2px);
  }

  .button:hover::after { transform: translate(2px, -2px); }
  .button:active, button:active { transform: scale(.98); }

  .button.secondary {
    padding-right: 20px;
    border: 1px solid rgba(21,60,41,.15);
    color: var(--forest);
    background: rgba(255,255,255,.82);
    box-shadow: none;
  }

  .button.secondary::after { display: none; }
  .button.secondary:hover { color: var(--forest); background: var(--sage); box-shadow: none; }
  .action-form button { color: white; background: var(--forest); box-shadow: 0 12px 28px rgba(21,60,41,.20); }
  .action-form button:hover { background: #1d5036; box-shadow: 0 18px 38px rgba(21,60,41,.26); }

  .premium-health-visual {
    min-height: 540px;
    position: relative;
    overflow: hidden;
    border-radius: 25px;
    background: var(--paper);
  }

  .premium-health-visual img {
    width: 100%;
    height: 100%;
    min-height: 540px;
    display: block;
    object-fit: cover;
  }

  .visual-caption {
    position: absolute;
    right: 18px;
    bottom: 18px;
    left: 18px;
    padding: 16px;
    border: 1px solid rgba(255,255,255,.48);
    border-radius: 18px;
    color: white;
    background: rgba(21,60,41,.86);
    box-shadow: 0 16px 40px rgba(21,60,41,.18);
  }

  .visual-caption strong { display: block; font-family: "Manrope"; font-size: 18px; }
  .visual-caption span { display: block; margin-top: 3px; font-size: 12px; opacity: .78; }

  .trust-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0,1fr));
    gap: 10px;
    margin-top: 14px;
  }

  .trust-item {
    padding: 18px;
    border: 1px solid rgba(21,60,41,.08);
    border-radius: 20px;
    background: rgba(255,255,255,.88);
    box-shadow: 0 10px 30px rgba(21,60,41,.06);
  }

  .trust-item strong { display: block; color: var(--forest); font: 800 25px "Manrope"; }
  .trust-item span { color: var(--muted); font-size: 13px; }

  .section-intro {
    max-width: 720px;
    padding: 72px 4px 22px;
  }

  .section-intro h1 { margin-top: 14px; font-size: clamp(38px, 5vw, 64px); }

  .card {
    padding: clamp(18px, 3vw, 28px);
    transition: transform 520ms var(--health-ease), box-shadow 520ms var(--health-ease), border-color 520ms var(--health-ease);
  }

  .section-grid > * { min-width: 0; }
  .card p { overflow-wrap: anywhere; }

  .card:hover {
    border-color: rgba(52,120,73,.18);
    box-shadow: 0 30px 74px rgba(21,60,41,.13);
    transform: translateY(-3px);
  }

  .support-grid { gap: 18px; }
  .support-grid .card { min-height: 220px; }

  .phone-preview {
    color: var(--ink);
    background: var(--paper);
    box-shadow: inset 0 0 0 1px rgba(21,60,41,.08);
  }

  .mini-stat div, .story {
    color: var(--ink);
    border: 1px solid rgba(21,60,41,.08);
    background: var(--sage);
  }

  .mini-stat span, .story span { color: var(--muted); }
  .story-icon, .flow-step, .node, .algo-point strong, .pipeline-step span { background: var(--forest); }

  .research-map {
    border-color: rgba(21,60,41,.08);
    background: #f4f8f3;
  }

  .pipeline-step, .method, .algo-visual {
    border-color: rgba(21,60,41,.09);
    background: rgba(255,255,255,.88);
    box-shadow: none;
  }

  .pipeline-step { transform: none; }
  .pipeline-step strong, .method strong { color: var(--forest); }
  .pipeline-step span { box-shadow: none; }

  .algorithm-lab { gap: 18px; }
  .algo-card { min-height: 300px; }
  .algo-visual { height: 148px; background-color: #f4f8f3; }
  .algo-point strong { box-shadow: none; }
  .viz-arrow { height: 2px; background: var(--green); opacity: .42; }
  .viz-arrow::after { display: none; }
  .line-viz { background: var(--green); }

  .producer-card {
    border-color: rgba(21,60,41,.08);
    background: rgba(255,255,255,.94);
    box-shadow: var(--health-shadow);
  }

  .producer-card:hover {
    border-color: rgba(52,120,73,.18);
    box-shadow: 0 30px 74px rgba(21,60,41,.14);
  }

  .contact-link {
    color: var(--forest);
    background: var(--sage);
    border-color: transparent;
    box-shadow: none;
  }

  .contact-link:hover { color: white; background: var(--forest); box-shadow: none; }

  .predict-layout {
    grid-template-columns: minmax(250px,.62fr) minmax(0,1.38fr);
    gap: 18px;
  }

  .profile {
    top: 86px;
    padding: 26px;
    background: var(--forest);
    color: white;
  }

  .profile h2 { color: white; }
  .profile .note { color: rgba(255,255,255,.70); }
  .profile .pill { color: #dff3d5; background: rgba(255,255,255,.10); }
  .profile .method-grid { grid-template-columns: 1fr; }
  .profile .method { color: white; border-color: rgba(255,255,255,.10); background: rgba(255,255,255,.07); }
  .profile .method strong { color: #dff3d5; }
  .profile .method span { color: rgba(255,255,255,.66); }

  .profile-guide { display: grid; gap: 9px; margin-top: 24px; }
  .profile-guide div {
    display: grid;
    grid-template-columns: 34px 1fr;
    gap: 11px;
    align-items: center;
    padding: 12px;
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 15px;
    background: rgba(255,255,255,.07);
  }
  .profile-guide strong {
    width: 34px;
    height: 34px;
    display: grid;
    place-items: center;
    border-radius: 50%;
    color: var(--forest);
    background: #dff3d5;
  }
  .profile-guide span { color: rgba(255,255,255,.76); font-size: 13px; line-height: 1.4; }

  .predict-form { padding: clamp(20px,4vw,38px); }
  .form-head { align-items: start; }

  .form-progress {
    height: 7px;
    overflow: hidden;
    border-radius: 999px;
    background: var(--sage);
  }

  .form-progress span {
    width: 25%;
    height: 100%;
    display: block;
    border-radius: inherit;
    background: var(--green);
    transition: transform 620ms var(--health-ease);
    transform-origin: left;
  }

  .step-meta {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    margin: 16px 0 4px;
    color: var(--muted);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .08em;
    text-transform: uppercase;
  }

  .form-step { display: block; padding-top: 20px; }
  .js .form-step { display: none; }
  .js .form-step.active { display: block; animation: stepIn 620ms var(--health-ease) both; }
  .form-step h2 { margin-bottom: 5px; font-size: clamp(25px,4vw,38px); }
  .form-step > p { max-width: 620px; margin-top: 0; color: var(--muted); }

  .field-grid {
    display: grid;
    grid-template-columns: repeat(2,minmax(0,1fr));
    gap: 12px;
    margin-top: 22px;
  }

  label {
    border-color: rgba(21,60,41,.09);
    border-radius: 17px;
    color: var(--forest);
    background: #f5f8f4;
    font-weight: 800;
  }

  input, select {
    border: 1px solid transparent;
    color: var(--ink);
    background: white;
  }

  input:focus, select:focus {
    border-color: rgba(52,120,73,.35);
    box-shadow: 0 0 0 4px rgba(52,120,73,.12);
  }

  .step-actions {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    margin-top: 24px;
  }

  .step-actions button {
    width: auto;
    min-width: 140px;
    margin: 0;
    padding: 0 20px;
  }

  .step-actions .step-back {
    color: var(--forest);
    background: var(--sage);
    box-shadow: none;
  }

  .review-grid {
    display: grid;
    grid-template-columns: repeat(2,minmax(0,1fr));
    gap: 9px;
    margin-top: 20px;
  }

  .review-item {
    padding: 14px;
    border-radius: 15px;
    background: var(--sage);
  }

  .review-item span { display: block; color: var(--muted); font-size: 11px; }
  .review-item strong { display: block; margin-top: 4px; color: var(--forest); }

  .result-card {
    width: min(760px,100%);
    margin: 48px auto 24px;
    padding: clamp(24px,5vw,52px);
  }

  .ring-arc {
    stroke: var(--green);
    filter: drop-shadow(0 5px 13px rgba(52,120,73,.20));
  }

  .ring-bg-circle { stroke: var(--sage); }
  .prob { color: var(--forest); }

  .band {
    color: var(--forest);
    background: #dcefd8;
    border: 1px solid rgba(52,120,73,.13);
    box-shadow: none;
  }

  .result-meaning, .next-actions, .reason-box {
    margin-top: 18px;
    padding: 22px;
    border: 1px solid rgba(21,60,41,.09);
    border-radius: 21px;
    text-align: left;
    background: #f5f8f4;
  }

  .next-actions { background: var(--forest); color: white; }
  .next-actions h2 { color: white; }
  .next-action { display: grid; grid-template-columns: 34px 1fr; gap: 12px; align-items: start; margin-top: 12px; }
  .next-action strong { width: 34px; height: 34px; display: grid; place-items: center; border-radius: 50%; color: var(--forest); background: #dff3d5; }
  .next-action span { color: rgba(255,255,255,.78); line-height: 1.45; }

  .reason-box { background: white; }
  .reason-box summary { cursor: pointer; color: var(--forest); }
  .reason-box[open] summary { margin-bottom: 12px; }
  .table-wrapper { border-color: rgba(21,60,41,.09); background: white; }
  .metric-table th { color: var(--forest); background: var(--sage); }
  .bar-fill { background: var(--green); }

  .evaluation-dashboard {
    display: grid;
    gap: 16px;
    margin-top: 24px;
  }

  .selected-model-banner {
    display: grid;
    grid-template-columns: 54px 1fr;
    gap: 16px;
    align-items: center;
    padding: 22px;
    border: 1px solid rgba(102,173,122,.34);
    border-radius: 21px;
    color: white;
    background: var(--forest);
  }

  .selected-model-icon {
    width: 54px;
    height: 54px;
    display: grid;
    place-items: center;
    border-radius: 50%;
    color: var(--forest);
    background: #dff3d5;
    font-size: 24px;
    font-weight: 900;
  }

  .selected-model-banner span,
  .evaluation-panel-head span {
    display: block;
    color: var(--green);
    font-size: 11px;
    font-weight: 900;
    letter-spacing: .1em;
    text-transform: uppercase;
  }

  .selected-model-banner span { color: #bfe3c3; }
  .selected-model-banner h2 { margin: 4px 0 3px; color: white; font-size: clamp(22px,3vw,32px); }
  .selected-model-banner p { margin: 0; color: rgba(255,255,255,.72); }
  .selected-model-empty { color: var(--forest); background: var(--sage); }
  .selected-model-empty h2 { color: var(--forest); }

  .evaluation-panels {
    display: grid;
    grid-template-columns: minmax(0,1.05fr) minmax(0,.95fr);
    gap: 16px;
  }

  .evaluation-panel {
    min-width: 0;
    padding: 22px;
    border: 1px solid rgba(21,60,41,.1);
    border-radius: 21px;
    background: #f9fbf8;
  }

  .evaluation-panel-head {
    display: flex;
    justify-content: space-between;
    gap: 18px;
    align-items: start;
  }

  .evaluation-panel-head h2 { margin: 5px 0 0; font-size: clamp(20px,2.5vw,28px); }
  .evaluation-panel-head p { max-width: 230px; margin: 0; color: var(--muted); font-size: 12px; line-height: 1.45; }

  .evaluation-legend, .roc-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 13px;
    margin-top: 18px;
    color: var(--muted);
    font-size: 11px;
    font-weight: 800;
  }

  .evaluation-legend span, .roc-legend span { display: inline-flex; gap: 6px; align-items: center; }
  .evaluation-legend i, .roc-legend i {
    width: 10px;
    height: 10px;
    border-radius: 3px;
    background: var(--metric-color, var(--curve-color));
  }

  .eval-models {
    display: grid;
    grid-template-columns: repeat(auto-fit,minmax(150px,1fr));
    gap: 10px;
    margin-top: 18px;
  }

  .eval-model-group {
    min-width: 0;
    padding: 12px;
    border: 1px solid rgba(21,60,41,.08);
    border-radius: 14px;
    background: white;
  }

  .eval-model-group > strong {
    display: block;
    min-height: 35px;
    color: var(--forest);
    font-size: 11px;
    line-height: 1.35;
  }

  .eval-bars {
    height: 125px;
    display: grid;
    grid-template-columns: repeat(6,minmax(9px,1fr));
    gap: 5px;
    align-items: end;
    margin-top: 8px;
    padding-top: 8px;
    border-bottom: 1px solid rgba(21,60,41,.15);
  }

  .eval-bar {
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: end;
    gap: 4px;
    min-width: 0;
  }

  .eval-bar i {
    width: 100%;
    height: var(--metric-value);
    min-height: 2px;
    border-radius: 5px 5px 1px 1px;
    background: var(--metric-color);
  }

  .eval-bar small {
    display: none;
    color: var(--muted);
    font-size: 8px;
    white-space: nowrap;
  }

  .eval-bar.unavailable i {
    height: 2px;
    background: repeating-linear-gradient(90deg,rgba(102,116,108,.35) 0 4px,transparent 4px 7px);
  }

  .evaluation-empty, .roc-unavailable {
    padding: 30px 20px;
    border: 1px dashed rgba(21,60,41,.22);
    border-radius: 14px;
    color: var(--muted);
    background: var(--sage);
    text-align: center;
  }

  .roc-chart { display: block; width: 100%; height: auto; margin-top: 12px; overflow: visible; }
  .roc-chart text { fill: var(--muted); font: 11px "DM Sans",sans-serif; }
  .roc-grid { stroke: rgba(21,60,41,.32); stroke-width: 1.5; }
  .roc-baseline { stroke: rgba(102,116,108,.55); stroke-width: 1.5; stroke-dasharray: 7 7; }
  .roc-model-curve {
    fill: none;
    stroke: var(--curve-color);
    stroke-width: 3.5;
    stroke-linejoin: round;
    stroke-linecap: round;
  }
  .roc-baseline-key {
    height: 2px !important;
    border-radius: 0 !important;
    background: repeating-linear-gradient(90deg,rgba(102,116,108,.7) 0 5px,transparent 5px 8px) !important;
  }
  .roc-unavailable { margin-top: 18px; min-height: 220px; display: grid; place-items: center; }

  .advice-card { border-color: rgba(21,60,41,.09); }
  .source-list a { color: var(--green); }

  @keyframes stepIn {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
  }

  @media (max-width: 920px) {
    .nav { top: 8px; }
    .hero, .predict-layout, .section-grid, .algorithm-lab, .advice-grid, .research-map, .producer-grid, .evaluation-panels { grid-template-columns: 1fr; }
    .profile { position: static; }
    .hero-copy, .premium-health-visual, .premium-health-visual img { min-height: 430px; }
  }

  @media (max-width: 640px) {
    main { width: min(100vw - 18px,1240px); padding-top: 9px; }
    .nav { border-radius: 21px; padding: 7px; align-items: flex-start; }
    .nav-links { justify-content: flex-end; }
    .nav-links a { min-height: 32px; padding: 6px 8px; font-size: 11px; }
    .hero { padding: 5px; }
    .hero-copy { min-height: 430px; padding: 27px; }
    .premium-health-visual, .premium-health-visual img { min-height: 320px; }
    .trust-strip, .field-grid, .review-grid { grid-template-columns: 1fr; }
    .form-head { display: block; }
    .form-head .pill { margin-top: 8px; }
    .step-actions { flex-direction: column-reverse; }
    .step-actions button { width: 100%; }
    .result-card { margin-top: 20px; }
    .evaluation-panel, .selected-model-banner { padding: 17px; }
    .evaluation-panel-head { display: block; }
    .evaluation-panel-head p { max-width: none; margin-top: 8px; }
    .eval-models { grid-template-columns: repeat(2,minmax(0,1fr)); }
  }

  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
      scroll-behavior: auto !important;
      animation-duration: 1ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 1ms !important;
    }
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
    notify = bool(chat_context.get("notify", False)) if chat_context else False
    widget = chat_widget_html(risk_tier, probability, notify)
    return f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>{title}</title>
      {STYLE}
      {CYBERPUNK_OVERRIDE}
      {PREMIUM_HEALTH_OVERRIDE}
      <script>document.documentElement.classList.add('js');</script>
    </head>
    <body>
      <main>
        <nav class="nav">
          <div class="brand"><img class="brand-logo" src="/static/obeast-logo.png" alt="O-Beast logo">O-Beast</div>
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
          Answer ten body and lifestyle questions. O-Beast combines them into an educational probability estimate.
        </p>
        <div class="profile-guide">
          <div><strong>1</strong><span>Share your body profile.</span></div>
          <div><strong>2</strong><span>Describe your usual routine.</span></div>
          <div><strong>3</strong><span>Review answers before prediction.</span></div>
        </div>
      </aside>

      <form class="predict-form" id="predictionForm" action="/predict-form" method="post">
        <div class="form-head">
          <div>
            <div class="kicker">Your profile</div>
            <h2 style="margin-top:12px">Start your risk check</h2>
          </div>
          <div class="pill" id="stepCounter">Step 1 of 4</div>
        </div>
        <div class="form-progress" aria-label="Form progress"><span id="formProgress"></span></div>
        <div class="step-meta"><span id="stepName">Body profile</span><span>Educational estimate</span></div>

        <section class="form-step active" data-step="1">
          <h2>First, your body profile.</h2>
          <p>These answers help calculate BMI and provide basic context for the model.</p>
          <div class="field-grid">
            <label>Age <input name="age" type="number" min="5" max="100" value="16" required></label>
            <label>Sex <select name="sex"><option value="M">Male</option><option value="F">Female</option></select></label>
            <label>Height in centimetres <input name="height_cm" type="number" min="80" max="230" step="0.1" value="170" required></label>
            <label>Weight in kilograms <input name="weight_kg" type="number" min="20" max="250" step="0.1" value="65" required></label>
          </div>
          <div class="step-actions"><span></span><button type="button" data-next>Continue</button></div>
        </section>

        <section class="form-step" data-step="2">
          <h2>How does a usual day feel?</h2>
          <p>Movement, sitting time, and sleep create important lifestyle signals.</p>
          <div class="field-grid">
            <label>Physical activity each week, hours <input name="physical_activity_hours_per_week" type="number" min="0" max="40" step="0.1" value="3" required></label>
            <label>Screen time each day, hours <input name="screen_time_hours_per_day" type="number" min="0" max="24" step="0.1" value="5" required></label>
            <label>Usual sleep each day, hours <input name="sleep_hours" type="number" min="0" max="16" step="0.1" value="7" required></label>
          </div>
          <div class="step-actions"><button class="step-back" type="button" data-back>Previous</button><button type="button" data-next>Continue</button></div>
        </section>

        <section class="form-step" data-step="3">
          <h2>Now, food and family context.</h2>
          <p>Frequency matters more than perfection. Use answers closest to your usual routine.</p>
          <div class="field-grid">
            <label>Fast-food meals each week <input name="fast_food_meals_per_week" type="number" min="0" max="30" value="2" required></label>
            <label>Sugary drinks each day <input name="sugary_drinks_per_day" type="number" min="0" max="20" step="0.1" value="1" required></label>
            <label>Family history of obesity <select name="family_history_obesity"><option value="0">No</option><option value="1">Yes</option></select></label>
          </div>
          <div class="step-actions"><button class="step-back" type="button" data-back>Previous</button><button type="button" data-next>Review answers</button></div>
        </section>

        <section class="form-step" data-step="4">
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
      var names=['Body profile','Daily routine','Food and family','Review'];

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
          physical_activity_hours_per_week:'Weekly activity',screen_time_hours_per_day:'Daily screen time',
          sleep_hours:'Daily sleep',fast_food_meals_per_week:'Weekly fast food',
          sugary_drinks_per_day:'Daily sugary drinks',family_history_obesity:'Family history'
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
    return page_shell("Advice - O-Beast", body, chat_context={"notify": True})


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
