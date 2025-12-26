import html
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import joblib
import nltk
import numpy as np
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from ollama import Client
from TurkishStemmer import TurkishStemmer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SYSTEM_MESSAGE = (
    "Sen yardımsever, nazik ve bilgili bir tıbbi asistansın. "
    "Kullanıcıların sağlıkla ilgili şikayetlerini dinler ve genel bilgilendirme yaparsın. "
    "Cevapların her zaman Türkçe olsun.\n\n"
    "ÖNEMLİ KURALLAR:\n"
    "- Asla kesin teşhis koyma ve reçeteli ilaç önerme.\n"
    "- Acil durum belirtisi varsa (nefes darlığı, göğüs ağrısı, bilinç kaybı, şiddetli kanama vb.) kullanıcıyı ACİLE yönlendir.\n"
    "- Acil değilse, uygun branşa veya aile hekimine başvurmasını öner.\n"
    "- Önce kısa bir değerlendirme yap, sonra maddeler halinde öneriler ver.\n"
    "- Eksik bilgi varsa en fazla 3 kısa soru sor (ör: yaş, kaç gündür, şiddet/ateş/nefes darlığı var mı?).\n"
    "- Yanıtların kısa, net ve maddeli olsun."
)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "sina")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def parse_gradio_auth(value: str) -> Optional[Tuple[str, str]]:
    if not value or ":" not in value:
        return None
    user, password = value.split(":", 1)
    if not user or not password:
        return None
    return user, password


GRADIO_SHARE = os.getenv("GRADIO_SHARE", "1") == "1"
GRADIO_QUEUE_SIZE = env_int("GRADIO_QUEUE_SIZE", 32)
GRADIO_AUTH = parse_gradio_auth(os.getenv("GRADIO_AUTH", ""))

GRADIO_CSS = """
:root {
  --bg: #0f1117;
  --panel: #141824;
  --panel-2: #1a1f2b;
  --border: #2a3140;
  --text: #f8fafc;
  --muted: #9aa3af;
  --accent: #22c55e;
  --primary: #f97316;
  --primary-hover: #ea6a0f;
  --info: #3b82f6;
  --warning: #f6c177;
}
body {
  background: var(--bg);
  color: var(--text);
}
.gradio-container {
  max-width: 1200px;
  margin: 0 auto;
  font-family: "Source Sans Pro", "Helvetica Neue", Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
}
.gradio-container .prose,
.gradio-container .markdown,
.gradio-container p,
.gradio-container span,
.gradio-container label {
  color: var(--text);
}
h1, h2, h3, h4 {
  color: var(--text);
}
h1 {
  font-size: 2.4rem;
  font-weight: 700;
  margin-bottom: 0.25rem;
}
h3 {
  font-size: 1.1rem;
  font-weight: 600;
}
#caption {
  color: var(--muted);
  font-size: 0.9rem;
  margin-top: -0.3rem;
  margin-bottom: 1.25rem;
}
.info-box {
  color: var(--muted);
  font-size: 0.9rem;
}
.warning-box {
  color: var(--warning);
  font-size: 0.9rem;
  margin-top: 0.5rem;
}
.pred-line {
  margin-bottom: 0.7rem;
}
.score {
  color: var(--accent);
  font-weight: 600;
}
#chat-display p {
  margin: 0 0 0.85rem 0;
}
#input-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 0.6rem;
  padding: 0.75rem;
}
#user-input label {
  color: var(--text);
}
#user-input textarea {
  height: 120px !important;
  background: var(--panel-2) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 0.4rem;
  box-shadow: none !important;
}
#user-input textarea:focus {
  border-color: var(--info) !important;
  box-shadow: 0 0 0 0.2rem rgba(59, 130, 246, 0.25) !important;
}
#user-input textarea::placeholder {
  color: #6b7280;
}
#send-btn {
  margin-top: 0.75rem;
}
#send-btn button {
  width: 100%;
  background: var(--primary) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 0.4rem;
}
#send-btn button:hover {
  background: var(--primary-hover) !important;
}
#send-btn button:focus {
  box-shadow: 0 0 0 0.2rem rgba(249, 115, 22, 0.35) !important;
}
footer,
.gradio-footer {
  display: none !important;
}
"""


def ensure_nltk_resources() -> None:
    resources = {
        "stopwords": "corpora/stopwords",
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
    }
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass


class ClassicMLClassifier:
    def __init__(self, model_dir: Path) -> None:
        ensure_nltk_resources()
        self.vectorizer = joblib.load(model_dir / "tfidf_vectorizer.joblib")
        self.models = {
            "TF-IDF + Logistic Regression": joblib.load(model_dir / "logreg_best.joblib"),
            "TF-IDF + Linear SVM": joblib.load(model_dir / "linearsvm_best.joblib"),
        }
        self.id2label = self._load_label_map(model_dir / "id2label.json")
        self.stopwords = set(stopwords.words("turkish"))
        self.stemmer = TurkishStemmer()
        self.re_digits = re.compile(r"\d+")
        self.re_non_turkish = re.compile(r"[^a-zçğıöşü\s]")
        self.re_multi_space = re.compile(r"\s+")

    @staticmethod
    def _load_label_map(path: Path) -> Dict[int, str]:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {int(k): v for k, v in data.items()}

    def _preprocess(self, text: str) -> str:
        text = "" if text is None else str(text)
        text = text.replace("İ", "i").replace("I", "ı").lower()
        text = self.re_digits.sub(" num ", text)
        text = self.re_non_turkish.sub(" ", text)
        tokens = word_tokenize(text, language="turkish")
        processed: List[str] = []
        for tok in tokens:
            if len(tok) <= 2 or tok in self.stopwords:
                continue
            stem = self.stemmer.stem(tok)
            processed.append(stem)
        cleaned = " ".join(processed)
        return self.re_multi_space.sub(" ", cleaned).strip()

    def predict(self, text: str) -> Dict[str, Dict[str, float]]:
        cleaned = self._preprocess(text)
        features = self.vectorizer.transform([cleaned])
        predictions: Dict[str, Dict[str, float]] = {}
        for name, model in self.models.items():
            label_idx = int(model.predict(features)[0])
            label = self.id2label.get(label_idx, str(label_idx))
            score = self._score(model, features, label_idx)
            predictions[name] = {"label": label, "score": score}
        return predictions

    @staticmethod
    def _score(model, features, target_idx: int) -> float:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            return float(probs[target_idx])
        if hasattr(model, "decision_function"):
            raw = model.decision_function(features)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            scores = raw[0]
            shifted = scores - scores.max()
            exp_scores = np.exp(shifted)
            probs = exp_scores / exp_scores.sum()
            return float(probs[target_idx])
        return float("nan")


class TransformerClassifier:
    def __init__(self, model_dir: Path, display_name: str) -> None:
        self.name = display_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.id2label = {
            int(k): v for k, v in self.model.config.id2label.items()
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> Tuple[str, float]:
        prepared = preprocess_for_transformer(text)
        encoded = self.tokenizer(
            prepared,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1)
            score, idx = torch.max(probs, dim=-1)
        label_idx = int(idx.item())
        label = self.id2label.get(label_idx, str(label_idx))
        return label, float(score.item())


def preprocess_for_transformer(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.strip()
    return re.sub(r"\s+", " ", text)


@lru_cache(maxsize=1)
def load_classic_ml() -> ClassicMLClassifier:
    model_dir = Path("saved_models/ml")
    if not model_dir.exists():
        raise FileNotFoundError(
            "Klasik ML modelleri bulunamadı. saved_models/ml klasörünü kontrol et."
        )
    return ClassicMLClassifier(model_dir)


@lru_cache(maxsize=1)
def load_transformer_models() -> Dict[str, TransformerClassifier]:
    base = Path("saved_models")
    models: Dict[str, TransformerClassifier] = {}
    bert_path = base / "berturk-doctorsitesi-best"
    xlmr_path = base / "xlmr-doctorsitesi-best"
    if bert_path.exists():
        models["BERTurk"] = TransformerClassifier(bert_path, "BERTurk")
    if xlmr_path.exists():
        models["XLM-R"] = TransformerClassifier(xlmr_path, "XLM-R")
    if not models:
        raise FileNotFoundError("Transformer modelleri bulunamadı.")
    return models


@lru_cache(maxsize=1)
def load_ollama_client() -> Client:
    host = OLLAMA_BASE_URL.rstrip("/")
    return Client(host=host)


def build_chat_messages(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_MESSAGE}] + history


def _extract_chunk_content(chunk) -> str:
    message = getattr(chunk, "message", None)
    if message is None and isinstance(chunk, dict):
        message = chunk.get("message")
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    return content or ""


def query_ollama(history: List[Dict[str, str]]) -> str:
    client = load_ollama_client()
    messages = build_chat_messages(history)
    response_chunks: List[str] = []
    try:
        stream = client.chat(model=OLLAMA_MODEL, messages=messages, stream=True)
        for chunk in stream:
            text_chunk = _extract_chunk_content(chunk)
            if text_chunk:
                response_chunks.append(text_chunk)
    except Exception as exc:
        raise RuntimeError(
            f"Ollama servisine ulaşılamadı veya sohbet isteği başarısız oldu: {exc}. "
            "Modeli `ollama serve` ile çalıştırdığından emin ol."
        ) from exc
    return "".join(response_chunks).strip()


def _escape_text(value: str) -> str:
    return html.escape("" if value is None else str(value))


def info_box(text: str) -> str:
    return f'<div class="info-box">{_escape_text(text)}</div>'


def warning_box(text: str) -> str:
    return f'<div class="warning-box">{_escape_text(text)}</div>'


def format_ml_predictions(predictions: Dict[str, Dict[str, float]]) -> str:
    if not predictions:
        return info_box("Henüz tahmin yok.")
    lines = []
    for model_name, result in predictions.items():
        label = result.get("label", "-")
        score = result.get("score")
        safe_model = _escape_text(model_name)
        safe_label = _escape_text(label)
        if score is None or np.isnan(score):
            lines.append(f"<div class=\"pred-line\"><strong>{safe_model}</strong> ➜ {safe_label}</div>")
        else:
            lines.append(
                f"<div class=\"pred-line\"><strong>{safe_model}</strong> ➜ "
                f"{safe_label} <span class=\"score\">({score:.2%})</span></div>"
            )
    return "".join(lines)


def format_transformer_predictions(predictions: Dict[str, Tuple[str, float]]) -> str:
    if not predictions:
        return info_box("Henüz tahmin yok.")
    lines = []
    for model_name, (label, score) in predictions.items():
        safe_model = _escape_text(model_name)
        safe_label = _escape_text(label)
        lines.append(
            f"<div class=\"pred-line\"><strong>{safe_model}</strong> ➜ "
            f"{safe_label} <span class=\"score\">({score:.2%})</span></div>"
        )
    return "".join(lines)


def format_chat_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return ""
    lines = []
    for message in history:
        avatar = "👤" if message["role"] == "user" else "🤖"
        role = message["role"].capitalize()
        lines.append(f"{avatar} **{role}**: {message['content']}")
    return "\n\n".join(lines)


def handle_submit(
    user_text: str,
    chat_history: List[Dict[str, str]],
) -> Tuple[str, List[Dict[str, str]], str, str, str, str]:
    cleaned = (user_text or "").strip()
    if not cleaned:
        chat_text = format_chat_history(chat_history)
        return chat_text, chat_history, gr.update(), gr.update(), warning_box(
            "Boş mesaj gönderemezsin."
        ), ""

    chat_history = list(chat_history)
    chat_history.append({"role": "user", "content": cleaned})

    try:
        assistant_reply = query_ollama(chat_history)
    except RuntimeError as exc:
        assistant_reply = f"⚠️ {exc}"

    chat_history.append({"role": "assistant", "content": assistant_reply})
    chat_text = format_chat_history(chat_history)

    ml_text = warning_box("Klasik ML modelleri bulunamadı.")
    tr_text = warning_box("Transformer modelleri bulunamadı.")

    try:
        ml_pipeline = load_classic_ml()
        ml_preds = ml_pipeline.predict(cleaned)
        ml_text = format_ml_predictions(ml_preds)
    except Exception as exc:
        ml_text = warning_box(str(exc))

    try:
        transformer_models = load_transformer_models()
        transformer_preds = {
            name: model.predict(cleaned)
            for name, model in transformer_models.items()
        }
        tr_text = format_transformer_predictions(transformer_preds)
    except Exception as exc:
        tr_text = warning_box(str(exc))

    return chat_text, chat_history, ml_text, tr_text, "", ""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Yapay Zeka Sağlık Asistanı", css=GRADIO_CSS) as demo:
        gr.Markdown("# Sina")
        gr.Markdown(
            "Sol: Klasik ML | Orta: LLM sohbeti | Sağ: Transformer klasifikasyonları",
            elem_id="caption",
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Klasik ML Modelleri")
                ml_output = gr.HTML(info_box("Henüz tahmin yok."), elem_id="ml-output")
            with gr.Column(scale=2):
                gr.Markdown("### LLM Sohbet Alanı")
                chat_display = gr.Markdown("", elem_id="chat-display")
                with gr.Group(elem_id="input-card"):
                    user_input = gr.Textbox(
                        placeholder="Örn. 3 gündür boğazım ağrıyor ama ateşim yok...",
                        label="Sorunu yaz",
                        lines=3,
                        elem_id="user-input",
                    )
                    send_btn = gr.Button("Gönder", variant="primary", elem_id="send-btn")
                warning_output = gr.HTML("", elem_id="warning-output")
            with gr.Column(scale=1):
                gr.Markdown("### Transformer Modelleri")
                tr_output = gr.HTML(info_box("Henüz tahmin yok."), elem_id="tr-output")

        history_state = gr.State([])

        send_btn.click(
            handle_submit,
            inputs=[user_input, history_state],
            outputs=[
                chat_display,
                history_state,
                ml_output,
                tr_output,
                warning_output,
                user_input,
            ],
        )
        user_input.submit(
            handle_submit,
            inputs=[user_input, history_state],
            outputs=[
                chat_display,
                history_state,
                ml_output,
                tr_output,
                warning_output,
                user_input,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    launch_kwargs = {"share": GRADIO_SHARE}
    if GRADIO_AUTH:
        launch_kwargs["auth"] = GRADIO_AUTH
    demo.queue(max_size=GRADIO_QUEUE_SIZE).launch(**launch_kwargs)
