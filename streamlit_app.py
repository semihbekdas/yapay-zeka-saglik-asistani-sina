import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import nltk
import numpy as np
import streamlit as st
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


@st.cache_resource
def load_classic_ml() -> ClassicMLClassifier:
    model_dir = Path("saved_models/ml")
    if not model_dir.exists():
        raise FileNotFoundError(
            "Klasik ML modelleri bulunamadı. saved_models/ml klasörünü kontrol et."
        )
    return ClassicMLClassifier(model_dir)


@st.cache_resource
def load_transformer_models() -> Dict[str, TransformerClassifier]:
    base = Path("saved_models")
    models = {}
    bert_path = base / "berturk-doctorsitesi-best"
    xlmr_path = base / "xlmr-doctorsitesi-best"
    if bert_path.exists():
        models["BERTurk"] = TransformerClassifier(bert_path, "BERTurk")
    if xlmr_path.exists():
        models["XLM-R"] = TransformerClassifier(xlmr_path, "XLM-R")
    if not models:
        raise FileNotFoundError("Transformer modelleri bulunamadı.")
    return models


@st.cache_resource
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


def init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[Dict[str, str]] = []
    if "ml_predictions" not in st.session_state:
        st.session_state.ml_predictions = {}
    if "transformer_predictions" not in st.session_state:
        st.session_state.transformer_predictions = {}


def render_predictions(title: str, predictions: Dict[str, Dict[str, float]]) -> None:
    st.subheader(title)
    if not predictions:
        st.info("Henüz tahmin yok.")
        return
    for model_name, result in predictions.items():
        label = result.get("label", "-")
        score = result.get("score")
        if score is None or np.isnan(score):
            st.markdown(f"**{model_name}** ➜ {label}")
        else:
            st.markdown(f"**{model_name}** ➜ {label} (`{score:.2%}`)")


def render_transformer_predictions(predictions: Dict[str, Tuple[str, float]]) -> None:
    st.subheader("Transformer Modelleri")
    if not predictions:
        st.info("Henüz tahmin yok.")
        return
    for model_name, result in predictions.items():
        label, score = result
        st.markdown(f"**{model_name}** ➜ {label} (`{score:.2%}`)")


def main() -> None:
    st.set_page_config(page_title="Yapay Zeka Sağlık Asistanı", layout="wide")
    st.title("Sina")
    st.caption("Sol: Klasik ML | Orta: LLM sohbeti | Sağ: Transformer klasifikasyonları")

    init_state()

    try:
        ml_pipeline = load_classic_ml()
    except Exception as exc:
        st.sidebar.error(str(exc))
        ml_pipeline = None

    try:
        transformer_models = load_transformer_models()
    except Exception as exc:
        st.sidebar.error(str(exc))
        transformer_models = {}

    left_col, center_col, right_col = st.columns([1, 2, 1])

    with left_col:
        render_predictions("Klasik ML Modelleri", st.session_state.ml_predictions)

    with center_col:
        st.subheader("LLM Sohbet Alanı")
        for message in st.session_state.chat_history:
            avatar = "👤" if message["role"] == "user" else "🤖"
            st.markdown(f"{avatar} **{message['role'].capitalize()}**: {message['content']}")

        with st.form("chat-form", clear_on_submit=True):
            user_text = st.text_area(
                "Sorunu yaz",
                height=120,
                placeholder="Örn. 3 gündür boğazım ağrıyor ama ateşim yok...",
            )
            submitted = st.form_submit_button("Gönder")

        if submitted:
            user_text = user_text.strip()
            if user_text:
                st.session_state.chat_history.append({"role": "user", "content": user_text})
                with st.spinner("LLM düşünüyor..."):
                    try:
                        assistant_reply = query_ollama(st.session_state.chat_history)
                    except RuntimeError as exc:
                        assistant_reply = f"⚠️ {exc}"
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": assistant_reply}
                )
                if ml_pipeline:
                    st.session_state.ml_predictions = ml_pipeline.predict(user_text)
                if transformer_models:
                    transformer_preds = {}
                    for name, model in transformer_models.items():
                        label, score = model.predict(user_text)
                        transformer_preds[name] = (label, score)
                    st.session_state.transformer_predictions = transformer_preds
                st.rerun()
            else:
                st.warning("Boş mesaj gönderemezsin.")

    with right_col:
        render_transformer_predictions(st.session_state.transformer_predictions)


if __name__ == "__main__":
    main()
