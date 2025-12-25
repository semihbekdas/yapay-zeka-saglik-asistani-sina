# 🏥 Sina: Yapay Zeka Sağlık Asistanı

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red.svg)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-LLM-green.svg)](https://ollama.com)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/SemihBekdas)

Türkçe tıbbi sorulara yanıt veren yapay zeka destekli sağlık asistanı. Fine-tuned LLM, klasik ML ve Transformer modelleri entegre edilmiş Streamlit arayüzü.

> ⚠️ **Yasal Uyarı:** Bu asistan sadece bilgilendirme amaçlıdır. Kesin tanı koymaz, teşhis yapmaz ve ilaç öneremez. Tıbbi şikayetleriniz için mutlaka bir sağlık kuruluşuna başvurun.

---

## 📋 Proje Özeti

Bu projede Türkçe tıbbi soru-cevap için üç farklı yaklaşım kullanılmıştır:

| Bileşen | Model | Dataset | Açıklama |
|---------|-------|---------|----------|
| 🤖 **LLM (Sina)** | [Llama 3.1 8B Fine-tuned](https://huggingface.co/SemihBekdas/Llama3.1-8B-TR-PatientQA-LoRA-v1) | 503K+ örnek | LoRA ile fine-tune edilmiş konuşma modeli |
| 📊 **Klasik ML** | TF-IDF + LogReg/SVM | 90K+ örnek | Branş tahmini (16 kategori) |
| 🔬 **Transformer** | BERTurk + XLM-R | 90K+ örnek | Branş tahmini (16 kategori) |

---

## 🖥️ Arayüz

Streamlit uygulaması 3 sütunlu bir arayüz sunar:

```
┌─────────────────┬─────────────────────────┬─────────────────┐
│   Klasik ML     │      LLM Sohbet         │   Transformer   │
│                 │                         │                 │
│  LogReg: %95    │  👤 Kullanıcı mesajı    │  BERTurk: %92   │
│  SVM: %94       │  🤖 Sina cevabı         │  XLM-R: %91     │
└─────────────────┴─────────────────────────┴─────────────────┘
```

- **Sol Sütun:** Klasik ML modelleri ile branş tahmini
- **Orta Sütun:** LLM ile interaktif sohbet
- **Sağ Sütun:** Transformer modelleri ile branş tahmini

---

## 🚀 Kurulum

### Gereksinimler

- Python 3.10+
- [Ollama](https://ollama.com/download) (Mac/Linux/Windows)
- 8GB+ RAM (LLM için önerilir)

### 1. Repo'yu Klonla

```bash
git clone https://github.com/semihbekdas/yapay-zeka-saglik-asistani-sina.git
cd yapay-zeka-saglik-asistani-sina
```

### 2. Python Ortamını Kur

```bash
# Sanal ortam oluştur
python3 -m venv venv

# Aktif et
source venv/bin/activate  # macOS/Linux
# veya
venv\Scripts\activate     # Windows

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### 3. Ollama Model Kurulumu

```bash
# Ollama'yı başlat (arka planda çalışır)
ollama serve

# Sina modelini oluştur (ilk seferde ~5GB indirir)
ollama create sina -f Modelfile

# Test et
ollama run sina
```

---

## ▶️ Çalıştırma

**Terminal 1:** Ollama sunucusu
```bash
ollama serve
```

**Terminal 2:** Streamlit uygulaması
```bash
source venv/bin/activate
streamlit run streamlit_app.py
```

Tarayıcıda **http://localhost:8501** adresine git.

---

## 📁 Proje Yapısı

```
├── streamlit_app.py              # Ana Streamlit uygulaması
├── Modelfile                     # Ollama model konfigürasyonu
├── requirements.txt              # Python bağımlılıkları
├── README.md                     # Bu dosya
│
├── 📓 Notebooks
│   ├── NlpPipeline.ipynb         # ML + Transformer eğitim pipeline
│   └── Llama3_1_(8B).ipynb       # LLM fine-tuning (LoRA)
│
├── 📊 saved_models/
│   ├── ml/                       # TF-IDF vektörizör + LogReg + SVM
│   │   ├── tfidf_vectorizer.joblib
│   │   ├── logreg_best.joblib
│   │   ├── linearsvm_best.joblib
│   │   └── id2label.json
│   ├── berturk-doctorsitesi-best/   # Fine-tuned BERTurk
│   └── xlmr-doctorsitesi-best/      # Fine-tuned XLM-R
│
└── 📄 Docs
    └── Rapor.pdf                 # Proje raporu
```

---

## 🔬 Metodoloji

### 1. Veri Seti

- **Kaynak:** [alibayram/doktorsitesi](https://huggingface.co/datasets/alibayram/doktorsitesi) + [kayrab/patient-doctor-qa-tr](https://huggingface.co/datasets/kayrab/patient-doctor-qa-tr-167732)
- **Ham Veri:** 150K+ soru-cevap çifti
- **Temizlik:** URL, telefon, email, doktor adı, ilaç dozları filtrelendi
- **Son Veri:** ~90K temiz örnek (ML/Transformer), 60K dengeli örnek (LLM)

### 2. Model Eğitimi

#### Klasik ML (NlpPipeline.ipynb)
- **Preprocessing:** Türkçe stemming, stopword temizliği, TF-IDF vektörizasyon
- **Modeller:** Logistic Regression, Linear SVM
- **Sonuç:** 16 branş sınıflandırması

#### Transformer (NlpPipeline.ipynb)
- **Modeller:** BERTurk, XLM-RoBERTa
- **Fine-tuning:** Hugging Face Transformers
- **Sonuç:** 16 branş sınıflandırması

#### LLM Fine-tuning (Llama3_1_(8B).ipynb)
- **Base Model:** Llama 3.1 8B Instruct
- **Teknik:** LoRA (Low-Rank Adaptation)
- **Framework:** Unsloth (2x hızlı fine-tuning)
- **Quantization:** Q4_K_M (GGUF format)
- **Platform:** Google Colab A100 GPU

### 3. Deployment

- **LLM Serving:** Ollama (lokal)
- **UI:** Streamlit
- **Model Hosting:** HuggingFace Hub

---

## ⚙️ Konfigürasyon

### Modelfile

```dockerfile
FROM hf.co/SemihBekdas/Llama3.1-8B-TR-PatientQA-LoRA-v1:Q4_K_M

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
PARAMETER repeat_penalty 1.1

SYSTEM """Sen yardımsever, nazik ve bilgili bir tıbbi asistansın.
Kullanıcıların sağlıkla ilgili şikayetlerini dinler ve genel bilgilendirme yaparsın.
Cevapların her zaman Türkçe olsun.

ÖNEMLİ KURALLAR:
- Asla kesin teşhis koyma ve reçeteli ilaç önerme.
- Acil durum belirtisi varsa kullanıcıyı ACİLE yönlendir.
- Acil değilse, uygun branşa veya aile hekimine başvurmasını öner.
- Yanıtların kısa, net ve maddeli olsun.
"""
```

### requirements.txt

```
streamlit>=1.38.0
torch>=2.1.0
transformers>=4.38.0
scikit-learn>=1.4.0
joblib>=1.3.0
nltk>=3.8.1
TurkishStemmer>=1.3
numpy>=1.24.0
ollama>=0.3.0
```

### Ortam Değişkenleri

| Değişken | Varsayılan | Açıklama |
|----------|------------|----------|
| `OLLAMA_MODEL` | `sina` | Kullanılacak Ollama modeli |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API adresi |

---

## 🛠️ Ollama Komutları

| Komut | Açıklama |
|-------|----------|
| `ollama serve` | Sunucuyu başlat |
| `ollama list` | Yüklü modelleri listele |
| `ollama ps` | Çalışan modelleri göster |
| `ollama run sina` | Modeli çalıştır |
| `ollama stop sina` | Modeli durdur |
| `ollama rm sina` | Modeli sil |
| `ollama create sina -f Modelfile` | Model oluştur |

---

## 📊 Desteklenen Tıbbi Branşlar (16 Kategori)

| # | Branş |
|---|-------|
| 1 | Beyin ve Sinir Cerrahisi |
| 2 | Çocuk Sağlığı ve Hastalıkları |
| 3 | Çocuk Nörolojisi |
| 4 | Endokrinoloji ve Metabolizma |
| 5 | Nefroloji |
| 6 | Dermatoloji |
| 7 | Fiziksel Tıp ve Rehabilitasyon |
| 8 | Genel Cerrahi |
| 9 | Kadın Hastalıkları ve Doğum |
| 10 | Jinekolojik Onkoloji |
| 11 | Üreme Endokrinolojisi ve İnfertilite |
| 12 | Kulak Burun Boğaz |
| 13 | Ortopedi ve Travmatoloji |
| 14 | Plastik Cerrahi |
| 15 | Psikiyatri |
| 16 | Üroloji |

---

## 🔧 Sorun Giderme

| Sorun | Çözüm |
|-------|-------|
| `ollama: command not found` | [Ollama'yı indir](https://ollama.com) |
| `Connection refused` | `ollama serve` çalıştır |
| `Model not found` | `ollama create sina -f Modelfile` çalıştır |
| `Out of memory` | RAM'i kontrol et (min 8GB) |
| `NLTK data missing` | `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"` |

---

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

---

## 👤 Geliştirici

**Semih Bekdaş**

- GitHub: [@semihbekdas](https://github.com/semihbekdas)
- HuggingFace: [SemihBekdas](https://huggingface.co/SemihBekdas)

---

## 🔗 Linkler

- 📦 [LLM Model (HuggingFace)](https://huggingface.co/SemihBekdas/Llama3.1-8B-TR-PatientQA-LoRA-v1)
- 📊 [Dataset (alibayram/doktorsitesi)](https://huggingface.co/datasets/alibayram/doktorsitesi)
- 📊 [Dataset (kayrab/patient-doctor-qa-tr)](https://huggingface.co/datasets/kayrab/patient-doctor-qa-tr-167732)
