# 🏥 Türkçe Sağlık Sorunları için Hibrit Sınıflandırma ve LLM Tabanlı Üretimsel Asistan (Sina)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red.svg)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-LLM-green.svg)](https://ollama.com)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/SemihBekdas)

Bu repo, Türkçe sağlık alanındaki hasta–doktor etkileşimlerini desteklemek için geliştirilen iki bileşenli bir NLP sistemi içerir:
1. **Branş Yönlendirme:** Hasta sorusundan uygun doktor uzmanlık alanını tahmin eden 16 sınıflı metin sınıflandırma
2. **Üretimsel Asistan:** Güvenli sınırlar içinde bilgilendirici yanıt üreten LLM tabanlı sohbet modülü

> ⚠️ **Yasal Uyarı:** Bu asistan yalnızca bilgilendirme amaçlıdır. Kesin tanı koymaz, teşhis yapmaz ve ilaç/doz öneremez. Tıbbi şikayetleriniz için mutlaka bir sağlık kuruluşuna başvurun.

---

## 📌 Özet ve Katkılar

- Türkçe sağlık sorularından **branş yönlendirme** (16 sınıf) ve **bilgilendirici yanıt üretimi** birlikte ele alındı.
- Klasik ML (TF-IDF + Logistic Regression / Linear SVM) ve transformer modelleri (BERTurk, XLM-R) aynı deneysel kurgu altında karşılaştırıldı.
- LLM tarafında Llama 3.1–8B-Instruct, **Unsloth + LoRA (r=16) + 4-bit nicemleme** ile Türkçe tıbbi diyalog verisine uyarlandı.
- Veri hazırlamada **PII/promo temizliği**, **gürültü azaltma**, **sınıf filtreleme ve dengeleme** uygulandı.

---

## 🖥️ Uygulama Arayüzü (Streamlit)

Uygulama 3 sütunlu bir arayüz sunar:

```
┌─────────────────┬─────────────────────────┬─────────────────┐
│   Klasik ML     │      LLM Sohbet         │   Transformer   │
│                 │                         │                 │
│  LogReg: %64    │  👤 Kullanıcı mesajı    │  BERTurk: %69   │
│  SVM: %65       │  🤖 Sina cevabı         │  XLM-R: %64     │
└─────────────────┴─────────────────────────┴─────────────────┘
```

- Sol: TF-IDF + Logistic Regression / Linear SVM tahminleri (etiket + olasılık)
- Orta: LLM sohbet alanı (Ollama üzerinden Sina modeli)
- Sağ: BERTurk ve XLM-R tahminleri (etiket + olasılık)

---

## ✅ Deneysel Sonuçlar (Test)

| Model | Veri | Accuracy | Macro F1 |
|------|------|---------:|---------:|
| TF-IDF + Logistic Regression | 48.816 dengeli örnek | 0.6418 | 0.6424 |
| TF-IDF + Linear SVM | 48.816 dengeli örnek | 0.6467 | 0.6481 |
| BERTurk (dbmdz/bert-base-turkish-cased) | 48.816 dengeli örnek | **0.6705** | **0.6882** |
| XLM-RoBERTa (xlm-roberta-base) | 48.816 dengeli örnek | 0.6289 | 0.6434 |

LLM fine-tuning aşamasında doğrulama kaybı **2.27 → 2.12** seviyesine düşmüştür (1 epoch SFT).

---

## 📊 Veri Setleri ve Hazırlama

| Görev | Veri Seti | Ham Boyut | Çalışmada Kullanılan |
|------|-----------|-----------|----------------------|
| Branş sınıflandırma | [alibayram/doktorsitesi](https://huggingface.co/datasets/alibayram/doktorsitesi) | 150.105 train / 37.527 test | 16 sınıf, 48.816 dengeli örnek (41.493 train / 7.323 val) + 17.888 test |
| LLM fine-tuning | [kayrab/patient-doctor-qa-tr-167732](https://huggingface.co/datasets/kayrab/patient-doctor-qa-tr-167732) | 503.196 train / 60.000 test | 20 sınıf, 60.000 dengeli örnek (54.000 train / 6.000 val) |

Lisans/erişim notu: `alibayram/doktorsitesi` veri seti HF üzerinde “gated” ve CC BY-NC 4.0 lisanslıdır; `kayrab/patient-doctor-qa-tr-167732` veri seti MIT lisanslıdır.

**Ortak temizlik adımları:**
- PII temizliği (URL, e‑posta, telefon vb.)
- Tanıtım/iletişim satırlarının silinmesi (randevu, tel, whatsapp, klinik vb.)
- Unicode ve boşluk normalizasyonu
- Yinelenen kayıtların kaldırılması

**LLM için ek filtreler:**
- İlaç/doz istekleri ve riskli kalıpların elenmesi
- Çok kısa/çok uzun örneklerin filtrelenmesi

---

## 🧪 Yöntemler

### 1) Klasik ML (TF-IDF + LogReg / Linear SVM)
- Türkçe lowercasing, sayısal normalizasyon, karakter temizliği
- Stop-word çıkarımı + TurkishStemmer ile kök bulma
- TF-IDF vektörleştirme: `ngram_range=(1,2)`, `max_features=20000`
- C ∈ {0.1, 0.5, 1.0, 3.0, 10.0} aralığında doğrulama Macro F1’e göre seçim

### 2) Transformer (BERTurk / XLM-R)
- Minimal temizlik + model tokenizer’ı
- Fine-tuning: 3 epoch, lr=2e-5, batch=16, max_len=128
- Streamlit inference sırasında tokenizer `max_length=256` ile kısaltma yapar

### 3) LLM (Llama 3.1–8B + Unsloth + LoRA)
- Temel model: `Meta-Llama-3.1-8B-Instruct`
- 4-bit nicemleme + LoRA (r=16)
- SFT: 1 epoch, efektif batch ≈ 64, `adamw_8bit`, cosine LR
- Chat şablonu: **system / user / assistant** formatı
- Kayıp sadece assistant yanıtı üzerinde hesaplanır (train_on_responses_only)

---

## 📁 Proje Yapısı

```
├── streamlit_app.py              # Ana Streamlit uygulaması
├── Modelfile                     # Ollama model konfigürasyonu
├── requirements.txt              # Python bağımlılıkları
├── saved_models/                 # Klasik ML + Transformer ağırlıkları
│   ├── ml/
│   ├── berturk-doctorsitesi-best/
│   └── xlmr-doctorsitesi-best/
├── outputs/                      # Eğitim grafikleri ve confusion matrix görselleri
├── NlpPipeline.ipynb             # ML + Transformer eğitim pipeline
├── Llama3_1_(8B).ipynb           # LLM fine-tuning (LoRA)
├── Rapor.pdf                     # Detaylı proje raporu
└── sunum/                        # Sunum dosyaları
```

> Not: Uygulama `saved_models/` klasöründeki ağırlıkları bekler. Bu klasörler yoksa aynı isimlerle yerleştirmeniz gerekir.

---

## 🧩 Modelleri Eğitme (NlpPipeline.ipynb)

Bu repodaki **klasik ML ve transformer modellerini** yeniden üretmek için `NlpPipeline.ipynb` çalıştırılmalıdır. Notebook, veri temizleme, dengeleme, eğitim ve değerlendirme adımlarını uçtan uca içerir ve çıktıları `saved_models/` altına kaydeder.

Özet akış:
1. `NlpPipeline.ipynb` içindeki veri hazırlama hücrelerini çalıştır.
2. TF-IDF + Logistic Regression / Linear SVM eğitimini tamamla.
3. BERTurk ve XLM-R fine-tuning adımlarını çalıştır.
4. Üretilen klasörler `saved_models/ml/`, `saved_models/berturk-doctorsitesi-best/`, `saved_models/xlmr-doctorsitesi-best/` altında oluşur.

> Uyarı: Notebook eğitimi GPU gerektirebilir. Transformer ve LLM eğitimleri için Colab/A100 gibi ortamlar önerilir.

---

## 🚀 Kurulum

### Gereksinimler
- Python 3.10+
- [Ollama](https://ollama.com/download) (Mac/Linux/Windows)
- 8GB+ RAM (LLM için daha yüksek bellek önerilir)

### 1) Repo'yu Klonla

```bash
git clone https://github.com/semihbekdas/yapay-zeka-saglik-asistani-sina.git
cd yapay-zeka-saglik-asistani-sina
```

### 2) Sanal Ortam ve Bağımlılıklar

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# veya
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 3) Ollama Modeli Kur

```bash
ollama serve
ollama create sina -f Modelfile
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

Tarayıcıda `http://localhost:8501` adresine git.

---

## ⚙️ Konfigürasyon

Streamlit uygulaması aşağıdaki ortam değişkenlerini okur:

```bash
export OLLAMA_MODEL=sina
export OLLAMA_BASE_URL=http://localhost:11434
```

`Modelfile` içeriği Ollama tarafındaki model davranışını belirler.

---

## 🛠️ Sorun Giderme

| Sorun | Çözüm |
|-------|-------|
| `ollama: command not found` | [Ollama'yı indir](https://ollama.com) |
| `Connection refused` | `ollama serve` çalıştır |
| `Model not found` | `ollama create sina -f Modelfile` çalıştır |
| `Out of memory` | Daha yüksek RAM/GPU kullan veya `num_ctx` değerini düşür |
| `NLTK data missing` | `python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"` |

---

## 🔍 Sınırlılıklar ve Gelecek Çalışmalar

- Etiketler arasında doğal örtüşmeler (özellikle alt branşlar) nedeniyle karışmalar oluşur.
- Veri gürültüsü, yazım hataları ve kısa/eksik sorular performansı sınırlar.
- LLM çıktılarının güvenliği için nitel değerlendirme ve uzman geri bildirimi güçlendirilmelidir.
- Daha kapsamlı hiperparametre araması ve hiyerarşik branş etiketleme performansı artırabilir.

---

## 📚 Kaynaklar

- [LLM Model (Hugging Face)](https://huggingface.co/SemihBekdas/Llama3.1-8B-TR-PatientQA-LoRA-v1)
- [Sınıflandırma Verisi](https://huggingface.co/datasets/alibayram/doktorsitesi)
- [LLM Eğitim Verisi](https://huggingface.co/datasets/kayrab/patient-doctor-qa-tr-167732)
- [BERTurk](https://huggingface.co/dbmdz/bert-base-turkish-cased)
- [Llama 3.1](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)

---

## 👤 Geliştirici

**Semih Bekdaş**

- GitHub: [@semihbekdas](https://github.com/semihbekdas)
- HuggingFace: [SemihBekdas](https://huggingface.co/SemihBekdas)
