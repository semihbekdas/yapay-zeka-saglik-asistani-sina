# 🏥 Sina: Yapay Zeka Sağlık Asistanı

Türkçe tıbbi sorulara yanıt veren yapay zeka asistanı. Fine-tuned LLM + ML + Transformer modelleri entegre edilmiş Streamlit arayüzü.

## 📋 Özellikler

| Bileşen | Model | Açıklama |
|---------|-------|----------|
| **LLM (Sina)** | Llama 3.1 8B (Fine-tuned) | Sohbet tabanlı tıbbi danışmanlık |
| **ML** | TF-IDF + LogReg/SVM | Branş tahmini (16 kategori) |
| **Transformer** | BERTurk + XLM-R | Branş tahmini (16 kategori) |

## 🚀 Kurulum

### 1. Gereksinimler

- Python 3.10+
- [Ollama](https://ollama.com/download) (Mac/Linux/Windows)
- 8GB+ RAM (LLM için)

### 2. Python Ortamı

```bash
cd /path/to/nlpproje

# Sanal ortam oluştur
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### 3. Ollama Model Kurulumu

```bash
# Ollama'yı başlat (arka planda çalışır)
ollama serve

# Modeli oluştur (ilk seferde ~5GB indirir)
cd /path/to/nlpproje
ollama create sina -f Modelfile

# Test et
ollama run sina
```

## ▶️ Çalıştırma

### Terminal 1: Ollama Server
```bash
ollama serve
```

### Terminal 2: Streamlit App
```bash
cd /path/to/nlpproje
source venv/bin/activate
streamlit run streamlit_app.py
```

Tarayıcıda **http://localhost:8501** adresine git.

## 🖥️ Arayüz

```
┌─────────────────┬──────────────────────┬─────────────────┐
│  Klasik ML      │     LLM Sohbet       │  Transformer    │
│                 │                      │                 │
│  LogReg: ...    │  👤 Kullanıcı mesajı │  BERTurk: ...   │
│  SVM: ...       │  🤖 LLM cevabı       │  XLM-R: ...     │
└─────────────────┴──────────────────────┴─────────────────┘
```

## 📁 Dosya Yapısı

```
nlpproje/
├── streamlit_app.py      # Ana uygulama
├── Modelfile             # Ollama model konfigürasyonu
├── requirements.txt      # Python bağımlılıkları
├── saved_models/
│   ├── ml/               # TF-IDF + LogReg + SVM
│   ├── berturk-doctorsitesi-best/
│   └── xlmr-doctorsitesi-best/
└── venv/                 # Python sanal ortamı
```

## ⚙️ Konfigürasyon

### Modelfile Ayarları

```dockerfile
FROM hf.co/SemihBekdas/Llama3.1-8B-TR-PatientQA-LoRA-v1:Q4_K_M

PARAMETER temperature 0.3     # Düşük = daha tutarlı cevaplar
PARAMETER top_p 0.9
PARAMETER num_ctx 4096        # Context penceresi
PARAMETER repeat_penalty 1.1
```

### Ortam Değişkenleri (opsiyonel)

```bash
export OLLAMA_MODEL="sina"
export OLLAMA_BASE_URL="http://localhost:11434"
```

## �️ Ollama Komutları

### Temel Komutlar

| Komut | Açıklama |
|-------|----------|
| `ollama serve` | Ollama sunucusunu başlat |
| `ollama list` | Yüklü modelleri listele |
| `ollama ps` | Çalışan modelleri göster |
| `ollama run <model>` | Modeli çalıştır (interaktif) |
| `ollama stop <model>` | Çalışan modeli durdur |

### Model Yönetimi

| Komut | Açıklama |
|-------|----------|
| `ollama pull <model>` | Modeli indir |
| `ollama rm <model>` | Modeli sil |
| `ollama create <isim> -f Modelfile` | Modelfile'dan model oluştur |
| `ollama show <model>` | Model bilgilerini göster |
| `ollama cp <kaynak> <hedef>` | Modeli kopyala |

### Örnek Kullanım

```bash
# Yeni model oluştur
ollama create sina -f Modelfile

# Modeli test et
ollama run sina

# Çalışan modelleri gör
ollama ps

# Modeli sil
ollama rm sina

# Tüm modelleri listele
ollama list
```

## �🔧 Sorun Giderme

| Sorun | Çözüm |
|-------|-------|
| `ollama: command not found` | Ollama'yı [ollama.com](https://ollama.com) adresinden indir |
| `Connection refused` | `ollama serve` komutunu çalıştır |
| `Model not found` | `ollama create sina -f Modelfile` komutunu çalıştır |
| `Out of memory` | Daha küçük model kullan veya RAM artır |

## 📊 Tıbbi Branşlar (16 Kategori)

1. Beyin ve Sinir Cerrahisi
2. Çocuk Sağlığı ve Hastalıkları
3. Çocuk Nörolojisi
4. Endokrinoloji ve Metabolizma
5. Nefroloji
6. Dermatoloji
7. Fiziksel Tıp ve Rehabilitasyon
8. Genel Cerrahi
9. Kadın Hastalıkları ve Doğum
10. Jinekolojik Onkoloji
11. Üreme Endokrinolojisi ve İnfertilite
12. Kulak Burun Boğaz
13. Ortopedi ve Travmatoloji
14. Plastik Cerrahi
15. Psikiyatri
16. Üroloji

## ⚠️ Yasal Uyarı

Bu asistan **sadece bilgilendirme amaçlıdır**. Kesin tanı koymaz, teşhis yapmaz ve ilaç öneremez. Tıbbi şikayetleriniz için mutlaka bir sağlık kuruluşuna başvurun.

## 📄 Lisans

MIT License
