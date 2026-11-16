# 📈 Pekiştirmeli Öğrenme (RL) Tabanlı Dinamik Portföy Optimizasyonu

Hisse senetlerinin tarihsel fiyatları ve teknik göstergelerinden sermaye dağılımını dinamik olarak öğrenen, **Derin Pekiştirmeli Öğrenme (PPO)** tabanlı bir portföy ajanı ve onu Markowitz ile Al-ve-Tut stratejilerine karşı adil biçimde kıyaslayan bir altyapı.

## Durum

| Modül | Sorumlu | Durum |
|---|---|---|
| `src/rl_agent/` | 2. Kişi | ✅ Ortam, eğitim, backtest, testler hazır. Sonuçlar henüz **sentetik** veriyle |
| `src/baseline/` | 3. Kişi | ✅ Markowitz (günlük/haftalık) ve Al-ve-Tut hazır |
| `src/data_loader/` | 1. Kişi | ⏳ Henüz yok. Gerçek veri gelmeden strateji değeri hakkında sonuç çıkarılamaz |
| `src/metrics_ui/` | 4. Kişi | ⏳ Henüz yok |

> ⚠️ Repodaki mevcut sonuç dosyaları sentetik veriden gelir ve stratejinin başarısını göstermez.

## Klasör yapısı

```text
RL_Portfolio_Optimization/
├── data/
│   ├── raw/            # Ham piyasa verileri (Git'e yüklenmez)
│   ├── processed/      # train/val/test fiyat ve özellik CSV'leri (sözleşme formatında)
│   └── results/        # <yöntem>_backtest_results.csv (RL, Markowitz, Al-ve-Tut)
├── docs/ARAYUZ_SOZLESMESI.md   # Modüller arası veri sözleşmesi (okumak zorunlu)
├── src/
│   ├── data_loader/    # [1. Kişi] Veri çekme ve özellik mühendisliği
│   ├── rl_agent/       # [2. Kişi] Gymnasium ortamı, PPO, backtest
│   ├── baseline/       # [3. Kişi] Markowitz ve Al-ve-Tut
│   └── metrics_ui/     # [4. Kişi] Risk metrikleri ve arayüz
├── tests/              # Sözleşme testleri
└── requirements.txt    # Çalışma bağımlılıkları (testler için requirements-dev.txt)
```

## Kurulum

```bash
git clone https://github.com/AhmetTalhaSivrikaya/RL_Portfolio_Optimization.git
cd RL_Portfolio_Optimization

python -m venv venv
venv\Scripts\activate          # Windows PowerShell   (Linux/macOS: source venv/bin/activate)

pip install -r requirements-dev.txt
```

## Hızlı başlangıç (sentetik veriyle uçtan uca)

```bash
python src/rl_agent/dummy_data_generator.py
python src/rl_agent/train.py --seeds 42 43 --timesteps 50000
python src/rl_agent/evaluate.py
python -m src.baseline.portfolio_baselines
python src/rl_agent/market_data_validator.py     # tüm dosyalar sözleşmeye uyuyor mu?
pytest
```

Ayrıntılar: [`src/rl_agent/README.md`](src/rl_agent/README.md), [`src/baseline/README.md`](src/baseline/README.md), [`docs/ARAYUZ_SOZLESMESI.md`](docs/ARAYUZ_SOZLESMESI.md).

## Ekip çalışma kuralları

- 🚫 **`main` dalına doğrudan push yasaktır.** Herkes kendi `feature/...` dalında ve kendi modül klasöründe çalışır.
- 🔀 **Pull Request zorunludur:** Birleştirmeden önce en az 1 ekip arkadaşının onayı alınır.
- 🔄 Çalışmaya başlamadan önce `git pull origin main`.
- ✅ **Göndermeden önce** `pytest` ve `python src/rl_agent/market_data_validator.py` hatasız geçmeli.
- 📊 **Kronolojik veri ayrımı:** Rastgele `train_test_split` yasak. Bölünme: train 2018–2022, val 2023, test 2024–2025.
- 💸 **Komisyon standardı:** `portföy_değeri × turnover × %0.1`; turnover yalnızca hisse bacaklarının alış+satış hacmidir (nakit ücretsiz). RL ve baseline'lar aynı tanımı kullanır.
- 💾 Büyük/ham veri, eğitilmiş modeller (`models/`) ve geçici sonuçlar Git'e eklenmez.