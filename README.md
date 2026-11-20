📈 Pekiştirmeli Öğrenme ile BIST30 Portföy Optimizasyonu

Bu proje, Borsa İstanbul'da (BIST30) işlem gören yüksek likiditeli 5 hisse senedi ve nakit (CASH) varlığı arasında dinamik sermaye dağılımı yapan Derin Pekiştirmeli Öğrenme (PPO - Proximal Policy Optimization) tabanlı bir algoritmik portföy yönetim sistemidir.

Geliştirilen ajan, geçmiş fiyat hareketlerini, teknik göstergeleri ve makro piyasa rejimini (XU100) analiz ederek işlem maliyetleri (komisyon ve turnover) altında risk-ayarlı getiriyi maksimize etmeyi öğrenir ve klasik Markowitz (Mean-Variance) ile Buy & Hold stratejileriyle kıyaslanır.

🎯 Temel Özellikler

Gerçekçi BIST30 Varlık Evreni: THYAO.IS, GARAN.IS, ASELS.IS, KCHOL.IS, BIMAS.IS hisseleri + güvenli liman olarak CASH (nakit) varlığı.

Piyasa Rejimi Farkındalığı (MKT_xu100): BIST100 endeksinin günlük getirisi, 20 günlük volatilitesi ve 50 günlük hareketli ortalamaya göre konumu (rejim göstergesi) doğrudan ajanın gözlem uzayına dahil edilir.

Sızıntısız (Leakage-Free) Ön İşleme: Eğitim (%70), doğrulama (%15) ve test (%15) setleri kronolojik olarak ayrılır; ölçeklendirme (StandardScaler) istatistikleri yalnızca eğitim verisi üzerinde hesaplanır.

Gerçekçi İşlem Maliyeti ve Rebalans Kısıtları:

Binde 1 (0.001) komisyon oranı ve tek yönlü turnover ($0.5 \times \sum \vert{}w_t - w_{t-1}\vert{}$) hesabı.

Gereksiz işlem maliyetini önlemek için rebalance_freq=5 (haftalık periyot) ve rebalance_threshold=0.05 eşik değeri.

Çoklu Seed (Multi-Seed) Eğitim ve Seçim: 42, 43 ve 44 seed değerleriyle eğitilen modeller arasından validation Sharpe oranına göre en iyi model (Seed 44) otomatik seçilir.

Etkileşimli Analiz Paneli: Streamlit ve Plotly tabanlı dashboard ile kümülatif getiri, drawdown, rolling Sharpe, portföy ağırlık değişimleri ve işlem maliyetleri interaktif olarak incelenir.

🏗️ Proje Mimarisi

RL_Portfolio_Optimization/
├── config.py                        # Merkezi varlık listesi, tarih aralıkları ve yol ayarları
├── requirements.txt                 # Proje bağımlılıkları
├── data/
│   ├── raw/                         # Ham piyasa ve BIST100 verileri (gitignore)
│   ├── processed/                   # Ölçeklenmiş ve bölünmüş train/val/test setleri
│   └── results/                     # RL, Markowitz ve Buy&Hold backtest çıktıları (CSV & JSON)
├── models/
│   └── ppo_portfolio/               # Eğitilmiş PPO modelleri (seed_42, seed_43, seed_44, best)
└── src/
    ├── data_pipeline/
    │   └── fetch_data.py            # Yahoo Finance veri çekme, özellik mühendisliği ve ölçekleme
    ├── baseline/
    │   └── portfolio_baselines.py   # Markowitz (PyPortfolioOpt) ve Buy & Hold referans modelleri
    ├── rl_agent/
    │   ├── env.py                   # Gymnasium uyumlu özel PortfolioEnv ortamı
    │   ├── train.py                 # Çoklu seed PPO eğitim ve validation seçim betiği
    │   ├── evaluate.py              # Test seti backtest ve metrik raporlama betiği
    │   ├── market_data_validator.py # Veri ve sonuç sözleşmesi doğrulayıcı
    │   └── test_env.py              # Ortam finansal hesaplama birim testleri
    └── metrics_ui/
        ├── metrics.py               # Sharpe, Sortino, Calmar, VaR/CVaR hesaplama modülü
        ├── app.py                   # Streamlit performans ve karşılaştırma paneli
        └── tests/
            └── test_metrics.py      # Finansal metrik birim testleri


📊 Deneysel Sonuçlar

1. Validation Seti Çoklu Seed Eğitim Özeti (Seed 42, 43, 44)

Ortalama Validation Sharpe: 2.07 ± 0.58

En İyi Model (Seed 44): Validation döneminde %147.58 getiri ve 2.71 Sharpe oranına ulaşarak Buy & Hold referansını (%56.34 getiri, 1.57 Sharpe) geride bırakmıştır.

2. Test Seti Strateji Karşılaştırması (Ortak Değerlendirme Dönemi)

Metrik

Buy & Hold

Markowitz (Max Sharpe)

PPO RL Ajanı (Seed 44)

Toplam Getiri (Net)

%59.76

%8.30

%35.11

Yıllık Getiri

%51.11

%7.28

%30.36

Sharpe Oranı

1.312

0.045

0.735

Sortino Oranı

1.971

0.064

1.042

Calmar Oranı

3.141

0.370

1.181

Kazanan Gün Oranı

%55.24

%50.70

%53.15

Toplam Komisyon

-

-

4,120 TL (56 rebalans)

Önemli Bulgu: Klasik Markowitz optimizasyonu geçmiş örneklem kovaryansına aşırı duyarlı olduğu için yalnızca 3 hisseye yoğunlaşmış ve test döneminde düşük performans göstermiştir. PPO ajanı ise piyasa rejimine göre hisse ve nakit (CASH) ağırlıklarını dinamik olarak kaydırarak klasik optimizasyon yöntemini açık ara geride bırakmıştır.

🚀 Kurulum ve Çalıştırma

1. Ortam Kurulumu

git clone https://github.com/suleyman7361/RL_Portfolio_Optimization.git
cd RL_Portfolio_Optimization
python -m pip install -r requirements.txt


2. Uçtan Uca Çalıştırma Adımları

# 1. Veriyi çek ve train/val/test olarak hazırla
python src/data_pipeline/fetch_data.py

# 2. Klasik baseline stratejilerini (Markowitz & Buy&Hold) çalıştır
python -m src.baseline.portfolio_baselines

# 3. PPO ajanını çoklu seed (42, 43, 44) ile eğit
python src/rl_agent/train.py

# 4. En iyi modeli test setinde değerlendir ve sonuçları kaydet
python src/rl_agent/evaluate.py --split test

# 5. Tüm veri ve sonuç dosyalarının sözleşme uyumluluğunu doğrula
python src/rl_agent/market_data_validator.py


3. Testleri Çalıştırma ve Streamlit Panelini Açma

# Birim testleri çalıştır (22 test)
pytest -v

# Etkileşimli performans panelini başlat
python -m streamlit run src/metrics_ui/app.py


🛠️ Kullanılan Teknolojiler

Pekiştirmeli Öğrenme: Gymnasium, Stable-Baselines3 (PPO)

Finansal Optimizasyon & Analiz: PyPortfolioOpt, ta (Technical Analysis), yfinance

Veri İşleme & Test: NumPy, Pandas, SciPy, Pytest

Görselleştirme & Arayüz: Streamlit, Plotly, Matplotlib
