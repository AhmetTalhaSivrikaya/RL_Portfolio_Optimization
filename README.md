# 📈 Pekiştirmeli Öğrenme (RL) Tabanlı Dinamik Portföy Optimizasyonu

Borsa İstanbul (BIST) hisse senetlerinin tarihsel fiyat hareketlerini, teknik indikatörlerini ve piyasa endeksi (`XU100`) dinamiklerini analiz ederek sermaye dağılımını dinamik olarak optimize eden **Derin Pekiştirmeli Öğrenme (Proximal Policy Optimization - PPO)** tabanlı portföy yönetim sistemi.

Proje; **özel Gymnasium borsa simülasyon ortamı**, **çoklu seed ile eğitilmiş PPO ajanı**, **klasik finansal referans modelleri (Markowitz Ortalama-Varyans Optimizasyonu ve Buy & Hold)** ve **etkileşimli Streamlit performans analizi panelinden** oluşur.

---

## 🎯 Temel Özellikler ve Metodoloji

* **Varlık Evreni ve Piyasa Bağlamı:** BIST30'un farklı sektörlerini temsil eden 5 hisse senedi (`THYAO.IS`, `GARAN.IS`, `ASELS.IS`, `KCHOL.IS`, `BIMAS.IS`), nakit pozisyonu (`W_CASH`) ve piyasa rejimi göstergesi (`MKT_xu100`) üzerinde çalışır.
* **Sızıntısız (Leak-Free) Kronolojik Veri Ayrımı:** Zaman serisi bütünlüğünü korumak için rastgele bölme yerine kronolojik ayrım uygulanır (`2018–2023` Eğitim & Doğrulama, `2024–2025` Out-of-Sample Test). Özellik ölçeklendirme (`StandardScaler`) yalnızca eğitim setine fit edilir.
* **Gerçekçi İşlem Maliyeti ve Rebalans Kısıtları:** Hem RL ortamında hem de referans modellerde tek yönlü turnover üzerinden **%0.1 (binde 1)** komisyon uygulanır. Ajan, gereksiz işlem maliyetini önlemek amacıyla periyodik yeniden dengeleme (`rebalance_freq = 5` işlem günü) ve minimum değişim eşiği (`rebalance_threshold = 0.05`) ile çalışır.
* **Risk-Duyarlı Ödül Fonksiyonu:** Logaritmik net portföy getirisine ek olarak turnover maliyeti ve zirveden düşüş (drawdown) cezalarını içeren ödül yapısı kullanılır.
* **Çoklu Seed Eğitim ve Doğrulama:** Rastgele başlangıç etkisini azaltmak için modeller 3 farklı seed (`42, 43, 44`) ile eğitilir; en yüksek validation Sharpe oranına sahip kontrol noktası (`best_seed = 44`) test değerlendirmesi için otomatik seçilir.

---

## 📊 Test Dönemi Performans Karşılaştırması (`2024-01-16` → `2025-12-31`)

Başlangıç sermayesi **100.000 TL**, komisyon oranı **%0.1** ve yıllık risksiz getiri oranı **%0** kabul edilerek elde edilen test sonuçları:

| Metrik | Buy & Hold (Eşit Ağırlık) | Markowitz (Max Sharpe) | PPO RL Ajanı (Seed 44) |
| :--- | :---: | :---: | :---: |
| **Son Portföy Değeri** | 159.760 TL | 108.298 TL | **135.107 TL** |
| **Toplam Net Getiri** | %59.76 | %8.30 | **%35.11** |
| **Yıllıklandırılmış Getiri** | %51.11 | %7.28 | **%30.36** |
| **Yıllık Volatilite** | %26.56 | %29.25 | **%27.74** |
| **Sharpe Oranı** | 1.312 | 0.045 | **0.735** |
| **Sortino Oranı** | 1.971 | 0.064 | **1.042** |
| **Max Drawdown** | -%16.27 | -%19.67 | **-%25.71** |
| **Calmar Oranı** | 3.141 | 0.370 | **1.181** |
| **Günlük VaR (%95)** | -%2.02 | -%2.31 | **-%2.56** |
| **Günlük CVaR (%95)** | -%3.27 | -%3.87 | **-%4.04** |
| **Toplam Rebalans / Komisyon** | Tek seferlik alım | Periyodik | **56 işlem / 4.120 TL** |

### Öne Çıkan Bulgular
* **Klasik Optimizasyona Üstünlük:** PPO ajanı, geçmiş kovaryans matrisine aşırı uyum sağlayarak yalnızca 3 hisseye yoğunlaşan klasik Markowitz modelini toplam getiride (**%35.11 vs %8.30**) ve Sharpe oranında (**0.735 vs 0.045**) açık ara geride bırakmıştır.
* **Dinamik Rejim Adaptasyonu:** Ağırlık dağılımı analizleri, ajanın düşüş dönemlerinde nakit (`CASH`) ağırlığını yükselterek savunmaya geçtiğini, yükseliş trendlerinde ise momentumu yüksek hisselere (`GARAN.IS`, `ASELS.IS`, `THYAO.IS`) dinamik olarak ağırlık kaydırdığını göstermektedir.


## KURULUM VE ÇALIŞTIRMA
### 1.KURULUM
git clone [https://github.com/AhmetTalhaSivrikaya/RL_Portfolio_Optimization.git](https://github.com/AhmetTalhaSivrikaya/RL_Portfolio_Optimization.git)
cd RL_Portfolio_Optimization

python -m venv venv
* Windows PowerShell:
.\venv\Scripts\Activate.ps1
 *Linux / macOS / WSL:
 source venv/bin/activate

python -m pip install -r requirements.txt

### 2.UÇTAN UCA PIPELINE

 1. Veri setini indir ve teknik indikatörleri üret
python -m src.data_loader.data_pipeline

 2. PPO ajanını çoklu seed (42, 43, 44) ile eğit ve en iyi modeli seç
python src/rl_agent/train.py

 3. En iyi RL modelini out-of-sample test setinde değerlendir
python src/rl_agent/evaluate.py --split test

 4. Markowitz ve Buy & Hold baseline sonuçlarını üret
python -m src.baseline.portfolio_baselines

 5. Tüm çıktı dosyalarının sözleşme ve ağırlık tutarlılığını doğrula
python src/rl_agent/market_data_validator.py

### 3.ETKİLEŞİMLİ PANELİ AÇMA
python -m streamlit run src/metrics_ui/app.py

### 4. TEST ÇALIŞTIRMA 
pytest -v

Bunu doğrudan kök dizindeki `README.md` dosyanın içine yapıştırıp kaydedebilir, ardından `git add README.md && git commit -m "Update README with final architecture and benchmark results" && git push origin main` komutuyla GitHub'a gönderebilirsin.
