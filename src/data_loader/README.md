Bu modül, pekiştirmeli öğrenme (RL) ajanı ve geleneksel finansal stratejiler (Markowitz, Buy & Hold) için ham borsa verilerini temizleyen, teknik indikatörler üreten ve hiçbir veri sızıntısı (data leakage) olmadan standart veri setlerine dönüştüren veri mühendisliği katmanıdır.

🛠️ Mimari ve Bileşenler
Modül, sorumluluklarına göre 6 ana Python dosyasından oluşur:

config.py (Konfigürasyon):

Çalışmada kullanılacak hedef hisse senetlerini (THYAO.IS, GARAN.IS, ASELS.IS, KCHOL.IS, BIMAS.IS), piyasa endeksini (XU100.IS), tarih aralığını (2018-01-01 - 2026-01-01) ve veri bölme oranlarını (%70 Train, %15 Val, %15 Test) tanımlar.

fetcher.py (Ham Veri Çekimi):

yfinance kütüphanesini kullanarak belirlenen hisselerin ve endeksin tarihsel OHLCV (Açılış, En Yüksek, En Düşük, Kapanış, Hacim) verilerini indirir.

İndirilen ham verileri data/raw/raw_market_data.csv altına yedekler.

cleaner.py (Temizleme ve Hizalama):

Piyasa tatillerini, eksik satırları ileriye dönük doldurma (ffill) ve eksik değer temizliği ile giderir.

Sıfır veya negatif fiyat hatalarını denetleyerek tüm varlıkları ortak bir tarih indeksinde ve artan sırada hizalar.

features.py (Özellik Mühendisliği):

Her hisse senedi için logaritmik getiriler, 20 günlük volatilite, hareketli ortalamalar (SMA 20, SMA 50), RSI (14) ve Bollinger Bantları gibi teknik indikatörler üretir.

İlk günlerdeki ısınma (warm-up) periyodundan kaynaklanan NaN değerleri temizler.

splitter.py (Sızıntısız Bölme ve Ölçeklendirme):

Veriyi tarih sırasını (kronolojik sıra) bozmadan Train, Validation ve Test dilimlerine ayırır.

Veri Sızıntısını Önleme (Data Leakage Prevention): StandardScaler parametreleri yalnızca Eğitim (Train) seti üzerinde öğrenilir (fit_transform); Doğrulama ve Test setlerine ise yalnızca dönüştürme (transform) uygulanır.

pipeline.py (Ana Boru Hattı):

Tüm bu adımları sırasıyla koordine eder ve modellerin ortaklaşa tüketeceği 6 temel CSV dosyasını data/processed/ klasörüne kaydeder.

📂 Üretilen Çıktılar (data/processed/)
Pipeline çalıştırıldığında data/processed/ altında şu dosyalar otomatik olarak oluşturulur:

train_prices.csv & train_features.csv

val_prices.csv & val_features.csv

test_prices.csv & test_features.csv