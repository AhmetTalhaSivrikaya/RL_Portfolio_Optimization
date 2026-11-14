# 📈 Pekiştirmeli Öğrenme (RL) Tabanlı Dinamik Portföy Optimizasyonu

Bu proje; finansal piyasalardaki hisse senetlerinin tarihsel fiyat hareketlerini ve teknik indikatörleri analiz ederek sermaye dağılımını dinamik olarak optimize eden, risk-getiri dengesini (Sharpe Oranı) maksimize etmeyi hedefleyen **Derin Pekiştirmeli Öğrenme (Reinforcement Learning)** tabanlı bir yapay zeka ajanıdır.

---

## 📂 Proje Klasör Mimarisi

```text
rl-portfolio-optimization/
├── data/
│   ├── raw/            # Ham piyasa verileri (Git'e yüklenmez)
│   └── processed/      # Temizlenmiş ve indikatör eklenmiş veri setleri
├── src/
│   ├── data_loader/    # [1. Kişi] Veri çekme & teknik indikatör mühendisliği
│   ├── rl_agent/       # [2. Kişi] Gymnasium borsa ortamı & PPO/SAC modeli
│   ├── baseline/       # [3. Kişi] Markowitz Portföy Teorisi & Buy-and-Hold kıyaslaması
│   └── metrics_ui/     # [4. Kişi] Finansal risk metrikleri & Streamlit arayüzü
├── notebooks/          # Deneysel analizler ve test defterleri
└── requirements.txt    # Ortak Python bağımlılıkları




Kurulum Adımları : 
# 1. Repoyu bilgisayarınıza klonlayın
git clone [https://github.com/PROJE_SAHIBI/rl-portfolio-optimization.git](https://github.com/PROJE_SAHIBI/rl-portfolio-optimization.git)
cd rl-portfolio-optimization

# 2. Sanal ortam (venv) oluşturun ve aktif edin
python3 -m venv venv
source venv/bin/activate       # Linux / macOS / WSL
# venv\Scripts\activate        # Windows PowerShell

# 3. Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt

# 4. Kendi sorumlu olduğunuz çalışma dalına (branch) geçin
git checkout feature/DAL_ADINIZ





Ekip Çalışma Kuralları ve Önemli Uyarılar
Kod çakışmalarını (merge conflict) ve veri hatalarını önlemek için tüm ekip üyelerinin aşağıdaki kurallara uyması zorunludur:

🚫 main Dalına Doğrudan Push Yasaktır: Hiçbir ekip üyesi doğrudan main dalına kod göndermemelidir. Herkes yalnızca kendi sorumlu olduğu feature/... dalında ve kendi modül klasöründe çalışmalıdır.

🔀 Pull Request (PR) Zorunluluğu: Bir modül veya özellik tamamlandığında GitHub üzerinden main dalına Pull Request açılmalı ve en az 1 ekip arkadaşının onayı alındıktan sonra birleştirilmelidir.

🔄 Güne Başlarken Güncelleme: Kod yazmaya başlamadan önce mutlaka ana daldaki güncel değişiklikleri kendi dalınıza çekin:
     git pull origin main

📊 Kronolojik Veri Ayrımı (Data Leakage Önlemi): Finansal zaman serilerinde kesinlikle rastgele veri bölme (train_test_split) yapılmamalıdır. Eğitim ve test verileri kronolojik olarak ayrılmalıdır (Örn: 2018–2023 Eğitim, 2024–2025 Backtest).

💸 İşlem Maliyeti (Komisyon) Standartı: Gerçekçi simülasyon sonuçları için hem RL ortamında hem de Markowitz baseline modelinde her alım-satım işlemi için %0.1 (binde 1) komisyon kesintisi standart olarak uygulanmalıdır.

💾 Büyük Veri Dosyaları: data/raw/ klasöründeki ham ve büyük boyutlu CSV dosyalarını GitHub'a yüklemeyin; yalnızca işlenmiş ve standartlaştırılmış veri setlerini paylaşın.
