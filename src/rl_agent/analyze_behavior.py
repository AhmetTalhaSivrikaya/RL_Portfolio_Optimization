import pandas as pd
import numpy as np


def analyze_agent_behavior(results_path: str = "data/results/rl_backtest_results.csv"):
    df = pd.read_csv(results_path)

    # Ağırlık sütunlarını otomatik bul (W_ ile başlayanlar)
    weight_cols = [col for col in df.columns if col.startswith("W_")]
    weights_df = df[weight_cols]

    print("=" * 60)
    print("🔍 RL AJANI PORTFÖY DAVRANIŞ VE ÇEŞİTLENDİRME RAPORU")
    print("=" * 60)

    # 1. Varlık Bazlı Ortalama, Minimum ve Maksimum Ağırlıklar
    print("\n1️⃣ Varlık Ağırlık İstatistikleri (Test Dönemi Boyunca):")
    stats = pd.DataFrame({
        "Ortalama (%)": (weights_df.mean() * 100).round(2),
        "Std Sapma (%)": (weights_df.std() * 100).round(2),
        "Min (%)": (weights_df.min() * 100).round(2),
        "Max (%)": (weights_df.max() * 100).round(2),
    })
    print(stats.to_string())

    # 2. Çeşitlendirme Skoru (Efektif Varlık Sayısı = 1 / sum(w^2))
    # Tamamen tek hisseye kilitlenirse = 1.0, 6 varlığa eşit dağılırsa = 6.0 çıkar
    hhi_per_day = (weights_df ** 2).sum(axis=1)
    effective_assets = (1.0 / hhi_per_day).mean()
    num_total_assets = len(weight_cols)

    print(f"\n2️⃣ Çeşitlendirme Skoru (Efektif Varlık Sayısı): {effective_assets:.2f} / {num_total_assets}")
    if effective_assets < 1.5:
        print("   ⚠️ UYARI: Ajan tek bir varlığa kilitlenmiş (Policy Collapse)! ent_coef değerini artırın.")
    elif effective_assets > num_total_assets * 0.85:
        print("   ℹ️ BİLGİ: Ajan neredeyse eşit ağırlıklı (1/N) davranıyor. Eğitim adımını (timesteps) artırın.")
    else:
        print("   ✅ BAŞARILI: Ajan dinamik ve sağlıklı bir portföy çeşitlendirmesi yapıyor.")

    # 3. Günlük Ortalama Turnover (İşlem Sıklığı ve Hacmi)
    daily_turnover = weights_df.diff().abs().sum(axis=1).dropna()
    avg_turnover = daily_turnover.mean() * 100
    active_rebalance_days = (daily_turnover > 0.05).mean() * 100  # %5 üzeri değişim olan günler

    print(f"\n3️⃣ İşlem Sıklığı (Turnover) Analizi:")
    print(f"   - Günlük Ortalama Portföy Değişimi (Turnover): %{avg_turnover:.2f}")
    print(f"   - Aktif Yeniden Dengeleme Yapılan Gün Oranı  : %{active_rebalance_days:.1f}")

    if avg_turnover > 60:
        print("   ⚠️ UYARI: Ajan aşırı sık al-sat yapıyor (komisyon maliyeti yüksek). Ödül fonksiyonundaki komisyon cezasını artırın.")
    elif avg_turnover < 1:
        print("   ⚠️ UYARI: Ajan hiç işlem yapmıyor (donmuş politika). Öğrenme oranını (learning_rate) gözden geçirin.")
    else:
        print("   ✅ BAŞARILI: İşlem sıklığı komisyon maliyetleriyle dengeli.")

    # 4. Ağırlıkların Zaman İçindeki Değişimi (İlk 5 ve Son 5 Gün)
    print("\n4️⃣ İlk 3 Gün vs Son 3 Gün Portföy Dağılımı:")
    preview = pd.concat([df[["Date"] + weight_cols].head(3), df[["Date"] + weight_cols].tail(3)])
    print(preview.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    analyze_agent_behavior()