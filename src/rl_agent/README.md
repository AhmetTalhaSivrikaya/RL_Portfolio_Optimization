# RL Ajanı (`src/rl_agent/`)

Gymnasium + Stable-Baselines3 (PPO) tabanlı portföy ajanı. Hisse senetleri ve nakit arasında günlük ağırlık kararları verir.

## Dosyalar

| Dosya | Görev |
|---|---|
| `env.py` | `PortfolioEnv`: borsa simülasyon ortamı |
| `train.py` | PPO eğitimi (çoklu seed, validation ile kontrol noktası seçimi) |
| `evaluate.py` | Deterministik backtest, sözleşme şemasında CSV üretir |
| `market_data_validator.py` | Veri ve sonuç dosyalarının sözleşmeye uygunluk denetimi |
| `dummy_data_generator.py` | Sözleşmeye uygun sentetik veri (gürültü veya `--signal` ile öğrenilebilir sinyal) |
| `analyze_behavior.py` | Ağırlık dağılımı ve çeşitlendirme tanılaması |
| `test_env.py` | Ortam birim testleri |

## Ortam tasarımı

- **Eylem:** `softmax(skor × action_scale)` ile toplamı 1 olan hedef ağırlıklar (hisseler + nakit).
- **Gözlem:** Son `window_size` (10) günün özellikleri + mevcut ağırlıklar.
- **Ödül:** `100 × log(1 + NET getiri) − drawdown cezası − turnover cezası`.
- **Komisyon:** `portföy_değeri × turnover × %0.1`; turnover = hisse bacaklarının `Σ|Δw|`'si (nakit bacağı ücretsiz, rotasyon = 2.0).
- **Rebalance:** `rebalance_freq = N` ile yalnızca her N. adımda işlem yapılır (CLI varsayılanı 5, yani haftalık). Ara günlerde komisyon yoktur.
- **Bölüm sonu:** Veri sonu `truncated`, sermayenin %80'ini kaybetmek `terminated`.

## Çalıştırma (proje kökünden)

```bash
python src/rl_agent/dummy_data_generator.py
python src/rl_agent/train.py --seeds 42 43 44 --timesteps 200000 --action-scale 1.2 --turnover-penalty 0.25 --rebalance-threshold 0.1 --rebalance-freq 5 --ent-coef 0.005 --log-std-init -0.5
python src/rl_agent/evaluate.py
python src/rl_agent/market_data_validator.py
pytest
```

## Bilinen sonuçlar ve sınırlamalar (dürüst özet)

- **Şu ana kadarki tüm sonuçlar SENTETİK veriyle elde edildi.** Strateji başarısı hakkında kanıt değildir; yalnızca hattın çalıştığını gösterir.
- **Churn sorunu:** Varsayılan ayarlarda ajan portföyü neredeyse her gün baştan kurar (günlük turnover ~1.4) ve komisyon sermayenin yarısını eritir.
  Haftalık rebalance + düşük `action_scale` + yüksek `turnover_penalty` bunu çözer (turnover ~0.06–0.09).
- **Sinyal öğrenme:** Çok güçlü sinyalli veride ajan öğrenir. Zayıf sinyalde (gerçeğe daha yakın) 3 seed ile sinyali güvenilir biçimde bulduğu gösterilemedi.
- **Model seçimi gürültülü:** 1 yıllık val dönemindeki yıllıklandırılmış Sharpe tahmininin standart hatası ≈ ±1.0 (2 yıllık test için ≈ ±0.7).
  "En iyi val Sharpe'lı seed" seçimi bu nedenle büyük ölçüde rastgeledir. Sonuçlar çok sayıda seed ile ortalama ± std olarak raporlanmalıdır.
- Gerçek BIST verisiyle al-ve-tut ve Markowitz'i komisyon sonrası geçmek garanti değildir; sonuç ne olursa olsun dürüstçe raporlanmalıdır.