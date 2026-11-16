# Baseline stratejileri

Proje kök dizininden çalıştırın (`data/processed/train_prices.csv` ve `<split>_prices.csv` hazır olmalı):

```bash
python -m src.baseline.portfolio_baselines                # test dönemi
python -m src.baseline.portfolio_baselines --split val    # val dönemi
```

Üretilen dosyalar (`data/results/`), RL değerlendiricisiyle **aynı şemadadır** (bkz. `docs/ARAYUZ_SOZLESMESI.md` §3):

| Dosya | Strateji |
|---|---|
| `markowitz_backtest_results.csv` | Eğitim verisinden bulunan maksimum-Sharpe ağırlıklarına **her gün** dönüş |
| `markowitz_weekly_backtest_results.csv` | Aynı ağırlıklar, **5 günde bir** dönüş (RL ile adil kıyas) |
| `buy_and_hold_backtest_results.csv` | Eşit ağırlıkla ilk gün al, sonra tut |

Kurallar: Markowitz ağırlıkları yalnızca kronolojik olarak önceki eğitim fiyatlarından tahmin edilir. Komisyon `portföy_değeri × turnover × %0.1`'dir;
turnover yalnızca hisse bacaklarının işlem hacmidir (nakit bacağı ücretsiz). `Daily_Return` **net** getiridir (komisyon dahil).
RL ortamıyla aynı ekonomi, `tests/test_contract.py` içinde birebir doğrulanır.