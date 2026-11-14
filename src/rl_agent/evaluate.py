"""
Backtest / Değerlendirme Betiği (v2)

v1'e göre değişiklikler:
  * Ortam, eğitimde kaydedilen train_config.json'dan kurulur (eğitim ve değerlendirme ortamı birebir aynı).
  * Model/seed seçimi: --seed verilmezse models/training_summary.json'daki en iyi seed (val Sharpe) kullanılır.
  * Tarih doğrudan ortamdan (info["date"]) alınır; tahmin yok.
  * Çıktı docs/ARAYUZ_SOZLESMESI.md §3 şemasındadır: NET Daily_Return (komisyon dahil), W_* ağırlıkları,
    ek olarak Turnover ve Commission. Dosya yazıldıktan sonra doğrulayıcıyla otomatik denetlenir.
  * Metrikler ve eşit ağırlıklı al-ve-tut referansı ekrana basılır; özet data/results/<yöntem>_backtest_summary.json'a yazılır.

Kullanım (proje kök dizininden):
    python src/rl_agent/evaluate.py                       # test seti, en iyi seed, en iyi kontrol noktası
    python src/rl_agent/evaluate.py --split val           # val seti (ayar yaparken)
    python src/rl_agent/evaluate.py --seed 43 --checkpoint last

DİKKAT: Test seti yalnızca ayarlar bittikten sonra, final raporlama için kullanılmalıdır.
"""
import argparse
import json
import os

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env import PortfolioEnv
from market_data_validator import INITIAL_BALANCE, load_and_validate_data, validate_results_file
from train import buy_and_hold_reference, compute_metrics


def resolve_seed(model_dir: str, seed):
    if seed is not None:
        return seed
    summary_path = os.path.join(model_dir, "training_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"{summary_path} bulunamadı. Önce eğitim yapın (python src/rl_agent/train.py) veya --seed verin."
        )
    with open(summary_path, encoding="utf-8") as fh:
        return int(json.load(fh)["best_seed"])


def evaluate_agent(
    data_dir: str = "data/processed",
    model_dir: str = "models",
    split: str = "test",
    seed=None,
    checkpoint: str = "best",
    method: str = "rl",
    results_dir: str = "data/results",
):
    os.makedirs(results_dir, exist_ok=True)
    seed = resolve_seed(model_dir, seed)
    seed_dir = os.path.join(model_dir, f"seed_{seed}")

    with open(os.path.join(seed_dir, "train_config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    env_params = cfg["env_params"]

    # 1. Veriyi yükle ve doğrula
    prices_df, features_df = load_and_validate_data(
        f"{data_dir}/{split}_prices.csv", f"{data_dir}/{split}_features.csv"
    )

    # 2-3. Ortamı eğitimdekiyle aynı ayarlarla kur, eğitimdeki normalizasyon istatistiklerini (güncelleme yok)
    #      ve modeli yükle. Veri boyutu eğitimdekiyle uyuşmuyorsa anlaşılır hata ver.
    raw_env = PortfolioEnv(prices_df, features_df, **env_params)
    try:
        vec_env = VecNormalize.load(
            os.path.join(seed_dir, f"{checkpoint}_vecnormalize.pkl"), DummyVecEnv([lambda: raw_env])
        )
        vec_env.training = False
        vec_env.norm_reward = False
        model = PPO.load(os.path.join(seed_dir, f"{checkpoint}_model"), env=vec_env, device="cpu")
    except (ValueError, AssertionError) as exc:
        raise SystemExit(
            "HATA: Model, bu veriyle uyumsuz (hisse veya özellik sayısı eğitimdekinden farklı).\n"
            "Veri değiştiyse modeli yeniden eğitin: python src/rl_agent/train.py\n"
            f"Ayrıntı: {exc}"
        )

    # 4. Deterministik backtest
    obs = vec_env.reset()
    records, rets, pvs, turns = [], [], [], []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = vec_env.step(action)
        info = infos[0]

        row = {
            "Date": info["date"],
            "Portfolio_Value": round(info["portfolio_value"], 2),
            "Daily_Return": round(info["step_return"], 8),  # NET getiri (komisyon dahil)
        }
        for name, w in zip(raw_env.asset_names, info["weights"][:-1]):
            row[f"W_{name}"] = round(float(w), 6)
        row["W_CASH"] = round(float(info["weights"][-1]), 6)
        row["Turnover"] = round(info["turnover"], 6)
        row["Commission"] = round(info["commission_cost"], 4)
        records.append(row)

        rets.append(info["step_return"])
        pvs.append(info["portfolio_value"])
        turns.append(info["turnover"])
        if dones[0]:
            break

    # 5. Kaydet ve sözleşmeye uygunluğu denetle
    out_csv = os.path.join(results_dir, f"{method}_backtest_results.csv")
    results_df = pd.DataFrame(records)
    results_df.to_csv(out_csv, index=False)
    validate_results_file(out_csv)

    # 6. Metrikler + referans
    m = compute_metrics(rets, pvs, turns)
    m["total_commission"] = float(results_df["Commission"].sum())
    ref = buy_and_hold_reference(prices_df, env_params["window_size"], env_params["commission_rate"])

    print(f"\n📊 Backtest ({split}, seed {seed}, {checkpoint} kontrol noktası) — "
          f"{results_df['Date'].iloc[0]} → {results_df['Date'].iloc[-1]}")
    print(f"   RL ajanı         : {INITIAL_BALANCE:,.0f} → {m['final_value']:,.2f} | getiri %{m['total_return_pct']:+.2f} | "
          f"Sharpe {m['sharpe']:+.2f} | maxDD %{m['max_drawdown_pct']:.1f} | ort. turnover {m['avg_turnover']:.3f} | "
          f"toplam komisyon {m['total_commission']:,.0f}")
    print(f"   Eşit ağırlık B&H : getiri %{ref['total_return_pct']:+.2f} | Sharpe {ref['sharpe']:+.2f} | "
          f"maxDD %{ref['max_drawdown_pct']:.1f}")

    summary = {"split": split, "seed": seed, "checkpoint": checkpoint, "method": method,
               "first_date": results_df["Date"].iloc[0], "last_date": results_df["Date"].iloc[-1],
               "metrics": m, "buy_and_hold_reference": ref, "env_params": env_params}
    with open(os.path.join(results_dir, f"{method}_backtest_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"✅ Sonuçlar kaydedildi: {out_csv}")
    return results_df, summary


def main():
    ap = argparse.ArgumentParser(description="Eğitilmiş PPO ajanını backtest et")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--seed", type=int, default=None, help="Verilmezse training_summary.json'daki en iyi seed")
    ap.add_argument("--checkpoint", choices=["best", "last"], default="best")
    ap.add_argument("--method", default="rl", help="Çıktı dosya öneki: <method>_backtest_results.csv")
    ap.add_argument("--results-dir", default="data/results")
    a = ap.parse_args()
    evaluate_agent(a.data_dir, a.model_dir, a.split, a.seed, a.checkpoint, a.method, a.results_dir)


if __name__ == "__main__":
    main()