import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import PortfolioEnv
from market_data_validator import load_and_validate_data

def evaluate_agent(
    prices_path: str = "data/processed/test_prices.csv",
    features_path: str = "data/processed/test_features.csv",
    model_path: str = "models/ppo_portfolio_agent.zip",
    vecnorm_path: str = "models/vec_normalize.pkl",
    output_csv_path: str = "data/results/rl_backtest_results.csv",
):
    os.makedirs("data/results", exist_ok=True)

    # 1. Test verisini yükle ve doğrula
    prices_df, features_df = load_and_validate_data(prices_path, features_path)

    # 2. Test ortamını kur ve eğitimdeki VecNormalize istatistiklerini yükle
    raw_env = PortfolioEnv(prices_df=prices_df, features_df=features_df)
    base_env = DummyVecEnv([lambda: raw_env])
    env = VecNormalize.load(vecnorm_path, base_env)
    env.training = False      # Test sırasında istatistikleri güncelleme!
    env.norm_reward = False   # Gerçek ödülleri gör

    # 3. Eğitilmiş modeli yükle
    model = PPO.load(model_path, env=env)

    # 4. Test dönemi boyunca simülasyonu çalıştır
    obs = env.reset()
    records = []

    while True:
        # deterministic=True -> Rastgele keşif yok, öğrenilen en iyi politika
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        info = infos[0]
        # DummyVecEnv otomatik reset atmadan önceki gerçek adım bilgisini al
        step_idx = raw_env.current_step if not dones[0] else len(raw_env.dates) - 1
        current_date = raw_env.dates[step_idx]

        row = {
            "Date": current_date,
            "Portfolio_Value": round(info["portfolio_value"], 2),
            "Daily_Return": round(info["step_return"], 6),
        }

        # Her hissenin ve nakdin ağırlığını sütun olarak ekle
        weights = info["weights"]
        for i, asset in enumerate(raw_env.asset_names):
            row[f"W_{asset}"] = round(float(weights[i]), 4)
        row["W_CASH"] = round(float(weights[-1]), 4)

        records.append(row)

        if dones[0]:
            break

    # 5. Sonuçları CSV olarak kaydet
    results_df = pd.DataFrame(records)
    results_df.to_csv(output_csv_path, index=False)

    initial_val = raw_env.initial_balance
    final_val = results_df["Portfolio_Value"].iloc[-1]
    total_return = ((final_val - initial_val) / initial_val) * 100

    print(f"Backtest tamamlandı! Sonuçlar kaydedildi: {output_csv_path}")
    print(f"Başlangıç Bakiyesi: {initial_val:,.2f} -> Bitiş Bakiyesi: {final_val:,.2f} (%{total_return:.2f})")


if __name__ == "__main__":
    evaluate_agent()