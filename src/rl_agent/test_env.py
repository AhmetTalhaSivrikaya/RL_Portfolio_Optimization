import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env import PortfolioEnv

# 1. Sahte (Dummy) Veri Oluşturma (1. kişiyi beklemeden test için)
np.random.seed(42)
days = 500
prices_df = pd.DataFrame(
    np.cumprod(1 + np.random.normal(0.0005, 0.02, (days, 5)), axis=0) * 100,
    columns=["HISSE_1", "HISSE_2", "HISSE_3", "HISSE_4", "HISSE_5"],
)
# Normalize edilmiş 10 adet sahte teknik indikatör sütunu
features_df = pd.DataFrame(np.random.randn(days, 10))

# 2. Ortamı oluştur ve Gymnasium standartlarına uygunluğunu denetle
env = PortfolioEnv(prices_df=prices_df, features_df=features_df)
check_env(env, warn=True)
print("✅ Ortam Gymnasium denetiminden başarıyla geçti!")

# 3. PPO Ajanı ile kısa bir eğitim testi yap
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)
print("✅ PPO ajanı sorunsuz şekilde eğitildi!")