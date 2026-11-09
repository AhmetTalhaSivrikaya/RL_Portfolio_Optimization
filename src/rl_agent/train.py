import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import PortfolioEnv
from market_data_validator import load_and_validate_data


def train_agent(
    prices_path: str = "data/processed/train_prices.csv",
    features_path: str = "data/processed/train_features.csv",
    model_save_path: str = "models/ppo_portfolio_agent",
    vecnorm_save_path: str = "models/vec_normalize.pkl",
    total_timesteps: int = 100_000,
):
    os.makedirs("models", exist_ok=True)

    # 1. Veriyi yükle ve sözleşmeye uygunluğunu denetle
    prices_df, features_df = load_and_validate_data(prices_path, features_path)

    # 2. Ortamı oluştur ve VecNormalize ile sar (Eğitim stabilitesi için kritik)
    base_env = DummyVecEnv([lambda: PortfolioEnv(prices_df=prices_df, features_df=features_df)])
    env = VecNormalize(base_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 3. PPO Ajanını tanımla
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,  # Keşif (exploration) katsayısı: Tek hisseye kilitlenmeyi önler
        verbose=1,
        seed=42,
    )

    print(f" Eğitim başlıyor ({total_timesteps} adım)...")
    model.learn(total_timesteps=total_timesteps)

    # 4. Modeli ve normalizasyon istatistiklerini kaydet
    model.save(model_save_path)
    env.save(vecnorm_save_path)
    print(f"✅ Model başarıyla kaydedildi: {model_save_path}.zip")


if __name__ == "__main__":
    train_agent()