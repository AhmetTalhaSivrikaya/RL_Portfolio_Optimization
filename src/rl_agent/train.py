"""
PPO Eğitim Betiği (v2)

v1'e göre değişiklikler:
  * Komut satırı argümanları (veri yolu, seed'ler, adım sayısı, ortam parametreleri).
  * Eğitim ortamı rastgele başlangıç + sabit bölüm uzunluğu kullanır (tek patikayı ezberlemeyi azaltır).
  * Periyodik olarak VALIDATION setinde (val_*.csv) değerlendirilir; en iyi model val Sharpe'ına göre seçilir.
  * Çoklu seed: her seed ayrı klasöre kaydedilir, özet tablo ve en iyi seed models/training_summary.json'a yazılır.
  * Eğitimde kullanılan ortam ayarları train_config.json olarak kaydedilir (evaluate.py aynı ortamı kurar).

Kullanım (proje kök dizininden):
    python src/rl_agent/train.py                                  # varsayılan: seed 42
    python src/rl_agent/train.py --seeds 42 43 44 --timesteps 200000
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization

from env import PortfolioEnv
from market_data_validator import INITIAL_BALANCE, load_and_validate_data, validate_split_continuity

TRADING_DAYS = 252


# ----------------------------------------------------------------------------------
# Metrikler / değerlendirme yardımcıları
# ----------------------------------------------------------------------------------
def compute_metrics(step_returns, portfolio_values, turnovers=None):
    r = np.asarray(step_returns, dtype=np.float64)
    pv = np.asarray(portfolio_values, dtype=np.float64)
    std = r.std()
    sharpe = float(r.mean() / std * np.sqrt(TRADING_DAYS)) if std > 1e-12 else 0.0
    peak = np.maximum.accumulate(np.concatenate([[INITIAL_BALANCE], pv]))[1:]
    return {
        "final_value": float(pv[-1]),
        "total_return_pct": float((pv[-1] / INITIAL_BALANCE - 1.0) * 100.0),
        "sharpe": sharpe,
        "max_drawdown_pct": float(((pv / peak) - 1.0).min() * 100.0),
        "avg_turnover": float(np.mean(turnovers)) if turnovers is not None and len(turnovers) else 0.0,
        "steps": int(len(r)),
    }


def run_episode(model, vec_env):
    """Deterministik tek bölüm çalıştırır (val/test değerlendirmesi). Net getiri üzerinden metrik döndürür."""
    obs = vec_env.reset()
    rets, pvs, turns = [], [], []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = vec_env.step(action)
        info = infos[0]
        rets.append(info["step_return"])
        pvs.append(info["portfolio_value"])
        turns.append(info["turnover"])
        if dones[0]:
            break
    return compute_metrics(rets, pvs, turns)


def buy_and_hold_reference(prices_df, window_size, commission_rate):
    """Karşılaştırma için: eşit ağırlıklı al-ve-tut (tek komisyon, nakitsiz). Ortamla aynı başlangıç gününden."""
    p = prices_df.values[window_size:]
    growth = (p / p[0]).mean(axis=1)  # eşit ağırlıklı, rebalance yok
    pv = INITIAL_BALANCE * (1.0 - commission_rate) * growth[1:]
    rets = np.diff(np.concatenate([[INITIAL_BALANCE], pv])) / np.concatenate([[INITIAL_BALANCE], pv[:-1]])
    return compute_metrics(rets, pv)


class ValidationCallback(BaseCallback):
    """Her `eval_freq` adımda val setinde değerlendirir; en iyi val Sharpe'lı modeli kaydeder."""

    def __init__(self, eval_vec, eval_freq, out_dir):
        super().__init__(verbose=0)
        self.eval_vec = eval_vec
        self.eval_freq = eval_freq
        self.out_dir = out_dir
        self.best_sharpe = -np.inf
        self.best_step = 0
        self.last_eval = 0
        self.history = []

    def _evaluate(self):
        # Eğitim ortamının obs normalizasyon istatistiklerini val ortamına kopyala
        sync_envs_normalization(self.training_env, self.eval_vec)
        m = run_episode(self.model, self.eval_vec)
        m["timesteps"] = int(self.num_timesteps)
        self.history.append(m)
        mark = ""
        if m["sharpe"] > self.best_sharpe:
            self.best_sharpe, self.best_step = m["sharpe"], m["timesteps"]
            self.model.save(os.path.join(self.out_dir, "best_model"))
            self.eval_vec.save(os.path.join(self.out_dir, "best_vecnormalize.pkl"))
            mark = "  ⭐ en iyi"
        print(
            f"   [val @ {m['timesteps']:>8,}] getiri %{m['total_return_pct']:+7.2f} | "
            f"Sharpe {m['sharpe']:+.2f} | maxDD %{m['max_drawdown_pct']:.1f} | "
            f"ort. turnover {m['avg_turnover']:.3f}{mark}"
        )

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            self.last_eval = self.num_timesteps
            self._evaluate()
        return True

    def _on_training_end(self) -> None:
        if self.last_eval != self.num_timesteps:
            self._evaluate()


# ----------------------------------------------------------------------------------
# Eğitim
# ----------------------------------------------------------------------------------
def train_one_seed(seed, train_data, val_data, env_params, args):
    (train_p, train_f), (val_p, val_f) = train_data, val_data
    out_dir = os.path.join(args.model_dir, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    max_ep = len(train_p) - 1 - env_params["window_size"]
    episode_length = args.episode_length if args.episode_length > 0 else None
    if episode_length is not None and episode_length > max_ep:
        print(f"⚠️ episode_length ({episode_length}) veri uzunluğunu aşıyor, {max_ep} olarak ayarlandı.")
        episode_length = max_ep

    # Eğitim ortamı: rastgele başlangıç + sabit uzunlukta bölümler
    train_vec = VecNormalize(
        DummyVecEnv([lambda: PortfolioEnv(train_p, train_f, random_start=episode_length is not None,
                                          episode_length=episode_length, **env_params)]),
        norm_obs=True, norm_reward=True, clip_obs=10.0,
    )
    train_vec.seed(seed)

    # Validation ortamı: tüm val dönemi, deterministik başlangıç, istatistikler güncellenmez
    eval_vec = VecNormalize(
        DummyVecEnv([lambda: PortfolioEnv(val_p, val_f, **env_params)]),
        training=False, norm_obs=True, norm_reward=False, clip_obs=10.0,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_vec,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        policy_kwargs=dict(log_std_init=args.log_std_init),
        verbose=0,
        seed=seed,
        device="cpu",
    )

    callback = ValidationCallback(eval_vec, eval_freq=args.eval_freq, out_dir=out_dir)
    print(f"\n=== Seed {seed}: {args.timesteps:,} adım eğitim başlıyor "
          f"(bölüm uzunluğu: {episode_length or 'tüm veri'}) ===")
    model.learn(total_timesteps=args.timesteps, callback=callback)

    model.save(os.path.join(out_dir, "last_model"))
    train_vec.save(os.path.join(out_dir, "last_vecnormalize.pkl"))
    pd.DataFrame(callback.history).to_csv(os.path.join(out_dir, "val_log.csv"), index=False)

    config = {"seed": seed, "env_params": env_params, "episode_length": episode_length,
              "timesteps": args.timesteps, "learning_rate": args.learning_rate, "ent_coef": args.ent_coef,
              "log_std_init": args.log_std_init, "gamma": args.gamma, "batch_size": args.batch_size,
              "n_epochs": args.n_epochs}
    with open(os.path.join(out_dir, "train_config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)

    best = next(h for h in callback.history if h["timesteps"] == callback.best_step)
    return {"seed": seed, "best_step": callback.best_step, **{f"val_{k}": v for k, v in best.items() if k != "timesteps"}}


def main():
    ap = argparse.ArgumentParser(description="PPO portföy ajanı eğitimi")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--eval-freq", type=int, default=10_000, help="Kaç adımda bir val değerlendirmesi")
    ap.add_argument("--episode-length", type=int, default=250, help="Eğitim bölüm uzunluğu (0: tüm veri)")
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--log-std-init", type=float, default=0.0, help="Başlangıç keşif gürültüsü: std = exp(değer)")
    # Ortam parametreleri (docs/ARAYUZ_SOZLESMESI.md §1 ile uyumlu varsayılanlar)
    ap.add_argument("--window-size", type=int, default=10)
    ap.add_argument("--commission", type=float, default=0.001)
    ap.add_argument("--action-scale", type=float, default=5.0)
    ap.add_argument("--cash-daily-return", type=float, default=0.0)
    ap.add_argument("--turnover-penalty", type=float, default=0.01)
    ap.add_argument("--rebalance-threshold", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    d = args.data_dir
    train_data = load_and_validate_data(f"{d}/train_prices.csv", f"{d}/train_features.csv")
    val_data = load_and_validate_data(f"{d}/val_prices.csv", f"{d}/val_features.csv")
    validate_split_continuity({"train": train_data, "val": val_data})

    env_params = {
        "window_size": args.window_size,
        "commission_rate": args.commission,
        "action_scale": args.action_scale,
        "cash_daily_return": args.cash_daily_return,
        "turnover_penalty": args.turnover_penalty,
        "rebalance_threshold": args.rebalance_threshold,
    }

    bh = buy_and_hold_reference(val_data[0], args.window_size, args.commission)
    print(f"\n📏 Referans (val, eşit ağırlık al-ve-tut): getiri %{bh['total_return_pct']:+.2f} | "
          f"Sharpe {bh['sharpe']:+.2f} | maxDD %{bh['max_drawdown_pct']:.1f}")

    results = [train_one_seed(s, train_data, val_data, env_params, args) for s in args.seeds]

    summary = pd.DataFrame(results).set_index("seed")
    best_seed = int(summary["val_sharpe"].idxmax())
    print("\n" + "=" * 70 + "\nSeed özeti (validation seti, en iyi kontrol noktası):")
    print(summary[["best_step", "val_total_return_pct", "val_sharpe", "val_max_drawdown_pct"]].round(2).to_string())
    print(f"\n🏆 En iyi seed (val Sharpe): {best_seed}  |  Referans al-ve-tut Sharpe: {bh['sharpe']:+.2f}")
    if len(args.seeds) > 1:
        print(f"   Seed'ler arası val Sharpe: ort. {summary['val_sharpe'].mean():+.2f} ± {summary['val_sharpe'].std():.2f}")

    with open(os.path.join(args.model_dir, "training_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"best_seed": best_seed, "buy_and_hold_val_reference": bh,
                   "seeds": results, "env_params": env_params}, fh, indent=2, ensure_ascii=False)
    print(f"✅ Kaydedildi: {args.model_dir}/seed_*/ ve {args.model_dir}/training_summary.json")


if __name__ == "__main__":
    main()