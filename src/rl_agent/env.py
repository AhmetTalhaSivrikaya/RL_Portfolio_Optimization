from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class PortfolioEnv(gym.Env):
    """
    Dinamik Portföy Optimizasyonu için Özel Gymnasium Ortamı (v2).

    Eylem  : N adet hisse + 1 adet Nakit için ham skorlar -> softmax -> portföy ağırlıkları (toplam = 1).
    Gözlem : Son `window_size` günün özellikleri (düzleştirilmiş) + mevcut portföy ağırlıkları.
    Ödül   : Net (komisyon sonrası) getirinin ortalama-varyans faydası - drawdown cezası - küçük turnover cezası.

    v1'e göre değişiklikler:
      * step_return artık komisyon DAHİL net getiridir (Portfolio_Value ile tutarlı).
      * action_scale: softmax öncesi ölçekleme. Önceki sürümde [-1,1] aralığı tek varlığı
        en fazla %59.6 ile sınırlıyordu; action_scale=5 ile ~%99.5'e çıkar.
      * Nakit getirisi parametrik (cash_daily_return).
      * Ödül: gürültülü getiri/volatilite oranı yerine ortalama-varyans faydası.
      * Turnover cezası komisyonla uyumlu hale getirildi (küçük düzenleyici terim).
      * Veri sonu `truncated`, iflas (<%20) `terminated` olarak düzeltildi
        (SB3 değer fonksiyonu bootstrap'i için doğru yorum).
      * Eğitimde rastgele başlangıç + sabit bölüm uzunluğu (episode_length) ile ezber riski azaltıldı.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices_df: pd.DataFrame,
        features_df: pd.DataFrame,
        initial_balance: float = 100_000.0,
        commission_rate: float = 0.001,  # Binde 1 (%0.1) komisyon
        window_size: int = 10,
        drawdown_threshold: float = 0.15,  # %15 üzeri düşüşte ceza
        drawdown_penalty_coef: float = 10.0,
        rebalance_threshold: float = 0.05,  # Bu değerin altındaki toplam ağırlık değişiminde işlem yapma (deadband)
        turnover_penalty: float = 0.01,  # Küçük düzenleyici terim (komisyon zaten net getiriye yansıyor)
        risk_aversion: float = 0.1,  # Ortalama-varyans faydasında risk kaçınma katsayısı
        action_scale: float = 5.0,  # softmax(action * action_scale)
        cash_daily_return: float = 0.0,  # Günlük nakit getirisi (örn. yıllık %40 faiz için ~0.40/252)
        random_start: bool = False,  # True: reset'te rastgele başlangıç noktası (EĞİTİM için)
        episode_length: Optional[int] = None,  # None: verinin sonuna kadar
    ):
        super().__init__()

        # Tarih indeksini ve varlık isimlerini sakla (Backtest raporu için)
        self.dates = (
            prices_df.index.astype(str).tolist()
            if isinstance(prices_df.index, pd.DatetimeIndex)
            else [str(i) for i in range(len(prices_df))]
        )
        self.asset_names = list(prices_df.columns)

        self.prices_df = prices_df.reset_index(drop=True)
        self.features_df = features_df.reset_index(drop=True)
        self._prices = self.prices_df.values.astype(np.float64)
        self._features = self.features_df.values.astype(np.float32)

        self.initial_balance = float(initial_balance)
        self.commission_rate = float(commission_rate)
        self.window_size = int(window_size)
        self.drawdown_threshold = float(drawdown_threshold)
        self.drawdown_penalty_coef = float(drawdown_penalty_coef)
        self.rebalance_threshold = float(rebalance_threshold)
        self.turnover_penalty = float(turnover_penalty)
        self.risk_aversion = float(risk_aversion)
        self.action_scale = float(action_scale)
        self.cash_daily_return = float(cash_daily_return)
        self.random_start = bool(random_start)
        self.episode_length = None if episode_length is None else int(episode_length)

        self.num_assets = self.prices_df.shape[1]
        self.num_actions = self.num_assets + 1  # Hisseler + Nakit

        max_steps = len(self.prices_df) - 1 - self.window_size
        if max_steps < 2:
            raise ValueError("Veri, window_size için fazla kısa.")
        if self.episode_length is not None and self.episode_length > max_steps:
            raise ValueError(
                f"episode_length ({self.episode_length}) veri uzunluğundan ({max_steps}) büyük olamaz."
            )

        # Eylem Uzayı: Her varlık için [-1, 1] arası ham skor (Softmax ile ağırlığa dönecek)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
        )

        # Gözlem Uzayı: (window_size * özellik_sayısı) + mevcut_portföy_ağırlıkları
        num_features = self.features_df.shape[1]
        obs_dim = (self.window_size * num_features) + self.num_actions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.reset()

    # ------------------------------------------------------------------ yardımcılar
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        z = np.asarray(x, dtype=np.float64) * self.action_scale
        e_x = np.exp(z - np.max(z))
        return e_x / e_x.sum()

    def _get_observation(self) -> np.ndarray:
        window_features = self._features[
            self.current_step - self.window_size : self.current_step
        ].flatten()
        return np.concatenate([window_features, self.weights]).astype(np.float32)

    # ------------------------------------------------------------------ Gymnasium API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        last_valid_step = len(self.prices_df) - 1  # son kullanılabilir fiyat indeksi
        if self.random_start and self.episode_length is not None:
            latest_start = last_valid_step - self.episode_length
            self.current_step = int(
                self.np_random.integers(self.window_size, latest_start + 1)
            )
            self.end_step = self.current_step + self.episode_length
        else:
            self.current_step = self.window_size
            self.end_step = (
                self.current_step + self.episode_length
                if self.episode_length is not None
                else last_valid_step
            )

        self.portfolio_value = self.initial_balance
        self.peak_value = self.initial_balance

        # Başlangıçta %100 Nakit (son eleman nakit pozisyonu)
        self.weights = np.zeros(self.num_actions, dtype=np.float32)
        self.weights[-1] = 1.0

        self.returns_history = []
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        # 1. Ham eylemi toplamı 1 olan hedef ağırlıklara çevir
        target_weights = self._softmax(action)

        # 2. Deadband: çok küçük ayarlamalarda işlem yapma
        turnover = float(np.sum(np.abs(target_weights - self.weights)))
        if turnover < self.rebalance_threshold:
            target_weights = self.weights.astype(np.float64).copy()
            turnover = 0.0

        # 3. Komisyon (portföy değerinden düşülür)
        value_before = self.portfolio_value
        commission_cost = value_before * turnover * self.commission_rate
        value_after_cost = value_before - commission_cost

        # 4. Bir sonraki güne geç ve getirileri hesapla
        prev_prices = self._prices[self.current_step]
        self.current_step += 1
        curr_prices = self._prices[self.current_step]

        asset_returns = (curr_prices - prev_prices) / (prev_prices + 1e-8)
        all_returns = np.append(asset_returns, self.cash_daily_return)

        gross_return = float(np.dot(target_weights, all_returns))
        self.portfolio_value = float(value_after_cost * (1.0 + gross_return))

        # NET getiri: komisyon dahil. Portfolio_Value ile birebir tutarlıdır.
        step_return = float(self.portfolio_value / value_before - 1.0)
        self.returns_history.append(step_return)

        # 5. Fiyat değişiminden sonra ağırlıkların doğal kaymasını güncelle
        updated_values = target_weights * (1.0 + all_returns)
        self.weights = (updated_values / np.sum(updated_values)).astype(np.float32)

        # 6. Ödül: ortalama-varyans faydası (getiri % cinsinden) - drawdown - turnover
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = float((self.peak_value - self.portfolio_value) / self.peak_value)

        r_pct = step_return * 100.0
        reward = r_pct - 0.5 * self.risk_aversion * (r_pct ** 2)

        if drawdown > self.drawdown_threshold:
            reward -= (drawdown - self.drawdown_threshold) * self.drawdown_penalty_coef

        reward -= turnover * self.turnover_penalty

        # 7. Bölüm sonu: iflas -> terminated, veri/bölüm sonu -> truncated
        terminated = bool(self.portfolio_value <= self.initial_balance * 0.2)
        truncated = bool(self.current_step >= self.end_step)

        info = {
            "date": self.dates[self.current_step],
            "portfolio_value": self.portfolio_value,
            "drawdown": drawdown,
            "weights": self.weights.copy(),
            "step_return": step_return,  # net (komisyon dahil)
            "gross_return": gross_return,
            "commission_cost": float(commission_cost),
            "turnover": turnover,
        }

        return self._get_observation(), float(reward), terminated, truncated, info