import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class PortfolioEnv(gym.Env):
    """
    Dinamik Portföy Optimizasyonu için Özel Gymnasium Ortamı.
    Eylem: N adet hisse + 1 adet Nakit için portföy ağırlıkları (toplamı 1).
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
        rebalance_threshold: float = 0.05,  # %5 altı ağırlık değişiminde işlem yapma (Deadband)
        turnover_penalty: float = 0.15,  # Aşırı al-sat ceza katsayısı
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
        self.initial_balance = float(initial_balance)
        self.commission_rate = float(commission_rate)
        self.window_size = int(window_size)
        self.drawdown_threshold = float(drawdown_threshold)
        self.rebalance_threshold = float(rebalance_threshold)
        self.turnover_penalty = float(turnover_penalty)

        self.num_assets = self.prices_df.shape[1]
        self.num_actions = self.num_assets + 1  # Hisseler + Nakit

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

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _get_observation(self) -> np.ndarray:
        # Son window_size günlük özellikleri düzleştir
        window_features = self.features_df.iloc[
            self.current_step - self.window_size : self.current_step
        ].values.flatten()

        # Mevcut ağırlıkları sonuna ekle
        obs = np.concatenate([window_features, self.weights]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.portfolio_value = self.initial_balance
        self.peak_value = self.initial_balance

        # Başlangıçta %100 Nakit (son eleman nakit pozisyonu)
        self.weights = np.zeros(self.num_actions, dtype=np.float32)
        self.weights[-1] = 1.0

        self.returns_history = []
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        # 1. Ham eylemi toplamı 1 olan portföy ağırlıklarına çevir
        target_weights = self._softmax(action)

        # 2. İşlem maliyeti (Komisyon) ve Ölü Bölge (Deadband) kontrolü
        turnover = float(np.sum(np.abs(target_weights - self.weights)))

        if turnover < self.rebalance_threshold:
            target_weights = self.weights.copy()
            turnover = 0.0

        commission_cost = self.portfolio_value * turnover * self.commission_rate
        self.portfolio_value = float(self.portfolio_value - commission_cost)

        # 3. Bir sonraki güne geç ve hisse getirilerini hesapla
        prev_prices = self.prices_df.iloc[self.current_step].values
        self.current_step += 1
        curr_prices = self.prices_df.iloc[self.current_step].values

        # Nakit getirisi 0 kabul edilir
        asset_returns = (curr_prices - prev_prices) / (prev_prices + 1e-8)
        all_returns = np.append(asset_returns, 0.0)

        # 4. Portföy değerini güncelle
        step_return = float(np.dot(target_weights, all_returns))
        self.portfolio_value = float(self.portfolio_value * (1.0 + step_return))
        self.returns_history.append(step_return)

        # Fiyat değişiminden sonra ağırlıkların doğal kaymasını güncelle
        updated_values = target_weights * (1.0 + all_returns)
        self.weights = (updated_values / np.sum(updated_values)).astype(np.float32)

        # 5. Ödül Fonksiyonu (Net Getiri / Sharpe - Drawdown Cezası - Turnover Cezası)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = float((self.peak_value - self.portfolio_value) / self.peak_value)

        # 5.1 Temel risk-ayarlı ödülü hesapla
        if len(self.returns_history) >= 20:
            recent_returns = np.array(self.returns_history[-20:])
            volatility = np.std(recent_returns) + 1e-6
            reward = (step_return / volatility) * 0.1
        else:
            reward = step_return * 10.0

        # 5.2 Kritik eşik aşılırsa Drawdown cezası düş
        if drawdown > self.drawdown_threshold:
            reward -= (drawdown - self.drawdown_threshold) * 5.0

        # 5.3 Aşırı al-sat (Turnover) cezası düş
        reward -= turnover * self.turnover_penalty

        # 6. Bölüm bitti mi kontrolü (Saf Python bool)
        terminated = bool(self.current_step >= len(self.prices_df) - 1)
        truncated = bool(self.portfolio_value <= self.initial_balance * 0.2)

        info = {
            "portfolio_value": self.portfolio_value,
            "drawdown": drawdown,
            "weights": self.weights.copy(),
            "step_return": step_return,
            "turnover": turnover,
        }

        return self._get_observation(), float(reward), terminated, truncated, info