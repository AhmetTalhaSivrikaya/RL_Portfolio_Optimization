from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class PortfolioEnv(gym.Env):
    """
    Dinamik Portföy Optimizasyonu için Gymnasium ortamı.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices_df: pd.DataFrame,
        features_df: pd.DataFrame,
        initial_balance: float = 100_000.0,
        commission_rate: float = 0.001,
        window_size: int = 10,
        drawdown_threshold: float = 0.15,
        drawdown_penalty_coef: float = 10.0,
        rebalance_threshold: float = 0.05,
        rebalance_freq: int = 1,
        turnover_penalty: float = 0.01,
        risk_aversion: float = 0.1,
        action_scale: float = 5.0,
        cash_daily_return: float = 0.0,
        random_start: bool = False,
        episode_length: Optional[int] = None,
    ):
        super().__init__()

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
        self.rebalance_freq = max(1, int(rebalance_freq))
        self.turnover_penalty = float(turnover_penalty)

        self.risk_aversion = float(risk_aversion)
        self.action_scale = float(action_scale)

        self.cash_daily_return = float(cash_daily_return)
        self.random_start = bool(random_start)
        self.episode_length = None if episode_length is None else int(episode_length)

        self.num_assets = self.prices_df.shape[1]
        self.num_actions = self.num_assets + 1

        max_steps = len(self.prices_df) - 1 - self.window_size
        if max_steps < 2:
            raise ValueError("Veri, window_size için fazla kısa.")
        if len(self.features_df) != len(self.prices_df):
            raise ValueError("prices_df ve features_df aynı sayıda satıra sahip olmalıdır.")
        if self.episode_length is not None:
            if self.episode_length < 1:
                raise ValueError("episode_length en az 1 olmalıdır.")
            if self.episode_length > max_steps:
                raise ValueError(
                    f"episode_length ({self.episode_length}) veri uzunluğundan ({max_steps}) büyük olamaz."
                )
        if self.num_assets < 1:
            raise ValueError("En az bir varlık bulunmalıdır.")

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
        )

        num_features = self.features_df.shape[1]
        obs_dim = (self.window_size * num_features) + self.num_actions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.reset()

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        z = np.asarray(x, dtype=np.float64) * self.action_scale
        e_x = np.exp(z - np.max(z))
        return e_x / e_x.sum()

    def _calculate_turnover(self, old_weights: np.ndarray, new_weights: np.ndarray) -> float:
        return float(0.5 * np.sum(np.abs(new_weights - old_weights)))

    def _calculate_reward(self, step_return: float, drawdown: float, turnover: float) -> float:
        if step_return <= -1.0:
            reward = -100.0
        else:
            reward = 100.0 * np.log1p(step_return)

        if drawdown > self.drawdown_threshold:
            excess_drawdown = drawdown - self.drawdown_threshold
            reward -= excess_drawdown * self.drawdown_penalty_coef

        # Getiri 100x ölçekli olduğu için turnover cezası da orantılı ölçeklenir
        reward -= turnover * self.turnover_penalty
        return float(reward)

    def _get_observation(self) -> np.ndarray:
        window_features = self._features[
            self.current_step - self.window_size : self.current_step
        ].flatten()
        return np.concatenate([window_features, self.weights]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        last_valid_step = len(self.prices_df) - 1
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

        self.steps_elapsed = 0
        self.portfolio_value = self.initial_balance
        self.peak_value = self.initial_balance

        self.weights = np.zeros(self.num_actions, dtype=np.float32)
        self.weights[-1] = 1.0
        self.returns_history = []

        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        is_rebalance_day = (self.steps_elapsed % self.rebalance_freq == 0)

        if is_rebalance_day:
            target_weights = self._softmax(action)
            turnover = self._calculate_turnover(self.weights, target_weights)
            if turnover < self.rebalance_threshold:
                target_weights = self.weights.astype(np.float64).copy()
                turnover = 0.0
        else:
            # Ara günlerde mevcut ağırlıklar korunur, komisyon ödenmez
            target_weights = self.weights.astype(np.float64).copy()
            turnover = 0.0

        value_before = self.portfolio_value
        commission_cost = value_before * turnover * self.commission_rate
        value_after_cost = value_before - commission_cost

        prev_prices = self._prices[self.current_step]
        self.current_step += 1
        self.steps_elapsed += 1
        curr_prices = self._prices[self.current_step]

        asset_returns = (curr_prices - prev_prices) / (prev_prices + 1e-8)
        all_returns = np.append(asset_returns, self.cash_daily_return)

        gross_return = float(np.dot(target_weights, all_returns))
        self.portfolio_value = float(value_after_cost * (1.0 + gross_return))

        step_return = float(self.portfolio_value / value_before - 1.0)
        self.returns_history.append(step_return)

        updated_values = target_weights * (1.0 + all_returns)
        total_value = np.sum(updated_values)
        self.weights = (updated_values / total_value).astype(np.float32)

        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = float((self.peak_value - self.portfolio_value) / self.peak_value)

        reward = self._calculate_reward(
            step_return=step_return,
            drawdown=drawdown,
            turnover=turnover,
        )

        terminated = bool(self.portfolio_value <= self.initial_balance * 0.2)
        truncated = bool(self.current_step >= self.end_step)

        info = {
            "date": self.dates[self.current_step],
            "portfolio_value": self.portfolio_value,
            "drawdown": drawdown,
            "weights": self.weights.copy(),
            "step_return": step_return,
            "gross_return": gross_return,
            "commission_cost": float(commission_cost),
            "turnover": turnover,
        }

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            info,
        )