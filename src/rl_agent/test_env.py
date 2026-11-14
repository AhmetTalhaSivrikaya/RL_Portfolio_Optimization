
import numpy as np
import pandas as pd
import pytest

from src.rl_agent.env import PortfolioEnv
from stable_baselines3.common.env_checker import check_env


@pytest.fixture
def env():
    """Testler için örnek portföy ortamı oluşturur."""

    np.random.seed(42)

    days = 100
    num_assets = 5

    prices_df = pd.DataFrame(
        np.cumprod(
            1 + np.random.normal(
                0.0005, 0.02, (days, num_assets)
            ),
            axis=0
        ) * 100,
        columns=[
            "HISSE_1",
            "HISSE_2",
            "HISSE_3",
            "HISSE_4",
            "HISSE_5"
        ]
    )

    features_df = pd.DataFrame(
        np.random.randn(days, 10)
    )

    return PortfolioEnv(
        prices_df=prices_df,
        features_df=features_df,
        initial_balance=100_000,
        commission_rate=0.001,
        window_size=10
    )


def test_environment_initialization(env):
    """Environment doğru oluşturuluyor mu?"""

    assert env.num_assets == 5
    assert env.num_actions == 6
    assert env.portfolio_value == 100_000

    assert np.isclose(
        np.sum(env.weights),
        1.0
    )


def test_observation_shape(env):
    """Observation boyutu doğru mu?"""

    observation, info = env.reset(seed=42)

    expected_size = (
        env.window_size * env.features_df.shape[1]
        + env.num_actions
    )

    assert observation.shape == (expected_size,)
    assert np.all(np.isfinite(observation))


def test_turnover_calculation(env):
    """Tam portföy değişiminde turnover %100 olmalı."""

    old_weights = np.array(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    new_weights = np.array(
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    )

    turnover = env._calculate_turnover(
        old_weights,
        new_weights
    )

    assert np.isclose(turnover, 1.0)


def test_no_trade_commission(env):
    """İşlem yapılmadığında komisyon sıfır olmalı."""

    env.reset(seed=42)

    # Nakit ağırlığını yüksek tutan aksiyon
    action = np.array(
        [-1.0, -1.0, -1.0, -1.0, -1.0, 1.0],
        dtype=np.float32
    )

    observation, reward, terminated, truncated, info = (
        env.step(action)
    )

    assert info["turnover"] == 0.0
    assert info["commission_cost"] == 0.0


def test_portfolio_weights_sum_to_one(env):
    """Her adım sonunda portföy ağırlıkları toplamı 1 olmalı."""

    env.reset(seed=42)

    action = np.array(
        [1.0, 0.0, -1.0, 0.5, -0.5, 0.0],
        dtype=np.float32
    )

    observation, reward, terminated, truncated, info = (
        env.step(action)
    )

    assert np.isclose(
        np.sum(info["weights"]),
        1.0,
        atol=1e-5
    )


def test_portfolio_value_is_positive(env):
    """Portföy değeri pozitif kalmalı."""

    env.reset(seed=42)

    action = np.zeros(
        env.num_actions,
        dtype=np.float32
    )

    observation, reward, terminated, truncated, info = (
        env.step(action)
    )

    assert info["portfolio_value"] > 0
    assert np.isfinite(reward)


def test_gymnasium_compatibility(env):
    """Environment Gymnasium standartlarına uygun mu?"""

    check_env(env, warn=True)


@pytest.fixture
def deterministic_env():
    """
    Finansal hesaplamaları kontrol etmek için
    sabit ve tahmin edilebilir fiyat verisi.
    """

    prices_df = pd.DataFrame({
        "ASSET_1": [
            100, 100, 100, 110, 110,
            110, 110, 110, 110, 110,
            110, 110, 110, 110, 110,
            110, 110, 110, 110, 110
        ],
        "ASSET_2": [
            100, 100, 100, 100, 100,
            100, 100, 100, 100, 100,
            100, 100, 100, 100, 100,
            100, 100, 100, 100, 100
        ]
    })

    features_df = pd.DataFrame(
        np.zeros((20, 4))
    )

    return PortfolioEnv(
        prices_df=prices_df,
        features_df=features_df,
        initial_balance=100_000,
        commission_rate=0.001,
        window_size=2,
        rebalance_threshold=0.0,
        risk_aversion=0.0,
        turnover_penalty=0.0,
        drawdown_penalty_coef=0.0
    )


def test_commission_calculation(deterministic_env):
    """İşlem komisyonu doğru hesaplanıyor mu?"""

    env = deterministic_env
    env.reset(seed=42)

    action = np.array(
        [1.0, -1.0, -1.0],
        dtype=np.float32
    )

    target_weights = env._softmax(action)

    expected_turnover = env._calculate_turnover(
        env.weights,
        target_weights
    )

    expected_commission = (
        100_000 * expected_turnover * 0.001
    )

    _, _, _, _, info = env.step(action)

    assert np.isclose(
        info["turnover"],
        expected_turnover
    )

    assert np.isclose(
        info["commission_cost"],
        expected_commission
    )


def test_portfolio_return_calculation(deterministic_env):
    """
    İlk hissenin fiyatı %10 yükseldiğinde
    portföy getirisi doğru hesaplanıyor mu?
    """

    env = deterministic_env
    env.reset(seed=42)

    action = np.array(
        [1.0, -1.0, -1.0],
        dtype=np.float32
    )

    target_weights = env._softmax(action)

    expected_turnover = env._calculate_turnover(
        env.weights,
        target_weights
    )

    expected_commission = (
        100_000 * expected_turnover * 0.001
    )

    expected_gross_return = (
        target_weights[0] * 0.10
    )

    expected_portfolio_value = (
        (100_000 - expected_commission)
        * (1 + expected_gross_return)
    )

    _, _, _, _, info = env.step(action)

    assert np.isclose(
        info["gross_return"],
        expected_gross_return
    )

    assert np.isclose(
        info["portfolio_value"],
        expected_portfolio_value
    )


def test_net_return_calculation(deterministic_env):
    """Net getiri komisyonu içeriyor mu?"""

    env = deterministic_env
    env.reset(seed=42)

    action = np.array(
        [1.0, -1.0, -1.0],
        dtype=np.float32
    )

    _, _, _, _, info = env.step(action)

    expected_return = (
        info["portfolio_value"] / 100_000
    ) - 1.0

    assert np.isclose(
        info["step_return"],
        expected_return
    )


def test_weights_update_after_price_change(deterministic_env):
    """
    Hisse fiyatı değiştiğinde portföy ağırlıkları
    doğru güncelleniyor mu?
    """

    env = deterministic_env
    env.reset(seed=42)

    action = np.array(
        [1.0, -1.0, -1.0],
        dtype=np.float32
    )

    target_weights = env._softmax(action)

    asset_returns = np.array([
        0.10,
        0.0,
        0.0
    ])

    expected_values = (
        target_weights * (1 + asset_returns)
    )

    expected_weights = (
        expected_values / np.sum(expected_values)
    )

    _, _, _, _, info = env.step(action)

    assert np.allclose(
        info["weights"],
        expected_weights,
        atol=1e-6
    )

    assert np.isclose(
        np.sum(info["weights"]),
        1.0
    )

def test_logarithmic_reward(deterministic_env):
    """Reward, net logaritmik getiriyle hesaplanmalı."""

    env = deterministic_env

    step_return = 0.05
    drawdown = 0.0
    turnover = 0.5

    reward = env._calculate_reward(
        step_return=step_return,
        drawdown=drawdown,
        turnover=turnover
    )

    expected_reward = (
        100 * np.log1p(step_return)
        - turnover * env.turnover_penalty
    )

    assert np.isclose(
        reward,
        expected_reward
    )


def test_drawdown_penalty(deterministic_env):
    """Drawdown eşiği aşılınca ödül azalmalı."""

    env = deterministic_env

    # Ceza katsayısını test için açıkça belirliyoruz.
    env.drawdown_threshold = 0.15
    env.drawdown_penalty_coef = 10.0

    step_return = 0.01
    turnover = 0.0

    reward_without_drawdown = env._calculate_reward(
        step_return,
        drawdown=0.0,
        turnover=turnover
    )

    reward_with_drawdown = env._calculate_reward(
        step_return,
        drawdown=0.25,
        turnover=turnover
    )

    assert reward_with_drawdown < reward_without_drawdown

def test_turnover_penalty(deterministic_env):
    """Turnover arttığında diğer koşullar sabitse ödül azalmalı."""

    env = deterministic_env

    # Ceza katsayısını test için açıkça belirliyoruz.
    env.turnover_penalty = 0.01

    reward_without_turnover = env._calculate_reward(
        step_return=0.01,
        drawdown=0.0,
        turnover=0.0
    )

    reward_with_turnover = env._calculate_reward(
        step_return=0.01,
        drawdown=0.0,
        turnover=1.0
    )

    assert reward_with_turnover < reward_without_turnover