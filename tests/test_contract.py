"""Ekip sözleşmesi testleri: RL ortamı, baseline'lar ve doğrulayıcı aynı kurallarla çalışmalı.

Çalıştırma (proje kökünden):  pytest
"""
import numpy as np
import pandas as pd
import pytest

from src.baseline.portfolio_baselines import _backtest, run_baselines, trade_turnover
from src.rl_agent.env import PortfolioEnv
from src.rl_agent.market_data_validator import DataContractError, validate_results_file

COMMISSION = 0.001
WINDOW = 10


def make_prices(days=120, assets=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=days)
    rets = rng.normal(0.0005, 0.02, size=(days, assets))
    data = 100 * np.cumprod(1 + rets, axis=0)
    return pd.DataFrame(data, index=idx, columns=[f"H{i}.IS" for i in range(assets)])


def make_env(prices, **kw):
    feats = pd.DataFrame(np.random.default_rng(1).standard_normal((len(prices), 4)), index=prices.index)
    return PortfolioEnv(prices, feats, commission_rate=COMMISSION, window_size=WINDOW, **kw)


def test_turnover_definition_is_identical_in_env_and_baseline():
    env = make_env(make_prices())
    rng = np.random.default_rng(3)
    for _ in range(200):
        old, new = rng.dirichlet(np.ones(6)), rng.dirichlet(np.ones(6))
        assert np.isclose(env._calculate_turnover(old, new), trade_turnover(old, new))


def test_cash_leg_is_free_and_rotation_pays_both_legs():
    # Nakit -> %100 hisse = 1.0; hisse A -> hisse B = 2.0
    cash = np.array([0, 0, 0, 0, 0, 1.0])
    a = np.array([1.0, 0, 0, 0, 0, 0])
    b = np.array([0, 1.0, 0, 0, 0, 0])
    assert np.isclose(trade_turnover(cash, a), 1.0)
    assert np.isclose(trade_turnover(a, b), 2.0)


def test_env_commission_equals_value_times_turnover_times_rate():
    env = make_env(make_prices(), rebalance_threshold=0.0, action_scale=3.0)
    env.reset(seed=0)
    rng = np.random.default_rng(5)
    for _ in range(15):
        before_value, before_w = env.portfolio_value, env.weights.astype(float).copy()
        action = rng.uniform(-1, 1, env.num_actions).astype(np.float32)
        target = env._softmax(action)
        _, _, _, _, info = env.step(action)
        expected = before_value * trade_turnover(before_w, target) * COMMISSION
        assert np.isclose(info["commission_cost"], expected, rtol=1e-9, atol=1e-9)


def test_env_step_return_is_net_of_commission():
    env = make_env(make_prices(), rebalance_threshold=0.0)
    env.reset(seed=0)
    rets = []
    rng = np.random.default_rng(6)
    for _ in range(30):
        _, _, _, _, info = env.step(rng.uniform(-1, 1, env.num_actions).astype(np.float32))
        rets.append(info["step_return"])
    assert np.isclose(np.prod(1 + np.array(rets)), info["portfolio_value"] / env.initial_balance)


def test_rl_env_and_baseline_engine_produce_identical_portfolio_values():
    """Aynı politika (eşit ağırlıklı hisse, her gün rebalance) iki motorda da aynı sonucu vermeli."""
    prices = make_prices(days=150)
    # action_scale çok büyük + nakit eylemi düşük -> nakit ağırlığı ~1e-22, hisseler eşit ağırlık
    env = make_env(prices, rebalance_threshold=0.0, action_scale=50.0)
    env.reset(seed=0)
    action = np.array([0, 0, 0, 0, 0, -1.0], dtype=np.float32)
    env_values, done = [], False
    while not done:
        _, _, term, trunc, info = env.step(action)
        env_values.append(info["portfolio_value"])
        done = term or trunc

    base = _backtest(
        prices, np.full(5, 0.2), rebalance_every=1,
        initial_balance=100_000.0, commission_rate=COMMISSION, window_size=WINDOW,
    )
    assert len(base) == len(env_values)
    assert np.allclose(base["Portfolio_Value"].values, env_values, rtol=1e-6, atol=0.02)


@pytest.fixture
def baseline_files(tmp_path):
    prices = make_prices(days=400)
    train, test = prices.iloc[:250], prices.iloc[250:]
    train.to_csv(tmp_path / "train_prices.csv", index_label="Date")
    test.to_csv(tmp_path / "test_prices.csv", index_label="Date")
    run_baselines(tmp_path / "train_prices.csv", tmp_path / "test_prices.csv", tmp_path / "out")
    return tmp_path / "out"


@pytest.mark.parametrize("name", ["markowitz", "markowitz_weekly", "buy_and_hold"])
def test_baseline_outputs_satisfy_the_contract(baseline_files, name):
    df = validate_results_file(str(baseline_files / f"{name}_backtest_results.csv"))
    assert np.isclose(np.prod(1 + df["Daily_Return"].values), df["Portfolio_Value"].iloc[-1] / 100_000.0, rtol=1e-6)


def test_buy_and_hold_pays_one_commission_on_first_day(baseline_files):
    df = pd.read_csv(baseline_files / "buy_and_hold_backtest_results.csv")
    assert np.isclose(df["Turnover"].iloc[0], 1.0, atol=1e-6)
    assert np.isclose(df["Commission"].iloc[0], 100_000 * COMMISSION, atol=1e-3)
    assert (df["Commission"].iloc[1:] == 0).all()


def test_validator_rejects_gross_daily_return(baseline_files, tmp_path):
    df = pd.read_csv(baseline_files / "markowitz_backtest_results.csv")
    df["Daily_Return"] = df["Daily_Return"] + df["Commission"] / df["Portfolio_Value"]  # komisyonu geri ekle (brüt)
    bad = tmp_path / "gross_backtest_results.csv"
    df.to_csv(bad, index=False)
    with pytest.raises(DataContractError):
        validate_results_file(str(bad))


def test_validator_rejects_wrong_turnover_definition(baseline_files, tmp_path):
    df = pd.read_csv(baseline_files / "buy_and_hold_backtest_results.csv")
    df.loc[0, ["Turnover", "Commission"]] *= 2  # nakit bacağını da sayan eski tanım
    bad = tmp_path / "turn_backtest_results.csv"
    df.to_csv(bad, index=False)
    with pytest.raises(DataContractError):
        validate_results_file(str(bad))