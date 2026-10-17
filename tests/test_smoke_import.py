def test_core_modules_import():
    import config, utils, session_manager, market_data, mt5_connector
    import order_manager, risk_manager, multi_bot, backtester, optimizer
    import performance, logic_version, bot_history, evaluation_matrix, event_bus
    import trailing_profiles, profile_manager
    import strategies
    from strategies import STRATEGY_REGISTRY
    assert len(STRATEGY_REGISTRY) == 14

def test_every_strategy_instantiates():
    from strategies import STRATEGY_REGISTRY, get_strategy
    for key in STRATEGY_REGISTRY:
        s = get_strategy(key)
        assert s.name
