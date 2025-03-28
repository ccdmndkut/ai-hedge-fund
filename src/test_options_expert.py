from tools.api import get_option_chain, get_prices, prices_to_df
from agents.options_expert import (
    analyze_historical_volatility,
    analyze_volatility_surface,
    generate_strategy_suggestions,
    analyze_advanced_greeks,
    analyze_liquidity,
    analyze_options_sentiment,
    calculate_risk_metrics
)
from datetime import datetime, timedelta

# Test ticker
TICKER = "AAPL"

# Get price data for historical volatility analysis
print(f"Testing options expert analysis for {TICKER}...")
start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
end_date = datetime.now().strftime("%Y-%m-%d")

prices = get_prices(TICKER, start_date, end_date)
if prices:
    prices_df = prices_to_df(prices)
    current_price = prices_df['close'].iloc[-1]
    print(f"Current price: ${current_price:.2f}")

    # Get options chain
    options_data = get_option_chain(TICKER)
    
    # Test each analysis component
    print("\n--- Testing Historical Volatility Analysis ---")
    hist_vol = analyze_historical_volatility(prices_df)
    print(f"Historical volatility: {hist_vol['historical_volatility']:.2f}%")
    print(f"HV percentile: {hist_vol['percentile']}")
    print(f"HV trend: {hist_vol['trend']}")
    
    print("\n--- Testing Volatility Surface Analysis ---")
    vol_surface = analyze_volatility_surface(options_data, current_price)
    print(f"Average IV: {vol_surface['average_iv']:.2f}%")
    print(f"IV skew: {vol_surface['skew']:.4f}")
    print(f"Skew strength: {vol_surface['skew_strength']}")
    print(f"Term structure: {vol_surface['term_structure']}")
    
    print("\n--- Testing Strategy Suggestions ---")
    strategies = generate_strategy_suggestions(options_data, current_price, hist_vol, vol_surface)
    for i, strategy in enumerate(strategies, 1):
        print(f"{i}. {strategy['name']} - Fit: {strategy['fit']}")
        print(f"   {strategy['description']}")
    
    print("\n--- Testing Advanced Greeks Analysis ---")
    adv_greeks = analyze_advanced_greeks(options_data, current_price)
    print(f"Key gamma levels: {', '.join([f'${level:.2f}' for level in adv_greeks['key_gamma_levels'][:3]])}")
    print(f"Vanna exposure: {adv_greeks['vanna_exposure']}")
    print(f"Vol of vol: {adv_greeks['vol_of_vol']}")
    print(f"Charm risk: {adv_greeks['charm_risk']}")
    
    print("\n--- Testing Liquidity Analysis ---")
    liquidity = analyze_liquidity(options_data)
    print(f"Average call spread: {liquidity['bid_ask_spreads']['average_call_spread']:.2f}%")
    print(f"Average put spread: {liquidity['bid_ask_spreads']['average_put_spread']:.2f}%")
    print(f"Total volume: {liquidity['volume_trends']['total_volume']}")
    print(f"Liquidity rating: {liquidity['liquidity_rating']}")
    
    print("\n--- Testing Options Sentiment Analysis ---")
    sentiment = analyze_options_sentiment(options_data, current_price)
    print(f"Sentiment: {sentiment['sentiment']} ({sentiment['sentiment_strength']})")
    print(f"Put/Call volume ratio: {sentiment['put_call_ratio']['volume_ratio']:.2f}")
    if sentiment['key_observations']:
        print("Key observations:")
        for obs in sentiment['key_observations']:
            print(f"- {obs}")
    
    print("\n--- Testing Risk Metrics Calculation ---")
    risk = calculate_risk_metrics(options_data, current_price, adv_greeks, hist_vol["historical_volatility"])
    print(f"Max loss: ${risk['max_loss']:.2f}")
    print(f"Max gain: {risk['max_gain']}")
    print(f"Probability of profit: {risk['probability_of_profit'] * 100:.1f}%")
    print(f"Risk/reward ratio: {risk['risk_reward_ratio']:.2f}")
    print(f"Breakeven points: ${risk['breakeven_points'][0]:.2f}, ${risk['breakeven_points'][1]:.2f}")
    
    print("\nOptions expert analysis is working correctly!")
else:
    print(f"Failed to get price data for {TICKER}") 