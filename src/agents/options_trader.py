import json
import pandas as pd
import numpy as np
from langchain_core.messages import HumanMessage

from graph.state import AgentState, show_agent_reasoning
from tools.api import get_prices, prices_to_df, get_option_chain
from utils.progress import progress


def options_trader_agent(state: AgentState):
    """
    Sophisticated options trading analysis system focused on intraday movements:
    1. Implied Volatility Analysis
    2. Options Chain Assessment 
    3. Greeks Analysis
    4. Unusual Options Activity Detection
    5. Strike Price Distribution Analysis
    """
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Initialize analysis for each ticker
    options_analysis = {}

    for ticker in tickers:
        progress.update_status("options_trader_agent", ticker, "Analyzing options data")

        # Get the historical price data for context
        prices = get_prices(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )

        if not prices:
            progress.update_status("options_trader_agent", ticker, "Failed: No price data found")
            continue

        # Convert prices to a DataFrame
        prices_df = prices_to_df(prices)
        
        # Get options chain data
        progress.update_status("options_trader_agent", ticker, "Fetching options chain")
        try:
            option_chain = get_option_chain(ticker)
            if not option_chain or not option_chain.get('calls') or not option_chain.get('puts'):
                progress.update_status("options_trader_agent", ticker, "Failed: No options data found")
                continue
        except Exception as e:
            progress.update_status("options_trader_agent", ticker, f"Failed: {str(e)}")
            continue
        
        # Extract current stock price
        current_price = prices_df['close'].iloc[-1]
        
        # Analyze implied volatility
        progress.update_status("options_trader_agent", ticker, "Analyzing implied volatility")
        iv_analysis = analyze_implied_volatility(option_chain, current_price)
        
        # Analyze options chain
        progress.update_status("options_trader_agent", ticker, "Analyzing options chain")
        chain_analysis = analyze_options_chain(option_chain, current_price)
        
        # Analyze Greeks
        progress.update_status("options_trader_agent", ticker, "Analyzing Greeks")
        greeks_analysis = analyze_greeks(option_chain)
        
        # Detect unusual options activity
        progress.update_status("options_trader_agent", ticker, "Detecting unusual options activity")
        unusual_activity = detect_unusual_activity(option_chain)
        
        # Analyze strike price distribution
        progress.update_status("options_trader_agent", ticker, "Analyzing strike distribution")
        strike_distribution = analyze_strike_distribution(option_chain, current_price)
        
        # Combine all analyses for a comprehensive options trading signal
        combined_signal = generate_combined_signal(
            iv_analysis,
            chain_analysis,
            greeks_analysis,
            unusual_activity,
            strike_distribution,
            current_price
        )
        
        # Generate detailed analysis report for this ticker
        options_analysis[ticker] = {
            "signal": combined_signal["signal"],
            "confidence": round(combined_signal["confidence"] * 100),
            "strategy_recommendations": combined_signal["strategy_recommendations"],
            "reasoning": {
                "iv_skew": f"IV Skew: {iv_analysis['skew']:.2f}",
                "put_call_ratio": f"Put/Call Ratio: {chain_analysis['put_call_ratio']:.2f}",
                "volume_distribution": f"Volume Distribution: {chain_analysis['volume_distribution']}",
                "delta_exposure": f"Delta Exposure: {greeks_analysis['delta_exposure']}",
                "key_support": ", ".join([f"${level:.2f}" for level in strike_distribution["support_levels"][:2]]) if strike_distribution["support_levels"] else "None",
                "key_resistance": ", ".join([f"${level:.2f}" for level in strike_distribution["resistance_levels"][:2]]) if strike_distribution["resistance_levels"] else "None"
            },
            "analysis_components": {
                "implied_volatility": {
                    "signal": iv_analysis["signal"],
                    "skew": iv_analysis["skew"],
                    "term_structure": iv_analysis["term_structure"]
                },
                "options_chain": {
                    "put_call_ratio": chain_analysis["put_call_ratio"],
                    "volume_distribution": chain_analysis["volume_distribution"]
                },
                "greeks": {
                    "key_gamma_levels": greeks_analysis["key_gamma_levels"],
                    "delta_exposure": greeks_analysis["delta_exposure"]
                },
                "unusual_activity": {
                    "unusual_strikes": unusual_activity["unusual_strikes"],
                    "high_volume_options": unusual_activity["high_volume_options"]
                },
                "strike_distribution": {
                    "key_support_levels": strike_distribution["support_levels"],
                    "key_resistance_levels": strike_distribution["resistance_levels"]
                }
            }
        }
        progress.update_status("options_trader_agent", ticker, "Done")

    # Create the options trader message
    message = HumanMessage(
        content=json.dumps(options_analysis),
        name="options_trader_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(options_analysis, "Options Trader")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["options_trader_agent"] = options_analysis

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }


def analyze_implied_volatility(option_chain, current_price):
    """
    Analyze implied volatility patterns across strikes and expirations
    """
    try:
        calls = pd.DataFrame(option_chain["calls"])
        puts = pd.DataFrame(option_chain["puts"])
        
        # Calculate IV skew (difference between OTM puts and OTM calls)
        # Higher values indicate fear of downside movement
        otm_puts = puts[puts["strike"] < current_price]
        otm_calls = calls[calls["strike"] > current_price]
        
        if not otm_puts.empty and not otm_calls.empty:
            avg_put_iv = otm_puts["impliedVolatility"].mean()
            avg_call_iv = otm_calls["impliedVolatility"].mean()
            iv_skew = avg_put_iv - avg_call_iv
            
            # Determine signal based on IV skew
            if iv_skew > 0.05:
                signal = "bearish"  # High put IV relative to calls indicates fear
            elif iv_skew < -0.05:
                signal = "bullish"  # High call IV relative to puts indicates optimism
            else:
                signal = "neutral"
                
            # Get term structure information (compare near-term vs longer-term IV)
            # This implementation is simplified for demonstration
            term_structure = "flat"  # Could be "contango" or "backwardation" with real data
            
            return {
                "signal": signal,
                "skew": float(iv_skew),
                "term_structure": term_structure
            }
        else:
            return {
                "signal": "neutral",
                "skew": 0.0,
                "term_structure": "flat"
            }
    except Exception as e:
        print(f"Error analyzing implied volatility: {str(e)}")
        return {
            "signal": "neutral",
            "skew": 0.0,
            "term_structure": "flat"
        }


def analyze_options_chain(option_chain, current_price):
    """
    Analyze the options chain for volume patterns and put-call ratios
    """
    try:
        calls = pd.DataFrame(option_chain["calls"])
        puts = pd.DataFrame(option_chain["puts"])
        
        # Calculate put-call ratio
        total_put_volume = puts["volume"].sum()
        total_call_volume = calls["volume"].sum()
        
        if total_call_volume > 0:
            put_call_ratio = total_put_volume / total_call_volume
        else:
            put_call_ratio = 1.0
        
        # Analyze volume distribution by strike proximity to current price
        atm_calls = calls[(calls["strike"] >= current_price * 0.95) & (calls["strike"] <= current_price * 1.05)]
        atm_puts = puts[(puts["strike"] >= current_price * 0.95) & (puts["strike"] <= current_price * 1.05)]
        
        # Determine where the volume is concentrated
        if not atm_calls.empty and not atm_puts.empty:
            atm_call_volume = atm_calls["volume"].sum()
            atm_put_volume = atm_puts["volume"].sum()
            
            if atm_call_volume > atm_put_volume * 1.5:
                volume_distribution = "calls_dominated"
            elif atm_put_volume > atm_call_volume * 1.5:
                volume_distribution = "puts_dominated"
            else:
                volume_distribution = "balanced"
        else:
            volume_distribution = "insufficient_data"
        
        return {
            "put_call_ratio": float(put_call_ratio),
            "volume_distribution": volume_distribution
        }
    except Exception as e:
        print(f"Error analyzing options chain: {str(e)}")
        return {
            "put_call_ratio": 1.0,
            "volume_distribution": "error"
        }


def analyze_greeks(option_chain):
    """
    Analyze option Greeks to identify key levels and exposures
    """
    try:
        calls = pd.DataFrame(option_chain["calls"])
        puts = pd.DataFrame(option_chain["puts"])
        
        # Combine calls and puts for gamma analysis
        all_options = pd.concat([calls, puts])
        
        # Find strikes with highest gamma (market inflection points)
        if 'gamma' in all_options.columns and not all_options.empty:
            gamma_by_strike = all_options.groupby('strike')['gamma'].sum().reset_index()
            gamma_by_strike = gamma_by_strike.sort_values('gamma', ascending=False)
            key_gamma_levels = gamma_by_strike.head(3)['strike'].tolist()
        else:
            # Simulate key gamma levels if gamma data isn't available
            key_gamma_levels = []
        
        # Calculate net delta exposure
        if 'delta' in calls.columns and 'delta' in puts.columns:
            call_delta = (calls['delta'] * calls['openInterest']).sum()
            put_delta = (puts['delta'] * puts['openInterest']).sum()
            delta_exposure = call_delta + put_delta
            
            if delta_exposure > 0:
                delta_exposure_signal = "bullish"
            elif delta_exposure < 0:
                delta_exposure_signal = "bearish"
            else:
                delta_exposure_signal = "neutral"
        else:
            delta_exposure_signal = "neutral"
        
        return {
            "key_gamma_levels": key_gamma_levels,
            "delta_exposure": delta_exposure_signal
        }
    except Exception as e:
        print(f"Error analyzing Greeks: {str(e)}")
        return {
            "key_gamma_levels": [],
            "delta_exposure": "neutral"
        }


def detect_unusual_activity(option_chain):
    """
    Detect unusual options activity that might signal smart money positioning
    """
    try:
        calls = pd.DataFrame(option_chain["calls"])
        puts = pd.DataFrame(option_chain["puts"])
        
        # Combine calls and puts
        all_options = pd.concat([calls, puts])
        all_options['type'] = np.where(all_options.index < len(calls), 'call', 'put')
        
        # Find options with unusually high volume relative to open interest
        if 'volume' in all_options.columns and 'openInterest' in all_options.columns:
            all_options['volume_oi_ratio'] = all_options['volume'] / all_options['openInterest'].replace(0, 1)
            unusual_volume = all_options[all_options['volume_oi_ratio'] > 3]
            
            # Extract the unusual strikes
            unusual_strikes = []
            for _, row in unusual_volume.iterrows():
                unusual_strikes.append({
                    'strike': float(row['strike']),
                    'type': row['type'],
                    'volume': int(row['volume']),
                    'openInterest': int(row['openInterest'])
                })
                
            # Find options with highest absolute volume
            high_volume_options = []
            top_volume = all_options.sort_values('volume', ascending=False).head(5)
            for _, row in top_volume.iterrows():
                high_volume_options.append({
                    'strike': float(row['strike']),
                    'type': row['type'],
                    'volume': int(row['volume']),
                    'expiration': row.get('expiration', 'unknown')
                })
        else:
            unusual_strikes = []
            high_volume_options = []
        
        return {
            "unusual_strikes": unusual_strikes,
            "high_volume_options": high_volume_options
        }
    except Exception as e:
        print(f"Error detecting unusual activity: {str(e)}")
        return {
            "unusual_strikes": [],
            "high_volume_options": []
        }


def analyze_strike_distribution(option_chain, current_price):
    """
    Analyze the distribution of open interest across strikes to identify support/resistance
    """
    try:
        calls = pd.DataFrame(option_chain["calls"])
        puts = pd.DataFrame(option_chain["puts"])
        
        # Identify potential support levels (high put open interest below current price)
        support_levels = []
        if 'openInterest' in puts.columns:
            otm_puts = puts[puts['strike'] < current_price]
            if not otm_puts.empty:
                top_put_oi = otm_puts.sort_values('openInterest', ascending=False).head(3)
                support_levels = top_put_oi['strike'].tolist()
        
        # Identify potential resistance levels (high call open interest above current price)
        resistance_levels = []
        if 'openInterest' in calls.columns:
            otm_calls = calls[calls['strike'] > current_price]
            if not otm_calls.empty:
                top_call_oi = otm_calls.sort_values('openInterest', ascending=False).head(3)
                resistance_levels = top_call_oi['strike'].tolist()
        
        return {
            "support_levels": [float(level) for level in support_levels],
            "resistance_levels": [float(level) for level in resistance_levels]
        }
    except Exception as e:
        print(f"Error analyzing strike distribution: {str(e)}")
        return {
            "support_levels": [],
            "resistance_levels": []
        }


def generate_combined_signal(iv_analysis, chain_analysis, greeks_analysis, 
                            unusual_activity, strike_distribution, current_price):
    """
    Generate a combined options trading signal and strategy recommendations
    """
    # Initialize signal components
    signal_components = {
        "iv_skew": 0,
        "put_call_ratio": 0,
        "volume_distribution": 0,
        "delta_exposure": 0
    }
    
    # Evaluate IV skew signal
    if iv_analysis["skew"] > 0.1:
        signal_components["iv_skew"] = -1  # Bearish
    elif iv_analysis["skew"] < -0.1:
        signal_components["iv_skew"] = 1   # Bullish
    
    # Evaluate put-call ratio
    if chain_analysis["put_call_ratio"] > 1.5:
        signal_components["put_call_ratio"] = -1  # Bearish
    elif chain_analysis["put_call_ratio"] < 0.7:
        signal_components["put_call_ratio"] = 1   # Bullish
    
    # Evaluate volume distribution
    if chain_analysis["volume_distribution"] == "calls_dominated":
        signal_components["volume_distribution"] = 1  # Bullish
    elif chain_analysis["volume_distribution"] == "puts_dominated":
        signal_components["volume_distribution"] = -1  # Bearish
    
    # Evaluate delta exposure
    if greeks_analysis["delta_exposure"] == "bullish":
        signal_components["delta_exposure"] = 1
    elif greeks_analysis["delta_exposure"] == "bearish":
        signal_components["delta_exposure"] = -1
    
    # Calculate weighted average signal
    weights = {
        "iv_skew": 0.30,
        "put_call_ratio": 0.25,
        "volume_distribution": 0.20,
        "delta_exposure": 0.25
    }
    
    signal_value = sum(signal_components[k] * weights[k] for k in signal_components)
    
    # Determine final signal
    if signal_value > 0.2:
        signal = "bullish"
        confidence = min(abs(signal_value), 1.0)
    elif signal_value < -0.2:
        signal = "bearish"
        confidence = min(abs(signal_value), 1.0)
    else:
        signal = "neutral"
        confidence = 0.5
    
    # Generate strategy recommendations based on the signal
    strategy_recommendations = []
    support_levels = strike_distribution["support_levels"]
    resistance_levels = strike_distribution["resistance_levels"]
    
    if signal == "bullish":
        # Recommend bullish option strategies
        if len(resistance_levels) > 0:
            nearest_resistance = min([r for r in resistance_levels if r > current_price], default=current_price * 1.05)
            strategy_recommendations.append({
                "strategy": "call_debit_spread",
                "description": f"Buy ATM calls, sell OTM calls near resistance at {nearest_resistance}",
                "risk_level": "moderate"
            })
        
        strategy_recommendations.append({
            "strategy": "cash_secured_put",
            "description": f"Sell ATM or slightly OTM puts to collect premium",
            "risk_level": "moderate"
        })
        
    elif signal == "bearish":
        # Recommend bearish option strategies
        if len(support_levels) > 0:
            nearest_support = max([s for s in support_levels if s < current_price], default=current_price * 0.95)
            strategy_recommendations.append({
                "strategy": "put_debit_spread",
                "description": f"Buy ATM puts, sell OTM puts near support at {nearest_support}",
                "risk_level": "moderate"
            })
        
        strategy_recommendations.append({
            "strategy": "covered_call",
            "description": "Sell ATM or slightly OTM calls against existing stock position",
            "risk_level": "low"
        })
        
    else:
        # Recommend neutral option strategies
        strategy_recommendations.append({
            "strategy": "iron_condor",
            "description": "Sell OTM put spread and OTM call spread to collect premium in range-bound environment",
            "risk_level": "moderate"
        })
        
        strategy_recommendations.append({
            "strategy": "calendar_spread",
            "description": "Sell near-term ATM options and buy longer-term ATM options",
            "risk_level": "moderate"
        })
    
    # Add an intraday-specific recommendation
    if iv_analysis["skew"] > 0.05 or iv_analysis["skew"] < -0.05:
        strategy_recommendations.append({
            "strategy": "intraday_iv_play",
            "description": "Take advantage of elevated implied volatility with day trades",
            "risk_level": "high",
            "timeframe": "intraday"
        })
    
    return {
        "signal": signal,
        "confidence": confidence,
        "strategy_recommendations": strategy_recommendations
    } 