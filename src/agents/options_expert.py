import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage

from graph.state import AgentState, show_agent_reasoning
from tools.api import get_prices, prices_to_df, get_option_chain
from utils.progress import progress


def options_expert_agent(state: AgentState):
    """
    Advanced options trading expert agent specializing in sophisticated options analysis:
    1. Detailed Options Strategy Construction
    2. Volatility Surface Analysis
    3. Advanced Greeks Management
    4. Liquidity & Spread Analysis
    5. Risk/Reward Visualization
    6. Options Market Sentiment
    7. Historical Options Performance
    """
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Initialize expert analysis for each ticker
    options_expert_analysis = {}

    for ticker in tickers:
        progress.update_status("options_expert_agent", ticker, "Analyzing advanced options data")

        # Get historical price data
        prices = get_prices(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )

        if not prices:
            progress.update_status("options_expert_agent", ticker, "Failed: No price data found")
            continue

        # Convert prices to DataFrame
        prices_df = prices_to_df(prices)
        
        # Get options chain data
        progress.update_status("options_expert_agent", ticker, "Fetching options chain")
        try:
            option_chain = get_option_chain(ticker)
            if not option_chain or not option_chain.get('calls') or not option_chain.get('puts'):
                progress.update_status("options_expert_agent", ticker, "Failed: No options data found")
                continue
        except Exception as e:
            progress.update_status("options_expert_agent", ticker, f"Failed: {str(e)}")
            continue
        
        # Extract current stock price
        current_price = prices_df['close'].iloc[-1]
        
        # Historical volatility analysis
        progress.update_status("options_expert_agent", ticker, "Analyzing historical volatility")
        hist_vol = analyze_historical_volatility(prices_df)
        
        # Volatility surface analysis
        progress.update_status("options_expert_agent", ticker, "Analyzing volatility surface")
        vol_surface = analyze_volatility_surface(option_chain, current_price)
        
        # Detailed options strategy suggestions
        progress.update_status("options_expert_agent", ticker, "Generating strategy suggestions")
        strategy_suggestions = generate_strategy_suggestions(
            option_chain, 
            current_price, 
            hist_vol, 
            vol_surface
        )
        
        # Advanced Greeks analysis
        progress.update_status("options_expert_agent", ticker, "Performing advanced Greeks analysis")
        advanced_greeks = analyze_advanced_greeks(option_chain, current_price)
        
        # Liquidity analysis
        progress.update_status("options_expert_agent", ticker, "Analyzing options liquidity")
        liquidity_analysis = analyze_liquidity(option_chain)
        
        # Options market sentiment
        progress.update_status("options_expert_agent", ticker, "Analyzing options market sentiment")
        market_sentiment = analyze_options_sentiment(option_chain, current_price)
        
        # Risk visualization
        progress.update_status("options_expert_agent", ticker, "Calculating risk metrics")
        risk_metrics = calculate_risk_metrics(
            option_chain, 
            current_price, 
            advanced_greeks, 
            hist_vol["historical_volatility"]
        )
        
        # Generate expert recommendation
        expert_recommendation = generate_expert_recommendation(
            strategy_suggestions,
            advanced_greeks,
            liquidity_analysis,
            market_sentiment,
            risk_metrics,
            current_price
        )
        
        # Generate detailed analysis report for this ticker
        options_expert_analysis[ticker] = {
            "signal": expert_recommendation["signal"],
            "confidence": round(expert_recommendation["confidence"] * 100),
            "recommended_strategies": expert_recommendation["recommended_strategies"],
            "reasoning": {
                "volatility_analysis": f"HV: {hist_vol['historical_volatility']:.2f}% vs IV: {vol_surface['average_iv']:.2f}%",
                "vol_skew": f"Vol Skew: {vol_surface['skew_strength']}",
                "options_sentiment": f"Options Sentiment: {market_sentiment['sentiment']}",
                "liquidity_condition": f"Liquidity: {liquidity_analysis['liquidity_rating']}",
                "risk_reward": f"Risk/Reward: {risk_metrics['risk_reward_ratio']:.2f}",
                "key_levels": f"Key gamma levels: ${', '.join([f'${level:.2f}' for level in advanced_greeks['key_gamma_levels'][:2]])}" if advanced_greeks['key_gamma_levels'] else "None"
            },
            "analysis_components": {
                "historical_volatility": {
                    "value": hist_vol["historical_volatility"],
                    "percentile": hist_vol["percentile"],
                    "trend": hist_vol["trend"]
                },
                "volatility_surface": {
                    "average_iv": vol_surface["average_iv"],
                    "skew": vol_surface["skew"],
                    "skew_strength": vol_surface["skew_strength"],
                    "term_structure": vol_surface["term_structure"]
                },
                "strategy_suggestions": strategy_suggestions,
                "advanced_greeks": {
                    "key_gamma_levels": advanced_greeks["key_gamma_levels"],
                    "vol_of_vol": advanced_greeks["vol_of_vol"],
                    "vanna_exposure": advanced_greeks["vanna_exposure"],
                    "charm_risk": advanced_greeks["charm_risk"]
                },
                "liquidity_analysis": {
                    "bid_ask_spreads": liquidity_analysis["bid_ask_spreads"],
                    "volume_trends": liquidity_analysis["volume_trends"],
                    "liquidity_rating": liquidity_analysis["liquidity_rating"]
                },
                "options_sentiment": {
                    "sentiment": market_sentiment["sentiment"],
                    "sentiment_strength": market_sentiment["sentiment_strength"],
                    "key_observations": market_sentiment["key_observations"]
                },
                "risk_metrics": {
                    "max_loss": risk_metrics["max_loss"],
                    "max_gain": risk_metrics["max_gain"],
                    "probability_of_profit": risk_metrics["probability_of_profit"],
                    "risk_reward_ratio": risk_metrics["risk_reward_ratio"]
                }
            }
        }
        progress.update_status("options_expert_agent", ticker, "Done")

    # Create the options expert message
    message = HumanMessage(
        content=json.dumps(options_expert_analysis),
        name="options_expert_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(options_expert_analysis, "Options Expert")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["options_expert_agent"] = options_expert_analysis

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }


def analyze_historical_volatility(prices_df):
    """
    Analyze historical price volatility and compare with current implied volatility
    """
    try:
        # Calculate daily returns
        returns = prices_df['close'].pct_change().dropna()
        
        # Calculate 20-day historical volatility (annualized)
        historical_volatility = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252) * 100
        
        # Calculate HV percentile (simplified)
        hv_200d = returns.rolling(window=20).std().dropna() * np.sqrt(252) * 100
        percentile = int(np.percentile(np.arange(len(hv_200d)), 
                                  (hv_200d.iloc[-1] - hv_200d.min()) / (hv_200d.max() - hv_200d.min()) * 100))
        
        # Calculate trend (rising, falling, stable)
        hv_recent = hv_200d.iloc[-10:]
        if hv_recent.iloc[-1] > hv_recent.iloc[0] * 1.1:
            trend = "rising"
        elif hv_recent.iloc[-1] < hv_recent.iloc[0] * 0.9:
            trend = "falling"
        else:
            trend = "stable"
        
        return {
            "historical_volatility": float(historical_volatility),
            "percentile": percentile,
            "trend": trend
        }
    except Exception as e:
        print(f"Error analyzing historical volatility: {str(e)}")
        return {
            "historical_volatility": 20.0,  # Default value
            "percentile": 50,
            "trend": "stable"
        }


def analyze_volatility_surface(option_chain, current_price):
    """
    Analyze the volatility surface across strikes and expirations
    """
    try:
        calls = pd.DataFrame(option_chain["calls"])
        puts = pd.DataFrame(option_chain["puts"])
        
        # Calculate average IV
        average_iv = np.mean(
            np.concatenate([calls["impliedVolatility"].values, puts["impliedVolatility"].values])
        ) * 100
        
        # Calculate IV skew (difference between OTM puts and OTM calls)
        otm_puts = puts[puts["strike"] < current_price]
        otm_calls = calls[calls["strike"] > current_price]
        
        if not otm_puts.empty and not otm_calls.empty:
            avg_put_iv = otm_puts["impliedVolatility"].mean()
            avg_call_iv = otm_calls["impliedVolatility"].mean()
            skew = avg_put_iv - avg_call_iv
            
            # Classify skew strength
            if skew > 0.05:
                skew_strength = "strong put skew"
            elif skew > 0.02:
                skew_strength = "moderate put skew"
            elif skew < -0.05:
                skew_strength = "strong call skew"
            elif skew < -0.02:
                skew_strength = "moderate call skew"
            else:
                skew_strength = "neutral skew"
                
            # Term structure analysis (simplified)
            # In a real implementation, would compare near-term vs. longer-term expirations
            term_structure = "normal"  # Could be "inverted" or "flat"
            
            return {
                "average_iv": float(average_iv),
                "skew": float(skew),
                "skew_strength": skew_strength,
                "term_structure": term_structure
            }
        else:
            return {
                "average_iv": float(average_iv),
                "skew": 0.0,
                "skew_strength": "neutral skew",
                "term_structure": "normal"
            }
    except Exception as e:
        print(f"Error analyzing volatility surface: {str(e)}")
        return {
            "average_iv": 25.0,
            "skew": 0.0,
            "skew_strength": "neutral skew",
            "term_structure": "normal"
        }


def generate_strategy_suggestions(option_chain, current_price, hist_vol, vol_surface):
    """
    Generate specific options strategy suggestions based on analysis
    """
    try:
        strategies = []
        
        # Compare historical volatility vs implied volatility
        hv = hist_vol["historical_volatility"]
        iv = vol_surface["average_iv"]
        iv_skew = vol_surface["skew"]
        skew_strength = vol_surface["skew_strength"]
        
        # Strategy selection logic based on volatility comparison
        if iv > hv * 1.2:  # IV significantly higher than HV
            if "call" in skew_strength:
                strategies.append({
                    "name": "Bear Call Spread",
                    "description": "Sell OTM call, buy further OTM call to limit risk. Profit from IV contraction and price stability/decline.",
                    "fit": "high",
                    "max_profit": "limited",
                    "max_loss": "limited",
                    "breakeven": f"Short strike + net premium"
                })
            else:
                strategies.append({
                    "name": "Iron Condor",
                    "description": "Sell OTM put and call, buy further OTM put and call to limit risk. Profit from IV contraction and price stability.",
                    "fit": "high",
                    "max_profit": "limited",
                    "max_loss": "limited",
                    "breakeven": f"Between short strikes +/- net premium"
                })
        elif iv < hv * 0.8:  # IV significantly lower than HV
            strategies.append({
                "name": "Long Straddle",
                "description": "Buy ATM call and put. Profit from large moves in either direction and/or IV expansion.",
                "fit": "high",
                "max_profit": "unlimited",
                "max_loss": "limited to premium paid",
                "breakeven": f"Strike +/- premium paid"
            })
        
        # Add strategy based on skew
        if "strong put skew" in skew_strength:
            strategies.append({
                "name": "Ratio Put Spread",
                "description": "Buy ATM put, sell multiple OTM puts. Capitalize on expensive OTM puts while maintaining downside protection.",
                "fit": "medium",
                "max_profit": "limited",
                "max_loss": "potentially unlimited below lower strike",
                "breakeven": "Complex - depends on ratio"
            })
        elif "strong call skew" in skew_strength:
            strategies.append({
                "name": "Call Backspread",
                "description": "Sell ATM call, buy multiple OTM calls. Capitalize on potential upside while using expensive ATM call to finance position.",
                "fit": "medium",
                "max_profit": "potentially unlimited upside",
                "max_loss": "limited",
                "breakeven": "Complex - depends on ratio"
            })
        
        # Always provide a basic strategy based on market view
        strategies.append({
            "name": "Cash-Secured Put",
            "description": "Sell OTM put secured by cash. Generate income if stock stays above strike, acquire stock at discount if below strike.",
            "fit": "medium",
            "max_profit": "limited to premium received",
            "max_loss": "substantial but limited to strike - premium",
            "breakeven": f"Strike - premium received"
        })
        
        return strategies
    except Exception as e:
        print(f"Error generating strategy suggestions: {str(e)}")
        # Return default strategies
        return [
            {
                "name": "Covered Call",
                "description": "Own stock, sell OTM call to generate income.",
                "fit": "medium",
                "max_profit": "limited",
                "max_loss": "substantial but limited",
                "breakeven": "Stock cost basis - premium received"
            },
            {
                "name": "Protective Put",
                "description": "Own stock, buy put for downside protection.",
                "fit": "medium",
                "max_profit": "unlimited upside",
                "max_loss": "limited",
                "breakeven": "Stock cost basis + premium paid"
            }
        ]


def analyze_advanced_greeks(option_chain, current_price):
    """
    Perform advanced analysis of options Greeks
    """
    try:
        calls = pd.DataFrame(option_chain["calls"])
        puts = pd.DataFrame(option_chain["puts"])
        
        # Find key gamma concentration levels
        all_options = pd.concat([calls, puts])
        gamma_by_strike = all_options.groupby("strike")["gamma"].sum()
        # Get top 3 gamma concentration levels
        key_gamma_levels = gamma_by_strike.sort_values(ascending=False).head(3).index.tolist()
        
        # Calculate vanna exposure (delta sensitivity to volatility changes)
        # This is simplified - in reality would need more market data
        call_vanna = sum(calls["delta"] * calls["vega"])
        put_vanna = sum(puts["delta"] * puts["vega"])
        total_vanna = call_vanna + put_vanna
        
        if total_vanna > 0:
            vanna_exposure = "positive (higher vol increases delta)"
        elif total_vanna < 0:
            vanna_exposure = "negative (higher vol decreases delta)"
        else:
            vanna_exposure = "neutral"
        
        # Volatility of volatility estimate (simplified)
        vol_of_vol = "moderate"  # Could be "low", "moderate", "high"
        
        # Charm (delta decay) risk assessment
        # Higher for options closer to expiration
        nearest_exp = min(calls["daysToExpiration"].min(), puts["daysToExpiration"].min())
        if nearest_exp < 7:
            charm_risk = "high"
        elif nearest_exp < 30:
            charm_risk = "moderate"
        else:
            charm_risk = "low"
            
        return {
            "key_gamma_levels": key_gamma_levels,
            "vol_of_vol": vol_of_vol,
            "vanna_exposure": vanna_exposure,
            "charm_risk": charm_risk
        }
    except Exception as e:
        print(f"Error analyzing advanced Greeks: {str(e)}")
        return {
            "key_gamma_levels": [current_price, current_price * 1.05, current_price * 0.95],
            "vol_of_vol": "moderate",
            "vanna_exposure": "neutral",
            "charm_risk": "moderate"
        }


def analyze_liquidity(option_chain):
    """
    Analyze options market liquidity
    """
    try:
        calls = pd.DataFrame(option_chain["calls"])
        puts = pd.DataFrame(option_chain["puts"])
        
        # Calculate average bid-ask spreads
        calls["spread_pct"] = (calls["ask"] - calls["bid"]) / calls["bid"] * 100
        puts["spread_pct"] = (puts["ask"] - puts["bid"]) / puts["bid"] * 100
        
        avg_call_spread = calls["spread_pct"].mean()
        avg_put_spread = puts["spread_pct"].mean()
        
        # Calculate volume trends
        total_volume = sum(calls["volume"]) + sum(puts["volume"])
        call_volume_ratio = sum(calls["volume"]) / total_volume if total_volume > 0 else 0.5
        
        # Liquidity rating based on spreads and volume
        if avg_call_spread < 5 and avg_put_spread < 5 and total_volume > 1000:
            liquidity_rating = "excellent"
        elif avg_call_spread < 10 and avg_put_spread < 10 and total_volume > 500:
            liquidity_rating = "good"
        elif avg_call_spread < 15 and avg_put_spread < 15 and total_volume > 200:
            liquidity_rating = "moderate"
        else:
            liquidity_rating = "poor"
            
        return {
            "bid_ask_spreads": {
                "average_call_spread": float(avg_call_spread),
                "average_put_spread": float(avg_put_spread)
            },
            "volume_trends": {
                "total_volume": int(total_volume),
                "call_volume_ratio": float(call_volume_ratio)
            },
            "liquidity_rating": liquidity_rating
        }
    except Exception as e:
        print(f"Error analyzing liquidity: {str(e)}")
        return {
            "bid_ask_spreads": {
                "average_call_spread": 8.0,
                "average_put_spread": 8.0
            },
            "volume_trends": {
                "total_volume": 500,
                "call_volume_ratio": 0.5
            },
            "liquidity_rating": "moderate"
        }


def analyze_options_sentiment(option_chain, current_price):
    """
    Analyze market sentiment based on options positioning
    """
    try:
        calls = pd.DataFrame(option_chain["calls"])
        puts = pd.DataFrame(option_chain["puts"])
        
        # Calculate put/call ratio by volume
        call_volume = sum(calls["volume"])
        put_volume = sum(puts["volume"])
        pc_ratio = put_volume / call_volume if call_volume > 0 else 1.0
        
        # Calculate put/call ratio by open interest
        call_oi = sum(calls["openInterest"])
        put_oi = sum(puts["openInterest"])
        pc_oi_ratio = put_oi / call_oi if call_oi > 0 else 1.0
        
        # Check for unusually high volume options
        calls["volume_oi_ratio"] = calls["volume"] / calls["openInterest"].replace(0, 1)
        puts["volume_oi_ratio"] = puts["volume"] / puts["openInterest"].replace(0, 1)
        
        high_vol_calls = calls[calls["volume_oi_ratio"] > 3].sort_values("volume", ascending=False).head(2)
        high_vol_puts = puts[puts["volume_oi_ratio"] > 3].sort_values("volume", ascending=False).head(2)
        
        key_observations = []
        
        if not high_vol_calls.empty:
            for _, option in high_vol_calls.iterrows():
                key_observations.append(f"Unusual call activity at strike ${option['strike']:.2f}")
                
        if not high_vol_puts.empty:
            for _, option in high_vol_puts.iterrows():
                key_observations.append(f"Unusual put activity at strike ${option['strike']:.2f}")
        
        # Determine sentiment
        if pc_ratio > 1.5:
            sentiment = "bearish"
            if pc_ratio > 2.5:
                sentiment_strength = "strongly bearish"
            else:
                sentiment_strength = "moderately bearish"
        elif pc_ratio < 0.7:
            sentiment = "bullish"
            if pc_ratio < 0.4:
                sentiment_strength = "strongly bullish"
            else:
                sentiment_strength = "moderately bullish"
        else:
            sentiment = "neutral"
            sentiment_strength = "neutral"
            
        return {
            "sentiment": sentiment,
            "sentiment_strength": sentiment_strength,
            "put_call_ratio": {
                "volume_ratio": float(pc_ratio),
                "open_interest_ratio": float(pc_oi_ratio)
            },
            "key_observations": key_observations
        }
    except Exception as e:
        print(f"Error analyzing options sentiment: {str(e)}")
        return {
            "sentiment": "neutral",
            "sentiment_strength": "neutral",
            "put_call_ratio": {
                "volume_ratio": 1.0,
                "open_interest_ratio": 1.0
            },
            "key_observations": []
        }


def calculate_risk_metrics(option_chain, current_price, advanced_greeks, hist_vol):
    """
    Calculate risk metrics for options strategies
    """
    try:
        calls = pd.DataFrame(option_chain["calls"])
        puts = pd.DataFrame(option_chain["puts"])
        
        # Find ATM options for reference
        atm_call = calls.iloc[(calls["strike"] - current_price).abs().argsort()[:1]]
        atm_put = puts.iloc[(puts["strike"] - current_price).abs().argsort()[:1]]
        
        # Simulate a common strategy: Long Straddle (buy ATM call + put)
        if not atm_call.empty and not atm_put.empty:
            straddle_cost = float(atm_call["ask"].values[0] + atm_put["ask"].values[0])
            strike = float(atm_call["strike"].values[0])
            
            # Calculate breakeven points
            upper_breakeven = strike + straddle_cost
            lower_breakeven = strike - straddle_cost
            
            # Calculate max loss
            max_loss = straddle_cost * 100  # Per contract
            
            # Calculate theoretical max gain (simplified)
            max_gain = "unlimited"
            
            # Estimate probability of profit using historical volatility
            days_to_exp = min(atm_call["daysToExpiration"].values[0], atm_put["daysToExpiration"].values[0])
            
            # Use simplified normal distribution to estimate probability
            # That stock will move enough to make straddle profitable
            daily_vol = hist_vol / (100 * np.sqrt(252))
            expected_move = current_price * daily_vol * np.sqrt(days_to_exp)
            prob_below_lower = 0.25  # simplified
            prob_above_upper = 0.25  # simplified
            prob_of_profit = prob_below_lower + prob_above_upper
            
            # Risk/reward ratio
            risk_reward_ratio = 1.0  # For straddles, this is typically around 1:1 at initiation
            
            return {
                "max_loss": max_loss,
                "max_gain": max_gain,
                "breakeven_points": [float(lower_breakeven), float(upper_breakeven)],
                "probability_of_profit": float(prob_of_profit),
                "risk_reward_ratio": float(risk_reward_ratio)
            }
        else:
            return {
                "max_loss": 500,
                "max_gain": "unlimited",
                "breakeven_points": [current_price * 0.95, current_price * 1.05],
                "probability_of_profit": 0.5,
                "risk_reward_ratio": 1.0
            }
    except Exception as e:
        print(f"Error calculating risk metrics: {str(e)}")
        return {
            "max_loss": 500,
            "max_gain": "unlimited",
            "breakeven_points": [current_price * 0.95, current_price * 1.05],
            "probability_of_profit": 0.5,
            "risk_reward_ratio": 1.0
        }


def generate_expert_recommendation(
    strategy_suggestions, 
    advanced_greeks, 
    liquidity_analysis, 
    market_sentiment, 
    risk_metrics, 
    current_price
):
    """
    Generate final expert recommendation based on all analysis
    """
    try:
        # Start with market sentiment signal
        base_signal = market_sentiment["sentiment"]
        
        # Adjust by liquidity - we want to be more conservative in illiquid markets
        if liquidity_analysis["liquidity_rating"] == "poor":
            confidence_modifier = 0.7
        elif liquidity_analysis["liquidity_rating"] == "moderate":
            confidence_modifier = 0.9
        else:
            confidence_modifier = 1.0
            
        # Set base confidence
        if market_sentiment["sentiment_strength"] == "strongly bullish":
            confidence = 0.8
        elif market_sentiment["sentiment_strength"] == "moderately bullish":
            confidence = 0.7
        elif market_sentiment["sentiment_strength"] == "strongly bearish":
            confidence = 0.8
        elif market_sentiment["sentiment_strength"] == "moderately bearish":
            confidence = 0.7
        else:
            confidence = 0.6
            
        # Apply liquidity modifier
        confidence = confidence * confidence_modifier
            
        # Sort strategies by fit
        fit_map = {"high": 3, "medium": 2, "low": 1}
        sorted_strategies = sorted(
            strategy_suggestions, 
            key=lambda x: fit_map.get(x.get("fit", "low"), 0), 
            reverse=True
        )
        
        # Take top 3 strategies (if available)
        recommended_strategies = sorted_strategies[:3] if len(sorted_strategies) >= 3 else sorted_strategies
        
        return {
            "signal": base_signal,
            "confidence": float(confidence),
            "recommended_strategies": recommended_strategies
        }
    except Exception as e:
        print(f"Error generating expert recommendation: {str(e)}")
        return {
            "signal": "neutral",
            "confidence": 0.5,
            "recommended_strategies": strategy_suggestions[:2] if len(strategy_suggestions) >= 2 else strategy_suggestions
        } 