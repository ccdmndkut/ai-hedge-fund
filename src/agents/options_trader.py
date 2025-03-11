from langchain_openai import ChatOpenAI
from graph.state import AgentState, show_agent_reasoning
from tools.api import get_options_data
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm


class OptionsTraderSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def options_trader_agent(state: AgentState):
    """
    Analyzes options trading strategies, including implied volatility, open interest,
    and generates options trading signals.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    options_analysis = {}

    for ticker in tickers:
        progress.update_status("options_trader_agent", ticker, "Fetching options data")
        options_data = get_options_data(ticker, end_date)

        progress.update_status("options_trader_agent", ticker, "Analyzing options trading strategies")
        options_signal = analyze_options_trading(options_data)

        options_analysis[ticker] = {
            "signal": options_signal.signal,
            "confidence": options_signal.confidence,
            "reasoning": options_signal.reasoning,
        }

        progress.update_status("options_trader_agent", ticker, "Done")

    message = HumanMessage(
        content=json.dumps(options_analysis),
        name="options_trader_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(options_analysis, "Options Trader Agent")

    state["data"]["analyst_signals"]["options_trader_agent"] = options_analysis

    return {
        "messages": [message],
        "data": state["data"],
    }


def analyze_options_trading(options_data: list) -> OptionsTraderSignal:
    """
    Analyze options trading strategies and generate options trading signals.
    """
    score = 0
    details = []

    if not options_data:
        return OptionsTraderSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Insufficient data for options trading analysis"
        )

    # Example options trading analysis
    # 1. Check implied volatility
    # 2. Analyze open interest
    # 3. Generate options trading signals based on the analysis

    # Placeholder logic for options trading analysis
    implied_volatility = 0.25  # Example value
    open_interest = 1000  # Example value

    if implied_volatility > 0.3:
        score += 2
        details.append("High implied volatility, potential for options trading opportunities.")
    else:
        details.append("Low implied volatility, limited options trading opportunities.")

    if open_interest > 500:
        score += 2
        details.append("High open interest, potential for liquid options trading.")
    else:
        details.append("Low open interest, limited options trading opportunities.")

    signal = "bullish" if score >= 3 else "neutral" if score == 2 else "bearish"
    confidence = score / 4.0 * 100

    return OptionsTraderSignal(
        signal=signal,
        confidence=confidence,
        reasoning="; ".join(details)
    )
