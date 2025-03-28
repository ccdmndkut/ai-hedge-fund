import os
import pandas as pd
import requests
import json
from datetime import datetime, timedelta

from data.cache import get_cache
from data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
)

# Global cache instance
_cache = get_cache()


def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data from cache or API."""
    # Check cache first
    if cached_data := _cache.get_prices(ticker):
        # Filter cached data by date range and convert to Price objects
        filtered_data = [Price(**price) for price in cached_data if start_date <= price["time"] <= end_date]
        if filtered_data:
            return filtered_data

    # If not in cache or no data in range, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

    # Parse response with Pydantic model
    price_response = PriceResponse(**response.json())
    prices = price_response.prices

    if not prices:
        return []

    # Cache the results as dicts
    _cache.set_prices(ticker, [p.model_dump() for p in prices])
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or API."""
    # Check cache first
    if cached_data := _cache.get_financial_metrics(ticker):
        # Filter cached data by date and limit
        filtered_data = [FinancialMetrics(**metric) for metric in cached_data if metric["report_period"] <= end_date]
        filtered_data.sort(key=lambda x: x.report_period, reverse=True)
        if filtered_data:
            return filtered_data[:limit]

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = f"https://api.financialdatasets.ai/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

    # Parse response with Pydantic model
    metrics_response = FinancialMetricsResponse(**response.json())
    # Return the FinancialMetrics objects directly instead of converting to dict
    financial_metrics = metrics_response.financial_metrics

    if not financial_metrics:
        return []

    # Cache the results as dicts
    _cache.set_financial_metrics(ticker, [m.model_dump() for m in financial_metrics])
    return financial_metrics


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """Fetch line items from API."""
    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = "https://api.financialdatasets.ai/financials/search/line-items"

    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": end_date,
        "period": period,
        "limit": limit,
    }
    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
    data = response.json()
    response_model = LineItemResponse(**data)
    search_results = response_model.search_results
    if not search_results:
        return []

    # Cache the results
    return search_results[:limit]


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or API."""
    # Check cache first
    if cached_data := _cache.get_insider_trades(ticker):
        # Filter cached data by date range
        filtered_data = [InsiderTrade(**trade) for trade in cached_data 
                        if (start_date is None or (trade.get("transaction_date") or trade["filing_date"]) >= start_date)
                        and (trade.get("transaction_date") or trade["filing_date"]) <= end_date]
        filtered_data.sort(key=lambda x: x.transaction_date or x.filing_date, reverse=True)
        if filtered_data:
            return filtered_data

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_trades = []
    current_end_date = end_date
    
    while True:
        url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={limit}"
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
        
        data = response.json()
        response_model = InsiderTradeResponse(**data)
        insider_trades = response_model.insider_trades
        
        if not insider_trades:
            break
            
        all_trades.extend(insider_trades)
        
        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(insider_trades) < limit:
            break
            
        # Update end_date to the oldest filing date from current batch for next iteration
        current_end_date = min(trade.filing_date for trade in insider_trades).split('T')[0]
        
        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_trades:
        return []

    # Cache the results
    _cache.set_insider_trades(ticker, [trade.model_dump() for trade in all_trades])
    return all_trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """Fetch company news from cache or API."""
    # Check cache first
    if cached_data := _cache.get_company_news(ticker):
        # Filter cached data by date range
        filtered_data = [CompanyNews(**news) for news in cached_data 
                        if (start_date is None or news["date"] >= start_date)
                        and news["date"] <= end_date]
        filtered_data.sort(key=lambda x: x.date, reverse=True)
        if filtered_data:
            return filtered_data

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_news = []
    current_end_date = end_date
    
    while True:
        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
        
        data = response.json()
        response_model = CompanyNewsResponse(**data)
        company_news = response_model.news
        
        if not company_news:
            break
            
        all_news.extend(company_news)
        
        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(company_news) < limit:
            break
            
        # Update end_date to the oldest date from current batch for next iteration
        current_end_date = min(news.date for news in company_news).split('T')[0]
        
        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_news:
        return []

    # Cache the results
    _cache.set_company_news(ticker, [news.model_dump() for news in all_news])
    return all_news


def get_option_chain(ticker: str) -> dict:
    """
    Fetch options chain data for a given ticker.
    Returns a dictionary with 'calls' and 'puts' keys containing options data.
    
    Due to API limitations in this example, this returns simulated options data.
    In a real implementation, this would fetch from a financial data provider like:
    - Interactive Brokers, TD Ameritrade, or other brokers' APIs
    - Market data providers like IEX Cloud, Polygon.io, etc.
    """
    # Check cache first
    if cached_data := _cache.get_option_chain(ticker):
        return cached_data
    
    # If not in cache, we'll generate synthetic options data based on current stock price
    # In a real implementation, this would fetch from an actual API
    try:
        # Get current stock price to generate realistic option chain
        prices = get_prices(
            ticker=ticker,
            start_date=(datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
        )
        
        if not prices:
            return {"calls": [], "puts": []}
        
        current_price = prices[-1].close
        
        # Generate synthetic options data
        options_data = _generate_synthetic_option_chain(ticker, current_price)
        
        # Cache the results
        _cache.set_option_chain(ticker, options_data)
        
        return options_data
    
    except Exception as e:
        print(f"Error fetching options data for {ticker}: {str(e)}")
        return {"calls": [], "puts": []}


def _generate_synthetic_option_chain(ticker: str, current_price: float) -> dict:
    """Generate synthetic options data for simulation purposes."""
    expiration_dates = [
        (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),  # 1 week
        (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),  # 1 month
        (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d"),  # 3 months
    ]
    
    # Generate strike prices around current price
    strike_step = round(current_price * 0.025, 2)  # 2.5% increments
    strikes = [round(current_price + (i - 10) * strike_step, 2) for i in range(21)]  # 10 below, 10 above current price
    
    calls = []
    puts = []
    
    for expiration in expiration_dates:
        days_to_expiry = (datetime.strptime(expiration, "%Y-%m-%d") - datetime.now()).days
        
        for strike in strikes:
            # Calculate synthetic Greeks and IV based on strike distance from current price
            distance_from_current = abs(strike - current_price) / current_price
            time_factor = days_to_expiry / 365
            
            # Calculate synthetic option prices and greeks
            call_price = max(0.1, (current_price - strike) + (current_price * 0.05 * time_factor))
            put_price = max(0.1, (strike - current_price) + (current_price * 0.05 * time_factor))
            
            # Basic Black-Scholes-inspired calculations (very simplified)
            implied_vol = 0.25 + distance_from_current  # Higher IV for strikes further from current price
            
            # Volume distribution: more volume near the money
            atm_factor = 1 - min(1, distance_from_current * 5)
            base_volume = int(5000 * atm_factor)
            volume_noise = int(base_volume * 0.5)  # 50% random variation
            
            # Realistic volume and open interest
            volume = max(1, base_volume + int(volume_noise * (0.5 - 0.5)))
            open_interest = max(10, volume * 4 + int(volume * 2 * (0.5 - 0.5)))
            
            # Add more volume to puts for lower strikes (put skew)
            if strike < current_price:
                put_vol_skew = 1 + (current_price - strike) / current_price
            else:
                put_vol_skew = 1
                
            # Generate call option
            call = {
                "ticker": ticker,
                "strike": strike,
                "expiration": expiration,
                "type": "call",
                "bid": round(call_price * 0.95, 2),
                "ask": round(call_price * 1.05, 2),
                "last": round(call_price, 2),
                "volume": volume,
                "openInterest": open_interest,
                "impliedVolatility": round(implied_vol, 4),
                "delta": round(max(0, min(1, 0.5 + (current_price - strike) / (current_price * 0.2))), 4),
                "gamma": round(max(0, 4 * atm_factor * time_factor), 4),
                "theta": round(-call_price * 0.01 / max(1, days_to_expiry), 4),
                "vega": round(current_price * 0.01 * time_factor, 4),
            }
            
            # Generate put option
            put = {
                "ticker": ticker,
                "strike": strike,
                "expiration": expiration,
                "type": "put",
                "bid": round(put_price * 0.95, 2),
                "ask": round(put_price * 1.05, 2),
                "last": round(put_price, 2),
                "volume": int(volume * put_vol_skew),
                "openInterest": int(open_interest * put_vol_skew),
                "impliedVolatility": round(implied_vol * put_vol_skew, 4),
                "delta": round(min(0, max(-1, -0.5 + (current_price - strike) / (current_price * 0.2))), 4),
                "gamma": round(max(0, 4 * atm_factor * time_factor), 4),
                "theta": round(-put_price * 0.01 / max(1, days_to_expiry), 4),
                "vega": round(current_price * 0.01 * time_factor, 4),
            }
            
            calls.append(call)
            puts.append(put)
    
    return {"calls": calls, "puts": puts}


def get_market_cap(
    ticker: str,
    end_date: str,
) -> float | None:
    """Get market capitalization for a ticker on a specific date."""
    metrics = get_financial_metrics(ticker, end_date, limit=1)
    if not metrics:
        return None
    
    return metrics[0].market_cap


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert a list of Price objects to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    
    # Convert numeric columns to float
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
    return df


def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Convenience function to get price data as a DataFrame."""
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)
