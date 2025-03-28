import os
import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta, date

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

# Rate limiting configuration for Polygon.io
_last_api_call_time = 0
_min_time_between_calls = 0.12  # ~8 requests per second (conservative for free tier)
_max_retries = 3
_retry_delay = 60  # seconds

def _handle_empty_data(data_type: str, ticker: str) -> None:
    """Log warning about empty data and ensure we don't break workflows."""
    print(f"Warning: No {data_type} data available for {ticker}. Using empty data.")
    # This function helps standardize how we handle missing data

def _make_polygon_api_call(url, headers, method='get', json_data=None):
    """Make a rate-limited API call to Polygon with retries for 429 errors."""
    global _last_api_call_time
    
    # Rate limiting
    current_time = time.time()
    time_since_last_call = current_time - _last_api_call_time
    if time_since_last_call < _min_time_between_calls:
        sleep_time = _min_time_between_calls - time_since_last_call
        time.sleep(sleep_time)
    
    # Update last call time
    _last_api_call_time = time.time()
    
    # Make the API call with retries
    for attempt in range(_max_retries):
        if method.lower() == 'get':
            response = requests.get(url, headers=headers)
        elif method.lower() == 'post':
            response = requests.post(url, headers=headers, json=json_data)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        # If successful or not a rate limit error, return the response
        if response.status_code != 429:
            return response
        
        # Handle rate limit error by waiting and retrying
        print(f"Rate limit exceeded. Waiting {_retry_delay} seconds before retry {attempt+1}/{_max_retries}")
        time.sleep(_retry_delay)
    
    # If we've exhausted retries, return the last response
    return response


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
    if api_key := os.environ.get("POLYGON_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        raise Exception("Polygon API key not found in environment variables")

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc"
    response = _make_polygon_api_call(url, headers)
    if response.status_code == 404:
        print(f"Warning: Price data for '{ticker}' not found in Polygon API")
        _handle_empty_data("price", ticker)
        return []
    elif response.status_code != 200:
        print(f"Warning: Error fetching price data: {ticker} - {response.status_code} - {response.text}")
        _handle_empty_data("price", ticker)
        return []

    # Parse response and convert to Price objects
    data = response.json()
    if "results" not in data or not data["results"]:
        _handle_empty_data("price", ticker)
        return []
    
    prices = []
    for item in data["results"]:
        # Convert timestamp (milliseconds) to ISO format string
        timestamp_ms = item["t"]
        time_str = datetime.fromtimestamp(timestamp_ms / 1000).strftime("%Y-%m-%d")
        
        price = Price(
            open=item["o"],
            close=item["c"],
            high=item["h"],
            low=item["l"],
            volume=item["v"],
            time=time_str
        )
        prices.append(price)
    
    price_response = PriceResponse(ticker=ticker, prices=prices)
    prices = price_response.prices

    if not prices:
        _handle_empty_data("price", ticker)
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
    if api_key := os.environ.get("POLYGON_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        raise Exception("Polygon API key not found in environment variables")
    
    # For financial metrics, we'll need to combine data from multiple endpoints:
    # 1. Get ticker details for market cap
    # 2. Get financials for ratios
    
    # 1. Get ticker details
    url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
    response = _make_polygon_api_call(url, headers)
    if response.status_code == 404:
        print(f"Warning: Ticker '{ticker}' not found in Polygon API")
        _handle_empty_data("financial metrics", ticker)
        return []
    elif response.status_code != 200:
        print(f"Warning: Error fetching ticker details: {ticker} - {response.status_code} - {response.text}")
        _handle_empty_data("financial metrics", ticker)
        return []
    
    ticker_details = response.json()["results"]
    market_cap = ticker_details.get("market_cap")
    
    # 2. Get financial ratios
    url = f"https://api.polygon.io/v3/reference/financials?ticker={ticker}&limit={limit}"
    response = _make_polygon_api_call(url, headers)
    if response.status_code == 404:
        print(f"Warning: Financial data for '{ticker}' not found in Polygon API")
        _handle_empty_data("financial ratios", ticker)
        return []
    elif response.status_code != 200:
        print(f"Warning: Error fetching financials: {ticker} - {response.status_code} - {response.text}")
        _handle_empty_data("financial ratios", ticker)
        return []
    
    financial_data = response.json()["results"]
    if not financial_data:
        _handle_empty_data("financial data", ticker)
        return []
    
    financial_metrics = []
    for item in financial_data:
        # Filter by end_date
        report_period = item.get("end_date", "")
        if report_period > end_date:
            continue
            
        # Extract available metrics
        ratios = item.get("ratios", {})
        
        metric = FinancialMetrics(
            ticker=ticker,
            report_period=report_period,
            period=period,  # Use the requested period
            currency="USD",  # Default to USD
            market_cap=market_cap,
            enterprise_value=None,  # Not directly available
            price_to_earnings_ratio=ratios.get("pe_ratio"),
            price_to_book_ratio=ratios.get("price_to_book_value"),
            price_to_sales_ratio=ratios.get("price_to_sales"),
            enterprise_value_to_ebitda_ratio=None,  # Not directly available
            enterprise_value_to_revenue_ratio=None,  # Not directly available
            free_cash_flow_yield=None,  # Not directly available
            peg_ratio=None,  # Not directly available
            gross_margin=ratios.get("gross_margin"),
            operating_margin=ratios.get("operating_margin"),
            net_margin=ratios.get("net_margin"),
            return_on_equity=ratios.get("return_on_equity"),
            return_on_assets=ratios.get("return_on_assets"),
            return_on_invested_capital=None,  # Not directly available
            asset_turnover=None,  # Not directly available
            inventory_turnover=None,  # Not directly available
            receivables_turnover=None,  # Not directly available
            days_sales_outstanding=None,  # Not directly available
            operating_cycle=None,  # Not directly available
            working_capital_turnover=None,  # Not directly available
            current_ratio=ratios.get("current_ratio"),
            quick_ratio=ratios.get("quick_ratio"),
            cash_ratio=None,  # Not directly available
            operating_cash_flow_ratio=None,  # Not directly available
            debt_to_equity=ratios.get("debt_to_equity"),
            debt_to_assets=None,  # Not directly available
            interest_coverage=None,  # Not directly available
            revenue_growth=None,  # Would need to calculate from sequential data
            earnings_growth=None,  # Would need to calculate from sequential data
            book_value_growth=None,  # Would need to calculate from sequential data
            earnings_per_share_growth=None,  # Would need to calculate from sequential data
            free_cash_flow_growth=None,  # Would need to calculate from sequential data
            operating_income_growth=None,  # Would need to calculate from sequential data
            ebitda_growth=None,  # Would need to calculate from sequential data
            payout_ratio=ratios.get("dividend_payout"),
            earnings_per_share=item.get("financials", {}).get("income_statement", {}).get("diluted_eps"),
            book_value_per_share=None,  # Not directly available
            free_cash_flow_per_share=None,  # Not directly available
        )
        financial_metrics.append(metric)
    
    # Sort by report_period in descending order
    financial_metrics.sort(key=lambda x: x.report_period, reverse=True)
    
    if not financial_metrics:
        _handle_empty_data("filtered financial metrics", ticker)
        return []
    
    # Limit results
    financial_metrics = financial_metrics[:limit]
    
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
    # With Polygon.io we'll pull financial statements and extract specific line items
    headers = {}
    if api_key := os.environ.get("POLYGON_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        raise Exception("Polygon API key not found in environment variables")
    
    url = f"https://api.polygon.io/v3/reference/financials?ticker={ticker}&limit={limit}"
    response = _make_polygon_api_call(url, headers)
    if response.status_code == 404:
        print(f"Warning: Financial data for '{ticker}' not found in Polygon API")
        _handle_empty_data("line items", ticker)
        return []
    elif response.status_code != 200:
        print(f"Warning: Error fetching financials: {ticker} - {response.status_code} - {response.text}")
        _handle_empty_data("line items", ticker)
        return []
    
    financial_data = response.json()["results"]
    if not financial_data:
        _handle_empty_data("line items", ticker)
        return []
    
    search_results = []
    for item in financial_data:
        # Filter by end_date
        report_period = item.get("end_date", "")
        if report_period > end_date:
            continue
        
        # Extract available financial data
        financials = item.get("financials", {})
        income_statement = financials.get("income_statement", {})
        balance_sheet = financials.get("balance_sheet", {})
        cash_flow_statement = financials.get("cash_flow_statement", {})
        
        # Create a flat dictionary of all financial data
        all_financials = {}
        for key, value in income_statement.items():
            all_financials[f"income_statement.{key}"] = value
        for key, value in balance_sheet.items():
            all_financials[f"balance_sheet.{key}"] = value
        for key, value in cash_flow_statement.items():
            all_financials[f"cash_flow_statement.{key}"] = value
        
        # Filter for requested line items
        line_item_data = {}
        for requested_item in line_items:
            # Try to find an exact match or a partial match
            for key, value in all_financials.items():
                if requested_item == key or requested_item.lower() in key.lower():
                    line_item_data[requested_item] = value
                    break
        
        if line_item_data:
            result = LineItem(
                ticker=ticker,
                report_period=report_period,
                period=period,  # Use the requested period
                currency="USD",  # Default to USD
                **line_item_data
            )
            search_results.append(result)
    
    # Sort by report_period in descending order
    search_results.sort(key=lambda x: x.report_period, reverse=True)
    
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
    if api_key := os.environ.get("POLYGON_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        raise Exception("Polygon API key not found in environment variables")
    
    # Polygon doesn't have a direct insider trades endpoint, so we'll use the ticker news endpoint
    # and filter for SEC Form 4 filings which indicate insider trading
    all_trades = []
    
    # This is a simplified implementation since we can't directly get insider trades
    # from Polygon in the same format as the original API
    # For a production system, you might want to integrate with SEC Edgar API
    # or another specialized data provider for insider trades
    
    # Return empty list for now - in a real implementation you would connect to SEC API
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
    if api_key := os.environ.get("POLYGON_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        raise Exception("Polygon API key not found in environment variables")
    
    all_news = []
    
    # Format dates for Polygon API
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
    # Convert to timestamp in milliseconds
    end_timestamp = int(end_date_obj.timestamp() * 1000)
    
    start_timestamp = None
    if start_date:
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        start_timestamp = int(start_date_obj.timestamp() * 1000)
    
    # Polygon API for news
    url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&order=desc&limit={min(limit, 50)}"
    if start_timestamp:
        url += f"&published_utc.gte={start_date}"
    if end_timestamp:
        url += f"&published_utc.lte={end_date}"
    
    response = _make_polygon_api_call(url, headers)
    if response.status_code == 404:
        print(f"Warning: News data for '{ticker}' not found in Polygon API")
        return []
    elif response.status_code != 200:
        print(f"Warning: Error fetching news: {ticker} - {response.status_code} - {response.text}")
        return []
    
    news_data = response.json()["results"]
    
    for item in news_data:
        # Convert timestamp to date string
        published_utc = item.get("published_utc", "")
        if published_utc:
            date_str = published_utc.split("T")[0]  # Extract date part
        else:
            continue  # Skip items without a date
        
        news_item = CompanyNews(
            ticker=ticker,
            title=item.get("title", ""),
            author=item.get("author", ""),
            source=item.get("publisher", {}).get("name", ""),
            date=date_str,
            url=item.get("article_url", ""),
            sentiment=None  # Polygon doesn't provide sentiment analysis
        )
        all_news.append(news_item)
    
    if not all_news:
        return []
    
    # Sort by date in descending order
    all_news.sort(key=lambda x: x.date, reverse=True)
    
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
                "daysToExpiration": days_to_expiry,
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
                "daysToExpiration": days_to_expiry,
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
    """Convert price objects to pandas DataFrame."""
    if not prices:
        return pd.DataFrame()
    
    data = [
        {
            "date": p.time,
            "open": p.open,
            "high": p.high,
            "low": p.low,
            "close": p.close,
            "volume": p.volume,
        }
        for p in prices
    ]
    
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch price data and convert to pandas DataFrame."""
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)
