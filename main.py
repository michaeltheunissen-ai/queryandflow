"""
Share Risk Classifier - FastAPI Backend
Fetches real financial data from Yahoo Finance (free, no API key)
Classifies risk using local LM Studio LLM (free, no cost)
"""

import json
import threading
import webbrowser
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import yfinance as yf
import httpx
import uvicorn
import os
import sys

app = FastAPI(title="Share Risk Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

SYSTEM_PROMPT = """You are a financial analyst specialising in share risk classification.
You will be given real financial metrics for a listed company. Classify it into exactly one of these four risk buckets:

LOW RISK: Stable predictable cash flows, low debt-to-equity (below 0.5), modest fair P/E (10-20x), low volatility (beta under 1.0), consistent dividends.

MEDIUM RISK: Moderate growth tied to economic cycles, standard debt (D/E 0.5-1.5), average P/E aligned with industry (15-30x), moderate price swings (beta 0.8-1.3).

HIGH RISK: Unpredictable or negative cash flows, heavy debt (D/E above 1.5) or high cash burn, stretched P/E above 40x or negative, high volatility (beta above 1.3), no dividends.

BUBBLE: Fundamentals completely detached from price, astronomical P/E above 100x or nonsensical, price driven purely by momentum/narrative, no earnings justification.

Respond ONLY with a valid JSON object, no markdown, no backticks, no explanation outside the JSON:
{
  "bucket": "low",
  "confidence": "high",
  "metrics_assessment": {
    "cashflow": {"value": "describe what you see", "signal": "positive"},
    "balance_sheet": {"value": "describe what you see", "signal": "positive"},
    "valuation": {"value": "describe what you see", "signal": "neutral"},
    "volatility": {"value": "describe what you see", "signal": "neutral"}
  },
  "reasoning": "2-3 sentence plain English explanation of why this bucket was chosen based on the numbers provided."
}

signal must be exactly: "positive", "neutral", or "negative"
bucket must be exactly: "low", "medium", "high", or "bubble"
confidence must be exactly: "high", "medium", or "low"
"""

class TickerRequest(BaseModel):
    ticker: str

def get_html():
    """Read the frontend HTML file."""
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    html_path = os.path.join(base, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/", response_class=HTMLResponse)
async def root():
    return get_html()

@app.get("/health")
async def health():
    """Check if LM Studio is reachable."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get("http://localhost:1234/v1/models")
            lm_ok = r.status_code == 200
    except:
        lm_ok = False
    return {"status": "ok", "lm_studio": lm_ok}

@app.post("/analyse")
async def analyse(req: TickerRequest):
    ticker_raw = req.ticker.strip().upper()
    
    # Fetch from Yahoo Finance
    try:
        t = yf.Ticker(ticker_raw)
        info = t.info
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch data for {ticker_raw}: {str(e)}")

    # Check we got something meaningful
    if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker_raw}' not found on Yahoo Finance. For JSE stocks try adding .JO (e.g. MTN.JO)")

    def safe(key, default="N/A"):
        val = info.get(key)
        if val is None or val == "":
            return default
        if isinstance(val, float):
            return round(val, 2)
        return val

    company_name = safe("longName") or safe("shortName") or ticker_raw
    exchange = safe("exchange", "")
    currency = safe("currency", "")
    sector = safe("sector", "")
    industry = safe("industry", "")

    pe_ratio = safe("trailingPE")
    forward_pe = safe("forwardPE")
    price_to_sales = safe("priceToSalesTrailing12Months")
    price_to_book = safe("priceToBook")
    beta = safe("beta")
    debt_to_equity = safe("debtToEquity")
    current_ratio = safe("currentRatio")
    roe = safe("returnOnEquity")
    roa = safe("returnOnAssets")
    profit_margin = safe("profitMargins")
    revenue_growth = safe("revenueGrowth")
    earnings_growth = safe("earningsGrowth")
    free_cashflow = safe("freeCashflow")
    operating_cashflow = safe("operatingCashflow")
    dividend_yield = safe("dividendYield")
    payout_ratio = safe("payoutRatio")
    market_cap = safe("marketCap")
    current_price = safe("currentPrice") or safe("regularMarketPrice")
    fifty_two_week_high = safe("fiftyTwoWeekHigh")
    fifty_two_week_low = safe("fiftyTwoWeekLow")
    analyst_rating = safe("recommendationKey", "")
    target_price = safe("targetMeanPrice")
    shares_short_ratio = safe("shortRatio")
    gross_margins = safe("grossMargins")
    ebitda_margins = safe("ebitdaMargins")

    # Build the data summary for the LLM
    data_summary = f"""
Company: {company_name}
Ticker: {ticker_raw}
Exchange: {exchange}
Sector: {sector}
Industry: {industry}
Currency: {currency}

VALUATION METRICS:
- Current Price: {current_price}
- 52-week High: {fifty_two_week_high}
- 52-week Low: {fifty_two_week_low}
- Trailing P/E: {pe_ratio}
- Forward P/E: {forward_pe}
- Price/Sales: {price_to_sales}
- Price/Book: {price_to_book}
- Market Cap: {market_cap}
- Analyst Target Price: {target_price}
- Analyst Recommendation: {analyst_rating}

PROFITABILITY:
- Profit Margin: {profit_margin}
- Gross Margin: {gross_margins}
- EBITDA Margin: {ebitda_margins}
- Return on Equity: {roe}
- Return on Assets: {roa}
- Revenue Growth: {revenue_growth}
- Earnings Growth: {earnings_growth}

BALANCE SHEET:
- Debt/Equity: {debt_to_equity}
- Current Ratio: {current_ratio}
- Free Cash Flow: {free_cashflow}
- Operating Cash Flow: {operating_cashflow}

INCOME & DIVIDENDS:
- Dividend Yield: {dividend_yield}
- Payout Ratio: {payout_ratio}

MARKET BEHAVIOUR:
- Beta: {beta}
- Short Ratio: {shares_short_ratio}
"""

    # Send to local LM Studio
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                LM_STUDIO_URL,
                json={
                    "model": "local-model",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Classify this stock based on the following real financial data:\n{data_summary}"}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 800
                }
            )
            response.raise_for_status()
            llm_data = response.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="LM Studio is not running. Please start LM Studio and load a model.")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LM Studio took too long to respond. Try a smaller/faster model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LM Studio error: {str(e)}")

    raw_text = llm_data["choices"][0]["message"]["content"].strip()

    # Parse LLM JSON response
    try:
        clean = raw_text.replace("```json", "").replace("```", "").strip()
        classification = json.loads(clean)
    except json.JSONDecodeError:
        # Try to extract JSON if model added extra text
        import re
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            try:
                classification = json.loads(match.group())
            except:
                classification = {
                    "bucket": "high",
                    "confidence": "low",
                    "metrics_assessment": {
                        "cashflow": {"value": "Parse error", "signal": "neutral"},
                        "balance_sheet": {"value": "Parse error", "signal": "neutral"},
                        "valuation": {"value": "Parse error", "signal": "neutral"},
                        "volatility": {"value": "Parse error", "signal": "neutral"}
                    },
                    "reasoning": f"Model response could not be parsed. Raw: {raw_text[:300]}"
                }
        else:
            classification = {
                "bucket": "high",
                "confidence": "low",
                "metrics_assessment": {
                    "cashflow": {"value": "Parse error", "signal": "neutral"},
                    "balance_sheet": {"value": "Parse error", "signal": "neutral"},
                    "valuation": {"value": "Parse error", "signal": "neutral"},
                    "volatility": {"value": "Parse error", "signal": "neutral"}
                },
                "reasoning": f"Model response could not be parsed. Raw: {raw_text[:300]}"
            }

    return JSONResponse({
        "ticker": ticker_raw,
        "company": company_name,
        "exchange": exchange,
        "sector": sector,
        "currency": currency,
        "current_price": current_price,
        "market_cap": market_cap,
        "pe_ratio": pe_ratio,
        "beta": beta,
        "debt_to_equity": debt_to_equity,
        "dividend_yield": dividend_yield,
        "classification": classification
    })

def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://localhost:8003")

if __name__ == "__main__":
    print("=" * 55)
    print("  Share Risk Classifier")
    print("  Powered by Yahoo Finance + Local LLM")
    print("=" * 55)
    print("  Make sure LM Studio is running on port 1234")
    print("  Opening browser at http://localhost:8003")
    print("  Press Ctrl+C to stop")
    print("=" * 55)
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="warning")
