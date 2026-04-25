import yfinance as yf
import pandas as pd

ticker = "GOOGL"

#downloaded = yf.download(ticker, period="max", interval = "1d")
downloaded = yf.download(ticker, start="2022-01-01", end = "2022-12-31", interval = "1d")
if downloaded is None or downloaded.empty:
	raise RuntimeError(f"Failed to download data for ticker: {ticker}")

df: pd.DataFrame = downloaded.reset_index()
df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

df.to_json(f"{ticker}_test_set.json", orient="records", date_format="iso", indent=4)

print("Data saved")
