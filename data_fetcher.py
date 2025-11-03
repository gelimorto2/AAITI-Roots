"""
Data fetching module for historical BTC price data.
Supports multiple data sources for robust data collection.
"""

import pandas as pd
import yfinance as yf
import ccxt
from datetime import datetime, timedelta
import time


class BTCDataFetcher:
    """Fetch historical BTC price data from various sources."""
    
    def __init__(self):
        self.exchange = ccxt.binance()
    
    def fetch_from_yfinance(self, start_date, end_date, interval='1h'):
        """
        Fetch BTC data from Yahoo Finance.
        
        Args:
            start_date: Start date (str or datetime)
            end_date: End date (str or datetime)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
        
        Returns:
            pd.DataFrame: OHLCV data with timestamp index
        """
        try:
            btc = yf.Ticker("BTC-USD")
            df = btc.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                print("Warning: No data retrieved from yfinance")
                return None
            
            # Rename columns to standard format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Keep only OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            print(f"Fetched {len(df)} records from yfinance")
            return df
            
        except Exception as e:
            print(f"Error fetching from yfinance: {e}")
            return None
    
    def fetch_from_binance(self, days_back=30, timeframe='1h'):
        """
        Fetch BTC data from Binance via CCXT.
        
        Args:
            days_back: Number of days to fetch
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
        
        Returns:
            pd.DataFrame: OHLCV data with timestamp index
        """
        try:
            since = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', timeframe, since=since)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            print(f"Fetched {len(df)} records from Binance")
            return df
            
        except Exception as e:
            print(f"Error fetching from Binance: {e}")
            return None
    
    def fetch_data(self, source='yfinance', **kwargs):
        """
        Main method to fetch data from specified source.
        
        Args:
            source: Data source ('yfinance' or 'binance')
            **kwargs: Additional arguments for specific fetchers
        
        Returns:
            pd.DataFrame: OHLCV data
        """
        if source == 'yfinance':
            return self.fetch_from_yfinance(**kwargs)
        elif source == 'binance':
            return self.fetch_from_binance(**kwargs)
        else:
            raise ValueError(f"Unknown source: {source}")


def save_data(df, filename):
    """Save DataFrame to CSV file."""
    df.to_csv(filename)
    print(f"Data saved to {filename}")


def load_data(filename):
    """Load DataFrame from CSV file."""
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    print(f"Data loaded from {filename} with {len(df)} records")
    return df


if __name__ == "__main__":
    # Example usage
    fetcher = BTCDataFetcher()
    
    # Fetch 30 days of hourly data
    df = fetcher.fetch_data(
        source='yfinance',
        start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
        end_date=datetime.now().strftime('%Y-%m-%d'),
        interval='1h'
    )
    
    if df is not None:
        print(f"\nData shape: {df.shape}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nLast few rows:")
        print(df.tail())
