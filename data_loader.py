"""
Data loading and preprocessing module for financial time series.
"""
import logging
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
import yfinance as yf

# Set up logger first so it's available for import error messages
logger = logging.getLogger(__name__)

# Try to import pandas_ta for technical indicators
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    logger.warning("pandas_ta not available. Installing it now...")
    try:
        import pip
        pip.main(['install', 'pandas_ta'])
        import pandas_ta as ta
        PANDAS_TA_AVAILABLE = True
    except:
        logger.error("Failed to install pandas_ta. Some technical indicators may not be available.")
        PANDAS_TA_AVAILABLE = False

class DataLoader:
    """
    Handles loading and preprocessing of financial time series data.
    """
    
    def __init__(self, symbol: str = '^VIX', 
                 start_date: str = '2012-01-01',
                 end_date: str = '2023-12-31'):
        """
        Initialize the DataLoader.
        
        Args:
            symbol: Stock/Index symbol (default: '^VIX' for CBOE Volatility Index)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data = None
        self.processed_data = None
    
    def load_data(self) -> pd.DataFrame:
        """
        Load financial data from Yahoo Finance.
        
        Returns:
            DataFrame containing the raw financial data
        """
        try:
            logger.info(f"Loading data for {self.symbol} from {self.start_date} to {self.end_date}")
            self.raw_data = yf.download(
                self.symbol, 
                start=self.start_date, 
                end=self.end_date,
                progress=False,
                auto_adjust=True,  # Adjust for stock splits and dividends
                group_by='ticker',  # Group by ticker for consistency
                threads=True       # Use threads for faster download
            )
            
            # Ensure we have data
            if self.raw_data.empty:
                raise ValueError(f"No data returned for {self.symbol} in the specified date range")
            
            # Handle MultiIndex columns if present
            if isinstance(self.raw_data.columns, pd.MultiIndex):
                # Keep only the second level (OHLCV) as column names
                self.raw_data.columns = self.raw_data.columns.get_level_values(1)
            
            # Ensure we have all required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in self.raw_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in downloaded data: {missing_columns}")
            
            # Ensure numeric data types
            for col in required_columns:
                self.raw_data[col] = pd.to_numeric(self.raw_data[col], errors='coerce')
            
            # Drop any rows with missing essential data
            self.raw_data = self.raw_data.dropna(subset=required_columns)
                
            logger.info(f"Successfully loaded {len(self.raw_data)} data points")
            
            # Debug info
            print(f"\nRaw data shape: {self.raw_data.shape}")
            print(f"Raw data columns: {self.raw_data.columns.tolist()}")
            print(f"First few rows of raw data:")
            print(self.raw_data.head())
            
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            logger.exception("Detailed error:")
            raise
    
    def _calculate_rsi(self, close_prices: np.ndarray, window: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index (RSI) without TA-Lib."""
        delta = pd.Series(close_prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, close_prices: np.ndarray, 
                       fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """
        Calculate MACD (Moving Average Convergence Divergence) without TA-Lib.
        
        Args:
            close_prices: Array of closing prices
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        try:
            close_series = pd.Series(close_prices)
            exp1 = close_series.ewm(span=fast, adjust=False).mean()
            exp2 = close_series.ewm(span=slow, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            return macd_line.values, signal_line.values, histogram.values
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            # Return arrays of NaN with same length as input on error
            nan_array = np.full_like(close_prices, np.nan)
            return nan_array, nan_array, nan_array
        
    def add_wavelet_features(self, data: pd.DataFrame, target_column: str = 'Close', 
                           levels: int = 4, wavelet: str = 'db4') -> pd.DataFrame:
        """
        Add wavelet-based features to the dataset.
        
        Args:
            data: Input DataFrame with time series data
            target_column: Name of the target column for wavelet decomposition
            levels: Number of wavelet decomposition levels
            wavelet: Type of wavelet to use (default: 'db4' - Daubechies 4)
            
        Returns:
            DataFrame with added wavelet features
        """
        from feature_engineering import WaveletTransform
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
            
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Initialize wavelet transform
        wt = WaveletTransform(wavelet=wavelet, levels=levels)
        
        try:
            # Apply wavelet transform to the target column
            coeffs = wt.decompose(df[target_column].values)
            
            # Add approximation and detail coefficients as features
            for i, coeff in enumerate(coeffs):
                # Pad coefficients to match original length if needed
                if len(coeff) < len(df):
                    padded_coeff = np.pad(coeff, (0, len(df) - len(coeff)), 'edge')
                else:
                    padded_coeff = coeff[:len(df)]
                
                # Add as new feature
                if i == 0:
                    df[f'wavelet_approx_{i}'] = padded_coeff
                else:
                    df[f'wavelet_detail_{i-1}'] = padded_coeff
            
            # Add energy features (sum of squared coefficients)
            for i in range(1, len(coeffs)):
                energy = np.sum(coeffs[i]**2)
                df[f'wavelet_energy_{i-1}'] = energy
            
            # Add entropy of wavelet coefficients
            for i in range(len(coeffs)):
                coeff = coeffs[i]
                if len(coeff) > 1:  # Need at least 2 points for entropy
                    # Normalize coefficients to get a probability distribution
                    coeff_norm = np.abs(coeff) / (np.sum(np.abs(coeff)) + 1e-10)
                    # Calculate Shannon entropy
                    entropy = -np.sum(coeff_norm * np.log2(coeff_norm + 1e-10))
                    if i == 0:
                        df['wavelet_entropy_approx'] = entropy
                    else:
                        df[f'wavelet_entropy_detail_{i-1}'] = entropy
            
            logger.info(f"Added {len(coeffs) * 2 - 1} wavelet features")
            
        except Exception as e:
            logger.error(f"Error in wavelet feature extraction: {e}")
            # If wavelet transform fails, return original data
            return data
            
        return df
        return macd, signal_line, macd - signal_line
    
    def _calculate_bollinger_bands(self, close_prices: np.ndarray, 
                                 window: int = 20, num_std: float = 2) -> tuple:
        """Calculate Bollinger Bands without TA-Lib."""
        rolling_mean = pd.Series(close_prices).rolling(window=window).mean()
        rolling_std = pd.Series(close_prices).rolling(window=window).std()
        upper = rolling_mean + (rolling_std * num_std)
        lower = rolling_mean - (rolling_std * num_std)
        return upper, rolling_mean, lower
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, 
                      close: np.ndarray, window: int = 14) -> np.ndarray:
        """Calculate Average True Range (ATR) without TA-Lib."""
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the given price data.
        
        Args:
            data: DataFrame containing price data (OHLCV)
            
        Returns:
            DataFrame with additional technical indicators
        """
        df = data.copy()
        
        # Debug: Print input data info
        print("\nInput data for technical indicators:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First few rows:")
        print(df.head())
        
        # Handle MultiIndex columns by keeping only the second level (price type)
        if isinstance(df.columns, pd.MultiIndex):
            print("Detected MultiIndex columns, flattening...")
            # Keep only the second level (price type) as column names
            df.columns = df.columns.get_level_values(1)
            print("Flattened columns:", df.columns.tolist())
        
        # Ensure all column names are strings and strip any whitespace
        df.columns = [str(col).strip() for col in df.columns]
        
        # Ensure we have the required columns (case-insensitive check)
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col.lower() for col in df.columns]
        missing_columns = [col for col in required_columns if col.lower() not in available_columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns for technical indicators: {missing_columns}")
            
        # Ensure consistent column names (title case)
        column_mapping = {col: col.title() for col in df.columns}
        df = df.rename(columns=column_mapping)
        
        try:
            # Calculate returns and log returns first as they're used in other calculations
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            if PANDAS_TA_AVAILABLE:
                # Use pandas_ta for technical indicators
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['MACD'], df['MACD_signal'], _ = ta.macd(df['Close'])
                df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.bbands(df['Close'])
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])
                
                # Rename columns to match our naming convention
                column_mapping = {
                    'RSI_14': 'RSI',
                    'MACD_12_26_9': 'MACD',
                    'MACD_12_26_9_SIGNAL': 'MACD_signal',
                    'BBU_20_2.0': 'BB_upper',
                    'BBM_20_2.0': 'BB_middle',
                    'BBL_20_2.0': 'BB_lower',
                    'ATRr_14': 'ATR'
                }
                
                # Only rename columns that exist in the DataFrame
                rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
                df.rename(columns=rename_dict, inplace=True)
            else:
                # Fallback to alternative implementations if pandas_ta is not available
                df['RSI'] = self._calculate_rsi(df['Close'].values)
                df['MACD'], df['MACD_signal'], _ = self._calculate_macd(df['Close'].values)
                df['BB_upper'], df['BB_middle'], df['BB_lower'] = self._calculate_bollinger_bands(df['Close'].values)
                df['ATR'] = self._calculate_atr(df['High'].values, df['Low'].values, df['Close'].values)
            
            # Handle NaN values in technical indicators
            indicator_columns = ['RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR']
            for col in indicator_columns:
                if col in df.columns:
                    # Forward fill first
                    df[col] = df[col].fillna(method='ffill')
                    # Then backward fill
                    df[col] = df[col].fillna(method='bfill')
                    # Finally fill remaining NaNs with column mean
                    df[col] = df[col].fillna(df[col].mean())
            
            # Volatility measures - use min_periods to avoid dropping too many rows
            df['Realized_Volatility'] = df['Returns'].rolling(window=20, min_periods=1).std() * np.sqrt(252)
            
            # Handle NaNs in volatility
            df['Realized_Volatility'] = df['Realized_Volatility'].fillna(df['Realized_Volatility'].mean())
            
            # Volume indicators
            if 'Volume' in df.columns:
                # Use min_periods=1 to ensure we keep more data points
                df['Volume_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
                # Ensure we're working with Series for the division
                volume = df['Volume'] if isinstance(df['Volume'], pd.Series) else df['Volume'].iloc[:, 0]
                volume_ma = df['Volume_MA'] if isinstance(df['Volume_MA'], pd.Series) else df['Volume_MA'].iloc[:, 0]
                # Handle division by zero
                df['Volume_Ratio'] = np.where(
                    volume_ma != 0,
                    volume / volume_ma,
                    1.0  # Default value when volume_ma is 0 (1.0 means equal volume)
                )
                
                # Handle NaNs in volume indicators
                df['Volume_MA'] = df['Volume_MA'].fillna(df['Volume'].mean())
                df['Volume_Ratio'] = df['Volume_Ratio'].fillna(1.0)
            
            # Instead of dropping all rows with any NaN, only drop rows with essential missing values
            # This is less strict than dropna() which would drop any row with any NaN
            essential_columns = ['Close', 'Returns', 'Log_Returns', 'RSI', 'MACD', 'MACD_signal']
            result = df.dropna(subset=essential_columns, how='all')
            
            # Fill any remaining NaN values with column means (except for the first few rows)
            result = result.fillna(result.mean(numeric_only=True))
            
            # Add a check for data quality
            if result.isna().sum().sum() > 0:
                logger.warning(f"Data quality warning: {result.isna().sum().sum()} NaN values remaining after preprocessing")
                logger.warning("NaN values per column:")
                logger.warning(result.isna().sum())
            
            # Add a check for extreme values
            for col in result.select_dtypes(include=[np.number]).columns:
                col_data = result[col]
                if col_data.std() == 0:
                    logger.warning(f"Column '{col}' has zero standard deviation - all values are identical")
                elif col_data.abs().max() > 1e6:
                    logger.warning(f"Column '{col}' has extreme values (max abs value: {col_data.abs().max():.2e})")
            
            # Scale the data to improve model training
            scaler = MinMaxScaler(feature_range=(-1, 1))
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            result[numeric_cols] = scaler.fit_transform(result[numeric_cols])
            
            # Debug: Print result info
            print("\nAfter calculating technical indicators:")
            print(f"Result shape: {result.shape}")
            print(f"Result columns: {result.columns.tolist()}")
            print(f"First few rows of result:")
            print(result.head())
            
            # Store processed data
            self.processed_data = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error in calculate_technical_indicators: {e}")
            logger.exception("Detailed error:")
            
            # If there's an error, return the original data with basic calculations
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Store processed data
            self.processed_data = df
            
            return df
        
    def preprocess_data(self, data: Optional[pd.DataFrame] = None, 
                       add_wavelet_features: bool = True,
                       wavelet_levels: int = 4,
                       wavelet_type: str = 'db4') -> pd.DataFrame:
        """
        Preprocess the financial time series data.
        
        Args:
            data: Optional input data. If None, uses self.raw_data
            add_wavelet_features: Whether to add wavelet-based features
            wavelet_levels: Number of wavelet decomposition levels
            wavelet_type: Type of wavelet to use (default: 'db4' - Daubechies 4)
            
        Returns:
            Processed DataFrame with technical indicators and features
        """
        if data is None:
            if self.raw_data is None:
                self.load_data()
            data = self.raw_data.copy()
        
        # Calculate technical indicators
        df = self.calculate_technical_indicators(data)
        
        # Add wavelet features if requested
        if add_wavelet_features:
            try:
                # Create a copy to avoid modifying original data
                wavelet_df = df.copy()
                wavelet_df = self.add_wavelet_features(
                    wavelet_df, 
                    target_column='Close',
                    levels=wavelet_levels,
                    wavelet=wavelet_type
                )
                
                # Handle NaN values in wavelet features
                wavelet_features = [col for col in wavelet_df.columns if col.startswith('wavelet')]
                for feature in wavelet_features:
                    # Fill NaNs with mean of the feature
                    wavelet_df[feature] = wavelet_df[feature].fillna(wavelet_df[feature].mean())
                    # If still NaN (all values were NaN), fill with 0
                    wavelet_df[feature] = wavelet_df[feature].fillna(0)
                
                logger.info("Successfully added wavelet features")
                df = wavelet_df
            except Exception as e:
                logger.error(f"Failed to add wavelet features: {e}")
                # If wavelet features fail, continue without them
                logger.warning("Continuing without wavelet features")
        
        # Store and return processed data
        self.processed_data = df
        return df
    
    def get_train_test_split(self, test_size: float = 0.2, 
                           target_col: str = 'Close',
                           sequence_length: int = 10) -> Tuple[np.ndarray, ...]:
        """
        Split the processed data into training and testing sets with sequences.
        
        Args:
            test_size: Proportion of data to use for testing (0-1)
            target_col: Name of the target column
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.processed_data is None:
            self.preprocess_data()
        
        # Debug: Print column names and first few rows
        print("\nColumns in processed data:", self.processed_data.columns.tolist())
        print("First few rows of processed data:")
        print(self.processed_data.head())
        
        # Ensure target column exists
        if target_col not in self.processed_data.columns:
            raise ValueError(f"Target column '{target_col}' not found in processed data")
        
        # Select features and target
        feature_cols = [col for col in self.processed_data.columns if col != target_col]
        X = self.processed_data[feature_cols].values
        y = self.processed_data[target_col].values.reshape(-1, 1)
        
        print(f"\nX shape: {X.shape}, y shape: {y.shape}")
        
        # Create sequences with more lenient NaN handling
        X_seq, y_seq = [], []
        n_samples = len(X) - sequence_length
        
        if n_samples <= 0:
            raise ValueError(f"Not enough samples for sequence_length={sequence_length}. "
                           f"Need at least {sequence_length + 1} samples.")
        
        # Create sequences while handling NaNs
        for i in range(n_samples):
            # Get sequence and target
            sequence = X[i:(i + sequence_length)]
            target = y[i + sequence_length - 1]
            
            # Handle NaNs in sequence by filling with mean
            if np.isnan(sequence).any():
                sequence = np.nan_to_num(sequence, nan=np.nanmean(sequence[~np.isnan(sequence)]))
            
            # Handle NaN target by skipping
            if np.isnan(target):
                continue
                
            X_seq.append(sequence)
            y_seq.append(target)
        
        # Convert to numpy arrays
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq).squeeze()
        
        # Debug info about sequence creation
        original_samples = n_samples
        remaining_samples = len(X_seq)
        logger.info(f"Created {remaining_samples} sequences from {original_samples} possible samples")
        
        # If we have too few sequences, raise an error
        if remaining_samples < 10:  # Minimum 10 sequences needed
            raise ValueError(f"Insufficient sequences created ({remaining_samples}). "
                           f"Consider reducing sequence_length or adjusting data preprocessing.")
        
        print(f"After sequence creation - X_seq shape: {X_seq.shape}, y_seq shape: {y_seq.shape}")
        
        # Ensure we have the right dimensions
        if len(X_seq.shape) == 2:
            X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
        
        # Split into train/test
        split_idx = int(len(X_seq) * (1 - test_size))
        
        # Ensure we have enough samples for both train and test
        if split_idx == 0 or split_idx == len(X_seq):
            raise ValueError(f"Insufficient data for train/test split. "
                           f"Total sequences: {len(X_seq)}, split index: {split_idx}")
        
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        print(f"After train/test split - X_train: {X_train.shape}, X_test: {X_test.shape}, "
              f"y_train: {y_train.shape}, y_test: {y_test.shape}")
        
        # Ensure y has the right shape for scaling
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
        if len(y_test.shape) == 1:
            y_test = y_test.reshape(-1, 1)
            
        return X_train, X_test, y_train, y_test
