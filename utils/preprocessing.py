import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer, OneHotEncoder, OrdinalEncoder, TargetEncoder
import category_encoders as ce
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from typing import List, Dict, Union, Optional, Any, Tuple

# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║                       Functions for Data Pre-Processing                          ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝

## Checking basic data information
def check_data_information(data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Check basic data information including data types, null values, duplicates, unique values,
    and various data characteristics like zero/negative values, empty strings, and cardinality.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The DataFrame to analyze
    cols : List[str]
        List of column names to check
    
    Returns:
    --------
    pd.DataFrame
        A DataFrame containing extended information about each column
    
    Examples:
    ---------
    >>> # Check information for all columns
    >>> info_df = check_data_information(df, df.columns.tolist())
    """
    list_item = []
    total_rows = len(data)
    duplicated_rows = data.duplicated().sum() # Calculate once as it's global
    
    for col in cols:
        series = data[col]
        
        # Calculate basic metrics
        dtype = series.dtype
        null_count = series.isna().sum()
        null_pct = round(100 * null_count / total_rows, 2)
        nunique = series.nunique()
        cardinality_ratio = round(nunique / total_rows, 4)
        
        # Format unique sample
        unique_vals = series.unique()
        unique_sample_count = min(5, len(unique_vals))
        unique_sample = ', '.join(map(str, unique_vals[:unique_sample_count]))
        
        # Initialize type-specific metrics
        zeros_count = 0
        zeros_pct = 0.0
        neg_count = 0
        neg_pct = 0.0
        empty_str_count = 0
        numeric_in_object = 0
        
        # Numeric Checks (Zero, Negative)
        if pd.api.types.is_numeric_dtype(series):
            zeros_count = (series == 0).sum()
            zeros_pct = round(100 * zeros_count / total_rows, 2)
            neg_count = (series < 0).sum()
            neg_pct = round(100 * neg_count / total_rows, 2)
            
        # Object/String Checks (Empty Strings, Numeric content)
        # Check for object or categorical that might contain strings
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            # Empty Strings / Whitespace
            # Convert to str, handling NaNs
            try:
                # We filter out NaNs first to avoid counting them as empty strings implies existing but empty
                non_null = series.dropna().astype(str)
                empty_str_count = non_null[non_null.str.strip() == ''].count()
                
                # Numeric-in-Object (Count values that can be coerced to numbers)
                numeric_in_object = pd.to_numeric(series, errors='coerce').notna().sum()
            except Exception:
                pass # Fail silently for complex objects if any

        list_item.append([
            col,                 # Feature name
            str(dtype),          # Data Type
            null_count,          # Count of null values
            null_pct,            # Percentage of null values
            zeros_count,         # Count of zero values (numeric only)
            zeros_pct,           # Percentage of zero values
            neg_count,           # Count of negative values (numeric only)
            neg_pct,             # Percentage of negative values
            empty_str_count,     # Count of empty/whitespace strings
            numeric_in_object,   # Count of numeric values in object column
            duplicated_rows,     # Count of duplicated rows in df
            nunique,             # Count of unique values
            cardinality_ratio,   # Ratio of unique values to total rows
            unique_sample        # Sample of unique values
        ])

    desc_df = pd.DataFrame(
        data=list_item,
        columns=[
            'Feature',
            'Data Type',
            'Null Values',
            'Null Percentage',
            'Zero Values',
            'Zero Percentage',
            'Negative Values',
            'Negative Percentage',
            'Empty Strings',
            'Numeric in Object',
            'Duplicated Values',
            'Unique Values',
            'Cardinality Ratio',
            'Unique Sample'
        ]
    )
    return desc_df

## Drop columns function
def drop_columns(data: pd.DataFrame, columns: Union[str, List[str]]) -> pd.DataFrame:
    """
    Drop specified columns from a DataFrame.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The DataFrame from which to drop columns
    columns : Union[str, List[str]]
        Column name(s) to drop. Can be a single column name or a list of column names.
    
    Returns:
    --------
    pd.DataFrame
        A new DataFrame with specified columns removed
    
    Examples:
    ---------
    >>> # Drop single column
    >>> df_new = drop_columns(df, 'column_to_drop')
    
    >>> # Drop multiple columns
    >>> df_new = drop_columns(df, ['col1', 'col2', 'col3'])
    """
    # Convert single column to list for consistent handling
    if isinstance(columns, str):
        columns = [columns]
    
    return data.drop(columns=columns, errors='ignore')

## Change binary column data type
def change_binary_dtype(data: pd.DataFrame, column: str, target_type: str = 'categorical') -> pd.Series:
    """
    Convert binary columns between numerical (0/1) and categorical (No/Yes) representations.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The DataFrame containing the binary column
    column : str
        The name of the column to convert
    target_type : str, default='categorical'
        The desired data type for the column:
        - 'categorical': Convert 0/1 to 'No'/'Yes'
        - 'numerical': Convert 'No'/'Yes' back to 0/1
    
    Returns:
    --------
    pd.Series
        The converted column
    
    Examples:
    ---------
    >>> # Convert numerical to categorical
    >>> df['status'] = change_binary_dtype(df, 'status', target_type='categorical')
    >>> # Result: 0 -> 'No', 1 -> 'Yes'
    
    >>> # Convert categorical back to numerical
    >>> df['status'] = change_binary_dtype(df, 'status', target_type='numerical')
    >>> # Result: 'No' -> 0, 'Yes' -> 1
    """
    if target_type == 'categorical':
        return data[column].map({0: 'No', 1: 'Yes'})
    elif target_type == 'numerical':
        return data[column].map({'No': 0, 'Yes': 1})
    else:
        raise ValueError("target_type must be either 'categorical' or 'numerical'")

## Handle missing values function
def handle_missing_values(data: pd.DataFrame, columns: List[str], strategy: str = 'median', imputer: Optional[Any] = None, n_neighbors: int = 5) -> Tuple[pd.DataFrame, Optional[Any]]:
    """
    Handle missing values using various imputation strategies.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataframe to process
    columns : List[str]
        List of column names to impute
    strategy : str, default='median'
        Imputation method:
        - 'median', 'mean', 'mode': Simple imputation
        - 'ffill', 'bfill': Forward/backward fill
        - 'knn': K-Nearest Neighbors imputation (advanced)
        - 'remove': Drop rows with missing values
    imputer : Optional[Any], default=None
        Pre-fitted imputer for test data (for 'knn')
    n_neighbors : int, default=5
        Number of neighbors for KNN imputation
    
    Returns:
    --------
    df_imputed : pd.DataFrame
        Dataframe with imputed values
    imputer : Optional[Any]
        The fitted imputer (for 'knn' or 'iterative' methods)
    
    Example:
    --------
    # Simple imputation
    df_imputed, _ = handle_missing_values(df, columns=['col1', 'col2'], strategy='median')
    
    # KNN imputation on training data
    X_train_imputed, imputer = handle_missing_values(X_train, columns=['col1', 'col2'], 
                                                       strategy='knn', n_neighbors=5)
    
    # Apply same imputer on test data
    X_test_imputed, _ = handle_missing_values(X_test, columns=['col1', 'col2'], 
                                               imputer=imputer)
    """
    if columns is None or len(columns) == 0:
        return data, None
    
    df_imputed = data.copy()
    
    # Validate columns exist
    missing_cols = [col for col in columns if col not in df_imputed.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")
    
    # Remove rows with missing values
    if strategy == 'remove':
        return df_imputed.dropna(subset=columns), None
    
    # Simple imputation methods
    elif strategy in ['median', 'mean', 'mode']:
        if strategy == 'median':
            df_imputed[columns] = df_imputed[columns].fillna(df_imputed[columns].median())
        elif strategy == 'mean':
            df_imputed[columns] = df_imputed[columns].fillna(df_imputed[columns].mean())
        elif strategy == 'mode':
            for col in columns:
                mode_val = df_imputed[col].mode()
                if len(mode_val) > 0:
                    df_imputed[col] = df_imputed[col].fillna(mode_val.iloc[0])
        return df_imputed, None
    
    # Forward/backward fill
    elif strategy in ['ffill', 'bfill']:
        df_imputed[columns] = df_imputed[columns].fillna(method=strategy)
        return df_imputed, None
    
    # KNN imputation (advanced)
    elif strategy == 'knn':
        if imputer is None:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df_imputed[columns] = imputer.fit_transform(df_imputed[columns])
        else:
            df_imputed[columns] = imputer.transform(df_imputed[columns])
        return df_imputed, imputer
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'median', 'mean', 'mode', 'ffill', 'bfill', 'knn', or 'remove'")

## Handle and detect outliers function
def filter_outliers(data: pd.DataFrame, columns: List[str], method: str = 'iqr', threshold: float = 1.5, detect_only: bool = False, return_mask: bool = False, verbose: bool = True) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
    """
    Unified function to detect and/or filter outliers using IQR or Z-score method.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataframe to process
    columns : List[str]
        List of column names to check for outliers
    method : str, default='iqr'
        Method to detect outliers: 'iqr' or 'zscore'
    threshold : float, default=1.5
        For IQR: multiplier for IQR range (default 1.5, typically 1.5-3)
        For Z-score: max absolute z-score (default 1.5, typically 2-3)
    detect_only : bool, default=False
        If True, returns summary DataFrame of outliers (detection mode)
        If False, returns filtered dataframe (filtering mode)
    return_mask : bool, default=False
        Only for filtering mode: if True, returns (filtered_data, mask)
    verbose : bool, default=True
        Only for detection mode: if True, prints summary statistics
    
    Returns:
    --------
    Detection mode (detect_only=True):
        pd.DataFrame: Summary with columns, bounds, counts, percentages
    
    Filtering mode (detect_only=False):
        pd.DataFrame or tuple: Filtered data, or (filtered_data, mask) if return_mask=True
    
    Examples:
    --------
    # Detection mode - analyze outliers
    summary = filter_outliers(df, columns=['col1', 'col2'], method='iqr', detect_only=True)
    
    # Filtering mode - remove outliers
    df_clean = filter_outliers(df, columns=['col1', 'col2'], method='iqr', threshold=1.5)
    
    # With mask for debugging
    df_clean, mask = filter_outliers(df, columns=['col1'], return_mask=True)
    """
    if columns is None or len(columns) == 0:
        if detect_only:
            return pd.DataFrame()
        return data if not return_mask else (data, np.array([True] * len(data)))

    if method.lower() not in ['iqr', 'zscore']:
        raise ValueError("Method must be either 'iqr' or 'zscore'")
    
    # Validate columns
    missing_cols = [col for col in columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")
    
    # Initialize tracking variables for detection mode
    if detect_only:
        outlier_counts = []
        non_outlier_counts = []
        is_outlier_list = []
        low_bounds = []
        high_bounds = []
        outlier_percentages = []
    
    # Start with all rows marked as True (non-outliers)
    filtered_entries = np.array([True] * len(data))
    
    # Loop through each column
    for col in columns:
        # IQR method
        if method.lower() == 'iqr':
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (IQR * threshold)
            upper_bound = Q3 + (IQR * threshold)
            filter_outlier = ((data[col] >= lower_bound) & (data[col] <= upper_bound))
            
        # Z-score method
        elif method.lower() == 'zscore':
            z_scores = np.abs(stats.zscore(data[col]))
            filter_outlier = (z_scores < threshold)
            
            mean = data[col].mean()
            std = data[col].std()
            lower_bound = mean - (threshold * std)
            upper_bound = mean + (threshold * std)
        
        # Store detection statistics
        if detect_only:
            outlier_count = len(data[~filter_outlier])
            non_outlier_count = len(data[filter_outlier])
            outlier_pct = round((outlier_count / len(data)) * 100, 2)
            
            outlier_counts.append(outlier_count)
            non_outlier_counts.append(non_outlier_count)
            is_outlier_list.append(data[col][~filter_outlier].any())
            low_bounds.append(lower_bound)
            high_bounds.append(upper_bound)
            outlier_percentages.append(outlier_pct)
        
        # Combine filters
        filtered_entries = filtered_entries & filter_outlier
    
    # Detection mode - return summary DataFrame
    if detect_only:
        if verbose:
            print(f'Amount of Rows: {len(data)}')
            print(f'Amount of Outlier Rows (Across All Columns): {len(data[~filtered_entries])}')
            print(f'Amount of Non-Outlier Rows (Across All Columns): {len(data[filtered_entries])}')
            print(f'Percentage of Outliers: {round(len(data[~filtered_entries]) / len(data) * 100, 2)}%')
            print()
        
        return pd.DataFrame({
            'Column Name': columns,
            'Outlier Exist': is_outlier_list,
            'Lower Limit': low_bounds,
            'Upper Limit': high_bounds,
            'Outlier Data': outlier_counts,
            'Non-Outlier Data': non_outlier_counts,
            'Outlier Percentage (%)': outlier_percentages
        })
    
    # Filtering mode - return filtered data
    if return_mask:
        return data[filtered_entries], filtered_entries
    return data[filtered_entries]

## Feature scaling function
def feature_scaling(data: pd.DataFrame, scaling_config: Optional[Dict[str, Any]] = None, columns: Optional[List[str]] = None, method: str = 'standard', scaler: Optional[Union[Dict[str, Any], Any]] = None, apply_log: bool = False) -> Tuple[pd.DataFrame, Union[Dict[str, Any], Any]]:
    """
    General feature scaling function with flexible options.
    Supports applying different scaling methods to different columns.
    
    SCALING METHOD GUIDE:
    ---------------------
    1. 'standard' (StandardScaler): Mean=0, Std=1
       - Best for: Normally distributed data, models assuming normal distribution (linear/logistic regression)
       - Sensitive to outliers
    
    2. 'minmax' (MinMaxScaler): Scale to [0, 1] range
       - Best for: Bounded features (counts, percentages), neural networks, image data
       - Very sensitive to outliers
    
    3. 'robust' (RobustScaler): Uses median and IQR
       - Best for: Data with many outliers, skewed distributions
       - More resistant to outliers than standard/minmax
    
    4. 'power' (PowerTransformer): Box-Cox or Yeo-Johnson transformation
       - Best for: Highly skewed data, making distributions more Gaussian
       - Automatically reduces skewness
       - Use 'yeo-johnson' for any data (handles zeros/negatives)
       - Use 'box-cox' only for strictly positive data
    
    5. 'quantile' (QuantileTransformer): Maps to uniform or normal distribution
       - Best for: Heavily skewed/non-linear data, reducing impact of extreme outliers
       - Most robust to outliers
       - Output: 'normal' for Gaussian-like, 'uniform' for evenly spread [0,1]
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataframe to scale
    scaling_config : Optional[Dict[str, Any]], optional
        Dictionary mapping scaling methods to column lists and options.
        Format: {
            'standard': {'columns': ['col1', 'col2'], 'apply_log': False},
            'minmax': {'columns': ['col3', 'col4'], 'apply_log': False},
            'robust': {'columns': ['col5', 'col6'], 'apply_log': True},
            'power': {'columns': ['col7'], 'method': 'yeo-johnson'},
            'quantile': {'columns': ['col8'], 'n_quantiles': 1000, 'output_distribution': 'normal'}
        }
        If provided, overrides 'columns', 'method', and 'apply_log' parameters.
    columns : Optional[List[str]], optional
        List of column names to scale (used when scaling_config is None)
    method : str, default='standard'
        Scaling method: 'standard', 'minmax', 'robust', 'power', or 'quantile' (used when scaling_config is None)
    scaler : Optional[Union[Dict[str, Any], Any]], default=None
        For single method: sklearn scaler object
        For multiple methods: dict mapping method names to fitted scaler objects
        If None, fits new scaler(s) (for training data)
    apply_log : bool, default=False
        Whether to apply log1p transformation before scaling (used when scaling_config is None)
    
    Returns:
    --------
    df_scaled : pd.DataFrame
        Dataframe with scaled features
    scaler : Union[Dict[str, Any], Any]
        For single method: the fitted scaler
        For multiple methods: dict of fitted scalers by method name
    
    Examples:
    ---------
    # Single method (legacy API - still works)
    X_train_scaled, scaler = feature_scaling(
        X_train, 
        columns=['col1', 'col2'], 
        method='standard'
    )
    X_test_scaled, _ = feature_scaling(X_test, columns=['col1', 'col2'], scaler=scaler)
    
    # Multiple methods (new API) - Training data
    X_train_scaled, scalers = feature_scaling(
        X_train,
        scaling_config={
            'minmax': {
                'columns': ['amount_of_bathrooms', 'amount_of_bedrooms']
            },
            'standard': {
                'columns': ['land_size_sqm', 'building_size_sqm'],
                'apply_log': True
            },
            'power': {
                'columns': ['price'],
                'method': 'yeo-johnson'  # or 'box-cox' for positive-only data
            },
            'quantile': {
                'columns': ['area'],
                'n_quantiles': 1000,
                'output_distribution': 'normal'  # or 'uniform'
            }
        }
    )
    
    # Multiple methods - Test data (reuse fitted scalers)
    X_test_scaled, _ = feature_scaling(
        X_test,
        scaling_config={
            'minmax': {'columns': ['amount_of_bathrooms', 'amount_of_bedrooms']},
            'standard': {'columns': ['land_size_sqm', 'building_size_sqm'], 'apply_log': True},
            'power': {'columns': ['price'], 'method': 'yeo-johnson'},
            'quantile': {'columns': ['area'], 'n_quantiles': 1000, 'output_distribution': 'normal'}
        },
        scaler=scalers
    )
    """
    df_scaled = data.copy()
    
    # --- MODE 1: Multiple scaling methods (new API) ---
    if scaling_config is not None:
        # Validate scaling_config structure
        if not isinstance(scaling_config, dict):
            raise ValueError("scaling_config must be a dictionary")
        
        # Collect all columns being scaled
        all_cols = []
        for method_name, config in scaling_config.items():
            if 'columns' not in config:
                raise ValueError(f"scaling_config['{method_name}'] must have 'columns' key")
            all_cols.extend(config['columns'])
        
        # Check for duplicates
        if len(all_cols) != len(set(all_cols)):
            duplicates = [col for col in set(all_cols) if all_cols.count(col) > 1]
            raise ValueError(f"Columns cannot be scaled with multiple methods: {duplicates}")
        
        # Validate all columns exist
        missing_cols = [col for col in all_cols if col not in df_scaled.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in dataframe: {missing_cols}")
        
        # Initialize scaler dict if not provided (training mode)
        if scaler is None:
            scaler = {}
            is_training = True
        else:
            if not isinstance(scaler, dict):
                raise ValueError("When using scaling_config, scaler must be a dict or None")
            is_training = False
        
        # Apply each scaling method
        for method_name, config in scaling_config.items():
            cols = config['columns']
            log_transform = config.get('apply_log', False)
            
            # Convert to float
            df_scaled[cols] = df_scaled[cols].astype(float)
            
            # Apply log transformation if requested
            if log_transform:
                for col in cols:
                    df_scaled[col] = np.log1p(df_scaled[col])
            
            # Get or create scaler
            if is_training:
                if method_name == 'standard':
                    method_scaler = StandardScaler()
                elif method_name == 'minmax':
                    method_scaler = MinMaxScaler()
                elif method_name == 'robust':
                    method_scaler = RobustScaler(quantile_range=(5, 95))
                elif method_name == 'power':
                    # PowerTransformer: auto handles skewness, makes data more Gaussian
                    power_method = config.get('method', 'yeo-johnson')  # 'yeo-johnson' or 'box-cox'
                    method_scaler = PowerTransformer(method=power_method, standardize=True)
                elif method_name == 'quantile':
                    # QuantileTransformer: maps to uniform or normal distribution
                    n_quantiles = config.get('n_quantiles', 1000)
                    output_dist = config.get('output_distribution', 'normal')  # 'normal' or 'uniform'
                    method_scaler = QuantileTransformer(n_quantiles=n_quantiles, 
                                                       output_distribution=output_dist,
                                                       random_state=42)
                else:
                    raise ValueError(f"Unknown scaling method: {method_name}. Use 'standard', 'minmax', 'robust', 'power', or 'quantile'")
                
                # Fit and transform
                df_scaled[cols] = method_scaler.fit_transform(df_scaled[cols])
                scaler[method_name] = method_scaler
            else:
                # Transform only (test data)
                if method_name not in scaler:
                    raise ValueError(f"Scaler for method '{method_name}' not found in provided scaler dict")
                df_scaled[cols] = scaler[method_name].transform(df_scaled[cols])
        
        return df_scaled, scaler
    
    # --- MODE 2: Single scaling method (legacy API) ---
    else:
        if columns is None:
            raise ValueError("Must specify 'columns' when not using scaling_config")
        
        # Validate columns exist
        missing_cols = [col for col in columns if col not in df_scaled.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in dataframe: {missing_cols}")
        
        # Convert to float
        df_scaled[columns] = df_scaled[columns].astype(float)
        
        # Apply log transformation if requested
        if apply_log:
            for col in columns:
                df_scaled[col] = np.log1p(df_scaled[col])
        
        # Initialize scaler if not provided (training mode)
        if scaler is None:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler(quantile_range=(5, 95))
            elif method == 'power':
                scaler = PowerTransformer(method='yeo-johnson', standardize=True)
            elif method == 'quantile':
                scaler = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=42)
            else:
                raise ValueError(f"Unknown scaling method: {method}. Use 'standard', 'minmax', 'robust', 'power', or 'quantile'")
            
            # Fit and transform (training data)
            df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
        else:
            # Only transform (test data)
            df_scaled[columns] = scaler.transform(df_scaled[columns])
        
        return df_scaled, scaler

## Feature encoding function
def feature_encoding(data: pd.DataFrame, ordinal_columns: Optional[List[str]] = None, 
                     nominal_columns: Optional[List[str]] = None, mean_encoding_columns: Optional[List[str]] = None,
                     binary_columns: Optional[List[str]] = None, frequency_columns: Optional[List[str]] = None,
                     ordinal_categories: Optional[Dict[str, List[Any]]] = None, drop_first: bool = True, 
                     preserve_dtypes: bool = True, handle_unknown: str = 'error',
                     target: Optional[pd.Series] = None, encoders: Optional[Dict[str, Any]] = None,
                     mean_target_type: str = 'continuous', mean_smooth: Union[str, float] = 'auto',
                     mean_cv: int = 5
                     ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    General feature encoding function supporting ordinal, one-hot, mean, label, and frequency encoding.
    
    Uses sklearn's TargetEncoder for mean encoding with built-in cross-fitting to prevent data leakage.
    Label encoding for binary features (2 unique values) converts them to 0/1.
    Frequency encoding replaces categories with their occurrence count from training data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataframe to encode
    ordinal_columns : Optional[List[str]], optional
        List of column names for ordinal encoding
    nominal_columns : Optional[List[str]], optional
        List of column names for one-hot encoding
    mean_encoding_columns : Optional[List[str]], optional
        List of column names for mean encoding.
        Each category is replaced with the mean of the target variable for that category.
        Best for high-cardinality categorical features.
    binary_columns : Optional[List[str]], optional
        List of column names for label encoding (binary features with only 2 unique values).
        Converts categories to 0/1 integers (e.g., 'yes'/'no' -> 1/0).
    frequency_columns : Optional[List[str]], optional
        List of column names for frequency encoding.
        Each category is replaced with its frequency count from training data.
        Best for high-cardinality categorical features where count is informative.
    ordinal_categories : Optional[Dict[str, List[Any]]], optional
        Dictionary mapping ordinal column names to their category order lists
    drop_first : bool, default=True
        Whether to drop first category in one-hot encoding (avoid dummy trap)
    preserve_dtypes : bool, default=True
        Whether to preserve original dtypes for non-encoded columns
    handle_unknown : str, default='error'
        How to handle unknown categories: 'error', 'use_encoded_value', 'ignore'
    target : Optional[pd.Series], default=None
        Target variable for mean encoding. REQUIRED when mean_encoding_columns is specified
        and encoders is None (i.e., for training data).
    encoders : Optional[Dict[str, Any]], default=None
        Dictionary of pre-fitted encoders (for validation/test data).
        If provided, uses these encoders instead of fitting new ones.
        Keys include: 'ordinal_*', 'onehot_*', 'mean_encoder', 'binary_*', 'frequency_*'
    mean_target_type : str, default='continuous'
        Type of target variable for mean encoding: 'continuous' for regression, 'binary' for classification.
    mean_smooth : Union[str, float], default='auto'
        Smoothing parameter for mean encoding. 'auto' uses empirical Bayes, or provide a float.
    mean_cv : int, default=5
        Number of cross-validation folds for internal cross-fitting to prevent data leakage.
        
    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Encoded dataframe and dictionary of fitted encoders
        
    Examples:
    ---------
    >>> # Fit on training data with all encoding types
    >>> X_train_encoded, encoders = feature_encoding(
    ...     X_train,
    ...     ordinal_columns=['Class'],
    ...     nominal_columns=['BuildingType'],
    ...     mean_encoding_columns=['District'],
    ...     binary_columns=['sales_channel'],  # Label encode binary features
    ...     frequency_columns=['route', 'booking_origin'],  # Frequency encode high cardinality
    ...     ordinal_categories={'Class': ['Low', 'Medium', 'High']},
    ...     target=y_train
    ... )
    
    >>> # Transform validation data using fitted encoders
    >>> X_val_encoded, _ = feature_encoding(
    ...     X_val,
    ...     ordinal_columns=['Class'],
    ...     nominal_columns=['BuildingType'],
    ...     mean_encoding_columns=['District'],
    ...     binary_columns=['sales_channel'],
    ...     frequency_columns=['route', 'booking_origin'],
    ...     ordinal_categories={'Class': ['Low', 'Medium', 'High']},
    ...     encoders=encoders
    ... )
    """
    # Initialize defaults
    ordinal_columns = ordinal_columns or []
    nominal_columns = nominal_columns or []
    mean_encoding_columns = mean_encoding_columns or []
    binary_columns = binary_columns or []
    frequency_columns = frequency_columns or []
    ordinal_categories = ordinal_categories or {}
    
    # Validate inputs
    all_encoding_cols = ordinal_columns + nominal_columns + mean_encoding_columns + binary_columns + frequency_columns
    missing_cols = [col for col in all_encoding_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")
    
    if not ordinal_columns and not nominal_columns and not mean_encoding_columns and not binary_columns and not frequency_columns:
        return data.copy(), encoders or {}
    
    for col in ordinal_columns:
        if col not in ordinal_categories:
            raise ValueError(f"Ordinal column '{col}' requires category order in ordinal_categories")
        unique_vals = data[col].unique()
        if not all(val in ordinal_categories[col] for val in unique_vals):
            print(f"Warning: Some values in '{col}' are not in the specified category order")
    
    if mean_encoding_columns and encoders is None and target is None:
        raise ValueError("target parameter is required when mean_encoding_columns is specified and encoders is None (training data)")
    
    # Copy dataframe and store metadata
    df_preprocessed = data.copy()
    original_dtypes = df_preprocessed.dtypes if preserve_dtypes else None
    
    # Preserve datetime columns separately
    datetime_columns = df_preprocessed.select_dtypes(include=['datetime64']).columns.tolist()
    datetime_data = df_preprocessed[datetime_columns].copy() if datetime_columns else None
    
    # Initialize encoder storage
    is_fitting = encoders is None
    fitted_encoders = {} if is_fitting else encoders.copy()
    
    # ===== STEP 0: Handle binary (label) encoding =====
    if binary_columns:
        if is_fitting:
            for col in binary_columns:
                # Validate binary (only 2 unique values)
                unique_vals = df_preprocessed[col].dropna().unique()
                if len(unique_vals) != 2:
                    raise ValueError(f"Binary column '{col}' must have exactly 2 unique values, found {len(unique_vals)}: {unique_vals}")
                
                # Create label mapping (alphabetically first -> 0, second -> 1)
                sorted_vals = sorted(unique_vals, key=str)
                label_map = {sorted_vals[0]: 0, sorted_vals[1]: 1}
                
                # Apply encoding
                df_preprocessed[col] = df_preprocessed[col].map(label_map)
                
                # Store the mapping in encoders
                fitted_encoders[f'binary_{col}'] = {'mapping': label_map, 'sorted_values': sorted_vals}
        else:
            # Transform using pre-fitted encoders
            for col in binary_columns:
                encoder_key = f'binary_{col}'
                if encoder_key in fitted_encoders:
                    label_map = fitted_encoders[encoder_key]['mapping']
                    df_preprocessed[col] = df_preprocessed[col].map(label_map)
                    # Handle unknown values - map to NaN then fill with mode (0 or 1)
                    if df_preprocessed[col].isna().any():
                        print(f"Warning: Unknown values in binary column '{col}', mapping to 0")
                        df_preprocessed[col] = df_preprocessed[col].fillna(0)
    
    # ===== STEP 1: Handle ordinal and one-hot encoding =====
    sklearn_encoding_cols = ordinal_columns + nominal_columns
    
    if sklearn_encoding_cols:
        # Prepare data by excluding datetime and mean encoding columns
        cols_to_exclude = datetime_columns + mean_encoding_columns
        df_for_sklearn = df_preprocessed.drop(columns=cols_to_exclude) if cols_to_exclude else df_preprocessed
        
        if is_fitting:
            # Build transformers for training
            transformers = []
            for col in ordinal_columns:
                # OrdinalEncoder doesn't support 'ignore', map it to 'use_encoded_value'
                if handle_unknown == 'ignore':
                    ord_handle_unknown = 'use_encoded_value'
                    ord_unknown_value = -1
                else:
                    ord_handle_unknown = handle_unknown
                    ord_unknown_value = None
                
                ord_encoder_params = {
                    'categories': [ordinal_categories[col]],
                    'dtype': np.float64,
                    'handle_unknown': ord_handle_unknown
                }
                if ord_unknown_value is not None:
                    ord_encoder_params['unknown_value'] = ord_unknown_value
                
                transformers.append((
                    f'ordinal_{col}',
                    OrdinalEncoder(**ord_encoder_params),
                    [col]
                ))
            
            for col in nominal_columns:
                transformers.append((
                    f'onehot_{col}',
                    OneHotEncoder(
                        drop='first' if drop_first else None,
                        sparse_output=False,
                        dtype=np.float64,
                        handle_unknown='ignore' if handle_unknown == 'ignore' else 'error'
                    ),
                    [col]
                ))
            
            # Fit and transform
            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough',
                verbose_feature_names_out=False
            )
            
            df_sklearn_encoded = preprocessor.fit_transform(df_for_sklearn)
            
            # Store fitted encoders
            for name, trans, _ in preprocessor.transformers_:
                if name != 'remainder':
                    fitted_encoders[name] = trans
            
            # Build column names
            encoded_column_names = list(ordinal_columns)
            for col in nominal_columns:
                oh = fitted_encoders[f'onehot_{col}']
                cats = oh.categories_[0]
                if drop_first and len(cats) > 1:
                    encoded_column_names.extend([f'{col}_{cat}' for cat in cats[1:]])
                else:
                    encoded_column_names.extend([f'{col}_{cat}' for cat in cats])
            
            passthrough_cols = [c for c in df_for_sklearn.columns if c not in sklearn_encoding_cols]
            encoded_column_names.extend(passthrough_cols)
            
            df_sklearn_encoded = pd.DataFrame(df_sklearn_encoded, columns=encoded_column_names, index=data.index)
        else:
            # Transform using pre-fitted encoders
            df_sklearn_encoded = df_for_sklearn.copy()
            
            # Apply ordinal encoders
            for col in ordinal_columns:
                encoder_key = f'ordinal_{col}'
                if encoder_key in fitted_encoders:
                    df_sklearn_encoded[col] = fitted_encoders[encoder_key].transform(df_sklearn_encoded[[col]]).flatten()
            
            # Apply one-hot encoders
            onehot_dfs = []
            for col in nominal_columns:
                encoder_key = f'onehot_{col}'
                if encoder_key in fitted_encoders:
                    oh_encoder = fitted_encoders[encoder_key]
                    oh_result = oh_encoder.transform(df_sklearn_encoded[[col]])
                    cats = oh_encoder.categories_[0]
                    
                    if drop_first and len(cats) > 1:
                        oh_col_names = [f'{col}_{cat}' for cat in cats[1:]]
                    else:
                        oh_col_names = [f'{col}_{cat}' for cat in cats]
                    
                    oh_df = pd.DataFrame(oh_result, columns=oh_col_names, index=data.index)
                    onehot_dfs.append(oh_df)
            
            # Remove original nominal columns and add one-hot encoded ones
            df_sklearn_encoded = df_sklearn_encoded.drop(columns=nominal_columns)
            if onehot_dfs:
                df_sklearn_encoded = pd.concat([df_sklearn_encoded] + onehot_dfs, axis=1)
        
        # Convert to numeric (excluding frequency columns that will be encoded separately)
        cols_to_convert = [col for col in df_sklearn_encoded.columns if col not in frequency_columns]
        for col in cols_to_convert:
            df_sklearn_encoded[col] = pd.to_numeric(df_sklearn_encoded[col], errors='coerce')
    else:
        # No sklearn encoding, just exclude mean encoding columns
        cols_to_keep = [c for c in df_preprocessed.columns if c not in mean_encoding_columns + datetime_columns]
        df_sklearn_encoded = df_preprocessed[cols_to_keep].copy()
    
    # ===== STEP 2: Handle frequency encoding =====
    # Apply frequency encoding using category_encoders.CountEncoder
    if frequency_columns:
        if is_fitting:
            for col in frequency_columns:
                # Ensure column exists and is string/object type (as expected by CountEncoder)
                if col not in df_preprocessed.columns:
                    raise ValueError(f"Frequency column '{col}' not found in dataframe")
                
                # Get source data from df_preprocessed
                source_data = df_preprocessed[col]
                
                # Create and fit CountEncoder (counts occurrences, normalize=False for raw counts)
                count_encoder = ce.CountEncoder(
                    cols=[col],
                    normalize=False,  # Use raw counts, not normalized frequencies
                    handle_unknown='value',  # Return 0 for unknown categories
                    handle_missing='value'  # Return 0 for missing values
                )
                
                # Fit and transform
                encoded_col = count_encoder.fit_transform(source_data.to_frame())
                
                # Apply to df_sklearn_encoded
                if col in df_sklearn_encoded.columns:
                    df_sklearn_encoded[col] = encoded_col[col].values
                else:
                    df_sklearn_encoded[col] = encoded_col[col].values
                
                # Store the fitted encoder
                fitted_encoders[f'frequency_{col}'] = count_encoder
        else:
            # Transform using pre-fitted CountEncoders
            for col in frequency_columns:
                encoder_key = f'frequency_{col}'
                if encoder_key not in fitted_encoders:
                    print(f"Warning: Frequency encoder for '{col}' not found. Skipping.")
                    continue
                
                count_encoder = fitted_encoders[encoder_key]
                
                # Get source data
                if col in df_preprocessed.columns:
                    source_data = df_preprocessed[col]
                elif col in df_sklearn_encoded.columns:
                    source_data = df_sklearn_encoded[col]
                else:
                    print(f"Warning: Frequency column '{col}' not found. Skipping.")
                    continue
                
                # Transform
                encoded_col = count_encoder.transform(source_data.to_frame())
                
                # Apply to df_sklearn_encoded
                if col in df_sklearn_encoded.columns:
                    df_sklearn_encoded[col] = encoded_col[col].values
                else:
                    df_sklearn_encoded[col] = encoded_col[col].values
    
    # ===== STEP 3: Handle mean encoding =====
    if mean_encoding_columns:
        if is_fitting:
            mean_encoder = TargetEncoder(
                target_type=mean_target_type,
                smooth=mean_smooth,
                cv=mean_cv
            )
            df_mean_encoded = mean_encoder.fit_transform(df_preprocessed[mean_encoding_columns], target)
            fitted_encoders['mean_encoder'] = mean_encoder
        else:
            mean_encoder = fitted_encoders['mean_encoder']
            df_mean_encoded = mean_encoder.transform(df_preprocessed[mean_encoding_columns])
        
        df_mean_encoded = pd.DataFrame(df_mean_encoded, columns=mean_encoding_columns, index=data.index)
        
        # Convert to numeric
        for col in mean_encoding_columns:
            df_mean_encoded[col] = pd.to_numeric(df_mean_encoded[col], errors='coerce')
        
        # Combine sklearn and mean encoded data
        df_final = pd.concat([df_sklearn_encoded, df_mean_encoded], axis=1)
    else:
        df_final = df_sklearn_encoded
    
    # ===== STEP 4: Restore datetime columns and dtypes =====
    if datetime_columns:
        for col in datetime_columns:
            df_final[col] = datetime_data[col]
    
    if preserve_dtypes:
        all_encoded_cols = (
            ordinal_columns + 
            [col for col in df_final.columns if any(col.startswith(f'{nc}_') for nc in nominal_columns)] +
            mean_encoding_columns +
            binary_columns +
            frequency_columns
        )
        
        for col in df_final.columns:
            if col in original_dtypes and col not in all_encoded_cols:
                try:
                    df_final[col] = df_final[col].astype(original_dtypes[col])
                except Exception as e:
                    print(f"Warning: Could not convert column '{col}' back to {original_dtypes[col]}. Error: {e}")
    
    return df_final, fitted_encoders


