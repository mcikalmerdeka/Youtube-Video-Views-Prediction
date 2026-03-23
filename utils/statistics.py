import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from datetime import datetime
from typing import List, Optional

# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║                       Functions for Statistical Summary                          ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝

## Describe numerical columns
def describe_numerical_combined(data: pd.DataFrame, col_series: List[str], hue: Optional[str] = None) -> pd.DataFrame:
    """
    Generate descriptive statistics for numerical columns in a dataframe,
    both overall and optionally grouped by a categorical variable.
    
    Parameters:
    data (pd.DataFrame): The dataframe containing the numerical columns
    col_series (List[str]): The list of numerical columns to describe
    hue (Optional[str], optional): The name of the categorical column to group by
    
    Returns:
    pd.DataFrame: A dataframe containing descriptive statistics, with hue classes if specified
    
    Examples:
    ---------
    >>> # Overall statistics only
    >>> stats = describe_numerical_combined(df, ['age', 'salary', 'experience'])
    
    >>> # Statistics grouped by a categorical variable
    >>> stats = describe_numerical_combined(df, ['age', 'salary'], hue='department')
    """
    # Overall statistics (original approach)
    overall_summary = data[col_series].describe().transpose().reset_index()
    overall_summary = overall_summary.rename(columns={'index': 'Feature'})
    
    # Add additional statistics for overall data
    overall_summary['range'] = overall_summary['max'] - overall_summary['min']
    overall_summary['IQR'] = overall_summary['75%'] - overall_summary['25%']
    overall_summary['CV'] = (overall_summary['std'] / overall_summary['mean']) * 100
    
    # Calculate skewness and kurtosis for numerical columns
    numerical_data = data[col_series].select_dtypes(include=['int64', 'float64']).dropna()
    overall_summary['skewness'] = [skew(numerical_data[col]) for col in numerical_data.columns]
    overall_summary['kurtosis'] = [kurtosis(numerical_data[col]) for col in numerical_data.columns]
    
    # Rename columns to indicate these are overall statistics
    overall_summary.columns = ['Feature'] + [f'overall_{col}' if col != 'Feature' else col 
                                           for col in overall_summary.columns[1:]]
    
    final_summary = overall_summary
    
    # If hue column is provided, add class-specific statistics
    if hue is not None:
        target_classes = sorted(data[hue].unique())
        class_summaries = []
        
        for target_class in target_classes:
            # Filter data for current class
            class_data = data[data[hue] == target_class]
            
            # Calculate basic statistics
            class_summary = class_data[col_series].describe().transpose().reset_index()
            class_summary = class_summary.rename(columns={'index': 'Feature'})
            
            # Add additional statistics
            class_summary['range'] = class_summary['max'] - class_summary['min']
            class_summary['IQR'] = class_summary['75%'] - class_summary['25%']
            class_summary['CV'] = (class_summary['std'] / class_summary['mean']) * 100
            
            # Calculate skewness and kurtosis
            numerical_class_data = class_data[col_series].select_dtypes(include=['int64', 'float64']).dropna()
            class_summary['skewness'] = [skew(numerical_class_data[col]) for col in numerical_class_data.columns]
            class_summary['kurtosis'] = [kurtosis(numerical_class_data[col]) for col in numerical_class_data.columns]
            
            # Rename columns to indicate which class they belong to
            class_summary.columns = ['Feature'] + [f'class_{target_class}_{col}' if col != 'Feature' else col 
                                                 for col in class_summary.columns[1:]]
            
            class_summaries.append(class_summary)
        
        # Combine all class summaries
        all_class_summaries = class_summaries[0]
        for summary in class_summaries[1:]:
            all_class_summaries = pd.merge(all_class_summaries, summary, on='Feature')
            
        # Merge with overall summary
        final_summary = pd.merge(final_summary, all_class_summaries, on='Feature')
        
        # Reorder columns to group statistics by type rather than by class
        # Get all column names except 'Feature'
        cols = final_summary.columns.tolist()
        cols.remove('Feature')
        
        # Group similar statistics together
        stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range', 'IQR', 'CV', 'skewness', 'kurtosis']
        new_cols = ['Feature']
        
        for stat in stats:
            stat_cols = [col for col in cols if stat in col]
            new_cols.extend(stat_cols)
            
        final_summary = final_summary[new_cols]
    
    return final_summary

## Describe categorical columns
def describe_categorical_combined(data: pd.DataFrame, col_series: List[str], hue: Optional[str] = None) -> pd.DataFrame:
    """
    Generate descriptive statistics for categorical columns in a dataframe,
    both overall and optionally grouped by a categorical variable.
    
    Parameters:
    data (pd.DataFrame): The dataframe containing the categorical columns
    col_series (List[str]): The list of categorical columns to describe
    hue (Optional[str], optional): The name of the categorical column to group by
    
    Returns:
    pd.DataFrame: A dataframe containing descriptive statistics, with hue classes if specified
    
    Examples:
    ---------
    >>> # Overall statistics only
    >>> stats = describe_categorical_combined(df, ['gender', 'department', 'job_title'])
    
    >>> # Statistics grouped by a categorical variable
    >>> stats = describe_categorical_combined(df, ['gender', 'job_title'], hue='department')
    """
    # Overall statistics
    cats_summary = data[col_series].describe().transpose().reset_index().rename(columns={'index': 'Feature'})
    
    # Add additional statistics for overall data
    cats_summary['bottom'] = [data[col].value_counts().idxmin() for col in col_series]
    cats_summary['freq_bottom'] = [data[col].value_counts().min() for col in col_series]
    cats_summary['top_percentage'] = [round(data[col].value_counts().max() / len(data) * 100, 2) 
                                    for col in col_series]
    cats_summary['bottom_percentage'] = [round(data[col].value_counts().min() / len(data) * 100, 2) 
                                       for col in col_series]
    
    # Add number of unique categories
    cats_summary['n_categories'] = [data[col].nunique() for col in col_series]
    
    # Rename columns to indicate these are overall statistics
    cats_summary.columns = ['Feature'] + [f'overall_{col}' if col != 'Feature' else col 
                                        for col in cats_summary.columns[1:]]
    
    final_summary = cats_summary
    
    # If hue column is provided, add class-specific statistics
    if hue is not None:
        target_classes = sorted(data[hue].unique())
        class_summaries = []
        
        for target_class in target_classes:
            # Filter data for current class
            class_data = data[data[hue] == target_class]
            
            # Calculate basic statistics
            class_summary = class_data[col_series].describe().transpose().reset_index()
            class_summary = class_summary.rename(columns={'index': 'Feature'})
            
            # Add additional statistics
            class_summary['bottom'] = [class_data[col].value_counts().idxmin() for col in col_series]
            class_summary['freq_bottom'] = [class_data[col].value_counts().min() for col in col_series]
            class_summary['top_percentage'] = [round(class_data[col].value_counts().max() / len(class_data) * 100, 2) 
                                             for col in col_series]
            class_summary['bottom_percentage'] = [round(class_data[col].value_counts().min() / len(class_data) * 100, 2) 
                                                for col in col_series]
            class_summary['n_categories'] = [class_data[col].nunique() for col in col_series]
            
            # Rename columns to indicate which class they belong to
            class_summary.columns = ['Feature'] + [f'class_{target_class}_{col}' if col != 'Feature' else col 
                                                 for col in class_summary.columns[1:]]
            
            class_summaries.append(class_summary)
        
        # Combine all class summaries
        all_class_summaries = class_summaries[0]
        for summary in class_summaries[1:]:
            all_class_summaries = pd.merge(all_class_summaries, summary, on='Feature')
            
        # Merge with overall summary
        final_summary = pd.merge(final_summary, all_class_summaries, on='Feature')
        
        # Reorder columns to group statistics by type rather than by class
        cols = final_summary.columns.tolist()
        cols.remove('Feature')
        
        # Group similar statistics together
        stats = ['count', 'unique', 'top', 'freq', 'bottom', 'freq_bottom', 
                'top_percentage', 'bottom_percentage', 'n_categories']
        new_cols = ['Feature']
        
        for stat in stats:
            stat_cols = [col for col in cols if stat in col]
            new_cols.extend(stat_cols)
            
        final_summary = final_summary[new_cols]
    
    return final_summary

## Describe date columns
def describe_date_columns(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
    """
    Comprehensive analysis of date columns including various temporal features
    
    Parameters:
    df (pd.DataFrame): DataFrame containing date columns
    date_columns (List[str]): List of column names containing dates
    
    Returns:
    pd.DataFrame: dates_summary statistics and temporal features for each date column
    
    Examples:
    ---------
    >>> # Analyze single date column
    >>> date_stats = describe_date_columns(df, ['purchase_date'])
    
    >>> # Analyze multiple date columns
    >>> date_stats = describe_date_columns(df, ['start_date', 'end_date', 'modified_date'])
    """
    dates_summary = df[date_columns].describe().transpose()
    
    # Basic range calculations
    dates_summary['date_range_days'] = dates_summary['max'] - dates_summary['min']
    dates_summary['date_range_months'] = dates_summary['date_range_days'].apply(lambda x: x.days / 30)
    dates_summary['date_range_years'] = dates_summary['date_range_days'].apply(lambda x: x.days / 365)
    
    # Additional temporal features
    for col in date_columns:
        # Distribution across time periods
        dates = df[col].dropna()
        dates_summary.loc[col, 'unique_years'] = dates.dt.year.nunique()
        dates_summary.loc[col, 'unique_months'] = dates.dt.to_period('M').nunique()
        dates_summary.loc[col, 'unique_days'] = dates.dt.date.nunique()
        
        # Temporal patterns
        dates_summary.loc[col, 'weekend_percentage'] = (dates.dt.dayofweek.isin([5, 6]).mean()) * 100
        dates_summary.loc[col, 'business_hours_percentage'] = (
            dates[dates.dt.hour.between(9, 17)].count() / dates.count()
        ) * 100 if hasattr(dates.dt, 'hour') else np.nan
        
        # Seasonality indicators
        dates_summary.loc[col, 'most_common_month'] = dates.dt.month.mode().iloc[0]
        dates_summary.loc[col, 'most_common_weekday'] = dates.dt.day_name().mode().iloc[0]
        
        # Gaps analysis
        sorted_dates = dates.sort_values()
        gaps = sorted_dates.diff().dropna()
        dates_summary.loc[col, 'max_gap_days'] = gaps.dt.days.max()
        dates_summary.loc[col, 'median_gap_days'] = gaps.dt.days.median()
        
        # Future/Past analysis
        now = datetime.now()
        dates_summary.loc[col, 'future_dates_percentage'] = (dates > now).mean() * 100
        
        # Regularity check
        dates_summary.loc[col, 'is_regular_interval'] = gaps.dt.days.std() < 1
    
    return dates_summary

## Distribution type analysis
def identify_distribution_types(
    df: pd.DataFrame,
    col_series: List[str],
    uniform_cols: Optional[List[str]] = None,
    multimodal_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Identifies and categorizes the distribution type of each numerical column
    based on skewness and kurtosis.

    Skewness thresholds:
        |skew| < 0.2  → Normal / Symmetric
        0.2 ≤ |skew| < 0.5  → Slightly skewed
        0.5 ≤ |skew| < 1.0  → Moderately skewed
        |skew| ≥ 1.0  → Highly skewed

    Kurtosis (excess, Fisher definition):
        < -0.5  → Platykurtic (light-tailed / flat)
        -0.5 to 0.5  → Mesokurtic (normal-like)
        > 0.5  → Leptokurtic (heavy-tailed / peaked)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col_series : List[str]
        Column names to analyze.
    uniform_cols : List[str], optional
        Columns to manually override as Uniform Distribution.
    multimodal_cols : List[str], optional
        Columns to manually override as Multi-modal Distribution.

    Returns
    -------
    pd.DataFrame
        Columns: Column Name, Mean, Median, Mode, Skewness, Kurtosis,
                 Skewness Type, Kurtosis Type.

    Examples
    --------
    >>> result = identify_distribution_types(df, df.select_dtypes('number').columns.tolist())
    >>> result = identify_distribution_types(df, num_cols, uniform_cols=['Age'], multimodal_cols=['Income'])
    """
    records = []

    for col in col_series:
        data = df[col].dropna()

        mean = round(data.mean(), 3)
        median = round(data.median(), 3)
        mode_vals = data.mode()
        mode = round(mode_vals.iloc[0], 3) if not mode_vals.empty else median

        skew_val = round(skew(data, nan_policy="omit"), 3)
        kurtosis_val = round(kurtosis(data, nan_policy="omit"), 3)

        abs_skew = abs(skew_val)
        if abs_skew < 0.2:
            skew_type = "Normal Distribution (Symmetric)"
        elif abs_skew < 0.5:
            direction = "Negative" if skew_val < 0 else "Positive"
            skew_type = f"Slightly {direction}ly Skewed"
        elif abs_skew < 1.0:
            direction = "Negative" if skew_val < 0 else "Positive"
            skew_type = f"Moderately {direction}ly Skewed"
        else:
            direction = "Negative" if skew_val < 0 else "Positive"
            skew_type = f"Highly {direction}ly Skewed"

        if kurtosis_val < -0.5:
            kurt_type = "Platykurtic (Light-tailed)"
        elif kurtosis_val > 0.5:
            kurt_type = "Leptokurtic (Heavy-tailed)"
        else:
            kurt_type = "Mesokurtic (Normal-like)"

        records.append({
            "Column Name": col,
            "Mean": mean,
            "Median": median,
            "Mode": mode,
            "Skewness": skew_val,
            "Kurtosis": kurtosis_val,
            "Skewness Type": skew_type,
            "Kurtosis Type": kurt_type,
        })

    dist = pd.DataFrame(records)

    if uniform_cols:
        mask = dist["Column Name"].isin(uniform_cols)
        dist.loc[mask, "Skewness Type"] = "Uniform Distribution"
        dist.loc[mask, "Kurtosis Type"] = "Platykurtic (Light-tailed)"

    if multimodal_cols:
        dist.loc[dist["Column Name"].isin(multimodal_cols), "Skewness Type"] = "Multi-modal Distribution"

    return dist
