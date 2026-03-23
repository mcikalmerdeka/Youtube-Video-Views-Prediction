import pandas as pd
from IPython.display import display
from typing import Tuple, List, Optional, Union, Any
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║                       Functions for Feature Selection                             ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝

## Correlation analysis in tabluar form
def calculate_correlation_tabular(df: pd.DataFrame, target_col: str, method: str = 'spearman', corr_type: str = 'feature_feature') -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Calculate and display correlation analysis in tabular form for feature-to-feature,
    feature-to-target correlations, or both.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing the features and target variable
    target_col : str
        Name of the target column in the DataFrame
    method : str, optional (default='spearman')
        Correlation method to use. Options include:
        - 'pearson': standard correlation coefficient
        - 'spearman': Spearman rank correlation
        - 'kendall': Kendall Tau correlation coefficient
    corr_type : str, optional (default='feature_feature')
        Type of correlation analysis to perform:
        - 'feature_feature': correlations between all features
        - 'feature_target': correlations between features and target variable
        - 'both': display both feature-feature and feature-target correlations
    
    Examples:
    ---------
    >>> # Calculate feature-to-feature correlations
    >>> feature_feature_df = calculate_correlation_tabular(df, target_col='target', method='spearman', corr_type='feature_feature')
    
    >>> # Calculate feature-to-target correlations
    >>> feature_target_df = calculate_correlation_tabular(df, target_col='target', method='spearman', corr_type='feature_target')
    
    >>> # Calculate both feature-to-feature and feature-to-target correlations
    >>> feature_feature_df, feature_target_df = calculate_correlation_tabular(df, target_col='target', method='spearman', corr_type='both')

    Returns:
    --------
    tuple or pandas.DataFrame
        If corr_type is 'both':
            Returns a tuple (feature_feature_df, feature_target_df)
        Otherwise:
            Returns a single DataFrame containing the correlation analysis results
            
        DataFrame columns for feature_feature:
            - 'A': First feature name
            - 'B': Second feature name
            - 'Corr Value': Absolute correlation value
            - 'Corr Type': 'Positive' or 'Negative'
        DataFrame columns for feature_target:
            - 'Feature': Feature name
            - 'Corr Value': Absolute correlation value
            - 'Corr Type': 'Positive' or 'Negative'

    Raises:
    -------
    ValueError
        If corr_type is not 'feature_feature', 'feature_target', or 'both'
    """
    
    # Input validation for correlation type
    if corr_type not in ['feature_feature', 'feature_target', 'both']:
        raise ValueError("corr_type must be either 'feature_feature', 'feature_target', or 'both'")
    
    def get_feature_feature_corr():
        """
        Helper function to calculate feature-to-feature correlations
        """
        # Calculate correlation matrix for all numeric columns
        corr_feature = df.corr(method=method, numeric_only=True)

        # Convert correlation matrix to long format for better visualization
        flat_cm = corr_feature.stack().reset_index()
        flat_cm.columns = ['A', 'B', 'Corr Value']

        # Apply filters to remove:
        # 1. Self-correlations (A to A)
        # 2. Target variable correlations
        # 3. Duplicate pairs (A-B vs B-A)
        flat_cm = flat_cm.loc[
            (flat_cm['A'] != flat_cm['B']) &  # Remove self-correlations
            (flat_cm['A'] != target_col) &     # Remove target from first column
            (flat_cm['B'] != target_col)       # Remove target from second column
        ]

        # Add correlation type (Positive/Negative) and convert to absolute values
        flat_cm['Corr Type'] = flat_cm['Corr Value'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')
        flat_cm['Corr Value'] = flat_cm['Corr Value'].abs()
        
        # Sort by correlation strength and remove duplicates
        flat_cm = flat_cm.sort_values(by='Corr Value', ascending=False, ignore_index=True)
        redundan_cm = flat_cm.drop_duplicates(subset=['Corr Value', 'Corr Type']).reset_index(drop=True)
        
        print("Correlation Between Features:")
        display(redundan_cm.iloc[:15].reset_index(drop=True))
        return redundan_cm
    
    def get_feature_target_corr():
        """
        Helper function to calculate feature-to-target correlations
        """
        # Calculate correlations between all features and the target variable
        corr_target = df.corrwith(df[target_col], method=method, numeric_only=True)
        
        # Convert to DataFrame for better manipulation
        corr_final = corr_target.reset_index(name='Corr Value').rename(columns={'index': 'Feature'})

        # Remove target variable self-correlation
        corr_final = corr_final.loc[corr_final['Feature'] != target_col]

        # Add correlation type and convert to absolute values
        corr_final['Corr Type'] = corr_final['Corr Value'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')
        corr_final['Corr Value'] = corr_final['Corr Value'].abs()
        
        # Sort by correlation strength
        corr_final = corr_final.sort_values('Corr Value', ascending=False, ignore_index=True)

        print("\nCorrelation of Features to Target:")
        display(corr_final.iloc[:15].reset_index(drop=True))
        return corr_final
    
    # Handle different correlation type requests
    if corr_type == 'feature_feature':
        return get_feature_feature_corr()
    elif corr_type == 'feature_target':
        return get_feature_target_corr()
    else:  # corr_type == 'both'
        # Calculate and return both types of correlations
        feature_feature_df = get_feature_feature_corr()
        feature_target_df = get_feature_target_corr()
        return feature_feature_df, feature_target_df

## Categorical features correlation analysis
def analyze_categorical_relationships(df: pd.DataFrame, features: List[str], target: str, alpha: float = 0.05) -> pd.DataFrame:
    """
    Analyze relationships between categorical features and target using Chi-Square test
    and calculate effect sizes using Cramér's V.

    Parameters:
    -----------
    df : pandas DataFrame
        The input dataframe containing the features and target
    features : List[str]
        List of categorical feature names
    target : str
        Name of the target variable
    alpha : float, optional (default=0.05)
        Significance level for the chi-square test

    Examples:
    ---------
    >>> # Analyze relationships between categorical features and target
    >>> results = analyze_categorical_relationships(df, features=['feature1', 'feature2'], target='target', alpha=0.05)

    Returns:
    --------
    pandas DataFrame
        Summary of the analysis including chi-square statistics, p-values, and effect sizes
    """
    results = []

    for feature in features:
        # Create contingency table
        contingency = pd.crosstab(df[feature], df[target])

        # Perform chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)

        # Calculate Cramér's V for effect size
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramer_v = np.sqrt(chi2 / (n * min_dim))

        # Calculate standardized residuals for post-hoc analysis
        observed = contingency.values
        standardized_residuals = (observed - expected) / np.sqrt(expected)

        # Determine significance and effect size interpretation
        is_significant = p_value < alpha

        # Interpret effect size (common thresholds for Cramér's V)
        if cramer_v < 0.1:
            effect_size = 'Negligible'
        elif cramer_v < 0.3:
            effect_size = 'Small'
        elif cramer_v < 0.5:
            effect_size = 'Medium'
        else:
            effect_size = 'Large'

        # Find categories with strongest associations
        max_residual_idx = np.unravel_index(
            np.abs(standardized_residuals).argmax(),
            standardized_residuals.shape
        )
        strongest_category = contingency.index[max_residual_idx[0]]
        strongest_association = 'Positive' if standardized_residuals[max_residual_idx] > 0 else 'Negative'

        results.append({
            'Feature': feature,
            'Chi-Square': chi2,
            'P-Value': p_value,
            "Cramér's V": cramer_v,
            'Effect Size': effect_size,
            'Is Significant': is_significant,
            'Strongest Category': strongest_category,
            'Association Direction': strongest_association
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Cramér\'s V', ascending=False)

    # Display results
    print("\nChi-Square Analysis Results:")
    print("-" * 80)
    display(results_df)

    # Visualize the results
    plt.figure(figsize=(10, 6))
    plt.bar(results_df['Feature'], results_df["Cramér's V"])
    plt.title("Effect Size (Cramér's V) by Feature")
    plt.xticks(rotation=45)
    plt.ylabel("Cramér's V")
    plt.tight_layout()
    plt.show()

    return results_df

## Variance Inflation Factor (VIF) analysis
def calculate_vif(df: pd.DataFrame, target_col: str, high_threshold: int = 10, moderate_threshold: int = 5) -> pd.DataFrame:
    """
    Calculate VIF for each feature in the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing the features and target variable
    target_col : str
        Name of target column to exclude
    high_threshold : int, optional, default=10
        Threshold for high multicollinearity
    moderate_threshold : int, optional, default=5
        Threshold for moderate multicollinearity

    Examples:
    ---------
    >>> # Calculate VIF for the dataset
    >>> vif_data = calculate_vif(df, target_col='target', high_threshold=10, moderate_threshold=5)

    Returns:
    --------
    pd.DataFrame
        DataFrame with VIF scores
    """
    # Create DataFrame of features only
    features_df = df.drop(columns=[target_col])

    # Initialize VIF DataFrame
    vif_data = pd.DataFrame()
    vif_data['Feature'] = features_df.columns

    # Calculate VIF for each feature
    try:
        vif_data['VIF'] = [variance_inflation_factor(features_df.values, i)
                          for i in range(features_df.shape[1])]
    except Exception as e:
        print(f"Error calculating VIF: {e}")
        return None

    # Sort by VIF score
    vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)

    # Add interpretation column
    vif_data['Interpretation'] = vif_data['VIF'].apply(lambda x:
        'High multicollinearity' if x > high_threshold
        else ('Moderate multicollinearity' if x > moderate_threshold
        else 'Low multicollinearity'))

    return vif_data

from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    f_classif,
    chi2,
    f_regression,
    mutual_info_regression
)

# Calculate feature importance scores using different statistical methods for classification or regression
def calculate_feature_importance(X: pd.DataFrame, y: pd.Series, task: str = 'classification', method: str = 'mutual_info', k: int = 'all') -> pd.DataFrame:
    """
    Calculate feature importance scores using different statistical methods for classification or regression.

    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    task : str, optional, default='classification'
        Task type ('classification' or 'regression')
    method : str, optional, default='mutual_info'
        Method for feature selection ('mutual_info', 'anova', 'chi2' for classification; 'f_regression', 'mutual_info' for regression')
    k : int or 'all', optional, default='all'
        Number of top features to select. Use 'all' to see scores for all features.

    Examples:
    ---------
    >>> # Calculate feature importance scores for classification
    >>> feature_importance = calculate_feature_importance(X=X, y=y, task='classification', method='mutual_info', k='all')
    >>> display(feature_importance.iloc[:15].reset_index(drop=True))

    >>> # Calculate feature importance scores for regression
    >>> feature_importance = calculate_feature_importance(X=X, y=y, task='regression', method='f_regression', k='all')
    >>> display(feature_importance.iloc[:15].reset_index(drop=True))

    Returns:
    --------
    pd.DataFrame
        DataFrame with feature names, scores, and p-values (if applicable)
    """
    # Select the appropriate scoring function based on task and method
    if task == 'classification':
        if method == 'mutual_info':
            score_func = mutual_info_classif
        elif method == 'anova':
            score_func = f_classif
        elif method == 'chi2':
            score_func = chi2
        else:
            raise ValueError("For classification, method must be 'mutual_info', 'anova', or 'chi2'")
    elif task == 'regression':
        if method == 'f_regression':
            score_func = f_regression
        elif method == 'mutual_info':
            score_func = mutual_info_regression
        else:
            raise ValueError("For regression, method must be 'f_regression' or 'mutual_info'")
    else:
        raise ValueError("Task must be 'classification' or 'regression'")

    # Instantiate SelectKBest with the chosen scoring function
    k_best = SelectKBest(score_func=score_func, k=k)

    # Fit the model and transform the data
    k_best.fit_transform(X, y)

    # Extract feature scores and names
    feature_scores = k_best.scores_
    feature_names = X.columns

    # Create a DataFrame to display feature scores
    # Note: p-values are not applicable for mutual_info_classif or mutual_info_regression, hence set to None
    feature_scores_df = pd.DataFrame({
        'Feature': feature_names,
        'Score': feature_scores,
        'P_Values': k_best.pvalues_ if method in ['anova', 'chi2', 'f_regression'] else [None] * len(feature_scores)
    })

    # Sort the DataFrame by scores in descending order
    feature_scores_df = feature_scores_df.sort_values(by='Score', ascending=False)

    return feature_scores_df