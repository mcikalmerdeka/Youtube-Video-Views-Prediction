import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower, GofChisquarePower
from statsmodels.stats.proportion import proportions_ztest
from typing import List, Dict, Union, Optional, Tuple, Any

# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║                       Functions for A/B Testing                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝

## Calculate sample size
def calculate_sample_size(baseline_rate: float, minimum_detectable_effect: float, power: float = 0.8, alpha: float = 0.05, metric_type: str = 'proportion') -> int:
    """
    Calculate the required sample size per variation for an A/B test.
    
    Parameters:
    -----------
    baseline_rate : float
        The current conversion rate or mean value (e.g., 0.10 for 10%).
    minimum_detectable_effect : float
        The minimum relative change you want to detect (e.g., 0.02 for 2% absolute change or relative lift).
        For proportions, this is the absolute difference (e.g., 0.10 -> 0.12 means MDE is 0.02).
    power : float, default=0.8
        The probability of rejecting the null hypothesis when it is false (1 - beta).
    alpha : float, default=0.05
        The significance level (probability of Type I error).
    metric_type : str, default='proportion'
        Type of metric: 'proportion' (for conversion rates) or 'continuous' (for means like revenue).
        
    Returns:
    --------
    int
        The required sample size per group.
        
    Examples:
    ---------
    >>> # For a conversion rate moving from 10% to 12% (absolute diff 0.02)
    >>> n = calculate_sample_size(0.10, 0.02, metric_type='proportion')
    
    >>> # For a continuous metric (requires effect size in terms of Cohen's d, simplified here)
    >>> # Assuming standard deviation is roughly equal to mean for this simple estimation
    >>> n = calculate_sample_size(100, 5, metric_type='continuous') 
    """
    if metric_type == 'proportion':
        # Using statsmodels for proportion power analysis
        # effect_size for proportions is often calculated via Cohen's h, but here we use a simplified approximation or exact method
        # For simplicity in this utility, we'll use a standard formula approximation or a library function if available.
        # Let's use statsmodels' NormalIndPower for proportions (z-test)
        from statsmodels.stats.power import NormalIndPower
        import statsmodels.stats.proportion as proportion
        
        p1 = baseline_rate
        p2 = baseline_rate + minimum_detectable_effect
        
        # Calculate effect size (Cohen's h)
        effect_size = proportion.proportion_effectsize(p1, p2)
        
        analysis = NormalIndPower()
        sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1.0)
        
    elif metric_type == 'continuous':
        # For continuous, we typically need standard deviation. 
        # Here we assume a standardized effect size (Cohen's d) is passed or derived.
        # If the user passes raw values, we might need more info. 
        # To keep it simple and consistent with the signature, we'll assume 'minimum_detectable_effect' 
        # is the raw difference and we'd need an estimate of std dev.
        # Since we don't have std dev, we'll assume a standardized effect size of MDE / baseline (very rough) 
        # OR we can update the docstring to say MDE is Cohen's d.
        # Let's assume MDE is the raw difference and we assume unit variance for a rough estimate, 
        # OR better, let's ask for effect_size directly in a robust tool.
        # Given the constraints, let's treat MDE as Cohen's d for continuous to be safe/standard.
        
        # However, to be more user friendly, let's assume MDE is the relative lift and we estimate d.
        # Let's stick to standard TTestIndPower which takes effect_size (Cohen's d).
        # We will interpret minimum_detectable_effect as Cohen's d for continuous.
        
        analysis = TTestIndPower()
        sample_size = analysis.solve_power(effect_size=minimum_detectable_effect, alpha=alpha, power=power, ratio=1.0)
        
    else:
        raise ValueError("metric_type must be 'proportion' or 'continuous'")
        
    return int(np.ceil(sample_size))

## Check Normality
def check_normality(data: Union[pd.Series, np.ndarray, List[float]]) -> Dict[str, Any]:
    """
    Perform Shapiro-Wilk test to check for normality.
    
    Parameters:
    -----------
    data : array-like
        The data to test.
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing statistic, p-value, and interpretation.
    """
    stat, p_value = stats.shapiro(data)
    return {
        'test': 'Shapiro-Wilk',
        'statistic': stat,
        'p_value': p_value,
        'is_normal': p_value > 0.05
    }

## Check Variance Homogeneity
def check_variance_homogeneity(group_a: Union[pd.Series, np.ndarray], group_b: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """
    Perform Levene's test to check for equal variances.
    
    Parameters:
    -----------
    group_a : array-like
        Data for group A.
    group_b : array-like
        Data for group B.
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing statistic, p-value, and interpretation.
    """
    stat, p_value = stats.levene(group_a, group_b)
    return {
        'test': 'Levene',
        'statistic': stat,
        'p_value': p_value,
        'equal_variance': p_value > 0.05
    }

## T-Test (Parametric)
def perform_t_test(group_a: Union[pd.Series, np.ndarray], group_b: Union[pd.Series, np.ndarray], equal_var: bool = True) -> Dict[str, Any]:
    """
    Perform independent t-test for continuous metrics.
    
    Parameters:
    -----------
    group_a : array-like
        Data for control group.
    group_b : array-like
        Data for treatment group.
    equal_var : bool, default=True
        If True, perform standard t-test. If False, perform Welch's t-test.
        
    Returns:
    --------
    Dict[str, Any]
        Test results including statistic, p-value, and decision.
    """
    t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=equal_var)
    
    return {
        'test': 'T-Test' if equal_var else "Welch's T-Test",
        'statistic': t_stat,
        'p_value': p_val,
        'significant': p_val < 0.05,
        'mean_a': np.mean(group_a),
        'mean_b': np.mean(group_b)
    }

## Mann-Whitney U Test (Non-Parametric)
def perform_mann_whitney_u_test(group_a: Union[pd.Series, np.ndarray], group_b: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """
    Perform Mann-Whitney U test for continuous metrics (non-parametric).
    
    Parameters:
    -----------
    group_a : array-like
        Data for control group.
    group_b : array-like
        Data for treatment group.
        
    Returns:
    --------
    Dict[str, Any]
        Test results including statistic, p-value, and decision.
    """
    u_stat, p_val = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
    
    return {
        'test': 'Mann-Whitney U',
        'statistic': u_stat,
        'p_value': p_val,
        'significant': p_val < 0.05,
        'median_a': np.median(group_a),
        'median_b': np.median(group_b)
    }

## Chi-Squared Test
def perform_chi_squared_test(group_a_conversions: int, group_a_total: int, group_b_conversions: int, group_b_total: int) -> Dict[str, Any]:
    """
    Perform Chi-squared test for conversion rates.
    
    Parameters:
    -----------
    group_a_conversions : int
        Number of conversions in group A.
    group_a_total : int
        Total sample size of group A.
    group_b_conversions : int
        Number of conversions in group B.
    group_b_total : int
        Total sample size of group B.
        
    Returns:
    --------
    Dict[str, Any]
        Test results including statistic, p-value, and decision.
    """
    # Contingency table
    #           Converted   Not Converted
    # Group A   conv_a      total_a - conv_a
    # Group B   conv_b      total_b - conv_b
    
    obs = np.array([
        [group_a_conversions, group_a_total - group_a_conversions],
        [group_b_conversions, group_b_total - group_b_conversions]
    ])
    
    chi2, p_val, dof, ex = stats.chi2_contingency(obs)
    
    rate_a = group_a_conversions / group_a_total
    rate_b = group_b_conversions / group_b_total
    
    return {
        'test': 'Chi-Squared',
        'statistic': chi2,
        'p_value': p_val,
        'significant': p_val < 0.05,
        'rate_a': rate_a,
        'rate_b': rate_b,
        'lift': (rate_b - rate_a) / rate_a if rate_a > 0 else 0
    }

## Z-Test for Proportions
def perform_z_test_proportions(group_a_conversions: int, group_a_total: int, group_b_conversions: int, group_b_total: int) -> Dict[str, Any]:
    """
    Perform Z-test for two proportions.
    
    Parameters:
    -----------
    group_a_conversions : int
        Number of conversions in group A.
    group_a_total : int
        Total sample size of group A.
    group_b_conversions : int
        Number of conversions in group B.
    group_b_total : int
        Total sample size of group B.
        
    Returns:
    --------
    Dict[str, Any]
        Test results including statistic, p-value, and decision.
    """
    count = np.array([group_a_conversions, group_b_conversions])
    nobs = np.array([group_a_total, group_b_total])
    
    stat, p_val = proportions_ztest(count, nobs)
    
    rate_a = group_a_conversions / group_a_total
    rate_b = group_b_conversions / group_b_total
    
    return {
        'test': 'Z-Test Proportions',
        'statistic': stat,
        'p_value': p_val,
        'significant': p_val < 0.05,
        'rate_a': rate_a,
        'rate_b': rate_b
    }

## High-level Analysis
def analyze_ab_test_results(data: pd.DataFrame, metric_col: str, group_col: str, metric_type: str = 'continuous', control_label: str = 'A', treatment_label: str = 'B') -> Dict[str, Any]:
    """
    Analyze A/B test results from a DataFrame, automatically selecting appropriate tests.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataframe containing the experiment data.
    metric_col : str
        The column name of the metric to analyze.
    group_col : str
        The column name indicating the group (e.g., 'group').
    metric_type : str, default='continuous'
        'continuous' (e.g., revenue, time) or 'binary' (e.g., conversion 0/1).
    control_label : str, default='A'
        Value in group_col representing the control group.
    treatment_label : str, default='B'
        Value in group_col representing the treatment group.
        
    Returns:
    --------
    Dict[str, Any]
        Comprehensive analysis results.
    """
    # Split data
    group_a = data[data[group_col] == control_label][metric_col].dropna()
    group_b = data[data[group_col] == treatment_label][metric_col].dropna()
    
    results = {
        'metric': metric_col,
        'sample_size_a': len(group_a),
        'sample_size_b': len(group_b)
    }
    
    if metric_type == 'continuous':
        # Check normality
        normality_a = check_normality(group_a)
        normality_b = check_normality(group_b)
        
        results['normality_check'] = {
            'group_a': normality_a,
            'group_b': normality_b
        }
        
        # Check variance homogeneity
        variance_check = check_variance_homogeneity(group_a, group_b)
        results['variance_check'] = variance_check
        
        # Decide test
        if normality_a['is_normal'] and normality_b['is_normal']:
            # Parametric
            test_res = perform_t_test(group_a, group_b, equal_var=variance_check['equal_variance'])
        else:
            # Non-parametric
            test_res = perform_mann_whitney_u_test(group_a, group_b)
            
        results.update(test_res)
        
    elif metric_type == 'binary':
        # Conversion counts
        conv_a = group_a.sum()
        total_a = len(group_a)
        conv_b = group_b.sum()
        total_b = len(group_b)
        
        # Use Chi-Squared by default for binary data analysis
        test_res = perform_chi_squared_test(conv_a, total_a, conv_b, total_b)
        results.update(test_res)
        
        # Also include Z-test for reference
        z_res = perform_z_test_proportions(conv_a, total_a, conv_b, total_b)
        results['z_test_p_value'] = z_res['p_value']
        
    else:
        raise ValueError("metric_type must be 'continuous' or 'binary'")
        
    return results