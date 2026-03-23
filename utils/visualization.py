import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from typing import List, Optional, Tuple, Any

# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║                       Functions for Data Visualization                           ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝

## Hisplot and kdeplot analysis
def plot_dynamic_hisplots_kdeplots(df: pd.DataFrame, col_series: List[str], plot_type: str = 'histplot', ncols: int = 6, figsize: Tuple[int, int] = (26, 18), hue: Optional[str] = None, multiple: str = 'layer', fill: Optional[bool] = None) -> None:
    """
    Creates a dynamic grid of histogram plots (with KDE) or KDE plots for multiple numerical columns.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot.
    col_series : List[str]
        List of column names to include in the plots.
    plot_type : str, optional, default='histplot'
        Type of plot to generate. Options are:
        - 'histplot': Histogram with KDE overlay
        - 'kdeplot': Kernel Density Estimation plot
    ncols : int, optional, default=6
        Number of columns in the subplot grid. Adjust this value to change grid width.
    figsize : Tuple[int, int], optional, default=(26, 18)
        Size of the figure to control plot dimensions.
    hue : Optional[str], optional, default=None
        Column name to use for color encoding. Creates separate distributions for each category.
    multiple : str, optional, default='layer'
        How to display multiple distributions. Options are:
        - 'layer': Distributions are overlaid
        - 'dodge': Distributions are placed side by side
    fill : Optional[bool], optional, default=None
        Whether to fill the area under the KDE curve.

    Returns:
    -------
    None
        Displays a grid of distribution plots.

    Examples:
    --------
    >>> # Create histogram plots with KDE
    >>> plot_dynamic_hisplots_kdeplots(df, ['col1', 'col2'], plot_type='histplot')

    >>> # Create KDE plots with categorical splitting
    >>> plot_dynamic_hisplots_kdeplots(
    ...     df,
    ...     ['col1', 'col2'],
    ...     plot_type='kdeplot',
    ...     hue='category',
    ...     multiple='layer'
    ... )

    Notes:
    -----
    - For histplots, KDE (Kernel Density Estimation) is automatically enabled
    - The y-axis label adjusts automatically based on the plot type
    """

    # Validate plot_type parameter
    if plot_type not in ['histplot', 'kdeplot']:
        raise ValueError("plot_type must be either 'histplot' or 'kdeplot'")

    # Calculate required number of rows based on number of plots and specified columns
    num_plots = len(col_series)
    nrows = math.ceil(num_plots / ncols)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    # Convert ax to array if it's a single subplot
    if num_plots == 1:
        ax = np.array([ax])
    else: 
        ax = ax.flatten()  # Flatten the axes array for easy indexing

    # Generate plots for each column
    for i, col in enumerate(col_series):
        if plot_type == 'histplot':
            sns.histplot(data=df, ax=ax[i], x=col, kde=True, hue=hue, multiple=multiple)
        else:  # kdeplot
            sns.kdeplot(data=df, ax=ax[i], x=col, hue=hue, multiple=multiple, fill=fill)

        ax[i].set_title(f'Distribution of {col}')
        ax[i].set_ylabel(f'{"Count" if plot_type == "histplot" else "Density"} of {col}')
        ax[i].set_xlabel(f'{col}')

    # Remove any unused subplots if total subplots exceed columns in cols
    for j in range(num_plots, len(ax)):
        fig.delaxes(ax[j])

    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.show()

# Boxplot and violinplot analysis
def plot_dynamic_boxplots_violinplots(df: pd.DataFrame, col_series: List[str], plot_type: str = 'boxplot', ncols: int = 6, figsize: Tuple[int, int] = (26, 18), orientation: str = 'v', hue: Optional[str] = None) -> None:
    """
    Creates a dynamic grid of either boxplots or violin plots for multiple numerical columns.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot.
    col_series : List[str]
        List of column names to include in the plots.
    plot_type : str, optional, default='boxplot'
        Type of plot to generate. Options are 'boxplot' or 'violinplot'.
    ncols : int, optional, default=6
        Number of columns in the subplot grid. Adjust this value to change grid width.
    figsize : Tuple[int, int], optional, default=(26, 18)
        Size of the figure to control plot dimensions.
    orientation : str, optional, default='v'
        Orientation of the plots. Use 'v' for vertical and 'h' for horizontal.
    hue : Optional[str], optional, default=None
        Column name to use for color encoding. Creates separate plots for each category.

    Returns:
    -------
    None
        Displays a grid of plots.

    Examples:
    --------
    >>> # Create vertical boxplots
    >>> plot_dynamic_boxplots_violinplots(df, ['col1', 'col2'], plot_type='boxplot', orientation='v')

    >>> # Create horizontal violin plots with categorical splitting
    >>> plot_dynamic_boxplots_violinplots(df, ['col1', 'col2'], plot_type='violinplot',
    ...                                     orientation='h', hue='category')
    """
    # Validate plot_type parameter
    if plot_type not in ['boxplot', 'violinplot']:
        raise ValueError("plot_type must be either 'boxplot' or 'violinplot'")

    # Calculate required number of rows based on number of plots and specified columns
    num_plots = len(col_series)
    nrows = math.ceil(num_plots / ncols)

    # Adjust figsize based on orientation
    if orientation == 'h':
        figsize = (figsize[1], figsize[0])  # Swap width and height for horizontal plots

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    # Convert ax to array if it's a single subplot
    if num_plots == 1:
        ax = np.array([ax])
    else: 
        ax = ax.flatten()  # Flatten the axes array for easy indexing

    # Generate plots for each column
    for i, col in enumerate(col_series):
        if plot_type == 'boxplot':
            if orientation == 'v':
                sns.boxplot(data=df, ax=ax[i], y=col, orient='v', hue=hue)
                ax[i].set_title(f'Boxplot of {col}')
            else:  # orientation == 'h'
                sns.boxplot(data=df, ax=ax[i], x=col, orient='h', hue=hue)
                ax[i].set_title(f'Boxplot of {col}')
        else: # violinplot
            if orientation == 'v':
                sns.violinplot(data=df, ax=ax[i], y=col, orient='v', hue=hue, inner_kws=dict(box_width=15, whis_width=2))
                ax[i].set_title(f'Violinplot of {col}')
            else:  # orientation == 'h'
                sns.violinplot(data=df, ax=ax[i], x=col, orient='h', hue=hue, inner_kws=dict(box_width=15, whis_width=2))
                ax[i].set_title(f'Violinplot of {col}')

    # Remove any unused subplots if total subplots exceed columns in cols
    for j in range(num_plots, len(ax)):
        fig.delaxes(ax[j])

    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.show()

## Countplot analysis
def plot_dynamic_countplot(df: pd.DataFrame, col_series: List[str], ncols: int = 6, figsize: Tuple[int, int] = (26, 18), stat: str = 'count', hue: Optional[str] = None, order: Optional[List[Any]] = None) -> None:
    """
    Plots a dynamic grid of countplot for a list of categorical columns from a DataFrame.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot.
    col_series : List[str]
        List of column names to include in the countplots.
    ncols : int, optional, default=6
        Number of columns in the subplot grid. Adjust this value to change grid width.
    figsize : Tuple[int, int], optional, default=(26, 18)
        Size of the figure to control plot dimensions.
    stat : str, optional, default='count'
        The statistic to compute for the bars
    hue : Optional[str], optional, default=None
        Column name to use for color encoding
    order : Optional[List[Any]], optional, default=None
        Specific ordering for the categorical variables

    Returns:
    -------
    None
        Displays a grid of countplots.
    
    Examples:
    ---------
    >>> # Basic countplot
    >>> plot_dynamic_countplot(df, ['gender', 'department', 'job_title'])
    
    >>> # Countplot with hue grouping
    >>> plot_dynamic_countplot(df, ['department', 'job_title'], hue='gender')
    
    >>> # With custom order
    >>> order_list = [['A', 'B', 'C'], ['Small', 'Medium', 'Large']]
    >>> plot_dynamic_countplot(df, ['category', 'size'], order=order_list, ncols=2)
    
    >>> # Show top 5 categories for each column (most frequent)
    >>> cats_cols = ['Category', 'Province', 'City']
    >>> plot_dynamic_countplot(
    ...     df=df,
    ...     col_series=cats_cols,
    ...     ncols=1,
    ...     figsize=(12, 10),
    ...     order=[df[cat].value_counts().iloc[:5].index.tolist() for cat in cats_cols]
    ... )
    
    >>> # Show bottom 5 categories for each column (least frequent)
    >>> plot_dynamic_countplot(
    ...     df=df,
    ...     col_series=cats_cols,
    ...     ncols=1,
    ...     figsize=(12, 10),
    ...     order=[df[cat].value_counts().iloc[-5:].index.tolist() for cat in cats_cols]
    ... )
    """

    # Calculate required number of rows based on number of plots and specified columns
    num_plots = len(col_series)
    nrows = math.ceil(num_plots / ncols)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    # Convert ax to array if it's a single subplot
    if num_plots == 1:
        ax = np.array([ax])
    else: 
        ax = ax.flatten()  # Flatten the axes array for easy indexing

    # Generate countplot for each column
    for i, col in enumerate(col_series):
        # Get the specific order for this column if provided
        if isinstance(order, list):
            current_order = order[i].tolist() if hasattr(order[i], 'tolist') else order[i]
        else:
            current_order = None

        sns.countplot(data=df,
                     ax=ax[i],
                     x=col,
                     stat=stat,
                     hue=hue,
                     order=current_order)
        if hue is not None:
            ax[i].set_title(f'Countplot sof {col} by {hue}s')
        else:
            ax[i].set_title(f'Countplot of {col}')
        ax[i].set_ylabel(f'Count of {col}')
        ax[i].set_xlabel(f'{col}')

        # Add value labels on top of bars
        for p in ax[i].patches:
            ax[i].annotate(
                f'{int(p.get_height())}',  # The text to display
                (p.get_x() + p.get_width() / 2., p.get_height()),  # The position
                ha='center',  # Horizontal alignment
                va='bottom'   # Vertical alignment
            )

    # Remove any unused subplots if total subplots exceed columns in cols
    for j in range(num_plots, len(ax)):
        fig.delaxes(ax[j])

    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.show()

## Stripplot/swarmplot analysis
def plot_dynamic_stripplot_swarmplot(df: pd.DataFrame, cat_col: str, nums_cols_series: List[str], hue_col: str = None, n_cols: int = 2, figsize: Tuple[int, int] = (14, 10), plot_type: str = 'stripplot') -> None:
    """
    Create multiple stripplots or swarmplots comparing a categorical column against numerical columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe containing the data
    cat_col : str
        The name of the categorical column to plot on x-axis
    nums_cols_series : List[str]
        List of numerical column names to plot on y-axis
    hue_col : str, optional, default=None
        Column name to use for color coding (default: None)
    n_cols : int, optional
        Number of columns in the subplot grid (default: 2)
    figsize : Tuple[int, int], optional, default=(14, 10)
        Figure size in inches (width, height) (default: (14, 10))
    plot_type : str, optional, default='stripplot'
        Type of plot to create: 'stripplot' or 'swarmplot' (default: 'stripplot')

    Examples:
    ---------
    >>> # Create stripplots for numerical columns
    >>> plot_dynamic_stripplot_swarmplot(df, cat_col='category', nums_cols_series=['col1', 'col2'], hue_col='hue_col', plot_type='stripplot')
    
    >>> # Create swarmplots for numerical columns
    >>> plot_dynamic_stripplot_swarmplot(df, cat_col='category', nums_cols_series=['col1', 'col2'], hue_col='hue_col', plot_type='swarmplot')
    
    Returns:
    --------
    None
        Display a grid of plots
    """
    # Validate plot_type
    if plot_type not in ['stripplot', 'swarmplot']:
        raise ValueError("plot_type must be either 'stripplot' or 'swarmplot'")
    
    # Calculate number of rows needed based on number of plots and columns
    n_plots = len(nums_cols_series)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Create subplots
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, 
                          figsize=figsize)
    
    # Flatten axes array for easier iteration
    ax = ax.flatten() if n_plots > 1 else [ax]
    
    # Create plots
    for i, num_col in enumerate(nums_cols_series):
        if plot_type == 'stripplot':
            sns.stripplot(data=df, ax=ax[i], 
                         x=cat_col, y=num_col, 
                         hue=hue_col)
            plot_name = 'Stripplot'
        else:  # swarmplot
            sns.swarmplot(data=df, ax=ax[i], 
                         x=cat_col, y=num_col, 
                         hue=hue_col)
            plot_name = 'Swarmplot'
            
        ax[i].set_title(f'{cat_col} {plot_name} for {num_col}')
        
        # Rotate x-axis labels if they're too long
        ax[i].tick_params(axis='x', rotation=45)
    
    # Hide empty subplots if any
    for i in range(len(nums_cols_series), len(ax)):
        ax[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

## Correlation heatmap analysis on numerical features and target
def plot_correlation_heatmap(df: pd.DataFrame, col_series: List[str], corr_method: str = 'pearson', figsize: Tuple[int, int] = (8, 6), cmap: str = 'coolwarm') -> None:
    """
    Plots a correlation heatmap for the specified columns in a dataframe.

    Parameters:
    ----------
    df : pd.DataFrame
        The input dataframe containing the data.
    col_series : List[str]
        List of column names to include in the correlation matrix.
    corr_method : str 
        Correlation method ('pearson', 'spearman', or 'kendall'). Default is 'pearson'.
    figsize : Tuple[int, int]
        Size of the heatmap figure (width, height). Default is (8, 6).
    cmap : str
        Color map for the heatmap. Default is 'coolwarm'.

    Returns:
    -------
    None
        Displays the correlation heatmap.
    
    Examples:
    --------
    >>> included_col = ['Income', 'Total_Spending', 'Age', 'CVR']
    >>> plot_correlation_heatmap(df_filtered_outliers, col_series=included_col, corr_method='pearson')
    >>> plot_correlation_heatmap(df_filtered_outliers, col_series=included_col, corr_method='spearman')
    """
    
    # Compute correlation matrix
    correlation_matrix = df[col_series].corr(method=corr_method)

    # Mask the upper triangle of the correlation matrix
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(data=correlation_matrix, cmap=cmap, annot=True, fmt='.3f', vmin=-1, vmax=1, mask=mask)
    plt.title(f'{corr_method.capitalize()} Correlation')
    
    plt.tight_layout()
    plt.show()
