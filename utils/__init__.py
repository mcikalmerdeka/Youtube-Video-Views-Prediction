# Core preprocessing functions (always available)
from .preprocessing import (
    check_data_information,
    drop_columns,
    change_binary_dtype,
    handle_missing_values,
    filter_outliers,
    feature_scaling,
    feature_encoding
)

# Feature selection functions (always available)
from .feature_selection import (
    calculate_correlation_tabular,
    analyze_categorical_relationships,
    calculate_vif,
    calculate_feature_importance
)
# Statistics functions (always available)
from .statistics import (
    describe_numerical_combined,
    describe_categorical_combined,
    describe_date_columns,
    identify_distribution_types
)

# Visualization functions (always available)
from .visualization import (
    plot_dynamic_hisplots_kdeplots,
    plot_dynamic_boxplots_violinplots,
    plot_dynamic_countplot,
    plot_correlation_heatmap
)

# ML regression functions
from .regression_evals_and_tuning import (
    eval_regression,
    compare_cv_metrics as compare_cv_metrics_regression,
    tune_pipelines as tune_pipelines_regression,
    tune_single_model as tune_single_model_regression,
    tune_all_models as tune_all_models_regression,
    get_model_pipeline as get_model_pipeline_regression,
    get_hyperparameters as get_hyperparameters_regression
)

# Translation utilities
from .translation_utils import (
    detect_indian_language_rows,
    translate_series,
    translate_dataframe,
    run_translation_pipeline,
    translate_all_text
)

# YouTube preprocessing functions
from .preprocess_youtube_data import (
    load_data,
    drop_leakage_columns,
    parse_datetime_features,
    calculate_text_features,
    calculate_sentiment_features,
    create_category_features,
    create_channel_features,
    create_target_variable,
    prepare_features_for_modeling,
    save_preprocessed_data,
    print_feature_summary,
    run_preprocessing
)

# Define what's exported with `from utils import *`
__all__ = [
    # Preprocessing
    'check_data_information',
    'drop_columns',
    'change_binary_dtype',
    'handle_missing_values',
    'filter_outliers',
    'feature_scaling',
    'feature_encoding',

    # Feature selection
    'calculate_correlation_tabular',
    'analyze_categorical_relationships',
    'calculate_vif',
    'calculate_feature_importance',
    
    # Statistics
    'describe_numerical_combined',
    'describe_categorical_combined',
    'describe_date_columns',
    'identify_distribution_types',
    
    # Visualization
    'plot_dynamic_hisplots_kdeplots',
    'plot_dynamic_boxplots_violinplots',
    'plot_dynamic_countplot',
    'plot_correlation_heatmap',
    
    # ML regression
    'eval_regression',
    'compare_cv_metrics_regression',
    'tune_pipelines_regression',
    'tune_single_model_regression',
    'tune_all_models_regression',
    'get_model_pipeline_regression',
    'get_hyperparameters_regression',
    
    # Translation utilities
    'detect_indian_language_rows',
    'translate_series',
    'translate_dataframe',
    'run_translation_pipeline',
    'translate_all_text',
    
    # YouTube preprocessing
    'load_data',
    'drop_leakage_columns',
    'parse_datetime_features',
    'calculate_text_features',
    'calculate_sentiment_features',
    'create_category_features',
    'create_channel_features',
    'create_target_variable',
    'prepare_features_for_modeling',
    'save_preprocessed_data',
    'print_feature_summary',
    'run_preprocessing',
]
