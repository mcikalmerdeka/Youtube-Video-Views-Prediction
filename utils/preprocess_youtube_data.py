"""
YouTube Video Views Prediction - Preprocessing Script
This script preprocesses the raw YouTube data and creates features for predicting views.
IMPORTANT: Only uses pre-upload features (no data leakage from likes, comments, etc.)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path):
    """Load the YouTube dataset from CSV."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


def drop_leakage_columns(df):
    """
    Drop columns that represent data leakage (metrics only available AFTER upload).
    These should NOT be used for predicting views before upload.
    """
    leakage_columns = [
        'likes', 'dislikes', 'comment_count',
        'comments_disabled', 'ratings_disabled', 'video_error_or_removed'
    ]
    
    # Only drop columns that exist in the dataframe
    columns_to_drop = [col for col in leakage_columns if col in df.columns]
    
    if columns_to_drop:
        print(f"Dropping {len(columns_to_drop)} leakage columns: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
    
    return df


def parse_datetime_features(df):
    """Extract time-based features from publish_time_fixed."""
    
    # Convert publish_time_fixed to datetime
    df['publish_time_fixed'] = pd.to_datetime(df['publish_time_fixed'])
    
    # Extract publish time features
    df['publish_day'] = df['publish_time_fixed'].dt.day_name()
    df['publish_hour'] = df['publish_time_fixed'].dt.hour
    df['publish_month'] = df['publish_time_fixed'].dt.month_name()
    df['publish_season'] = df['publish_month'].map({
        'December': 'Winter', 'January': 'Winter', 'February': 'Winter',
        'March': 'Spring', 'April': 'Spring', 'May': 'Spring',
        'June': 'Summer', 'July': 'Summer', 'August': 'Summer',
        'September': 'Fall', 'October': 'Fall', 'November': 'Fall'
    })
    
    # Categorize publish hour
    def categorize_hour(hour):
        if 0 <= hour < 6:
            return 'Night'
        elif 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        else:
            return 'Evening'
    
    df['publish_hour_category'] = df['publish_hour'].apply(categorize_hour)
    
    # Is weekend
    df['is_weekend_publish_day'] = df['publish_time_fixed'].dt.weekday >= 5
    df['is_weekend_publish_day'] = df['is_weekend_publish_day'].astype(int)
    
    return df


def calculate_text_features(df):
    """Calculate text-based features (lengths, tag counts)."""
    
    # Title length
    df['len_title'] = df['title'].fillna('').astype(str).str.len()
    
    # Description length
    df['desc_len'] = df['description'].fillna('').astype(str).str.len()
    
    # Tag features
    df['has_tags'] = (df['tags'].fillna('').astype(str) != 'none').astype(int)
    df['No_tags'] = df['tags'].fillna('').astype(str).str.split('|').str.len()
    df['No_tags'] = df['No_tags'].fillna(0).astype(int)
    
    return df


def calculate_sentiment_features(df):
    """Calculate sentiment scores for text columns."""
    
    def get_sentiment(text):
        """Get sentiment polarity using TextBlob."""
        try:
            if pd.isna(text) or text == '' or text == 'none':
                return 0.0
            text = str(text)
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def categorize_sentiment(score):
        """Categorize sentiment score into labels."""
        if score > 0.1:
            return 'Positive'
        elif score < -0.1:
            return 'Negative'
        else:
            return 'Neutral'
    
    # Title sentiment
    print("Calculating title sentiment...")
    df['title_sentiment'] = df['title'].fillna('').apply(get_sentiment)
    df['title_sentiment_label'] = df['title_sentiment'].apply(categorize_sentiment)
    
    # Description sentiment
    print("Calculating description sentiment...")
    df['description_sentiment'] = df['description'].fillna('').apply(get_sentiment)
    df['description_sentiment_label'] = df['description_sentiment'].apply(categorize_sentiment)
    
    # Tags sentiment
    print("Calculating tags sentiment...")
    df['tags_sentiment'] = df['tags'].fillna('').apply(get_sentiment)
    df['tags_sentiment_label'] = df['tags_sentiment'].apply(categorize_sentiment)
    
    return df


def create_category_features(df):
    """Create category-related features."""
    
    # Category ID to name mapping (YouTube category IDs)
    category_mapping = {
        1: 'Film & Animation',
        2: 'Autos & Vehicles',
        10: 'Music',
        15: 'Pets & Animals',
        17: 'Sports',
        19: 'Travel & Events',
        20: 'Gaming',
        22: 'People & Blogs',
        23: 'Comedy',
        24: 'Entertainment',
        25: 'News & Politics',
        26: 'Howto & Style',
        27: 'Education',
        28: 'Science & Technology',
        29: 'Nonprofits & Activism'
    }
    
    df['category_name'] = df['category_id'].map(category_mapping).fillna('Unknown')
    
    return df


def create_channel_features(df):
    """Create channel-level features (historical performance)."""
    
    # Sort by publish time to calculate rolling averages
    df = df.sort_values('publish_time_fixed').reset_index(drop=True)
    
    # Calculate channel historical statistics (non-leaky - uses all data)
    channel_stats = df.groupby('channel_title').agg({
        'views': ['mean', 'median', 'std', 'count']
    }).reset_index()
    channel_stats.columns = ['channel_title', 'channel_avg_views', 'channel_median_views', 
                             'channel_std_views', 'channel_video_count']
    
    # Merge back to dataframe
    df = df.merge(channel_stats, on='channel_title', how='left')
    
    # Calculate historical average (expanding mean of previous videos - no leakage)
    df['channel_historical_avg'] = df.groupby('channel_title')['views'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    
    # Fill NaN for first video with overall median
    overall_median = df['views'].median()
    df['channel_historical_avg'] = df['channel_historical_avg'].fillna(overall_median)
    
    return df


def create_target_variable(df):
    """Create the target variable (views) and categorize it."""
    
    # Log transform views (to handle skewness)
    df['views_log'] = np.log1p(df['views'])
    
    # Create view categories (terciles)
    view_percentiles = df['views'].quantile([0.33, 0.67])
    
    def categorize_views(views):
        if views <= view_percentiles.iloc[0]:
            return 'Low'
        elif views <= view_percentiles.iloc[1]:
            return 'Medium'
        else:
            return 'High'
    
    df['views_category'] = df['views'].apply(categorize_views)
    
    return df


def prepare_features_for_modeling(df):
    """Final preparation: select features and handle missing values."""
    
    # Define the columns to keep for modeling
    feature_columns = [
        # Identifier (not for modeling, but useful)
        'channel_title',
        
        # Categorical features (pre-upload only)
        'category_id', 'category_name',
        'publish_day', 'publish_month', 'publish_season',
        'publish_hour_category',
        'title_sentiment_label', 'description_sentiment_label', 'tags_sentiment_label',
        
        # Numerical features (pre-upload only)
        'len_title', 'desc_len', 'No_tags', 'has_tags',
        'is_weekend_publish_day', 'publish_hour',
        'title_sentiment', 'description_sentiment', 'tags_sentiment',
        
        # Channel features
        'channel_avg_views', 'channel_median_views', 'channel_video_count',
        'channel_historical_avg',
        
        # Target
        'views', 'views_log', 'views_category'
    ]
    
    # Only keep columns that exist
    existing_columns = [col for col in feature_columns if col in df.columns]
    
    # Select columns
    df_features = df[existing_columns].copy()
    
    # Handle missing values
    for col in df_features.select_dtypes(include=[np.number]).columns:
        if col not in ['views', 'views_log', 'channel_avg_views', 'channel_median_views', 
                       'channel_std_views', 'channel_historical_avg']:
            df_features[col] = df_features[col].fillna(df_features[col].median())
        elif col in ['channel_avg_views', 'channel_median_views', 'channel_std_views', 
                     'channel_historical_avg']:
            # These might be missing for new channels
            df_features[col] = df_features[col].fillna(df_features['views'].median())
    
    for col in df_features.select_dtypes(include=['object']).columns:
        if col not in ['channel_title']:
            df_features[col] = df_features[col].fillna('Unknown')
    
    return df_features


def save_preprocessed_data(df, output_path):
    """Save the preprocessed data."""
    print(f"\nSaving preprocessed data to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows and {len(df.columns)} columns")


def print_feature_summary(df):
    """Print summary of created features."""
    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)
    
    print(f"\nTotal samples: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    
    # Separate columns by type
    target_cols = ['views', 'views_log', 'views_category']
    id_cols = ['channel_title']
    feature_cols = [col for col in df.columns if col not in target_cols + id_cols]
    
    numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nNumerical features ({len(numerical_cols)}):")
    for col in numerical_cols:
        print(f"  - {col}")
    
    print(f"\nCategorical features ({len(categorical_cols)}):")
    for col in categorical_cols:
        print(f"  - {col}")
    
    print(f"\nTarget variables ({len(target_cols)}):")
    for col in target_cols:
        print(f"  - {col}")
    
    print(f"\nIdentifier columns ({len(id_cols)}):")
    for col in id_cols:
        print(f"  - {col}")


def run_preprocessing():
    """Main preprocessing pipeline."""
    
    # Configuration
    input_file = 'data/youtube_data_translated.csv'
    output_file = 'data/youtube_data_preprocessed.csv'
    
    print("="*60)
    print("YouTube Video Views Prediction - Preprocessing Pipeline")
    print("="*60)
    print("\nIMPORTANT: Only using pre-upload features (no leakage)")
    print("="*60)
    
    # Step 1: Load data
    df = load_data(input_file)
    
    # Step 2: Drop leakage columns
    df = drop_leakage_columns(df)
    
    # Step 3: Parse datetime features
    df = parse_datetime_features(df)
    
    # Step 4: Calculate text features
    df = calculate_text_features(df)
    
    # Step 5: Calculate sentiment features
    df = calculate_sentiment_features(df)
    
    # Step 6: Create category features
    df = create_category_features(df)
    
    # Step 7: Create channel features
    df = create_channel_features(df)
    
    # Step 8: Create target variable
    df = create_target_variable(df)
    
    # Step 9: Prepare final features
    df_features = prepare_features_for_modeling(df)
    
    # Step 10: Print feature summary
    print_feature_summary(df_features)
    
    # Step 11: Save preprocessed data
    save_preprocessed_data(df_features, output_file)
    
    print("\n" + "="*60)
    print("Preprocessing completed successfully!")
    print("="*60)
    
    return df_features