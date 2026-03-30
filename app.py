"""
YouTube Video Views Prediction - Gradio Inference App

This app allows users to predict video view counts based on various features
like channel history, content metadata, and publishing patterns.

Author: Muhammad Cikal Merdeka
Date: 2025
"""

import gradio as gr
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Set style for matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


# Global cache for lazy loading
_model_cache = None


def get_model_artifacts():
    """Lazy load model and preprocessing artifacts - only loads when first called."""
    global _model_cache
    if _model_cache is None:
        try:
            model = joblib.load(PROJECT_ROOT / 'models' / 'random_forest_model.joblib')
            encoders = joblib.load(PROJECT_ROOT / 'models' / 'encoders.joblib')
            scalers = joblib.load(PROJECT_ROOT / 'models' / 'scalers.joblib')
            _model_cache = (model, encoders, scalers)
            print("✅ Model artifacts loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model artifacts: {e}")
            _model_cache = (None, None, None)
    return _model_cache


def calculate_sentiment(text):
    """Calculate sentiment score using TextBlob."""
    if pd.isna(text) or text == '' or text == 'none':
        return 0.0
    try:
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0.0


def parse_datetime_features(publish_time_str):
    """Extract time-based features from publish_time_fixed string."""
    try:
        dt = pd.to_datetime(publish_time_str)
        
        publish_day = dt.day_name()
        publish_hour = dt.hour
        publish_month = dt.month_name()
        
        # Categorize hour
        if 0 <= publish_hour < 6:
            publish_hour_category = 'Night'
        elif 6 <= publish_hour < 12:
            publish_hour_category = 'Morning'
        elif 12 <= publish_hour < 18:
            publish_hour_category = 'Afternoon'
        else:
            publish_hour_category = 'Evening'
        
        # Is weekend
        is_weekend = 1 if dt.weekday() >= 5 else 0
        
        return {
            'publish_day': publish_day,
            'publish_hour': publish_hour,
            'publish_month': publish_month,
            'publish_hour_category': publish_hour_category,
            'is_weekend_publish_day': is_weekend
        }
    except:
        return {
            'publish_day': 'Monday',
            'publish_hour': 12,
            'publish_month': 'January',
            'publish_hour_category': 'Afternoon',
            'is_weekend_publish_day': 0
        }


def get_category_name(category_id):
    """Map category ID to category name."""
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
    return category_mapping.get(category_id, 'Unknown')


def preprocess_input(data, encoders, scalers):
    """
    Preprocess input data from translated CSV format.
    Applies full preprocessing pipeline: text features, sentiment, datetime, encoding, scaling.
    """
    df = data.copy()
    
    # Step 1: Parse datetime features from publish_time_fixed
    if 'publish_time_fixed' in df.columns:
        datetime_features = df['publish_time_fixed'].apply(parse_datetime_features)
        datetime_df = pd.DataFrame(datetime_features.tolist())
        df = pd.concat([df, datetime_df], axis=1)
    
    # Step 2: Calculate text features (lengths, tag counts)
    df['len_title'] = df['title'].fillna('').astype(str).str.len()
    df['desc_len'] = df['description'].fillna('').astype(str).str.len()
    
    # Tag features - count tags (split by | or comma)
    df['has_tags'] = (df['tags'].fillna('').astype(str) != 'none').astype(int)
    df['No_tags'] = df['tags'].fillna('').astype(str).str.split('|').str.len()
    df['No_tags'] = df['No_tags'].fillna(0).astype(int)
    
    # Step 3: Calculate sentiment features
    df['title_sentiment'] = df['title'].fillna('').apply(calculate_sentiment)
    df['description_sentiment'] = df['description'].fillna('').apply(calculate_sentiment)
    df['tags_sentiment'] = df['tags'].fillna('').apply(calculate_sentiment)
    
    # Step 4: Create category features
    if 'category_id' in df.columns:
        df['category_name'] = df['category_id'].apply(get_category_name)
    
    # Step 5: One-hot encoding for categorical features
    # Category encoding
    if 'category_name' in df.columns:
        encoder_key = 'onehot_category_name'
        if encoder_key in encoders:
            oh_encoder = encoders[encoder_key]
            oh_result = oh_encoder.transform(df[['category_name']])
            cats = oh_encoder.categories_[0]
            
            oh_col_names = [f'category_name_{cat}' for cat in cats[1:]]
            oh_df = pd.DataFrame(oh_result, columns=oh_col_names, index=df.index)
            
            df = df.drop(columns=['category_name'])
            df = pd.concat([df, oh_df], axis=1)
    
    # Publish day encoding
    if 'publish_day' in df.columns:
        encoder_key = 'onehot_publish_day'
        if encoder_key in encoders:
            oh_encoder = encoders[encoder_key]
            oh_result = oh_encoder.transform(df[['publish_day']])
            cats = oh_encoder.categories_[0]
            
            oh_col_names = [f'publish_day_{cat}' for cat in cats[1:]]
            oh_df = pd.DataFrame(oh_result, columns=oh_col_names, index=df.index)
            
            df = df.drop(columns=['publish_day'])
            df = pd.concat([df, oh_df], axis=1)
    
    # Publish month encoding
    if 'publish_month' in df.columns:
        encoder_key = 'onehot_publish_month'
        if encoder_key in encoders:
            oh_encoder = encoders[encoder_key]
            oh_result = oh_encoder.transform(df[['publish_month']])
            cats = oh_encoder.categories_[0]
            
            oh_col_names = [f'publish_month_{cat}' for cat in cats[1:]]
            oh_df = pd.DataFrame(oh_result, columns=oh_col_names, index=df.index)
            
            df = df.drop(columns=['publish_month'])
            df = pd.concat([df, oh_df], axis=1)
    
    # Publish hour category encoding
    if 'publish_hour_category' in df.columns:
        encoder_key = 'onehot_publish_hour_category'
        if encoder_key in encoders:
            oh_encoder = encoders[encoder_key]
            oh_result = oh_encoder.transform(df[['publish_hour_category']])
            cats = oh_encoder.categories_[0]
            
            oh_col_names = [f'publish_hour_category_{cat}' for cat in cats[1:]]
            oh_df = pd.DataFrame(oh_result, columns=oh_col_names, index=df.index)
            
            df = df.drop(columns=['publish_hour_category'])
            df = pd.concat([df, oh_df], axis=1)
    
    # Step 6: Feature scaling
    scaling_config = {
        'robust': {
            'columns': [
                'channel_median_views',
                'channel_avg_views', 
                'channel_historical_avg',
                'channel_video_count',
                'desc_len',
                'No_tags'
            ]
        },
        'minmax': {
            'columns': [
                'title_sentiment',
                'description_sentiment', 
                'tags_sentiment',
                'len_title',
                'publish_hour'
            ]
        }
    }
    
    for method_name, config in scaling_config.items():
        if method_name in scalers:
            cols = config['columns']
            existing_cols = [col for col in cols if col in df.columns]
            if existing_cols:
                df[existing_cols] = df[existing_cols].astype(float)
                df[existing_cols] = scalers[method_name].transform(df[existing_cols])
    
    # Step 7: Select features to match training (base_cols + category_name_cols only)
    # Based on notebook cell: selected_cols = base_cols + category_name_cols
    base_columns = [
        'channel_avg_views', 'channel_historical_avg', 'channel_median_views', 
        'channel_video_count', 'desc_len', 'title_sentiment', 'description_sentiment',
        'tags_sentiment', 'No_tags', 'len_title', 'publish_hour'
    ]
    
    # Only keep category_name encoded columns (not publish_day, publish_month, etc.)
    category_name_cols = [col for col in df.columns if col.startswith('category_name')]
    
    expected_columns = base_columns + category_name_cols
    
    # Ensure all expected columns are present (fill missing with 0)
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Get the exact column order from the model if available
    model, _, _ = get_model_artifacts()
    if model is not None and hasattr(model, 'feature_names_in_'):
        # Use the model's stored feature order
        training_columns = list(model.feature_names_in_)
        # Ensure all training columns exist
        for col in training_columns:
            if col not in df.columns:
                df[col] = 0
        # Reorder using model's exact order
        df = df[training_columns]
    else:
        # Fallback to expected order
        df = df[expected_columns]
    
    return df


def predict_views(channel_avg_views, channel_median_views, channel_historical_avg,
                  channel_video_count, title, description, tags, category_id,
                  publish_time):
    """Make prediction for a single video."""
    
    # Lazy load model artifacts
    model, encoders, scalers = get_model_artifacts()
    
    if model is None:
        return "Error: Model not loaded", None, None
    
    # Create input DataFrame with translated data format
    input_data = pd.DataFrame({
        'channel_avg_views': [float(channel_avg_views)],
        'channel_median_views': [float(channel_median_views)],
        'channel_historical_avg': [float(channel_historical_avg)],
        'channel_video_count': [int(channel_video_count)],
        'title': [title],
        'description': [description],
        'tags': [tags],
        'category_id': [int(category_id)],
        'publish_time_fixed': [publish_time]
    })
    
    # Preprocess (parse datetime, calculate sentiment, encode, scale)
    processed_data = preprocess_input(input_data, encoders, scalers)
    
    # Predict
    prediction = model.predict(processed_data)[0]
    
    # Format result with category
    if prediction < 100000:
        category = "Low"
        emoji = "🔴"
    elif prediction < 500000:
        category = "Medium"
        emoji = "🟡"
    else:
        category = "High"
        emoji = "🟢"
    
    result = f"{emoji} **Predicted Views: {prediction:,.0f}**\n\n**Category:** {category}"
    
    # Create prediction gauge
    gauge_fig, ax = plt.subplots(figsize=(10, 2))
    # Normalize for visualization (max ~5M views)
    max_views = 5000000
    gauge_value = min(prediction / max_views * 100, 100)
    # Color based on view category
    if prediction < 100000:
        color = '#dc3545'  # Red for low
    elif prediction < 500000:
        color = '#ffc107'  # Yellow for medium
    else:
        color = '#28a745'  # Green for high
    ax.barh([0], [gauge_value], color=color, height=0.4)
    ax.barh([0], [100], color='lightgray', height=0.4, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Relative View Count (%)', fontsize=10)
    ax.axvline(x=50, color='red', linestyle='--', linewidth=2)
    ax.text(gauge_value, 0, f'{prediction:,.0f}', 
            ha='center', va='center', fontweight='bold', fontsize=11)
    ax.set_yticks([])
    ax.set_title('Predicted Views Gauge', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Feature importance
    importance_fig = create_feature_importance_plot()
    
    return result, gauge_fig, importance_fig


def create_feature_importance_plot():
    """Create feature importance plot."""
    # Lazy load model artifacts
    model, _, _ = get_model_artifacts()
    
    if model is None or not hasattr(model, 'feature_importances_'):
        return None
    
    # Use model's feature names if available
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)
    else:
        # Fallback feature names
        feature_names = [
            'channel_avg_views', 'channel_historical_avg', 'channel_median_views', 
            'channel_video_count', 'desc_len', 'title_sentiment', 'description_sentiment',
            'tags_sentiment', 'No_tags', 'len_title', 'publish_hour'
        ]
    
    importances = model.feature_importances_
    
    # Ensure lengths match
    if len(importances) != len(feature_names):
        feature_names = feature_names[:len(importances)]
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True).tail(10)  # Show top 10
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
    ax.set_xlabel('Importance Score', fontsize=10)
    ax.set_title('Top 10 Feature Importance (Random Forest)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars, importance_df['Importance']):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
               f'{val:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    return fig


# Create the Gradio interface
with gr.Blocks(title="YouTube Views Predictor") as demo:
    
    gr.Markdown("""
    # 📺 YouTube Video Views Predictor
    
    Predict video view counts before upload using Machine Learning
    """)
    
    with gr.Tabs():
        
        # Tab 1: Manual Entry
        with gr.Tab("📝 Manual Entry"):
            gr.Markdown("### Enter video details to get a prediction")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Channel History**")
                    channel_avg_views = gr.Number(
                        label="Channel Average Views", 
                        value=500000, 
                        minimum=0,
                        info="Average views of the channel's previous videos"
                    )
                    channel_median_views = gr.Number(
                        label="Channel Median Views", 
                        value=400000, 
                        minimum=0,
                        info="Median views of the channel's previous videos"
                    )
                    channel_historical_avg = gr.Number(
                        label="Channel Historical Average", 
                        value=450000, 
                        minimum=0,
                        info="Historical average of channel performance"
                    )
                    channel_video_count = gr.Number(
                        label="Channel Video Count", 
                        value=50, 
                        minimum=1,
                        info="Total number of videos from the channel"
                    )
                
                with gr.Column():
                    gr.Markdown("**Video Content**")
                    title = gr.Textbox(
                        label="Video Title", 
                        value="Amazing Video Title",
                        info="Title of the video (sentiment will be analyzed)"
                    )
                    description = gr.Textbox(
                        label="Video Description", 
                        value="This is an interesting video description.",
                        lines=3,
                        info="Video description (sentiment will be analyzed)"
                    )
                    tags = gr.Textbox(
                        label="Tags", 
                        value="music|entertainment|viral",
                        info="Pipe-separated tags (e.g., tag1|tag2|tag3)"
                    )
                    category_id = gr.Dropdown(
                        label="Category",
                        choices=[
                            ("Film & Animation", 1),
                            ("Autos & Vehicles", 2),
                            ("Music", 10),
                            ("Pets & Animals", 15),
                            ("Sports", 17),
                            ("Travel & Events", 19),
                            ("Gaming", 20),
                            ("People & Blogs", 22),
                            ("Comedy", 23),
                            ("Entertainment", 24),
                            ("News & Politics", 25),
                            ("Howto & Style", 26),
                            ("Education", 27),
                            ("Science & Technology", 28),
                            ("Nonprofits & Activism", 29)
                        ],
                        value=24
                    )
                
                with gr.Column():
                    gr.Markdown("**Publishing Details**")
                    publish_time = gr.Textbox(
                        label="Publish Time",
                        value="2024-01-15 14:30:00",
                        info="Format: YYYY-MM-DD HH:MM:SS"
                    )
            
            predict_btn = gr.Button("🔮 Predict Views", variant="primary")
            
            with gr.Row():
                result_output = gr.Markdown(label="Prediction Result")
            
            with gr.Row():
                gauge_plot = gr.Plot(label="Prediction Gauge")
                importance_plot = gr.Plot(label="Feature Importance")
            
            predict_btn.click(
                fn=predict_views,
                inputs=[channel_avg_views, channel_median_views, channel_historical_avg,
                       channel_video_count, title, description, tags, category_id,
                       publish_time],
                outputs=[result_output, gauge_plot, importance_plot]
            )
        
        # Tab 2: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ### About This App
            
            This app uses a **Random Forest Regressor** trained on YouTube trending video data to predict 
            video view counts before upload.
            
            #### Model Features:
            - **Channel history**: average views, median views, video count, historical average
            - **Content features**: title length, description length, tag count, sentiment scores
            - **Publishing details**: category, publish time (parsed into day, hour, month)
            
            #### How to Use:
            Fill in the video details in the Manual Entry tab and click **Predict Views** to get your prediction.
            
            #### Model Performance:
            The model was trained on 36,000+ video records and achieves MAE ~174K views on validation.
            
            #### Technical Details:
            - **Algorithm**: Random Forest Regressor
            - **Preprocessing**: Datetime parsing, sentiment analysis (TextBlob), one-hot encoding, feature scaling
            - **Framework**: Gradio for the web interface
            """)
    
    gr.Markdown("---")

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
