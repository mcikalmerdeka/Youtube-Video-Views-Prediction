# YouTube Video Views Prediction

![Project Header](./assets/Project%20Header.jpg)

A machine learning solution to predict YouTube video view counts before upload, enabling content creators and marketers to optimize their content strategies and maximize audience engagement.

## Project Overview

End-to-end data science project that analyzes video metadata, channel performance history, and publishing patterns to predict video view counts. Includes comprehensive EDA, advanced text preprocessing with translation, feature engineering with sentiment analysis, XGBoost regression model, and preprocessing utilities for model deployment.

## Key Results

- **Model Algorithm**: XGBoost Regressor with hyperparameter tuning
- **Performance Metrics**:
  - R² Score: 0.XX (explained variance)
  - RMSE: XX,XXX views
  - MAE: XX,XXX views
- **Dataset Size**: 36,791 trending videos (32,562 after preprocessing)
- **Features Used**: 41 engineered features including channel history, sentiment scores, and publishing patterns

## Project Structure

```
├── notebook_fix.ipynb          # EDA and model training notebook
├── pyproject.toml              # Project dependencies (uv/pip)
├── requirements.txt            # Pip-compatible dependencies
├── README.md                   # This file
├── data/                       # Dataset files
│   ├── youtube_statistics.xlsx
│   ├── youtube_data_translated.csv
│   ├── youtube_data_preprocessed.csv
│   └── youtube_data_text_preprocessed.csv
├── output/                     # Generated visualizations and plots
├── utils/                      # Reusable preprocessing and ML functions
│   ├── __init__.py
│   ├── preprocessing.py        # General preprocessing utilities
│   ├── text_preprocessing.py   # NLP text processing
│   ├── preprocess_youtube_data.py  # Domain-specific preprocessing
│   ├── feature_selection.py    # Feature selection utilities
│   ├── regression_evals_and_tuning.py  # ML model tuning
│   ├── visualization.py        # Plotting functions
│   ├── statistics.py           # Statistical analysis
│   ├── translation_utils.py    # Language translation utilities
│   └── ab_testing.py           # A/B testing utilities
└── reference/                  # Reference materials and documentation
```

## Quick Start

### Prerequisites

- Python 3.12+
- uv (recommended) or pip

### Installation

```bash
# Clone repository
git clone https://github.com/mcikalmerdeka/Youtube-Video-Views-Prediction.git
cd Youtube-Video-Views-Prediction

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies (using pip)
pip install -r requirements.txt

# Or using uv (faster alternative)
uv sync
```

### Run the Notebook

```bash
# Start Jupyter notebook
uv run jupyter notebook

# Or with activated environment:
jupyter notebook notebook_fix.ipynb
```

## Features

- **Multilingual Text Processing**: Automatic translation of Indian language content (Hindi, Tamil, etc.) to English
- **Advanced Text Preprocessing**: 
  - HTML/URL/email removal
  - Sentiment analysis using TextBlob
  - Lemmatization and stopword removal
  - Word cloud visualizations
- **Feature Engineering**:
  - Channel historical performance metrics (avg/median views)
  - Sentiment scores for title, description, and tags
  - Temporal features (publish day, hour, season)
  - Content metadata (length metrics, tag counts)
- **Outlier Detection**: IQR-based outlier removal with training-derived thresholds
- **Model Training**: XGBoost regression with cross-validation
- **Feature Selection**: Correlation analysis and VIF for multicollinearity detection

## Technical Stack

- Python 3.12+
- XGBoost (gradient boosting regression)
- scikit-learn (preprocessing, model evaluation)
- pandas, numpy (data processing)
- matplotlib, seaborn (visualization)
- TextBlob (sentiment analysis)
- deep-translator, langid (translation)
- wordcloud (text visualization)
- category-encoders (categorical encoding)

## Business Problem

A digital media analytics company in India aims to forecast YouTube video performance to support content creators and marketers in optimizing their content strategies. However, the current approach lacks predictive capabilities, making it challenging to estimate the potential success of a video before publishing.

### Problem Statement

How can we develop a machine learning system that predicts YouTube video view counts using only pre-upload features to enable data-driven content optimization?

### Business Metrics

- **Mean Absolute Percentage Error (MAPE)** [MAIN]: Measure the accuracy of the predicted view count compared to the actual view count
- **Engagement Rate (ER)** [SECONDARY]: Percentage of interactions relative to video views to gauge engagement potential

### Goals

- **Primary**: Predict YouTube video performance before upload using historical data and video metadata
- **Secondary**: Assist content creators and marketers with data-driven decisions in content planning

### Objectives

1. Build a predictive model capable of estimating the number of views a video will generate
2. Identify key features influencing viewership trends to provide actionable insights
3. Help creators align their strategies with emerging trends and maximize audience reach

## Model Methodology

### Data Preprocessing Pipeline

1. **Data Understanding**:
   - Initial dataset: 36,791 rows × 18 columns
   - Features include video metadata, engagement metrics, and timestamps
   - Mixed English and Indian languages (Hindi, Tamil, Punjabi, etc.)

2. **Text Translation & Preprocessing**:
   - Language detection using langid
   - Translation of Indian language content to English via deep-translator
   - Text cleaning: HTML removal, URL stripping, accent removal
   - Lemmatization and stopword removal
   - Sentiment analysis for title, description, and tags

3. **Feature Engineering**:
   - **Channel Features**: Historical average views, median views, video count per channel
   - **Temporal Features**: Publish day, month, hour, season, weekend indicator
   - **Content Features**: Title length, description length, tag count, sentiment scores
   - **Category Features**: One-hot encoded video categories (Entertainment, Music, News, etc.)

4. **Data Cleaning**:
   - Removed 4,229 duplicate rows
   - Outlier detection using IQR method (threshold: 1.5)
   - Applied training-derived thresholds to validation set to prevent data leakage

5. **Train/Validation/Test Split**:
   - Training: 75% (17,824 samples after outlier removal)
   - Validation: 20% (4,753 samples)
   - Test: 5% (1,839 samples)
   - Stratified by category_name to maintain distribution

### Feature Selection

Selected 41 features based on correlation analysis and VIF:

- **Channel Historical Metrics** (High Importance):
  - channel_historical_avg: Average views of channel's previous videos
  - channel_avg_views: Mean views across channel videos
  - channel_median_views: Median views for robust central tendency
  - channel_video_count: Number of videos from the channel

- **Temporal Features**:
  - publish_hour: Hour of day (0-23)
  - publish_day: Day of week (one-hot encoded)
  - publish_month: Month of year (one-hot encoded)
  - is_weekend_publish_day: Weekend indicator

- **Content & Sentiment Features**:
  - len_title: Title character length
  - desc_len: Description length
  - No_tags: Number of tags
  - title_sentiment: Title sentiment score (-1 to 1)
  - description_sentiment: Description sentiment score
  - tags_sentiment: Tags sentiment score

- **Category Features** (One-hot encoded):
  - Entertainment, Music, News & Politics, Comedy, etc.

### Model Training

- **Algorithm**: XGBoost Regressor
- **Validation Strategy**: Train/Validation/Test split (75%/20%/5%)
- **Cross-Validation**: 5-fold CV for robust performance estimation
- **Target Variable**: Video view count (regression task)
- **Evaluation Metrics**: R², RMSE, MAE, MAPE

### Addressing Data Leakage

Critical correction made during development: Removed post-upload features (likes, dislikes, comment_count) that would only be available after video publication. Model now uses only pre-upload features available to creators before publishing.

## Data Dictionary

| Feature | Description | Type | Importance |
|---------|-------------|------|------------|
| channel_historical_avg | Average views of channel's previous videos | Numerical | High |
| channel_avg_views | Mean views across all channel videos | Numerical | High |
| channel_median_views | Median views for the channel | Numerical | Medium |
| channel_video_count | Number of videos from the channel | Numerical | Medium |
| title_sentiment | Sentiment score of video title (-1 to 1) | Numerical | Medium |
| description_sentiment | Sentiment score of description | Numerical | Medium |
| tags_sentiment | Sentiment score of tags | Numerical | Low |
| len_title | Length of video title in characters | Numerical | Low |
| desc_len | Length of description | Numerical | Low |
| No_tags | Number of tags used | Numerical | Low |
| publish_hour | Hour of publication (0-23) | Numerical | Medium |
| is_weekend_publish_day | Whether published on weekend | Binary | Low |
| category_name_* | Video category (one-hot encoded) | Categorical | Medium |
| publish_day_* | Day of week (one-hot encoded) | Categorical | Low |
| publish_month_* | Month (one-hot encoded) | Categorical | Low |

## Key Insights

1. **Channel History is the Strongest Predictor**: Historical average views of a channel is the most important feature for predicting future video success
2. **Sentiment Matters**: Title and description sentiment scores provide meaningful signal for view prediction
3. **Timing Effects**: Publish hour and day of week show measurable impact on view counts
4. **Content Categories**: Entertainment and Music categories show distinct view patterns
5. **Multilingual Challenge**: Mixed language content required translation preprocessing to enable text feature extraction

## Files Managed by Git LFS

- `data/*.csv` - Large dataset files
- `output/*.png` - Generated visualization images

## Author

**Muhammad Cikal Merdeka** | Data Analyst/Data Scientist

- [GitHub](https://github.com/mcikalmerdeka)
- [LinkedIn](https://www.linkedin.com/in/mcikalmerdeka)
- [Email](mailto:mcikalmerdeka@gmail.com)

---
