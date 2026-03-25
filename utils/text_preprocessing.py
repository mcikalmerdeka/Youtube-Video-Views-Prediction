import re
import unicodedata
from typing import Dict, List, Optional, Union, Tuple, Any, Set
from collections import Counter

import numpy as np
import pandas as pd

# Optional dependencies with graceful fallback
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    WordCloud = None

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    from bs4 import MarkupResemblesLocatorWarning
    import warnings
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import contractions
    CONTRACTIONS_AVAILABLE = True
except ImportError:
    CONTRACTIONS_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║                  Functions for Text Preprocessing and Analysis                   ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝

class TextPreprocessor:
    """
    A comprehensive text preprocessing class for NLP tasks.
    
    This class provides methods for cleaning, normalizing, and transforming
    text data. It supports various preprocessing steps including HTML removal,
    URL/email removal, normalization, tokenization, lemmatization, and stemming.
    
    Parameters:
    -----------
    language : str, default='english'
        Language for stopwords (e.g., 'english', 'french', 'german')
    custom_stopwords : Optional[Set[str]], default=None
        Additional stopwords to remove beyond the language defaults
    
    Examples:
    ---------
    >>> # Initialize preprocessor with default settings
    >>> preprocessor = TextPreprocessor()
    
    >>> # Initialize with custom stopwords
    >>> preprocessor = TextPreprocessor(
    ...     language='english',
    ...     custom_stopwords={'custom', 'words', 'here'}
    ... )
    
    >>> # Process single text
    >>> clean_text = preprocessor.process_text("Your text here with URLs http://example.com")
    
    >>> # Process DataFrame columns
    >>> df_clean = preprocessor.preprocess_dataframe(df, columns=['title', 'description'])
    """
    
    # Pre-compiled regex patterns for performance
    _URL_PATTERN = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        r'|(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+(?:/\S*)?'
        r'|bit\.ly/\S+|goo\.gl/\S+|t\.co/\S+|tinyurl\.com/\S+'
    )
    _EMAIL_PATTERN = re.compile(r'\S+@\S+')
    _MENTION_PATTERN = re.compile(r'[@#]\w+')
    _WHITESPACE_PATTERN = re.compile(r'\s+')
    _NON_ALPHANUM_PATTERN = re.compile(r'[^a-zA-Z0-9\s]')
    _NON_ALPHA_PATTERN = re.compile(r'[^a-zA-Z\s]')
    
    def __init__(
        self, 
        language: str = 'english',
        custom_stopwords: Optional[Set[str]] = None
    ):
        self.language = language
        self._stopwords: Set[str] = set()
        self._lemmatizer = None
        self._stemmer = None
        
        # Initialize NLTK components if available
        if NLTK_AVAILABLE:
            try:
                self._stopwords = set(stopwords.words(language))
                self._lemmatizer = WordNetLemmatizer()
                self._stemmer = PorterStemmer()
            except LookupError:
                # NLTK data not downloaded
                pass
        
        # Add custom stopwords
        if custom_stopwords:
            self._stopwords.update(custom_stopwords)
    
    def _validate_dependencies(self) -> None:
        """Check if required optional dependencies are available."""
        if not NLTK_AVAILABLE:
            raise ImportError(
                "NLTK is required for text preprocessing. "
                "Install with: pip install nltk"
            )
    
    def add_stopwords(self, words: Union[str, List[str], Set[str]]) -> None:
        """
        Add custom stopwords to the existing set.
        
        Parameters:
        -----------
        words : Union[str, List[str], Set[str]]
            Stopword(s) to add. Can be a single word, list, or set.
        
        Examples:
        ---------
        >>> preprocessor = TextPreprocessor()
        >>> preprocessor.add_stopwords('custom')
        >>> preprocessor.add_stopwords(['word1', 'word2', 'word3'])
        """
        if isinstance(words, str):
            self._stopwords.add(words.lower())
        else:
            self._stopwords.update(word.lower() for word in words)
    
    def remove_stopwords(self, words: Union[str, List[str], Set[str]]) -> None:
        """
        Remove words from the stopwords set.
        
        Parameters:
        -----------
        words : Union[str, List[str], Set[str]]
            Stopword(s) to remove. Can be a single word, list, or set.
        
        Examples:
        ---------
        >>> preprocessor = TextPreprocessor()
        >>> preprocessor.remove_stopwords('not')  # Keep negations
        >>> preprocessor.remove_stopwords(['good', 'bad'])  # Keep sentiment words
        """
        if isinstance(words, str):
            self._stopwords.discard(words.lower())
        else:
            for word in words:
                self._stopwords.discard(word.lower())
    
    def replace_newlines(self, text: str) -> str:
        """
        Replace newline characters and escaped newlines with spaces.
        
        Parameters:
        -----------
        text : str
            Input text to process
        
        Returns:
        --------
        str
            Text with newlines replaced by spaces
        
        Examples:
        ---------
        >>> preprocessor = TextPreprocessor()
        >>> preprocessor.replace_newlines("Hello\\nWorld\\n\\nTest")
        'Hello World Test'
        """
        if pd.isna(text):
            return text
        # Replace both escaped and actual newlines
        text = text.replace('\\n', ' ').replace('\n', ' ').replace('\r', ' ')
        # Normalize multiple spaces
        return self._WHITESPACE_PATTERN.sub(' ', text).strip()
    
    def remove_html(self, text: str) -> str:
        """
        Remove HTML tags from text using BeautifulSoup.
        
        Parameters:
        -----------
        text : str
            Input text containing HTML
        
        Returns:
        --------
        str
            Text with HTML tags removed
        
        Examples:
        ---------
        >>> preprocessor = TextPreprocessor()
        >>> preprocessor.remove_html("<p>Hello <b>World</b></p>")
        'Hello World'
        """
        if pd.isna(text) or not BS4_AVAILABLE:
            return text
        return BeautifulSoup(text, "html.parser").get_text(separator=' ')
    
    def expand_contractions(self, text: str) -> str:
        """
        Expand contractions (e.g., "don't" -> "do not").
        
        Parameters:
        -----------
        text : str
            Input text with contractions
        
        Returns:
        --------
        str
            Text with contractions expanded
        
        Examples:
        ---------
        >>> preprocessor = TextPreprocessor()
        >>> preprocessor.expand_contractions("I don't think it's working")
        "I do not think it is working"
        """
        if pd.isna(text) or not CONTRACTIONS_AVAILABLE:
            return text
        return contractions.fix(text)
    
    def remove_accents(self, text: str) -> str:
        """
        Remove accented characters by converting to ASCII.
        
        Parameters:
        -----------
        text : str
            Input text with accented characters
        
        Returns:
        --------
        str
            Text with accents removed
        
        Examples:
        ---------
        >>> preprocessor = TextPreprocessor()
        >>> preprocessor.remove_accents("café résumé")
        'cafe resume'
        """
        if pd.isna(text):
            return text
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from text.
        
        Parameters:
        -----------
        text : str
            Input text containing URLs
        
        Returns:
        --------
        str
            Text with URLs removed
        
        Examples:
        ---------
        >>> preprocessor = TextPreprocessor()
        >>> preprocessor.remove_urls("Visit http://example.com for more info")
        'Visit  for more info'
        """
        if pd.isna(text):
            return text
        return self._URL_PATTERN.sub('', text)
    
    def remove_emails(self, text: str) -> str:
        """
        Remove email addresses from text.
        
        Parameters:
        -----------
        text : str
            Input text containing email addresses
        
        Returns:
        --------
        str
            Text with emails removed
        
        Examples:
        ---------
        >>> preprocessor = TextPreprocessor()
        >>> preprocessor.remove_emails("Contact me at test@example.com")
        'Contact me at '
        """
        if pd.isna(text):
            return text
        return self._EMAIL_PATTERN.sub('', text)
    
    def remove_social_handles(self, text: str) -> str:
        """
        Remove social media handles (@mentions and #hashtags).
        
        Parameters:
        -----------
        text : str
            Input text containing handles
        
        Returns:
        --------
        str
            Text with social handles removed
        
        Examples:
        ---------
        >>> preprocessor = TextPreprocessor()
        >>> preprocessor.remove_social_handles("Follow @user and check #hashtag")
        'Follow  and check '
        """
        if pd.isna(text):
            return text
        return self._MENTION_PATTERN.sub('', text)
    
    def remove_special_chars(
        self, 
        text: str, 
        keep_punctuation: bool = False
    ) -> str:
        """
        Remove special characters and optionally keep punctuation.
        
        Parameters:
        -----------
        text : str
            Input text to clean
        keep_punctuation : bool, default=False
            If True, keeps basic punctuation marks (. , ! ? -)
        
        Returns:
        --------
        str
            Cleaned text with special characters removed
        
        Examples:
        ---------
        >>> preprocessor = TextPreprocessor()
        >>> preprocessor.remove_special_chars("Hello! How are you? @#$%")
        'Hello How are you '
        >>> preprocessor.remove_special_chars("Hello! How are you?", keep_punctuation=True)
        'Hello! How are you?'
        """
        if pd.isna(text):
            return text
        
        if keep_punctuation:
            # Keep letters, spaces, and basic punctuation
            pattern = r'[^a-zA-Z\s.,!?\-]'
        else:
            # Keep only letters and spaces
            pattern = r'[^a-zA-Z\s]'
        
        text = re.sub(pattern, '', text)
        return self._WHITESPACE_PATTERN.sub(' ', text).strip()
    
    def correct_spelling(self, text: str) -> str:
        """
        Correct spelling using TextBlob.
        
        Note: This is computationally expensive and should be used sparingly.
        
        Parameters:
        -----------
        text : str
            Input text with potential spelling errors
        
        Returns:
        --------
        str
            Text with corrected spelling
        
        Examples:
        ---------
        >>> preprocessor = TextPreprocessor()
        >>> preprocessor.correct_spelling("Ths is a tst sentnce")
        'This is a test sentence'
        """
        if pd.isna(text) or not TEXTBLOB_AVAILABLE:
            return text
        return str(TextBlob(text).correct())
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into individual words.
        
        Parameters:
        -----------
        text : str
            Input text to tokenize
        
        Returns:
        --------
        List[str]
            List of tokens
        
        Examples:
        ---------
        >>> preprocessor = TextPreprocessor()
        >>> preprocessor.tokenize("Hello world test")
        ['Hello', 'world', 'test']
        """
        self._validate_dependencies()
        if pd.isna(text):
            return []
        return word_tokenize(str(text).lower())
    
    def remove_stopwords_from_tokens(
        self, 
        tokens: List[str]
    ) -> List[str]:
        """
        Remove stopwords from a list of tokens.
        
        Parameters:
        -----------
        tokens : List[str]
            List of word tokens
        
        Returns:
        --------
        List[str]
            Tokens with stopwords removed
        
        Examples:
        ---------
        >>> preprocessor = TextPreprocessor()
        >>> preprocessor.remove_stopwords_from_tokens(['the', 'quick', 'brown', 'fox'])
        ['quick', 'brown', 'fox']
        """
        return [token for token in tokens if token.lower() not in self._stopwords]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize a list of tokens.
        
        Parameters:
        -----------
        tokens : List[str]
            List of word tokens
        
        Returns:
        --------
        List[str]
            Lemmatized tokens
        
        Examples:
        ---------
        >>> preprocessor = TextPreprocessor()
        >>> preprocessor.lemmatize_tokens(['running', 'better', 'dogs'])
        ['running', 'better', 'dog']
        """
        self._validate_dependencies()
        if self._lemmatizer is None:
            return tokens
        return [self._lemmatizer.lemmatize(token) for token in tokens]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Stem a list of tokens using Porter Stemmer.
        
        Parameters:
        -----------
        tokens : List[str]
            List of word tokens
        
        Returns:
        --------
        List[str]
            Stemmed tokens
        
        Examples:
        ---------
        >>> preprocessor = TextPreprocessor()
        >>> preprocessor.stem_tokens(['running', 'better', 'dogs'])
        ['run', 'better', 'dog']
        """
        self._validate_dependencies()
        if self._stemmer is None:
            return tokens
        return [self._stemmer.stem(token) for token in tokens]
    
    def process_text(
        self,
        text: Union[str, None],
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Process text with specified preprocessing options.
        
        This is the main method for text preprocessing that applies a configurable
        pipeline of text cleaning operations.
        
        Parameters:
        -----------
        text : Union[str, None]
            Input text to process. If None or NaN, returns empty string.
        options : Optional[Dict[str, Any]], default=None
            Dictionary of preprocessing options. If None, uses default options.
            Available options:
            - replace_newlines: bool, default=True - Replace newline characters
            - remove_html: bool, default=True - Remove HTML tags
            - remove_urls: bool, default=True - Remove URLs
            - remove_emails: bool, default=True - Remove email addresses
            - remove_social: bool, default=True - Remove @mentions and #hashtags
            - expand_contractions: bool, default=True - Expand contractions
            - remove_accents: bool, default=True - Remove accented characters
            - lowercase: bool, default=True - Convert to lowercase
            - remove_special_chars: bool, default=True - Remove special characters
            - keep_punctuation: bool, default=False - Keep punctuation marks
            - spell_correction: bool, default=False - Correct spelling (slow)
            - remove_stopwords: bool, default=True - Remove stopwords
            - lemmatize: bool, default=True - Apply lemmatization
            - stem: bool, default=False - Apply stemming (don't use with lemmatize)
            - min_token_length: int, default=1 - Minimum token length to keep
        
        Returns:
        --------
        str
            Processed text
        
        Examples:
        ---------
        >>> preprocessor = TextPreprocessor()
        
        >>> # Process with default options
        >>> clean_text = preprocessor.process_text("Hello World! http://example.com")
        
        >>> # Process with custom options
        >>> options = {
        ...     'remove_stopwords': False,
        ...     'lemmatize': False,
        ...     'stem': True
        ... }
        >>> clean_text = preprocessor.process_text("Hello World!", options=options)
        """
        # Handle null values
        if pd.isna(text):
            return ''
        
        text = str(text)
        
        # Default options
        default_options = {
            'replace_newlines': True,
            'remove_html': True,
            'remove_urls': True,
            'remove_emails': True,
            'remove_social': True,
            'expand_contractions': True,
            'remove_accents': True,
            'lowercase': True,
            'remove_special_chars': True,
            'keep_punctuation': False,
            'spell_correction': False,
            'remove_stopwords': True,
            'lemmatize': True,
            'stem': False,
            'min_token_length': 1
        }
        
        # Merge with user options
        if options is not None:
            default_options.update(options)
        opts = default_options
        
        # Pre-processing steps
        if opts['replace_newlines']:
            text = self.replace_newlines(text)
        
        if opts['remove_html']:
            text = self.remove_html(text)
        
        if opts['remove_urls']:
            text = self.remove_urls(text)
        
        if opts['remove_emails']:
            text = self.remove_emails(text)
        
        if opts['remove_social']:
            text = self.remove_social_handles(text)
        
        if opts['expand_contractions']:
            text = self.expand_contractions(text)
        
        if opts['remove_accents']:
            text = self.remove_accents(text)
        
        if opts['lowercase']:
            text = text.lower()
        
        if opts['remove_special_chars']:
            text = self.remove_special_chars(text, keep_punctuation=opts['keep_punctuation'])
        
        if opts['spell_correction']:
            text = self.correct_spelling(text)
        
        # Tokenization and post-processing
        if opts['remove_stopwords'] or opts['lemmatize'] or opts['stem']:
            tokens = self.tokenize(text)
            
            if opts['remove_stopwords']:
                tokens = self.remove_stopwords_from_tokens(tokens)
            
            # Apply lemmatization or stemming (not both)
            if opts['lemmatize'] and not opts['stem']:
                tokens = self.lemmatize_tokens(tokens)
            elif opts['stem']:
                tokens = self.stem_tokens(tokens)
            
            # Filter by minimum length
            if opts['min_token_length'] > 1:
                tokens = [t for t in tokens if len(t) >= opts['min_token_length']]
            
            text = ' '.join(tokens)
        
        return text.strip()
    
    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        columns: List[str],
        options: Optional[Dict[str, Any]] = None,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Process multiple text columns in a DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing text columns to process
        columns : List[str]
            List of column names to preprocess
        options : Optional[Dict[str, Any]], default=None
            Preprocessing options (see process_text for details)
        inplace : bool, default=False
            If True, modify the DataFrame in place. Otherwise, return a copy.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with preprocessed text columns
        
        Examples:
        ---------
        >>> preprocessor = TextPreprocessor()
        
        >>> # Process specific columns
        >>> df_clean = preprocessor.preprocess_dataframe(
        ...     df, 
        ...     columns=['title', 'description', 'content']
        ... )
        
        >>> # Process with custom options
        >>> options = {'remove_stopwords': False, 'lemmatize': True}
        >>> df_clean = preprocessor.preprocess_dataframe(
        ...     df,
        ...     columns=['text'],
        ...     options=options
        ... )
        """
        if not inplace:
            df = df.copy()
        
        for col in columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: self.process_text(x, options))
        
        return df
    
    def preprocess_dataframe_vectorized(
        self,
        df: pd.DataFrame,
        columns: List[str],
        options: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Process multiple text columns using vectorized operations for better performance.
        
        This method is faster for large DataFrames but may use more memory.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing text columns to process
        columns : List[str]
            List of column names to preprocess
        options : Optional[Dict[str, Any]], default=None
            Preprocessing options (see process_text for details)
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with preprocessed text columns
        
        Examples:
        ---------
        >>> preprocessor = TextPreprocessor()
        >>> df_clean = preprocessor.preprocess_dataframe_vectorized(
        ...     df,
        ...     columns=['title', 'description']
        ... )
        """
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: self.process_text(x, options))
        
        return df


class TextVisualizer:
    """
    A class for creating word clouds and analyzing text data.
    
    This class provides methods for visualizing text data through word clouds
    and frequency analysis. It's useful for exploratory data analysis of
    text corpora.
    
    Parameters:
    -----------
    style : Optional[str], default=None
        Matplotlib style to use for plots (e.g., 'seaborn', 'ggplot')
    
    Examples:
    ---------
    >>> # Initialize with default style
    >>> visualizer = TextVisualizer()
    
    >>> # Initialize with custom style
    >>> visualizer = TextVisualizer(style='seaborn-darkgrid')
    
    >>> # Create word clouds
    >>> fig = visualizer.create_wordclouds(df, text_columns=['title', 'content'])
    
    >>> # Analyze word frequencies
    >>> fig = visualizer.analyze_word_frequencies(df, text_columns=['title'])
    """
    
    def __init__(self, style: Optional[str] = None):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for TextVisualizer. "
                "Install with: pip install matplotlib"
            )
        if not WORDCLOUD_AVAILABLE:
            raise ImportError(
                "wordcloud is required for TextVisualizer. "
                "Install with: pip install wordcloud"
            )
        
        if style:
            plt.style.use(style)
        
        self._default_wordcloud_params = {
            'background_color': 'white',
            'max_words': 200,
            'width': 800,
            'height': 400,
            'contour_width': 3,
            'contour_color': 'steelblue'
        }
    
    def create_wordclouds(
        self,
        df: pd.DataFrame,
        text_columns: List[str],
        mask_path: Optional[str] = None,
        background_color: str = 'white',
        max_words: int = 200,
        width: int = 800,
        height: int = 400,
        colormap: str = 'viridis',
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create word clouds for multiple text columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the text columns
        text_columns : List[str]
            List of column names to visualize
        mask_path : Optional[str], default=None
            Path to an image file for shaping the word cloud
        background_color : str, default='white'
            Background color of the word clouds
        max_words : int, default=200
            Maximum number of words per cloud
        width : int, default=800
            Width of each word cloud in pixels
        height : int, default=400
            Height of each word cloud in pixels
        colormap : str, default='viridis'
            Matplotlib colormap for word colors
        figsize : Optional[Tuple[int, int]], default=None
            Figure size (width, height) in inches. Auto-calculated if None.
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure containing all word clouds
        
        Examples:
        ---------
        >>> visualizer = TextVisualizer()
        >>> fig = visualizer.create_wordclouds(
        ...     df,
        ...     text_columns=['title', 'description'],
        ...     max_words=100,
        ...     colormap='plasma'
        ... )
        >>> fig.savefig('wordclouds.png', dpi=300, bbox_inches='tight')
        """
        n_clouds = len(text_columns)
        
        # Auto-calculate figure size if not provided
        if figsize is None:
            figsize = (15, 5 * n_clouds)
        
        fig, axes = plt.subplots(n_clouds, 1, figsize=figsize)
        
        # Handle single column case
        if n_clouds == 1:
            axes = [axes]
        
        # Load mask if provided
        mask = None
        if mask_path:
            mask = np.array(Image.open(mask_path))
        
        # Create word cloud for each column
        for idx, column in enumerate(text_columns):
            # Combine all text, handling NaN values
            text = ' '.join(df[column].dropna().astype(str))
            
            if not text.strip():
                # Skip empty columns
                axes[idx].text(0.5, 0.5, f'No text in {column}',
                              ha='center', va='center', fontsize=14)
                axes[idx].set_title(f'Word Cloud - {column}', fontsize=16)
                axes[idx].axis('off')
                continue
            
            # Create word cloud
            wc = WordCloud(
                background_color=background_color,
                max_words=max_words,
                width=width,
                height=height,
                mask=mask,
                contour_width=3,
                contour_color='steelblue',
                colormap=colormap
            )
            wc.generate(text)
            
            # Display
            axes[idx].imshow(wc, interpolation='bilinear')
            axes[idx].axis('off')
            axes[idx].set_title(f'Word Cloud - {column}', fontsize=16, pad=20)
        
        plt.tight_layout(pad=3.0)
        return fig
    
    def create_combined_wordcloud(
        self,
        df: pd.DataFrame,
        text_columns: List[str],
        **kwargs
    ) -> plt.Figure:
        """
        Create a single word cloud from combined text columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the text columns
        text_columns : List[str]
            List of column names to combine and visualize
        **kwargs
            Additional arguments passed to WordCloud
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure with the word cloud
        
        Examples:
        ---------
        >>> visualizer = TextVisualizer()
        >>> fig = visualizer.create_combined_wordcloud(
        ...     df,
        ...     text_columns=['title', 'description', 'content'],
        ...     max_words=300,
        ...     colormap='coolwarm'
        ... )
        """
        # Combine text from all columns
        combined_text = ' '.join(
            df[text_columns].fillna('').astype(str).values.flatten()
        )
        
        if not combined_text.strip():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No text available',
                   ha='center', va='center', fontsize=14)
            ax.set_title('Word Cloud - Combined Text', fontsize=16)
            ax.axis('off')
            return fig
        
        return self.create_single_wordcloud(combined_text, **kwargs)
    
    def create_single_wordcloud(
        self,
        text: str,
        title: Optional[str] = 'Word Cloud',
        **kwargs
    ) -> plt.Figure:
        """
        Create a single word cloud with custom settings.
        
        Parameters:
        -----------
        text : str
            Text to generate word cloud from
        title : Optional[str], default='Word Cloud'
            Title for the plot
        **kwargs
            Additional WordCloud parameters:
            - background_color: str, default='white'
            - max_words: int, default=200
            - width: int, default=800
            - height: int, default=400
            - colormap: str, default='viridis'
            - mask: numpy array for custom shape
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure with the word cloud
        
        Examples:
        ---------
        >>> visualizer = TextVisualizer()
        >>> text = "Python data science machine learning visualization"
        >>> fig = visualizer.create_single_wordcloud(
        ...     text,
        ...     title='My Word Cloud',
        ...     colormap='plasma',
        ...     max_words=50
        ... )
        """
        # Merge default parameters with custom ones
        params = self._default_wordcloud_params.copy()
        params.update(kwargs)
        
        wc = WordCloud(**params)
        
        if text.strip():
            wc.generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, pad=20)
        
        plt.tight_layout()
        return fig
    
    def _calculate_word_frequencies(
        self, 
        df: pd.DataFrame, 
        column: str, 
        top_n: int = 20
    ) -> List[Tuple[str, int]]:
        """
        Calculate word frequencies for a text column.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the text column
        column : str
            Column name to analyze
        top_n : int, default=20
            Number of top frequent words to return
        
        Returns:
        --------
        List[Tuple[str, int]]
            List of (word, frequency) tuples
        """
        text = ' '.join(df[column].dropna().astype(str))
        words = text.split()
        return Counter(words).most_common(top_n)
    
    def _plot_frequency_bars(
        self,
        word_freq: List[Tuple[str, int]],
        ax: plt.Axes,
        title: str,
        show_values: bool = True
    ) -> None:
        """
        Create a horizontal bar plot of word frequencies.
        
        Parameters:
        -----------
        word_freq : List[Tuple[str, int]]
            List of (word, frequency) pairs
        ax : plt.Axes
            Matplotlib axes to plot on
        title : str
            Plot title
        show_values : bool, default=True
            Whether to show frequency values on bars
        """
        if not word_freq:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            ax.axis('off')
            return
        
        words, freqs = zip(*word_freq)
        y_pos = np.arange(len(words))
        
        # Create horizontal bar plot
        bars = ax.barh(y_pos, freqs, color='steelblue', alpha=0.7)
        
        # Customize the plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_title(title, pad=20, fontsize=12)
        ax.set_xlabel('Frequency')
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add value labels on bars
        if show_values:
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2,
                       f'{int(width)}',
                       ha='left', va='center', fontsize=9)
    
    def analyze_word_frequencies(
        self,
        df: pd.DataFrame,
        text_columns: List[str],
        top_n: int = 20,
        show_percentages: bool = True,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Analyze and visualize word frequencies for text columns.
        
        Creates frequency bar charts showing the most common words in each
        column, optionally including percentage distributions.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the text columns
        text_columns : List[str]
            List of column names to analyze
        top_n : int, default=20
            Number of top frequent words to display
        show_percentages : bool, default=True
            Whether to show percentage distributions alongside counts
        figsize : Optional[Tuple[int, int]], default=None
            Figure size (width, height) in inches. Auto-calculated if None.
        save_path : Optional[str], default=None
            Path to save the figure (e.g., 'frequencies.png')
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure containing frequency visualizations
        
        Examples:
        ---------
        >>> visualizer = TextVisualizer()
        >>> fig = visualizer.analyze_word_frequencies(
        ...     df,
        ...     text_columns=['title', 'description'],
        ...     top_n=15,
        ...     save_path='word_frequencies.png'
        ... )
        """
        n_cols = len(text_columns)
        n_plots = n_cols * (2 if show_percentages else 1)
        
        # Auto-calculate figure size
        if figsize is None:
            height = max(8, 4 * n_cols)
            width = 20 if show_percentages else 12
            figsize = (width, height)
        
        fig, axes = plt.subplots(n_cols, 2 if show_percentages else 1, 
                                  figsize=figsize, squeeze=False)
        
        for idx, column in enumerate(text_columns):
            # Calculate frequencies
            word_freq = self._calculate_word_frequencies(df, column, top_n)
            
            # Plot frequency counts
            self._plot_frequency_bars(
                word_freq, 
                axes[idx][0], 
                f'Word Frequencies - {column}'
            )
            
            # Plot percentages if requested
            if show_percentages:
                total_words = sum(freq for _, freq in word_freq) if word_freq else 1
                word_freq_pct = [(word, (freq/total_words)*100) for word, freq in word_freq]
                self._plot_frequency_bars(
                    word_freq_pct,
                    axes[idx][1],
                    f'Word Frequencies (%) - {column}'
                )
                axes[idx][1].set_xlabel('Percentage (%)')
        
        plt.tight_layout(pad=3.0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_word_frequency_table(
        self,
        df: pd.DataFrame,
        text_columns: List[str],
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get word frequency data as a DataFrame for further analysis.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the text columns
        text_columns : List[str]
            List of column names to analyze
        top_n : int, default=20
            Number of top frequent words to include
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: 'column', 'word', 'frequency', 'percentage'
        
        Examples:
        ---------
        >>> visualizer = TextVisualizer()
        >>> freq_df = visualizer.get_word_frequency_table(
        ...     df,
        ...     text_columns=['title', 'description'],
        ...     top_n=50
        ... )
        >>> print(freq_df.head())
        """
        results = []
        
        for column in text_columns:
            word_freq = self._calculate_word_frequencies(df, column, top_n)
            total = sum(freq for _, freq in word_freq) if word_freq else 1
            
            for word, freq in word_freq:
                results.append({
                    'column': column,
                    'word': word,
                    'frequency': freq,
                    'percentage': round((freq / total) * 100, 2)
                })
        
        return pd.DataFrame(results)


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║                        Convenience Functions                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝

def quick_clean(
    text: Union[str, pd.Series],
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_html: bool = True
) -> Union[str, pd.Series]:
    """
    Quickly clean text with common preprocessing steps.
    
    This is a convenience function for quick text cleaning without
    initializing the full TextPreprocessor class.
    
    Parameters:
    -----------
    text : Union[str, pd.Series]
        Input text or pandas Series to clean
    lowercase : bool, default=True
        Convert text to lowercase
    remove_urls : bool, default=True
        Remove URLs from text
    remove_emails : bool, default=True
        Remove email addresses
    remove_html : bool, default=True
        Remove HTML tags
    
    Returns:
    --------
    Union[str, pd.Series]
        Cleaned text (same type as input)
    
    Examples:
    ---------
    >>> # Clean single string
    >>> clean_text = quick_clean("Hello! Visit http://example.com")
    
    >>> # Clean pandas Series
    >>> df['clean_text'] = quick_clean(df['text'])
    """
    preprocessor = TextPreprocessor()
    
    if isinstance(text, pd.Series):
        return text.apply(lambda x: preprocessor.process_text(
            x,
            options={
                'lowercase': lowercase,
                'remove_urls': remove_urls,
                'remove_emails': remove_emails,
                'remove_html': remove_html,
                'remove_stopwords': False,
                'lemmatize': False
            }
        ))
    else:
        return preprocessor.process_text(
            text,
            options={
                'lowercase': lowercase,
                'remove_urls': remove_urls,
                'remove_emails': remove_emails,
                'remove_html': remove_html,
                'remove_stopwords': False,
                'lemmatize': False
            }
        )


def extract_ngrams(
    text: Union[str, List[str]],
    n: int = 2,
    preprocessor: Optional[TextPreprocessor] = None
) -> List[Tuple[str, ...]]:
    """
    Extract n-grams from text.
    
    Parameters:
    -----------
    text : Union[str, List[str]]
        Input text string or list of tokens
    n : int, default=2
        Size of n-grams (2 for bigrams, 3 for trigrams, etc.)
    preprocessor : Optional[TextPreprocessor], default=None
        TextPreprocessor instance for text cleaning. If None, uses default.
    
    Returns:
    --------
    List[Tuple[str, ...]]
        List of n-gram tuples
    
    Examples:
    ---------
    >>> # Extract bigrams
    >>> bigrams = extract_ngrams("machine learning is great", n=2)
    [('machine', 'learning'), ('learning', 'is'), ('is', 'great')]
    
    >>> # Extract trigrams with preprocessing
    >>> preprocessor = TextPreprocessor()
    >>> trigrams = extract_ngrams("Machine learning algorithms", n=3, preprocessor=preprocessor)
    """
    if preprocessor is None:
        preprocessor = TextPreprocessor()
    
    if isinstance(text, str):
        text = preprocessor.process_text(text)
        tokens = text.split()
    else:
        tokens = text
    
    if len(tokens) < n:
        return []
    
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def get_text_stats(df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
    """
    Calculate basic text statistics for DataFrame columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing text columns
    text_columns : List[str]
        List of column names to analyze
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with text statistics per column:
        - total_documents: Number of non-null documents
        - total_words: Total word count
        - avg_words_per_doc: Average words per document
        - unique_words: Number of unique words
        - avg_word_length: Average word length
    
    Examples:
    ---------
    >>> stats = get_text_stats(df, ['title', 'description'])
    >>> print(stats)
    """
    stats = []
    
    for col in text_columns:
        if col not in df.columns:
            continue
        
        # Get non-null texts
        texts = df[col].dropna().astype(str)
        total_docs = len(texts)
        
        if total_docs == 0:
            stats.append({
                'column': col,
                'total_documents': 0,
                'total_words': 0,
                'avg_words_per_doc': 0,
                'unique_words': 0,
                'avg_word_length': 0
            })
            continue
        
        # Calculate statistics
        word_counts = texts.str.split().str.len()
        all_words = ' '.join(texts).split()
        unique_words = len(set(all_words))
        total_words = len(all_words)
        avg_word_length = np.mean([len(word) for word in all_words]) if all_words else 0
        
        stats.append({
            'column': col,
            'total_documents': total_docs,
            'total_words': total_words,
            'avg_words_per_doc': round(word_counts.mean(), 2),
            'unique_words': unique_words,
            'avg_word_length': round(avg_word_length, 2)
        })
    
    return pd.DataFrame(stats)
