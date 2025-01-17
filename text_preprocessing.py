import pandas as pd
import nltk
import re
import unicodedata
import contractions
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from textblob import TextBlob

# Download required NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

# Initialize class for text preprocessing
class TextPreprocessor:
    def __init__(self, language='english'):
        self.language = language
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words(language))
        
        # Add custom stop words if needed
        custom_stop_words = {'amp', 'rt', 'etc', 'eg', 'watch', 'follow', 'subscribe', 'click', 'link'}
        self.stop_words.update(custom_stop_words)

    # Function to replace actual newline characters and \n string literals with space 
    def replace_newline(self, text):
        """Replace newline characters with space, handling both escaped and actual newlines"""
        # First replace escaped newlines, then actual newlines
        text = text.replace('\\n', ' ').replace('\n', ' ')
        # Remove multiple consecutive spaces
        return re.sub(r'\s+', ' ', text)

    # Function to remove html tags
    def remove_html(self, text):
        """Remove HTML tags from text"""
        return BeautifulSoup(text, "html.parser").get_text()

    # Function to expand contractions
    def expand_contractions(self, text):
        """Expand contractions like don't to do not"""
        return contractions.fix(text)

    # Function to remove accented characters
    def remove_accented_chars(self, text):
        """Remove accented characters from text"""
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # Function to remove special characters and numbers
    def remove_special_chars(self, text, keep_punctuation=False):
        """
        Remove special characters and numbers
        
        Parameters:
        -----------
        text : str
            Input text to process
        keep_punctuation : bool
            If True, keeps basic punctuation marks
        """
        if keep_punctuation:
            # Keep letters, spaces, and basic punctuation
            text = re.sub(r'[^a-zA-Z\s.,!?-]', '', text)
        else:
            # Keep only letters and spaces
            text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        return ' '.join(text.split())

    # Function to correct spelling
    def correct_spelling(self, text):
        """Correct spelling using TextBlob"""
        return str(TextBlob(text).correct())

    # Function to remove urls
    def remove_urls(self, text):
        """
        Remove URLs from text, handling various URL formats including shortened URLs
        """
        # Enhanced URL pattern to catch more variants
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            r'|(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+(?:/\S*)?'
            r'|bit\.ly/\S+'
            r'|goo\.gl/\S+'
        )
        return url_pattern.sub('', text)

    # Function to remove emails
    def remove_emails(self, text):
        """Remove email addresses from text"""
        email_pattern = re.compile(r'\S+@\S+')
        return email_pattern.sub('', text)

    # Function to remove social media handles and references
    def remove_social_media_handles(self, text):
        """Remove social media handles and references"""
        # Remove @mentions and #hashtags
        text = re.sub(r'[@#]\w+', '', text)
        # Remove common social media platform references
        social_platforms = r'(?i)(facebook|twitter|instagram|youtube|google\s*plus|linkedin)'
        text = re.sub(social_platforms, '', text)
        return text

    # Function to remove common YouTube-specific phrases
    def remove_common_youtube_phrases(self, text):
        """Remove common YouTube-specific phrases"""
        phrases = [
            r'(?i)subscribe to',
            r'(?i)like and subscribe',
            r'(?i)watch more',
            r'(?i)click the bell',
            r'(?i)follow us on',
            r'(?i)check out',
            r'(?i)watch full episode',
            r'(?i)brought to you by',
            r'(?i)sponsored by',
            r'(?i)download\s+\w+',
            r'(?i)join us on',
            r'(?i)this episode was brought to you by'
        ]
        for phrase in phrases:
            text = re.sub(phrase, '', text)
        return text

    # Function to process text with specified options
    def process_text(self, text, options=None):
        """
        Process text with specified options
        
        Parameters:
        -----------
        text : str
            Input text to process
        options : dict
            Dictionary of preprocessing options (default: all True)
            {
                'replace_newline': bool,
                'remove_html': bool,
                'expand_contractions': bool,
                'remove_accented': bool,
                'spell_correction': bool,
                'remove_urls': bool,
                'remove_emails': bool,
                'remove_social': bool,
                'remove_youtube_phrases': bool,
                'keep_punctuation': bool,
                'lemmatize': bool,
                'stem': bool,
                'remove_stopwords': bool
            }
        """
        # Skip processing if text is missing
        if pd.isnull(text):
            return text

        # Default options
        default_options = {
            'replace_newline': True,
            'remove_html': True,
            'expand_contractions': True,
            'remove_accented': True,
            'spell_correction': False,  # Disabled by default as it's slow
            'remove_urls': True,
            'remove_emails': True,
            'remove_social': True,
            'remove_youtube_phrases': True,
            'keep_punctuation': False,
            'lemmatize': True,
            'stem': False,  # Don't use both stemming and lemmatization
            'remove_stopwords': True
        }
        
        # Set options to default if not provided
        if options is None:
            options = default_options
        else:
            options = {**default_options, **options}

        # Apply preprocessing steps
        if options['replace_newline']:
            text = self.replace_newline(text)

        if options['remove_html']:
            text = self.remove_html(text)
            
        if options['remove_urls']:
            text = self.remove_urls(text)
            
        if options['remove_emails']:
            text = self.remove_emails(text)
            
        if options['remove_social']:
            text = self.remove_social_media_handles(text)
            
        if options['remove_youtube_phrases']:
            text = self.remove_common_youtube_phrases(text)
            
        if options['expand_contractions']:
            text = self.expand_contractions(text)
            
        if options['remove_accented']:
            text = self.remove_accented_chars(text)
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = self.remove_special_chars(text, keep_punctuation=options['keep_punctuation'])
        
        if options['spell_correction']:
            text = self.correct_spelling(text)

        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if options['remove_stopwords']:
            tokens = [token for token in tokens if token not in self.stop_words]
            
        # Apply lemmatization or stemming
        if options['lemmatize']:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        elif options['stem']:
            tokens = [self.stemmer.stem(token) for token in tokens]
            
        return ' '.join(tokens)

    # Final function to preprocess a DataFrame
    def preprocess_dataframe(self, df, columns, options=None):
        """Process multiple columns in a DataFrame"""
        df_processed = df.copy()
        for col in columns:
            df_processed[col] = df_processed[col].apply(lambda x: self.process_text(x, options))
        return df_processed

# # ======================================== Class Usage Example =========================================================
# preprocessor = TextPreprocessor()

# # # Example options
# # options = {
# #     'spell_correction': False,  # Disable spell correction for speed
# #     'stem': False,             # Use lemmatization instead of stemming
# #     'lemmatize': True
# # }

# # Process DataFrame
# columns_to_preprocess = ['title', 'tags', 'description']
# preprocessed_text_df = preprocessor.preprocess_dataframe(data, columns_to_preprocess, options=None)