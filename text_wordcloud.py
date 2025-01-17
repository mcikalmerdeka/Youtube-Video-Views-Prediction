from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

class TextVisualizer:
    """
    A class for creating word clouds and analyzing word frequencies in text data.
    
    Parameters:
    -----------
    style : str, optional
        The matplotlib style to use for plots (default: None)
    """
    
    def __init__(self, style=None):
        if style:
            plt.style.use(style)
            
        self.default_wordcloud_params = {
            'background_color': 'white',
            'max_words': 200,
            'width': 800,
            'height': 400,
            'contour_width': 3,
            'contour_color': 'steelblue'
        }
    
    def create_wordclouds(self, df, text_columns, mask_path=None, background_color='white', 
                         max_words=200, width=800, height=400):
        """
        Create word clouds for individual columns and combined text.
        
        Parameters:
        -----------
        df : pandas DataFrame
            DataFrame containing the text columns
        text_columns : list
            List of column names to process
        mask_path : str, optional
            Path to mask image file (if you want to shape the word cloud)
        background_color : str, optional
            Background color of the word cloud
        max_words : int, optional
            Maximum number of words to include in each word cloud
        width : int, optional
            Width of the word cloud image
        height : int, optional
            Height of the word cloud image
        """
        # Set up the matplotlib figure
        n_clouds = len(text_columns) + 1  # +1 for combined cloud
        fig = plt.figure(figsize=(15, 4 * n_clouds))
        
        # Load mask if provided
        mask = None
        if mask_path:
            mask = np.array(Image.open(mask_path))
        
        # Create word cloud object
        wc = WordCloud(
            background_color=background_color,
            max_words=max_words,
            width=width,
            height=height,
            mask=mask,
            contour_width=3,
            contour_color='steelblue'
        )
        
        # Generate word clouds for individual columns
        for idx, column in enumerate(text_columns, 1):
            # Combine all text in the column, handling NaN values
            text = ' '.join(df[column].dropna().astype(str))
            
            # Generate word cloud
            wc.generate(text)
            
            # Add subplot
            plt.subplot(n_clouds, 1, idx)
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud - {column}', fontsize=16, pad=20)
        
        # Generate word cloud for combined text
        combined_text = ' '.join(df[text_columns].fillna('').astype(str).values.flatten())
        wc.generate(combined_text)
        
        # Add subplot for combined cloud
        plt.subplot(n_clouds, 1, n_clouds)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Combined Text', fontsize=16, pad=20)
        
        plt.tight_layout(pad=3.0)
        
        return fig
    
    def _plot_frequencies(self, words_freq, ax, title):
        """
        Helper method to create frequency plots.
        
        Parameters:
        -----------
        words_freq : list of tuples
            List of (word, frequency) pairs
        ax : matplotlib.axes.Axes
            The axes to plot on
        title : str
            Title for the plot
        """
        words, freqs = zip(*words_freq)
        y_pos = np.arange(len(words))
        
        # Create horizontal bar plot
        bars = ax.barh(y_pos, freqs)
        
        # Customize the plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_title(title, pad=20, fontsize=12)
        ax.set_xlabel('Frequency')
        
        # Add value labels on the bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{int(width)}',
                   ha='left', va='center', fontsize=10)
            
        # Adjust layout
        ax.grid(True, axis='x', alpha=0.3)
        
        return words_freq

    def analyze_word_frequencies(self, df, text_columns, top_n=20, save_plots=False):
        """
        Analyze and visualize word frequencies for each column and combined text.
        
        Parameters:
        -----------
        df : pandas DataFrame
            DataFrame containing the text columns
        text_columns : list
            List of column names to process
        top_n : int, optional
            Number of top frequent words to display
        save_plots : bool, optional
            Whether to save the plots to files
        """
        print("Word Frequency Analysis")
        print("=" * 50)
        
        # Calculate figure size based on number of columns
        total_plots = len(text_columns) + 1  # +1 for combined text
        fig, axes = plt.subplots(total_plots, 2, figsize=(20, 6 * total_plots))
        
        # Analyze individual columns
        for idx, column in enumerate(text_columns):
            # Combine all text in the column
            text = ' '.join(df[column].dropna().astype(str))
            words = text.split()
            
            # Calculate word frequencies
            word_freq = Counter(words).most_common(top_n)
            
            # Print frequencies
            print(f"\nTop {top_n} words in {column}:")
            print("-" * 30)
            for word, freq in word_freq:
                print(f"{word}: {freq}")
            
            # Create frequency plot
            self._plot_frequencies(word_freq, axes[idx][0], f'Word Frequencies - {column}')
            
            # Create percentage plot
            total_words = sum(freq for _, freq in word_freq)
            word_freq_pct = [(word, (freq/total_words)*100) for word, freq in word_freq]
            self._plot_frequencies(word_freq_pct, axes[idx][1], f'Word Frequencies (%) - {column}')
        
        # Analyze combined text
        combined_text = ' '.join(df[text_columns].fillna('').astype(str).values.flatten())
        combined_words = combined_text.split()
        combined_freq = Counter(combined_words).most_common(top_n)
        
        # Print combined frequencies
        print(f"\nTop {top_n} words in combined text:")
        print("-" * 30)
        for word, freq in combined_freq:
            print(f"{word}: {freq}")
        
        # Create combined frequency plots
        self._plot_frequencies(combined_freq, axes[-1][0], 'Word Frequencies - Combined Text')
        
        # Create combined percentage plot
        total_words = sum(freq for _, freq in combined_freq)
        combined_freq_pct = [(word, (freq/total_words)*100) for word, freq in combined_freq]
        self._plot_frequencies(combined_freq_pct, axes[-1][1], 'Word Frequencies (%) - Combined Text')
        
        # Adjust layout and display
        plt.tight_layout(pad=3.0)
        
        if save_plots:
            plt.savefig('word_frequencies.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_single_wordcloud(self, text, **kwargs):
        """
        Create a single word cloud with custom settings.
        
        Parameters:
        -----------
        text : str
            Text to generate word cloud from
        **kwargs : dict
            Additional parameters to pass to WordCloud
        """
        # Merge default parameters with custom ones
        params = {**self.default_wordcloud_params, **kwargs}
        
        wc = WordCloud(**params)
        wc.generate(text)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        
        return plt.gcf()

# # # ======================================== Class Usage Example =========================================================
# # Initialize the visualizer
# visualizer = TextVisualizer()

# # Select columns to analyze
# columns_to_analyze = ['title', 'tags', 'description']

# # Create combined columns word clouds
# fig_clouds = visualizer.create_wordclouds(
#     data,
#     columns_to_analyze,
#     background_color='white',
#     max_words=150
# )

# # Save the combined columns word clouds
# fig_clouds.savefig('wordclouds.png', dpi=300, bbox_inches='tight')
# plt.close()

# # Analyze word frequencies
# fig_freq = visualizer.analyze_word_frequencies(
#     data, 
#     columns_to_analyze, 
#     top_n=20
# )

# # Create individual word clouds
# for column in columns_to_analyze:
#     text = ' '.join(data[column].dropna().astype(str))
#     fig = visualizer.create_single_wordcloud(
#         text,
#         title=f'Word Cloud - {column}',
#         colormap='viridis'
#     )
#     fig.savefig(f'wordcloud_{column}.png', dpi=300, bbox_inches='tight')
#     plt.close()