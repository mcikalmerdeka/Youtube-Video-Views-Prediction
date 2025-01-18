# Function to convert output of dataframe into markdown format for copy-pasting
def df_to_markdown(df, max_rows=None, max_cols=None, datetime_format="%Y-%m-%d %H:%M:%S", use_custom=False):
    """
    Convert a pandas DataFrame to a markdown-formatted table string with proper datetime handling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to convert
    max_rows : int, optional
        Maximum number of rows to include
    max_cols : int, optional
        Maximum number of columns to include
    datetime_format : str, optional
        Format string for datetime columns
    use_custom : bool, optional
        If True, uses custom markdown formatting instead of pandas to_markdown
        
    Returns:
    --------
    str
        Markdown-formatted table string
    """
    try:
        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()
        
        # Limit rows if specified
        if max_rows is not None:
            df_copy = df_copy.head(max_rows)
        
        # Limit columns if specified
        if max_cols is not None:
            df_copy = df_copy.iloc[:, :max_cols]
        
        # Format datetime columns
        for col in df_copy.select_dtypes(include=['datetime64']).columns:
            df_copy[col] = df_copy[col].dt.strftime(datetime_format)
        
        if not use_custom:
            try:
                # Try using pandas to_markdown (requires tabulate package)
                return df_copy.to_markdown()
            except ImportError:
                print("Warning: tabulate package not found. Falling back to custom formatting.")
                use_custom = True
        
        if use_custom:
            # Custom markdown formatting logic
            df_str = df_copy.to_string()
            lines = df_str.split('\n')
            headers = lines[0].split()
            
            markdown_lines = []
            markdown_lines.append('| ' + ' | '.join(headers) + ' |')
            markdown_lines.append('|' + '|'.join(['---' for _ in headers]) + '|')
            
            for line in lines[1:]:
                values = line.split()
                markdown_lines.append('| ' + ' | '.join(values) + ' |')
            
            return '\n'.join(markdown_lines)
            
    except Exception as e:
        print(f"Error converting DataFrame to markdown: {str(e)}")
        return str(df)