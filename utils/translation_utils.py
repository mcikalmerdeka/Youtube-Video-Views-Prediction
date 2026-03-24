"""NLP utilities for language detection and translation of Indian languages."""

import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from deep_translator import GoogleTranslator
from langid.langid import LanguageIdentifier, model
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

# ── Default export directory ───────────────────────────────────────────────────
DEFAULT_DATA_DIR = Path("data")

# ── Constants ──────────────────────────────────────────────────────────────────

INDIAN_LANG_CODES = frozenset(
    ['hi', 'ml', 'ta', 'te', 'kn', 'bn', 'gu', 'pa', 'or', 'as', 'mr']
)

# Unicode ranges for Indian scripts — used as a fast pre-filter before ML detection
_INDIAN_SCRIPT_RE = re.compile(r'[\u0900-\u0DFF]')  # All major Indian Unicode blocks

# Fast English detection - if text matches this, skip expensive ML detection
# Matches: ASCII letters, numbers, common punctuation, spaces only
_ENGLISH_ONLY_RE = re.compile(r'^[\x00-\x7F]+$')

# Common English words - if text contains mostly these, likely English
_COMMON_ENGLISH_WORDS = frozenset([
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
    'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
    'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
    'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them',
    'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over',
    'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work',
    'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
    'give', 'day', 'most', 'us', 'is', 'was', 'are', 'were', 'been', 'has',
    'had', 'did', 'does', 'doing', 'done', 'video', 'watch', 'subscribe',
    'channel', 'youtube', 'music', 'song', 'official', 'ft', 'feat', 'live'
])

# ── Build a constrained langid identifier ─────────────────────────────────────
_identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
_identifier.set_languages(list(INDIAN_LANG_CODES) + ['en'])


# ── Core detection helpers ─────────────────────────────────────────────────────

def _has_indian_script(text: str) -> bool:
    """Fast Unicode-range pre-filter. No ML needed — O(n) regex scan."""
    return bool(_INDIAN_SCRIPT_RE.search(text))


def _is_obviously_english(text: str) -> bool:
    """
    Fast heuristics to detect obviously English text without ML.
    Returns True if text is very likely English (safe to skip ML detection).
    """
    if not isinstance(text, str) or not text.strip():
        return True  # Empty/missing = treat as English (no translation needed)
    
    text_lower = text.lower()
    words = text_lower.split()
    
    # If it only contains ASCII characters and spaces
    if _ENGLISH_ONLY_RE.match(text):
        # Further check: if it contains mostly common English words
        if len(words) == 0:
            return True
        
        # Count how many words are common English words
        english_word_count = sum(1 for word in words if word in _COMMON_ENGLISH_WORDS)
        
        # If >30% of words are common English words, likely English
        if english_word_count / len(words) > 0.3:
            return True
        
        # If text is short (< 5 words) and all ASCII, assume English
        if len(words) < 5:
            return True
    
    return False


def _is_indian_language(text: str, confidence_threshold: float = 0.7) -> bool:
    """
    Multi-stage detection:
      1. Fast ASCII/English check - skip ML for obviously English text
      2. Unicode script check - catch Indian characters instantly
      3. Constrained langid ML model - for romanised/transliterated text
    
    Returns True if the text is likely an Indian language.
    """
    if not isinstance(text, str) or not text.strip():
        return False

    # Stage 0: Fast English filter - skip ML for obviously English text
    if _is_obviously_english(text):
        return False

    # Stage 1: Unicode fast path — if Indian characters are present, done
    if _has_indian_script(text):
        return True

    # Stage 2: ML detection for romanised/transliterated text
    try:
        lang, confidence = _identifier.classify(text)
        return lang in INDIAN_LANG_CODES and confidence >= confidence_threshold
    except Exception:
        return False


def _detect_batch(texts: list[tuple[str, float]]) -> list[tuple[str, bool]]:
    """
    Process a batch of texts for language detection.
    Used for parallel processing.
    
    Parameters
    ----------
    texts : list[tuple[str, float]]
        List of (text, confidence_threshold) tuples
    
    Returns
    -------
    list[tuple[str, bool]]
        List of (text, is_indian) results
    """
    return [(text, _is_indian_language(text, threshold)) for text, threshold in texts]


# ── Row-level detection with deduplication ────────────────────────────────────

def _row_contains_indian(row: pd.Series, columns: list[str]) -> bool:
    """Return True if any of the specified columns contain Indian language text."""
    return any(
        _is_indian_language(row[col])
        for col in columns
        if col in row.index and pd.notnull(row[col])
    )


def detect_indian_language_rows(
    df: pd.DataFrame,
    columns: list[str],
    confidence_threshold: float = 0.7,
    use_parallel: bool = True,
    max_workers: Optional[int] = None,
    detection_batch_size: int = 1000,
    export: bool = False,
    data_dir: Path = DEFAULT_DATA_DIR,
    filename_prefix: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into Indian-language rows and non-Indian-language rows.
    
    Optimized version with:
    - Fast English pre-filtering (skips ML for obviously English text)
    - Parallel processing support
    - Progress tracking
    - Optional CSV export

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame
    columns : list[str]
        Columns to inspect for language detection
    confidence_threshold : float, optional
        Minimum langid confidence to classify as Indian (0–1), default 0.7
    use_parallel : bool, optional
        Use parallel processing for detection, default True
    max_workers : int, optional
        Number of parallel workers (None = use all CPUs)
    detection_batch_size : int, optional
        Batch size for parallel detection, default 1000
    export : bool, optional
        Export results to CSV files, default False
    data_dir : Path, optional
        Directory to save CSV files, default "data"
    filename_prefix : str, optional
        Prefix for CSV filenames (default: auto-generated with timestamp)

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (indian_df, non_indian_df) — both preserve the original index

    Examples
    --------
    >>> # Fast mode - one column at a time
    >>> indian_titles, other_titles = detect_indian_language_rows(
    ...     df,
    ...     columns=['title'],
    ...     use_parallel=True
    ... )
    
    >>> # All columns at once with export
    >>> indian_df, non_indian_df = detect_indian_language_rows(
    ...     df,
    ...     columns=['title', 'tags', 'description'],
    ...     use_parallel=True,
    ...     max_workers=4,
    ...     export=True,
    ...     filename_prefix="my_data"
    ... )
    
    >>> # Load exported data later
    >>> indian_df = pd.read_csv("data/my_data_indian.csv")
    >>> non_indian_df = pd.read_csv("data/my_data_non_indian.csv")
    """
    # Collect all unique non-null strings across all target columns
    print(f"Collecting unique texts from columns: {columns}...")
    all_texts: set[str] = set()
    for col in columns:
        if col in df.columns:
            all_texts.update(df[col].dropna().unique())
    
    total_unique = len(all_texts)
    print(f"Found {total_unique:,} unique texts to check")
    
    if total_unique == 0:
        return df, df.iloc[0:0]

    # Prepare data for detection
    text_list = list(all_texts)
    
    if use_parallel and total_unique > 100:
        # Parallel detection for large datasets
        print(f"Running parallel detection with {max_workers or 'auto'} workers...")
        
        # Split into batches
        batches = [
            [(text, confidence_threshold) for text in text_list[i:i + detection_batch_size]]
            for i in range(0, len(text_list), detection_batch_size)
        ]
        
        text_cache: dict[str, bool] = {}
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            futures = {executor.submit(_detect_batch, batch): i for i, batch in enumerate(batches)}
            
            # Collect results with progress bar
            with tqdm(total=len(batches), desc="Detection batches") as pbar:
                for future in as_completed(futures):
                    batch_idx = futures[future]
                    try:
                        results = future.result()
                        for text, is_indian in results:
                            text_cache[text] = is_indian
                    except Exception as e:
                        print(f"Batch {batch_idx} failed: {e}")
                    pbar.update(1)
    else:
        # Sequential detection for small datasets
        print("Running sequential detection...")
        text_cache: dict[str, bool] = {}
        for text in tqdm(text_list, desc="Detecting languages"):
            text_cache[text] = _is_indian_language(text, confidence_threshold)

    # Build mask using cached results
    def _row_is_indian(row: pd.Series) -> bool:
        return any(
            text_cache.get(row[col], False)
            for col in columns
            if col in row.index and pd.notnull(row[col])
        )

    print("Classifying rows...")
    tqdm.pandas(desc="Classifying rows")
    mask = df.apply(_row_is_indian, axis=1)

    # Use df.index explicitly to prevent index misalignment
    indian_df = df.loc[mask]
    non_indian_df = df.loc[~mask]
    
    print(f"Results: {len(indian_df):,} Indian-language rows, {len(non_indian_df):,} others")
    
    # Export to CSV if requested
    if export:
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename prefix with timestamp if not provided
        if filename_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = f"language_detection_{timestamp}"
        else:
            prefix = filename_prefix
        
        indian_path = data_dir / f"{prefix}_indian.csv"
        non_indian_path = data_dir / f"{prefix}_non_indian.csv"
        
        print(f"\n── Exporting to CSV ──")
        indian_df.to_csv(indian_path, index=False)
        non_indian_df.to_csv(non_indian_path, index=False)
        print(f"   Indian rows saved to: {indian_path}")
        print(f"   Non-Indian rows saved to: {non_indian_path}")
        print(f"\nTo load later:")
        print(f"   indian_df = pd.read_csv('{indian_path}')")
        print(f"   non_indian_df = pd.read_csv('{non_indian_path}')")

    return indian_df, non_indian_df


# ── Translation helpers ────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _translate_batch_with_retry(
    texts: list[str],
    src: str = "auto",
    dest: str = "en",
) -> list[str]:
    """Translate a list of strings with automatic retry on failure."""
    return GoogleTranslator(source=src, target=dest).translate_batch(texts)


def translate_series(
    series: pd.Series,
    src: str = "auto",
    dest: str = "en",
    batch_size: int = 50,
    inter_batch_delay: float = 0.5,
) -> pd.Series:
    """
    Translate a pandas Series efficiently.

    Parameters
    ----------
    series : pd.Series
        Series containing text to translate
    src : str, optional
        Source language code, default 'auto' (auto-detect)
    dest : str, optional
        Target language code, default 'en' (English)
    batch_size : int, optional
        Number of strings per API call, default 50
    inter_batch_delay : float, optional
        Seconds to wait between batches (rate-limit courtesy), default 0.5

    Returns
    -------
    pd.Series
        Series with translated text
    """
    result = series.copy()

    # Work only with non-null values
    non_null_mask = series.notna()
    unique_values = series[non_null_mask].unique().tolist()

    if not unique_values:
        return result

    # Translate only unique values
    translation_map: dict[str, str] = {}
    batches = [
        unique_values[i : i + batch_size]
        for i in range(0, len(unique_values), batch_size)
    ]

    for batch in tqdm(batches, desc=f"Translating ({src}→{dest})"):
        try:
            translated = _translate_batch_with_retry(batch, src=src, dest=dest)
            translation_map.update(dict(zip(batch, translated)))
        except Exception as e:
            print(f"[Warning] Batch failed after retries: {e}. Keeping originals.")
            translation_map.update({text: text for text in batch})  # fallback

        if inter_batch_delay > 0:
            time.sleep(inter_batch_delay)

    # Map translations back to the full Series
    result[non_null_mask] = series[non_null_mask].map(translation_map)
    return result


def translate_dataframe(
    df: pd.DataFrame,
    columns: list[str],
    src: str = "auto",
    dest: str = "en",
    batch_size: int = 50,
    inter_batch_delay: float = 0.5,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Translate selected columns of a DataFrame in-place or as a copy.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame
    columns : list[str]
        Column names to translate
    src : str, optional
        Source language code, default 'auto' (auto-detect)
    dest : str, optional
        Target language code, default 'en' (English)
    batch_size : int, optional
        Strings per API call, default 50
    inter_batch_delay : float, optional
        Seconds to wait between batches (rate-limit courtesy), default 0.5
    inplace : bool, optional
        Modify df directly if True, otherwise return a copy, default False

    Returns
    -------
    pd.DataFrame
        DataFrame with translated columns

    Examples
    --------
    >>> columns_to_translate = ["title", "tags", "description"]
    >>> indian_language_english_df = translate_dataframe(
    ...     indian_language_df,
    ...     columns=columns_to_translate,
    ...     src="auto",   # auto-detect Indian language
    ...     dest="en",
    ... )
    >>> final_df = pd.concat(
    ...     [indian_language_english_df, non_indian_language_df],
    ...     ignore_index=True
    ... )
    """
    out = df if inplace else df.copy()

    for col in columns:
        if col not in out.columns:
            print(f"[Warning] Column '{col}' not found, skipping.")
            continue
        print(f"\n── Column: '{col}' ──")
        out[col] = translate_series(
            out[col],
            src=src,
            dest=dest,
            batch_size=batch_size,
            inter_batch_delay=inter_batch_delay,
        )

    return out


def run_translation_pipeline(
    df: pd.DataFrame,
    columns: list[str],
    src: str = "auto",
    dest: str = "en",
    use_parallel: bool = True,
    max_workers: Optional[int] = None,
    export: bool = False,
    data_dir: Path = DEFAULT_DATA_DIR,
    filename_prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Full pipeline: detect and split Indian-language rows, translate them,
    then recombine into a single DataFrame.

    This is a thin orchestrator — each step stays independent and debuggable.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame
    columns : list[str]
        Columns to check for Indian language and translate
    src : str, optional
        Source language code, default 'auto' (auto-detect)
    dest : str, optional
        Target language code, default 'en' (English)
    use_parallel : bool, optional
        Use parallel processing for detection, default True
    max_workers : int, optional
        Number of parallel workers (None = use all CPUs)
    export : bool, optional
        Export results to CSV files, default False
    data_dir : Path, optional
        Directory to save CSV files, default "data"
    filename_prefix : str, optional
        Prefix for CSV filenames (default: auto-generated with timestamp)

    Returns
    -------
    pd.DataFrame
        Final DataFrame with all rows, Indian-language rows translated

    Examples
    --------
    >>> # Fast approach - one column at a time
    >>> final_df = run_translation_pipeline(df, columns=['title'])
    
    >>> # All columns with parallel processing and export
    >>> final_df = run_translation_pipeline(
    ...     df,
    ...     columns=['title', 'tags', 'description'],
    ...     use_parallel=True,
    ...     max_workers=4,
    ...     export=True,
    ...     filename_prefix="my_translated_data"
    ... )
    
    >>> # Load final translated data later
    >>> final_df = pd.read_csv("data/my_translated_data_final.csv")
    """
    print("── Step 1: Language Detection ──")
    indian_df, non_indian_df = detect_indian_language_rows(
        df, 
        columns,
        use_parallel=use_parallel,
        max_workers=max_workers,
        export=export,
        data_dir=data_dir,
        filename_prefix=filename_prefix,
    )

    if len(indian_df) == 0:
        print("No Indian-language rows found. Returning original DataFrame.")
        if export:
            data_dir.mkdir(parents=True, exist_ok=True)
            if filename_prefix is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                prefix = f"translation_{timestamp}"
            else:
                prefix = filename_prefix
            final_path = data_dir / f"{prefix}_final.csv"
            df.to_csv(final_path, index=False)
            print(f"   Original data saved to: {final_path}")
        return df

    print(f"\n── Step 2: Translation ──")
    translated_df = translate_dataframe(indian_df, columns=columns, src=src, dest=dest)

    print("\n── Step 3: Recombine ──")
    final_df = (
        pd.concat([translated_df, non_indian_df])
        .sort_index()  # restore original row order
        .reset_index(drop=True)
    )
    print(f"   Final shape: {final_df.shape}")
    
    # Export final dataframe if requested
    if export:
        data_dir.mkdir(parents=True, exist_ok=True)
        
        if filename_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = f"translation_{timestamp}"
        else:
            prefix = filename_prefix
        
        final_path = data_dir / f"{prefix}_final.csv"
        final_df.to_csv(final_path, index=False)
        
        print(f"\n── Export Complete ──")
        print(f"   Final translated data saved to: {final_path}")
        print(f"\nTo load this data later:")
        print(f"   import pandas as pd")
        print(f"   final_df = pd.read_csv('{final_path}')")
    
    return final_df


# ── Alternative: Skip detection and translate everything ───────────────────────

def translate_all_text(
    df: pd.DataFrame,
    columns: list[str],
    dest: str = "en",
    batch_size: int = 50,
    inter_batch_delay: float = 0.5,
    export: bool = False,
    data_dir: Path = DEFAULT_DATA_DIR,
    filename_prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Translate ALL text without language detection.
    
    Use this if:
    - You know most of your data is in Indian languages
    - Detection is taking too long
    - You want to translate everything as a batch
    
    Much faster than detection + translation, but will translate English text too.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame
    columns : list[str]
        Column names to translate
    dest : str, optional
        Target language code, default 'en' (English)
    batch_size : int, optional
        Strings per API call, default 50
    inter_batch_delay : float, optional
        Seconds to wait between batches, default 0.5
    export : bool, optional
        Export results to CSV files, default False
    data_dir : Path, optional
        Directory to save CSV files, default "data"
    filename_prefix : str, optional
        Prefix for CSV filenames (default: auto-generated with timestamp)

    Returns
    -------
    pd.DataFrame
        DataFrame with all specified columns translated

    Examples
    --------
    >>> # Fastest option - skip detection entirely
    >>> final_df = translate_all_text(
    ...     df, 
    ...     columns=['title', 'tags', 'description'],
    ...     export=True,
    ...     filename_prefix="all_translated"
    ... )
    
    >>> # Load later
    >>> final_df = pd.read_csv("data/all_translated_final.csv")
    """
    result = df.copy()
    
    for col in columns:
        if col not in result.columns:
            print(f"[Warning] Column '{col}' not found, skipping.")
            continue
        print(f"\n── Translating column: '{col}' ──")
        result[col] = translate_series(
            result[col],
            src="auto",
            dest=dest,
            batch_size=batch_size,
            inter_batch_delay=inter_batch_delay,
        )
    
    # Export if requested
    if export:
        data_dir.mkdir(parents=True, exist_ok=True)
        
        if filename_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = f"translation_{timestamp}"
        else:
            prefix = filename_prefix
        
        final_path = data_dir / f"{prefix}_final.csv"
        result.to_csv(final_path, index=False)
        
        print(f"\n── Export Complete ──")
        print(f"   Translated data saved to: {final_path}")
        print(f"\nTo load this data later:")
        print(f"   import pandas as pd")
        print(f"   final_df = pd.read_csv('{final_path}')")
    
    return result
