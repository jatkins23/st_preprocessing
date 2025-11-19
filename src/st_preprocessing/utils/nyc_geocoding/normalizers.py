"""
Street and borough name normalizers.

Provides implementations for cleaning and standardizing street names
and borough names according to NYC conventions.
"""

import re
from typing import Optional, List, Dict, Mapping
import pandas as pd

from .base import Normalizer
from .models import NormalizationResult

# Ordinal word to number mapping shared by normalizers
_ORDINAL_WORDS: Mapping[str, str] = {
    "FIRST": "1", "SECOND": "2", "THIRD": "3",
    "FOURTH": "4", "FIFTH": "5", "SIXTH": "6",
    "SEVENTH": "7", "EIGHTH": "8", "NINTH": "9",
    "TENTH": "10", "ELEVENTH": "11", "TWELFTH": "12",
    "THIRTEENTH": "13", "FOURTEENTH": "14",
    "FIFTEENTH": "15", "SIXTEENTH": "16",
    "SEVENTEENTH": "17", "EIGHTEENTH": "18",
    "NINETEENTH": "19", "TWENTIETH": "20",
}
_ORDINAL_TENS: Mapping[str, int] = {
    "TWENTY": 20,
    "THIRTY": 30,
    "FORTY": 40,
    "FIFTY": 50,
    "SIXTY": 60,
    "SEVENTY": 70,
    "EIGHTY": 80,
    "NINETY": 90,
}
_ORDINAL_ONES: Mapping[str, int] = {
    "FIRST": 1,
    "SECOND": 2,
    "THIRD": 3,
    "FOURTH": 4,
    "FIFTH": 5,
    "SIXTH": 6,
    "SEVENTH": 7,
    "EIGHTH": 8,
    "NINTH": 9,
}


def _convert_spelled_ordinals_text(text: str) -> str:
    """
    Convert spelled ordinal words (including compound forms like "Forty Second")
    into their numeric equivalents.
    """
    def replace_compound(match: re.Match) -> str:
        tens_word = match.group("tens")
        ones_word = match.group("ones")
        tens_val = _ORDINAL_TENS.get(tens_word.upper(), 0)
        ones_val = _ORDINAL_ONES.get(ones_word.upper(), 0) if ones_word else 0
        total = tens_val + ones_val
        return str(total) if total > 0 else match.group(0)

    # Handle compound ordinals like "Forty Second" or standalone tens like "Seventieth"
    text = re.sub(
        r"\b(?P<tens>twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
        r"(?:[\s-]+(?P<ones>first|second|third|fourth|fifth|sixth|seventh|eighth|ninth))?\b",
        replace_compound,
        text,
        flags=re.I,
    )

    for word, num in _ORDINAL_WORDS.items():
        text = re.sub(rf"\b{word}\b", num, text, flags=re.I)
    return text


# Street type mappings for normalized forms
_STREET_TYPE_CANON: Dict[str, str] = {
    "AVENUE": "AVE",
    "STREET": "ST",
    "PLACE": "PL",
    "BOULEVARD": "BLVD",
    "ROAD": "RD",
    "COURT": "CT",
    "PARKWAY": "PKWY",
    "TERRACE": "TER",
    "LANE": "LN",
    "DRIVE": "DR",
    "AV": "AVE",  # Abbreviation for avenue
}

# Regex patterns for ordinal streets/avenues/places
# Matches: "1 STREET", "1st STREET", "1 ST", "1st ST", etc.
_RE_ORD_ST = re.compile(r"\b(\d+)(?:st|nd|rd|th)?\s+(?:STREET|ST)\b", re.I)
_RE_ORD_AVE = re.compile(r"\b(\d+)(?:st|nd|rd|th)?\s+(?:AVENUE|AVE)\b", re.I)
_RE_ORD_PL = re.compile(r"\b(\d+)(?:st|nd|rd|th)?\s+(?:PLACE|PL)\b", re.I)

# Borough name aliases from NYC Geoclient
BOROUGH_ALIASES: Mapping[str, str] = {
    "mn": "Manhattan",
    "bx": "Bronx",
    "bk": "Brooklyn",
    "qn": "Queens",
    "qns": "Queens",
    "si": "Staten Island",
    "manhattan": "Manhattan",
    "bronx": "Bronx",
    "brooklyn": "Brooklyn",
    "queens": "Queens",
    "staten island": "Staten Island",
    "new york": "Manhattan",
    "nyc": "Manhattan",
    "kings": "Brooklyn",
    "kings county": "Brooklyn",
    "richmond": "Staten Island",
    "richmond county": "Staten Island",
}

# Street abbreviations that should be expanded
_STREET_ABBREVS: Dict[str, str] = {
    "FT": "FORT",
    "MT": "MOUNT",
    # NOTE: EAST/WEST NOT included here - these are street name parts (EAST BROADWAY, EAST SIDE)
    # not just directional abbreviations. Geoclient expects full "EAST"/"WEST" forms.
}

# Known street name variant mappings (variant → canonical)
_STREET_NAME_MAPPINGS: Dict[str, str] = {
    "PRESIDENT STS": "PRESIDENT ST",
    "TATE ST": "STATE ST",
    "HYLAND BLVD": "HYLAN BLVD",
    "ELLWOOD ST": "ELWOOD ST",
    "OLIVER ST": "OLIVE ST",
    "LAGUARDIA PL": "LA GUARDIA PL",
    "LAIGHT ST": "LIGHT ST",
    "W72ND ST": "W 72 ST",
    "W79TH ST": "W 79 ST",
}

VALID_BOROUGHS = {"Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"}


class StreetNormalizer(Normalizer):
    """
    Normalizes street names to a normalized form.
    
    Handles:
    - Ordinal suffixes (1st → 1, 2nd → 2, etc.)
    - Street type abbreviations (Avenue → AVE, Street → ST, etc.)
    - Whitespace normalization
    - Case standardization
    """
    
    def __init__(self, aggressive: bool = False):
        """
        Initialize street normalizer.
        
        Args:
            aggressive: If True, apply more aggressive transformations
        """
        self.aggressive = aggressive
    
    def normalize(self, value: str, context: Optional[str] = None) -> str:
        """
        Normalize a single street name.
        
        Args:
            value: Street name to normalize
            context: Unused for streets (kept for interface compatibility)
            
        Returns:
            Normalized street name
        """
        if not value:
            return ""
        
        t = str(value).strip()
        
        # Remove trailing/leading punctuation and periods (E. → E, ST. → ST)
        t = re.sub(r"[.,]+", "", t)

        # Convert spelled ordinals early so "First Avenue" aligns with "1st Avenue"
        t = _convert_spelled_ordinals_text(t)
        
        # Strip ordinal suffixes from bare numbers (e.g., "1st" -> "1")
        t = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1", t, flags=re.I)
        
        # Convert ordinal number names (1st, 2nd, 3rd, etc.)
        t = _RE_ORD_ST.sub(r"\1 ST", t)
        t = _RE_ORD_AVE.sub(r"\1 AVE", t)
        t = _RE_ORD_PL.sub(r"\1 PL", t)

        # Expand common abbreviations/directions before type normalization
        t = self._expand_abbreviations(t)
        
        # Canonicalize street types
        t = self._normalize_street_types(t)

        # Apply known variant mappings (e.g., TATE → STATE) for deduplication
        t = self._apply_street_mappings(t)

        # Normalize separators/hyphenation consistently
        t = re.sub(r"\s*/\s*", "/", t)
        t = re.sub(r"\s*-\s*", "-", t)
        
        # Normalize whitespace and case
        t = re.sub(r"\s+", " ", t).strip().upper()
        
        return t

    def _normalize_street_types(self, text: str) -> str:
        """Replace street type variations with normalized forms."""
        t = text
        for key, value in _STREET_TYPE_CANON.items():
            t = re.sub(rf"\b{key}\b", value, t, flags=re.I)
        return t

    def _expand_abbreviations(self, text: str) -> str:
        """Expand short abbreviations and directional prefixes used in NYC data."""
        t = text
        for abbrev, full in _STREET_ABBREVS.items():
            t = re.sub(rf"\b{abbrev}\b", full, t, flags=re.I)

        # Only shorten EAST/WEST/NORTH/SOUTH to compass directions when followed by numbers
        t = re.sub(r"\bEAST\s+(\d)", r"E \1", t, flags=re.I)
        t = re.sub(r"\bWEST\s+(\d)", r"W \1", t, flags=re.I)
        t = re.sub(r"\bNORTH\s+(\d)", r"N \1", t, flags=re.I)
        t = re.sub(r"\bSOUTH\s+(\d)", r"S \1", t, flags=re.I)
        return t

    def _apply_street_mappings(self, text: str) -> str:
        """Apply curated variant → canonical mappings for known tricky cases."""
        t = text
        for variant, canonical in _STREET_NAME_MAPPINGS.items():
            t = re.sub(rf"\b{re.escape(variant)}\b", canonical, t, flags=re.I)
        return t
    
    def normalize_batch(self, values: List[str]) -> List[str]:
        """Normalize multiple street names."""
        return [self.normalize(v) for v in values]


class BoroughNormalizer(Normalizer):
    """
    Normalizes borough names to normalized forms.
    
    Handles:
    - Aliases and abbreviations (mn → Manhattan, bk → Brooklyn, etc.)
    - Case insensitivity
    - Whitespace trimming
    """
    
    def normalize(self, value: str, context: Optional[str] = None) -> str:
        """
        Normalize a borough name.
        
        Args:
            value: Borough name to normalize
            context: Unused (kept for interface compatibility)
            
        Returns:
            Normalized borough name (one of VALID_BOROUGHS or empty string)
        """
        if not value:
            return ""
        
        s = str(value).strip().lower()
        
        # Look up in aliases
        if s in BOROUGH_ALIASES:
            return BOROUGH_ALIASES[s]
        
        # Try title case match
        titled = s.title()
        if titled in VALID_BOROUGHS:
            return titled
        
        # Special case: "all", "citywide", "multiple" → empty (all boroughs)
        if s in {"citywide", "multiple", "all"}:
            return ""
        
        return ""
    
    def normalize_batch(self, values: List[str]) -> List[str]:
        """Normalize multiple borough names."""
        return [self.normalize(v) for v in values]


class StandardTokenizer(Normalizer):
    """
    Advanced tokenizer for street names.
    
    Handles:
    - Punctuation removal
    - Ordinal conversion (First → 1, Second → 2, etc.)
    - Street type standardization
    - Word order normalization
    """
    
    _COMPASS_ABBREV: Mapping[str, str] = {
        "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W",
        "NORTHEAST": "NE", "NORTHWEST": "NW",
        "SOUTHEAST": "SE", "SOUTHWEST": "SW",
    }
    
    def normalize(self, value: str, context: Optional[str] = None) -> str:
        """
        Tokenize and standardize a street name.
        
        Args:
            value: Street name to tokenize
            context: Unused (kept for interface compatibility)
            
        Returns:
            Standardized token string
        """
        if not value:
            return ""
        
        t = str(value).upper()
        
        # Remove punctuation (except hyphens, slashes, ampersands)
        t = re.sub(r"[.,]", " ", t)
        t = re.sub(r"[^\w\s\-/&]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        
        # Convert spelled ordinals
        t = self._convert_spelled_ordinals(t)
        
        # Remove numeric ordinal suffixes (1st, 2nd, 3rd, etc.)
        t = re.sub(r"\b(\d+)(?:ST|ND|RD|TH)\b", r"\1", t, flags=re.I)
        
        # Expand street name abbreviations (FT → FORT, MT → MOUNT)
        # Do this BEFORE street type normalization
        t = self._expand_abbreviations(t)
        
        # Standardize street types (STREET → ST, AVENUE → AVE, etc.)
        # Do this BEFORE variant mappings so "TATE STREET" → "TATE ST" 
        t = self._normalize_street_types(t)
        
        # Apply street name variant mappings (TATE ST → STATE ST, etc.)
        # Do this AFTER street type normalization
        t = self._apply_street_mappings(t)
        
        # Normalize separators
        t = re.sub(r"\s*/\s*", "/", t)
        t = re.sub(r"\s*-\s*", "-", t)
        t = re.sub(r"\s+", " ", t).strip()
        
        return t
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand street name abbreviations (FT → FORT, MT → MOUNT, etc.)."""
        t = text
        for abbrev, full in _STREET_ABBREVS.items():
            t = re.sub(rf"\b{abbrev}\b", full, t, flags=re.I)
        
        # Smart direction normalization: Only convert EAST/WEST to E/W when followed by a number
        # This preserves legitimate street names like "EAST BROADWAY" but normalizes "EAST 90 STREET"
        t = re.sub(r"\bEAST\s+(\d)", r"E \1", t, flags=re.I)
        t = re.sub(r"\bWEST\s+(\d)", r"W \1", t, flags=re.I)
        
        return t
    
    def _apply_street_mappings(self, text: str) -> str:
        """Apply known street name variant mappings."""
        t = text
        for variant, canonical in _STREET_NAME_MAPPINGS.items():
            t = re.sub(rf"\b{re.escape(variant)}\b", canonical, t, flags=re.I)
        return t
    
    def _convert_spelled_ordinals(self, text: str) -> str:
        """Convert spelled ordinals (First → 1, etc.)."""
        return _convert_spelled_ordinals_text(text)
    
    def _normalize_street_types(self, text: str) -> str:
        """Replace street type variations with normalized forms."""
        t = text
        for key, value in _STREET_TYPE_CANON.items():
            t = re.sub(rf"\b{key}\b", value, t, flags=re.I)
        return t


class CompositeNormalizer(Normalizer):
    """
    Combines multiple normalizers in sequence.
    
    Useful for applying multiple transformations in a pipeline.
    """
    
    def __init__(self, normalizers: List[Normalizer]):
        """
        Initialize with a list of normalizers to apply in order.
        
        Args:
            normalizers: List of normalizers to apply sequentially
        """
        self.normalizers = normalizers
    
    def normalize(self, value: str, context: Optional[str] = None) -> str:
        """Apply all normalizers in sequence."""
        result = value
        for normalizer in self.normalizers:
            result = normalizer.normalize(result, context)
        return result
    
    def normalize_batch(self, values: List[str]) -> List[str]:
        """Apply all normalizers to batch."""
        result = values
        for normalizer in self.normalizers:
            result = normalizer.normalize_batch(result)
        return result


class DataPreparer:
    """
    Prepare raw CSV data for geocoding.
    
    Delegates to the full PrepareData normalizer which handles:
    - Street name cleaning and standardization
    - Optional API normalization
    - Pair ordering (shorter street first)
    - Deduplication by unique_key
    - Rejection of invalid rows
    """
    
    def prepare_csv(self, csv_path: str, limit: Optional[int] = None, dedupe: bool = True) -> pd.DataFrame:
        """
        Load and clean CSV data for geocoding using full normalization pipeline.
        
        Args:
            csv_path: Path to CSV file with columns: street1, street2, borough
            limit: Optional limit on number of rows to process
            dedupe: If True, drop duplicate rows by unique_key (default: True)
            
        Returns:
            Cleaned DataFrame ready for geocoding with normalized streets and unique_key
        """
        from pathlib import Path
        from normalizer_prepare import PrepareData, NormalizeConfig
        
        # Use the original PrepareData normalizer for full pipeline
        config = NormalizeConfig(
            street_cols=("street1", "street2"),
            borough_col="borough",
            use_api="never",  # Don't call API during prep, just normalize locally
        )
        
        preparer = PrepareData(config=config)
        kept, rejected = preparer.prepare_file(Path(csv_path), dedupe=dedupe)
        
        if limit:
            kept = kept.head(limit)
        
        print(f"[DataPreparer] Loaded {len(kept)} rows, rejected {len(rejected)} invalid rows")
        if len(rejected) > 0:
            print(f"[DataPreparer] {len(rejected)} rows had missing/empty streets")
        
        return kept
