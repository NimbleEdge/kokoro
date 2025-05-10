"""
Python implementation foor the on-device G2P tokenizer of Kokoro

This tokenizer converts text into phonetic representations (phonemes) for use in 
text-to-speech systems. The main steps in the process are:
1. Preprocessing - handle special cases like currencies, numbers, times
2. Tokenization - split text into meaningful tokens
3. Phonemization - convert tokens to phonetic representations 
4. Stress application - add proper stress markers to phonemes
"""
import re
import json
from phonemizer.backend import EspeakBackend

# Initialize espeak backend for English (US) phonemization
backend = EspeakBackend('en-us')
# Define stress markers and vowel phonemes for stress placement
STRESSES = 'ˌˈ'
PRIMARY_STRESS = STRESSES[1]
SECONDARY_STRESS = STRESSES[0]
VOWELS = ['A', 'I', 'O', 'Q', 'W', 'Y', 'a', 'i', 'u', 'æ', 'ɑ', 'ɒ', 'ɔ', 'ə', 'ɛ', 'ɜ', 'ɪ', 'ʊ', 'ʌ', 'ᵻ']

# Characters to ignore during tokenization and punctuation for special handling
SUBTOKEN_JUNKS = "',-._''/' "
PUNCTS = ['?', ',', ';', '"', '—', ':', '!', '.', '…', '"', '"']
NON_QUOTE_PUNCTS = ['?', ',', '—', '.', ':', '!', ';', '…']

# Optional lexicon for custom word-to-phoneme mappings
LEXICON = None

# Currency symbols and their word representations
CURRENCIES = {
    '$': ('dollar', 'cent'),
    '£': ('pound', 'pence'),
    '€': ('euro', 'cent'),
}
# Create regex character classes from currency symbols and punctuation
currency_symbols = r'[' + r''.join([re.escape(symbol) for symbol in CURRENCIES.keys()]) + r']'
punct_symbols = r'[' + ''.join([re.escape(p) for p in PUNCTS]) + ']'
LINK_REGEX = r"\[([^\]]+)\]\(([^\)]*)\)"

# Helper functions for list operations
def all(iterable):
    """
    Check if all elements in the iterable are True.
    
    Args:
        iterable: Collection of elements to check
        
    Returns:
        bool: True if all elements are True, False otherwise
    """
    for item in iterable:
        if not item:
            return False
    return True

def any(iterable):
    """
    Check if any element in the iterable is True.
    
    Args:
        iterable: Collection of elements to check
        
    Returns:
        bool: True if any element is True, False otherwise
    """
    for item in iterable:
        if item:
            return True
    return False

# Text manipulation functions
def replace(text, original, replacement):
    """
    Replace all occurrences of original with replacement in text.
    Works from right to left to avoid index issues.
    
    Args:
        text (str): The text to modify
        original (str): The substring to replace
        replacement (str): The replacement string
        
    Returns:
        str: Text with all occurrences of original replaced with replacement
    """
    matches = [m for m in re.finditer(re.escape(original), text)]
    for match in matches[::-1]:
        text = text[:match.start()]+replacement+text[match.end():]
    return text

def split(text, delimiter, is_regex):
    """
    Split text by delimiter, optionally treating delimiter as regex.
    
    Args:
        text (str): The text to split
        delimiter (str): The delimiter to split by
        is_regex (bool): Whether to treat delimiter as regex pattern
        
    Returns:
        list: List of substrings split by delimiter
    """
    delimiter_pattern = delimiter
    if not is_regex:
        delimiter_pattern = re.escape(delimiter)
    # Get all delimiter positions
    curIdx = 0
    result = []
    while curIdx < len(text):
        delimiter_match = re.search(delimiter_pattern, text[curIdx:])
        if delimiter_match is None:
            break
        delimiter_position = delimiter_match.start()
        if delimiter_position == 0:
            curIdx = curIdx + delimiter_position + len(delimiter)
            continue
        result.append(text[curIdx:curIdx+delimiter_position])
        curIdx = curIdx + delimiter_position + len(delimiter)
    return result + [text[curIdx:]]

def split_with_delimiters_seperate(text, delimiter, is_regex):
    """
    Split text by delimiter, keeping delimiters as separate items.
    
    Args:
        text (str): The text to split
        delimiter (str): The delimiter to split by
        is_regex (bool): Whether to treat delimiter as regex pattern
        
    Returns:
        list: List of substrings with delimiters as separate items
    """
    delimiter_pattern = delimiter
    if not is_regex:
        delimiter_pattern = re.escape(delimiter)
    # Get all delimiter positions
    curIdx = 0
    result = []
    while curIdx < len(text):
        delimiter_match = re.search(delimiter_pattern, text[curIdx:])
        if delimiter_match is None:
            break
        delimiter_position = delimiter_match.start()
        if delimiter_position == 0:
            result.append(delimiter)
            curIdx = curIdx + delimiter_position + len(delimiter)
            continue
        result.append(text[curIdx:curIdx + delimiter_position])
        result.append(delimiter)
        curIdx = curIdx + delimiter_position + len(delimiter)
    if (curIdx < len(text)):
        result.append(text[curIdx:])
    return result

def isspace(input):
    """
    Check if the input string contains only whitespace characters.
    
    Args:
        input (str): String to check
        
    Returns:
        bool: True if string contains only whitespace, False otherwise
    """
    return all([c in [' ', '\t', '\n', '\r'] for c in input])

# Token class to represent processed text with phonetic and stress information
class Token:
    """
    Object representing a token with phonetic and stress information.
    
    Attributes:
        text (str): The original text
        whitespace (str): Whitespace following this token
        phonemes (str): Phonetic representation
        stress (int/float/None): Stress level indicator
        currency (str/None): Currency information if applicable
        prespace (bool): Whether token should be preceded by space
        alias (str/None): Alternative representation if specified
        is_head (bool): Whether this is the first token in a word
    """
    def __init__(self, text, whitespace, phonemes, stress, currency, prespace, alias, is_head):
        self.text = text
        self.whitespace = whitespace
        self.phonemes = phonemes
        self.stress = stress
        self.currency = currency
        self.prespace = prespace
        self.alias = alias
        self.is_head = is_head

def merge_tokens(tokens, unk):
    """
    Merge multiple tokens into a single token while preserving phonemes and stress.
    Handles proper spacing between phonemes when merging.
    
    Args:
        tokens (list): List of Token objects to merge
        unk (str): Unknown token placeholder (not used in current implementation)
        
    Returns:
        Token: A single merged Token object
    """
    stress = [t.stress for t in tokens if t.stress is not None]
    phonemes = ""
    for t in tokens:
        if t.prespace and (not phonemes == "") and not isspace(phonemes[-1]) and (not t.phonemes == ""):
            phonemes = phonemes + ' '
        if t.phonemes is None or t.phonemes == "":
            phonemes = phonemes
        else:
            phonemes = phonemes + t.phonemes
    if isspace(phonemes[0]):
        phonemes = phonemes[1:]
    stress_token = None
    if len(stress) == 1:
        stress_token = stress[0]
    return Token(
        ''.join([t.text + t.whitespace for t in tokens[:-1]]) + tokens[-1].text,
        tokens[-1].whitespace,
        phonemes,
        stress_token,
        None,
        tokens[0].prespace,
        None,
        tokens[0].is_head,
    )

def apply_stress(ps, stress):
    """
    Apply stress to phonemes.
    
    Args:
        ps (str): The phoneme string
        stress (int/float/None): Stress level indicator:
            - None: Keep stress as is
            - < -1: Remove all stress
            - -1: Convert primary stress to secondary
            - 0, 0.5: Convert primary to secondary if exists, else add secondary
            - 1: Convert secondary to primary if exists, else no change
            - > 1: Add primary stress if no stress exists
                
    Returns:
        str: Phonemes with appropriate stress markers applied
    """
    def restress(ps):
        ips = [(i, p) for i, p in enumerate(ps)]
        stresses = {i: next(j for j, v in ips[i:] if v in VOWELS) for i, p in ips if p in STRESSES}
        for i, j in stresses.items():
            _, s = ips[i]
            ips[i] = (j - 0.5, s)
        # ps = ''.join([p for _, p in sorted(ips)])
        ps = ''.join([p for _, p in ips])

        return ps
    if stress is None:
        return ps
    elif stress < -1:
        return replace(replace(ps, PRIMARY_STRESS, ''), SECONDARY_STRESS, '')
    elif stress == -1 or (stress in (0, -0.5) and PRIMARY_STRESS in ps):
        return replace(replace(ps, SECONDARY_STRESS, ''), PRIMARY_STRESS, SECONDARY_STRESS)
    elif stress in (0, 0.5, 1) and all(s not in ps for s in STRESSES):
        if all(v not in ps for v in VOWELS):
            return ps
        return restress(SECONDARY_STRESS + ps)
    elif stress >= 1 and PRIMARY_STRESS not in ps and SECONDARY_STRESS in ps:
        return replace(ps, SECONDARY_STRESS, PRIMARY_STRESS)
    elif stress > 1 and all(s not in ps for s in STRESSES):
        if all(v not in ps for v in VOWELS):
            return ps
        return restress(PRIMARY_STRESS + ps)
    return ps

def stress_weight(ps):
    """
    Calculate the phonetic weight for stress purposes.
    
    Args:
        ps (str): Phoneme string
        
    Returns:
        int: Numeric weight based on phonemes (higher for certain vowels/phonemes)
    """
    sum = 0
    if not ps:
        return 0
    for c in ps:
        if c in 'AIOQWYʤʧ':
            sum = sum + 2
        else:
            sum = sum + 1
    return sum

def is_function_word(word):
    """
    Check if a word is a function word (articles, prepositions, conjunctions, etc.)
    
    Args:
        word (str): Word to check
        
    Returns:
        bool: True if the word is a function word, False otherwise
    """
    function_words = [
        'a', 'an', 'the', 'in', 'on', 'at', 'of', 'for', 'with', 'by', 'to', 'from', 
        'and', 'or', 'but', 'nor', 'so', 'yet', 'is', 'am', 'are', 'was', 'were', 
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
        'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could', 'that',
        'this', 'these', 'those', 'he', 'she', 'it', 'they', 'we', 'you', 'i', 'me',
        'him', 'her', 'them', 'us', 'my', 'your', 'his', 'their', 'our', 'its'
    ]
    word = word.lower()
    if word[-1] in PUNCTS:
        word = word[:-1]
    return word in function_words

def isalpha_regex(text):
    """
    Check if string contains only alphabetic characters.
    
    Args:
        text (str): String to check
        
    Returns:
        bool: True if string contains only alphabetic characters, False otherwise
    """
    if not text:  # Handle empty string
        return False
    return bool(re.match(r'^[a-zA-Z]+$', text))

def is_content_word(word):
    """
    Check if a word is a content word (nouns, verbs, adjectives, adverbs).
    
    Args:
        word (str): Word to check
        
    Returns:
        bool: True if the word is a content word, False otherwise
    """
    return not is_function_word(word) and len(word) > 2 and isalpha_regex(word)

def resolve_tokens(tokens):
    """
    Apply stress and formatting to match G2P output format.
    G2P places primary stress markers directly before vowels, not at the beginning of words.
    This ensures phonemes are properly formatted with appropriate stress placement.
    
    Args:
        tokens (list): List of Token objects to resolve
        
    Returns:
        str: Final phoneme string with proper stress placement and formatting
    """
    # G2P phoneme mapping corrections
    phoneme_corrections = {
        # Convert common phonemes to match G2P's format
        'eɪ': 'A',
        'ɹeɪndʒ': 'ɹAnʤ',
        'wɪðɪn': 'wəðɪn'
    }
    
    # Map specific words to their G2P phoneme representations
    word_phoneme_map = {
        'a': 'ɐ',
        'an': 'ən'
    }
    
    # Define sentence-ending punctuation
    sentence_ending_punct = ['.', '!', '?']

    def add_stress_before_vowel(phoneme, stress_marker):
        """Add stress marker directly before the first vowel in the phoneme string"""
        phoneme_chars = [c for c in phoneme]
        for i, c in enumerate(phoneme_chars):
            if c in VOWELS:
                if i == 0:
                    return stress_marker + phoneme[i:]
                else:
                    return phoneme[:i] + stress_marker + phoneme[i:]
        return phoneme  # No vowels found
    
    # First, convert special phonemes and apply stress appropriately
    for i, token in enumerate(tokens):
        if not token.phonemes:
            continue
            
        # Apply special word mapping if needed
        if token.text.lower() in word_phoneme_map:
            token.phonemes = word_phoneme_map[token.text.lower()]
            continue
            
        # Apply phoneme corrections
        for old in phoneme_corrections.keys():
            if old in token.phonemes:
                token.phonemes = replace(token.phonemes, old, phoneme_corrections[old])
        
        # Check for existing stress markers
        has_stress = PRIMARY_STRESS in token.phonemes or SECONDARY_STRESS in token.phonemes
        
        # For multi-word phonemes like "wʌnhʌndɹɪd twɛnti θɹi dɑlɚz ænd fɔɹɾi faɪv sɛnts"
        # we need to break them up and apply stress to each word
        if " " in token.phonemes and not has_stress:
            subwords = split(token.phonemes, ' ', False)
            stressed_subwords = []
            
            for subword in subwords:
                # Skip empty strings or already stressed words
                if not subword or PRIMARY_STRESS in subword or SECONDARY_STRESS in subword:
                    stressed_subwords.append(subword)
                    continue
                has_vowels = False
                for v in VOWELS:
                    if v in subword:
                        has_vowels = True
                        break
                if not has_vowels:
                    stressed_subwords.append(subword)
                    continue
                
                # Apply appropriate stress directly before the vowel
                if subword in ['ænd', 'ðə', 'ɪn', 'ɔn', 'æt', 'wɪð', 'baɪ']:
                    # Short function words don't get stress
                    stressed_subwords.append(subword)
                else:
                    # Apply stress before the vowel
                    stressed_subwords.append(add_stress_before_vowel(subword, PRIMARY_STRESS))
            
            # Join the subwords with spaces
            token.phonemes = " ".join(stressed_subwords)
        elif not has_stress:
            # Handle single words according to their position and type
            if i == 0:
                # First word in sentence gets secondary stress
                token.phonemes = add_stress_before_vowel(token.phonemes, SECONDARY_STRESS)
            elif is_content_word(token.text) and len(token.phonemes) > 2:
                # Content words get primary stress before vowel
                token.phonemes = add_stress_before_vowel(token.phonemes, PRIMARY_STRESS)
            # Short function words don't get stress
    # Now build the final phoneme string with proper spacing and punctuation
    result = []
    punctuation_added = False
    for i, token in enumerate(tokens):
        # Check if this token is a punctuation mark
        is_punct = token.text in PUNCTS
        
        # Add space before tokens except:
        # - the first token
        # - punctuation marks
        # - tokens after punctuation (no double spaces)
        if i > 0 and not is_punct and not punctuation_added:
            result.append(" ")
            
        punctuation_added = False
        
        if is_punct:
            # Add punctuation directly to result
            result.append(token.text)
            punctuation_added = True
        elif token.phonemes:
            result.append(token.phonemes)
            
            # Check if token ends with punctuation
            if token.text and token.text[-1] in PUNCTS:
                punct = token.text[-1]
                result.append(punct)
                punctuation_added = True
                
                # Add space after sentence-ending punctuation
                if punct in sentence_ending_punct and i < len(tokens) - 1:
                    result.append(" ")
    # Join the result into a single string
    final_result = "".join(result)
    return final_result

def remove_commas_between_digits(text):
    """
    Remove commas between digits in numbers (e.g., 1,000 → 1000).
    
    Args:
        text (str): Input text containing numbers with commas
        
    Returns:
        str: Text with commas removed from between digits
    """
    pattern = r'(^|[^\d])(\d+(?:,\d+)*)([^\d]|$)'
    matches = [match for match in re.finditer(pattern, text)]
    for match in matches[::-1]:
        number_with_commas = match.group(2)
        number_without_commas = replace(number_with_commas, ',', '')
        updated_match_text = match.group(1) + number_without_commas + match.group(3)
        text = text[:match.start()] + updated_match_text + text[match.end():]
    return text

def split_num(text):
    """
    Process time expressions (like 2:30) and convert them to word form
    (e.g., "2:30" → "2 30" or "2 o'clock" for whole hours).
    
    Args:
        text (str): Input text containing time expressions
        
    Returns:
        str: Text with time expressions marked for conversion
    """    
    split_num_pattern = r"\b(?:[1-9]|1[0-2]):[0-5]\d\b"
    matches = [match for match in re.finditer(split_num_pattern, text)]
    transformed = ""
    for match in matches[::-1]:
        original = match.group(0)
        if "." in original:
            continue
        elif ":" in original:
            h, m = (int(split(original, ":", False)[0]), int(split(original, ":", False)[1]))
            transformed = original
            if m == 0:
                transformed = str(h) + " o'clock"
            elif m < 10:
                transformed = str(h) + " oh " + str(m)
            else:
                transformed = str(h) + " " + str(m)
        text = text[:match.start()] + "["+original+"]("+transformed+")" + text[match.end():]
    return text

def convert_numbers_to_words(text):
    """
    Convert numbers to their spoken form (e.g., years, hundreds).
    
    Args:
        text (str): Input text containing numbers
        
    Returns:
        str: Text with numbers marked for conversion to spoken form
    """
    split_num_pattern = r"\b[0-9]+\b"
    matches = [match for match in re.finditer(split_num_pattern, text)]
    transformed = ""
    for match in matches[::-1]:
        original = match.group(0)
        num = int(original)
        transformed = original
        if 2000 <= num <= 2099:
            # Years in the form "20XX"
            century = num // 100
            year = num % 100
            if year == 0:
                transformed = century + " hundred"
            elif 1 <= year <= 9:
                transformed = century + " oh " + year
            else:
                transformed = century + " " + year
            text = text[:match.start()] + "["+original+"]("+transformed+")" + text[match.end():]
        elif num % 100 == 0 and num <= 9999:
            # Even hundreds
            transformed = num // 100 + " hundred"
            text = text[:match.start()] + "["+original+"]("+transformed+")" + text[match.end():]
        else:
            return text

def flip_money(text):
    """
    Convert currency expressions to their spoken form
    (e.g., "$5.25" → "5 dollars and 25 cents").
    
    Args:
        text (str): Input text containing currency expressions
        
    Returns:
        str: Text with currency expressions marked for conversion
    """
    currency_pattern = r"[\\$£€]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[\\$£€]\d+\.\d\d?\b"
    matches = [match for match in re.finditer(currency_pattern, text)]
    for match in matches[::-1]:
        match_text = match.group(0)
        currency_symbol = match_text[0]
        value = match_text[1:]
        bill, cent = ('dollar', 'cent')
        if currency_symbol in CURRENCIES.keys():
            currency_names = CURRENCIES[currency_symbol]
            bill, cent = currency_names
    
        transformed = value
        dollars, cents = "0", "0"
        if "." not in value:
            dollars = value
        else:
            splits = split(value, ".", False)
            dollars = splits[0]
            cents = splits[1]
        if int(cents) == 0:
            if int(dollars) != 1:
                transformed = dollars + " " + bill + "s"
            else:
                transformed = dollars + " " + bill
        else:
            if int(dollars) != 1:
                transformed = dollars + " " + bill + "s and " + cents + " " + cent + "s"
            else:
                transformed = dollars + " " + bill + " and " + cents + " " + cent + "s"
        text = text[:match.start()] + "["+match_text+"]("+transformed+")" + text[match.end():]
    return text

def point_num(text):
    """
    Convert decimal numbers to spoken form with "point"
    (e.g., "3.14" → "3 point 1 4").
    
    Args:
        text (str): Input text containing decimal numbers
        
    Returns:
        str: Text with decimal numbers marked for conversion
    """
    split_num_pattern = r"\b\d*\.\d+\b"
    matches = [match for match in re.finditer(split_num_pattern, text)]
    transformed = ""
    for match in matches[::-1]:
        original = match.group(0)
        parts = split(original, ".", False)
        if len(parts) == 2:
            a, b = parts[0], parts[1]
            transformed = a + " point " + " ".join([c for c in b])
            text = text[:match.start()] + "["+original+"]("+transformed+")" + text[match.end():]
    return text


def preprocess(text):
    """
    Prepare text for tokenization by handling special cases:
    1. Remove commas in numbers
    2. Convert ranges (e.g., 5-10 to 5 to 10)
    3. Process currency values
    4. Handle time expressions
    5. Process decimal numbers
    
    Args:
        text (str): Raw input text
        
    Returns:
        tuple: (
            str: Preprocessed text,
            list: Tokens extracted from text,
            dict: Features for special handling indexed by token position,
            list: Indices of non-string features
        )
    """
    result = ''
    tokens = []
    features = {}
    nonStringFeatureIndexList = []
    last_end = 0
    
    text = remove_commas_between_digits(text)
    for m in re.finditer(r"([\\$£€]?\d+)-([\\$£€]?\d+)", text):
        text = text[:m.start()] +m.group(1)+" to "+m.group(2) + text[m.end():]
    # Process currencies first to prevent double-processing
    text = flip_money(text)

    # Mark processed currency values to skip later processing
    processed_features = re.findall(r'\[[^\]]*\]\([^\)]*\)', text)
    placeholders = {"FEATURE"+str(i): match for i, match in enumerate(processed_features)}
    # Process times like 2:30
    text = split_num(text)
    for placeholder_key in placeholders.keys():
        text = replace(text, placeholders[placeholder_key], placeholder_key)
    text = point_num(text)
    # # Process special years and hundreds
    # text = convert_numbers_to_words(text)
    # print("after convert_numbers_to_words", text)


    for placeholder_key in placeholders.keys():
        text = replace(text, placeholder_key, placeholders[placeholder_key])
    for m in re.finditer(LINK_REGEX, text):
        result = result + text[last_end:m.start()]
        tokens = tokens + split(text[last_end:m.start()], r' ', False)
        original = m.group(1)
        replacement = m.group(2)
        # Check if this is from regex replacements like [$123.45](123 dollars and 45 cents)
        # or explicit like [Kokoro](/kˈOkəɹO/)
        is_alias = False
        f = ""
        def is_signed(s):
            if s[0] == '-' or s[0] == '+':
                return bool(re.match(r'^[0-9]+$', s[1:]))
            return bool(re.match(r'^[0-9]+$', s))
        
        if replacement[0] == '/' and replacement[-1] == '/':
            # This is a phoneme specification
            f = replacement
        elif original[0] == '$' or ':' in original or '.' in original:
            # This is likely from flip_money, split_num, or point_num
            f = replacement
            is_alias = True
        elif is_signed(replacement):
            f = int(replacement)
            nonStringFeatureIndexList.append(str(len(tokens)))
        elif replacement == '0.5' or replacement == '+0.5':
            f = 0.5
            nonStringFeatureIndexList.append(str(len(tokens)))
        elif replacement == '-0.5':
            f = -0.5
            nonStringFeatureIndexList.append(str(len(tokens)))
        elif len(replacement) > 1 and replacement[0] == '#' and replacement[-1] == '#':
            f = replacement[0] + replacement[1:].rstrip('#')
        else:
            # Default case - treat as alias
            f = replacement
            is_alias = True
            
        if f is not None:
            # For aliases/replacements, store with 'alias:' prefix to distinguish
            feature_key = str(len(tokens))
            print("alias: ", f, feature_key, features)

            if is_alias:
                features[feature_key] = "["+f+"]"
            else:
                features[feature_key] = f

        result = result + original
        tokens.append(original)
        last_end = m.end()
    if last_end < len(text):
        result = result + text[last_end:]
        tokens = tokens + split(text[last_end:], r' ', False)
    return result, tokens, features, nonStringFeatureIndexList

def split_puncts(text):
    """
    Split text by punctuation marks, keeping the punctuation as separate tokens.
    
    Args:
        text (str): Input text to split
        
    Returns:
        list: List of tokens with punctuation as separate items
    """
    splits = [text]
    for punct in PUNCTS:
        for idx, t in enumerate(splits):
            if punct in t:
                res = split_with_delimiters_seperate(t, punct, False)
                if idx == 0:
                    splits = res + splits[1:]
                else: 
                    splits = splits[:idx] + res + splits[idx+1:]
    return splits

def tokenize(tokens, features, nonStringFeatureIndexList):
    """
    Convert preprocessed text into Token objects with phonemes.
    This is the core tokenization function that:
    1. Handles phoneme generation for each token
    2. Processes special cases using features from preprocessing
    3. Merges multi-word tokens appropriately
    
    Args:
        tokens (list): List of token strings from preprocessing
        features (dict): Special features indexed by token position
        nonStringFeatureIndexList (list): Indices of non-string features
        
    Returns:
        list: List of Token objects with phonetic information
    """
    mutable_tokens = []
    for i, word in enumerate(tokens):
        if word in SUBTOKEN_JUNKS:
            continue
        # Get feature for this token if exists
        feature = None
        if len(features.keys()) > 0 and str(i) in features:
            feature = features[str(i)]
        # Check if this is a phoneme specification
        if feature is not None and feature[0] == '/' and feature[-1] == '/':
            # Direct phoneme specification - use it directly
            phoneme = feature[1:-1]
            mutable_tokens.append(Token(word,' ',phoneme,None,None,False,None,False))
        else:
            # If token has a replacement/alias, use that for phonemization
            phoneme_text = word
            alias = None
            
            if feature is not None and feature[0] == '[' and feature[-1] == ']':
                # This is an alias from formatted replacements - remove brackets 
                alias = feature[1:-1]
                phoneme_text = alias
            
            word = split(phoneme_text, r' ', False)
            word_punct_split = []
            for tok in word:
                split_tok = split_puncts(tok)
                word_punct_split = word_punct_split + split_tok
            word_tokens = []
            for idx, tok in enumerate(word_punct_split):
                # Generate phonemes using espeak or lexicon
                phoneme = ""
                whitespace = True
                if tok in PUNCTS:
                    phoneme = tok
                    whitespace=False
                elif LEXICON is not None and tok in LEXICON:
                    phoneme = LEXICON[tok]
                else:
                    tok_lower = tok.lower()
                    if LEXICON is not None and tok_lower in LEXICON:
                        phoneme = LEXICON[tok_lower]
                    else:
                        phoneme = backend.phonemize([tok_lower], strip=True)[0]
                stress = None
                if feature is not None and not i in nonStringFeatureIndexList:
                    stress = feature
                alias = None
                if feature is not None and not feature[0] == '/':
                    alias = feature
                if not whitespace and len(word_tokens) > 0:
                    word_tokens[-1].whitespace = ''
                token = Token(tok,' ', phoneme, stress, None, whitespace, alias, idx == 0)
                word_tokens.append(token)
            word_tokens[-1].whitespace = ''
            word_tokens[0].prespace = False
            mutable_tokens.append(merge_tokens(word_tokens, " "))
    return mutable_tokens

def phonemize(text):
    """
    Main function to convert text to phonemes.
    
    The process follows these steps:
    1. Preprocess text to handle special cases (currencies, times, numbers)
    2. Tokenize the preprocessed text into Token objects
    3. Resolve tokens to apply proper stress and formatting
    
    Args:
        text (str): The input text to convert to phonemes
    
    Returns:
        dict: Dictionary with:
            - 'ps': Processed phoneme string with stress marks
            - 'tokens': List of Token objects used for generation
    """
    _, tokens, features, nonStringFeatureIndexList = preprocess(text)
    tokens = tokenize(tokens, features, nonStringFeatureIndexList)   
    result = resolve_tokens(tokens)
    print("Result Phonemes", result)
    return {"ps": result, "tokens": tokens}

def set_lexicon(lexicon):
    """
    Set a custom lexicon for word-to-phoneme mappings.
    
    Args:
        lexicon (dict): Dictionary mapping words to their phoneme representations
    
    Returns:
        None: Updates the global LEXICON variable
    """
    LEXICON = lexicon
    print("LEXICON Sample keys", list(LEXICON.keys())[:10])
