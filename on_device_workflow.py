"""
Python implementation for the on-device G2P tokenizer and workflow utilities of Kokoro

This script provides a complete on-device workflow for:

1. Text-to-Speech Generation:
   - Tokenization: Convert text into phonetic representations
     a. Preprocessing - handle special cases like currencies, numbers, times
     b. Tokenization - split text into meaningful tokens
     c. Phonemization - convert tokens to phonetic representations 
     d. Stress application - add proper stress markers to phonemes
   - Audio Generation: Convert phonemes to high-quality speech

2. Language Model Integration:
   - User Query Processing: Send user text to the LLM
   - Response Streaming: Get text responses as they're generated
   - Context Management: Maintain conversation history for contextual replies
   
Key Entry Points for Native Apps:
- run_text_to_speech_model: Convert text to speech audio
- prompt_llm: Send user input to the LLM
- get_next_str: Retrieve streaming LLM responses
- set_context: Set conversation history
- init: Initialize the model with a custom lexicon

Each of these functions accepts a dictionary (HashMap) of parameters and returns 
a dictionary of results, making them easy to call from Kotlin/Swift.
"""
from nimbleedge import ne_re as re
from nimbleedge import nimblenet as nm

# Get hardware information and optimize thread usage for model performance
hardware_info = nm.get_hardware_info()
num_cores  = int(hardware_info["numCores"])
nm.set_xnnpack_num_threads(num_cores / 2  + 1)

# Initialize models - kokoro for text-to-speech and llama-3 for LLM responses
kokoro_model = nm.Model("kokoro_")
llm = nm.llm("llama-3_")

# Phoneme-related constants for stress markers and vowel sounds
STRESSES = 'ˌˈ'
PRIMARY_STRESS = STRESSES[1]
SECONDARY_STRESS = STRESSES[0]
VOWELS = ['A', 'I', 'O', 'Q', 'W', 'Y', 'a', 'i', 'u', 'æ', 'ɑ', 'ɒ', 'ɔ', 'ə', 'ɛ', 'ɜ', 'ɪ', 'ʊ', 'ʌ', 'ᵻ']

# Tokenization and punctuation constants
SUBTOKEN_JUNKS = "',-._''/' "
PUNCTS = ['?', ',', ';', '"', '—', ':', '!', '.', '…', '"', '"']
NON_QUOTE_PUNCTS = ['?', ',', '—', '.', ':', '!', ';', '…']
LEXICON = None

# Global variables for handling the text stream and LLM state
text_stream = None
IS_LLM_INITIATED = False

# LLM prompt formatting constants
SYSTEM_PROMPT_BEGIN = "<|start_header_id|>system<|end_header_id|>"
SYSTEM_PROMPT_TEXT = """You are NimbleEdge AI, a chatbot running on device using the NimbleEdge platform. Keep answers short and concise."""
PROMPT_END = "<|eot_id|>\n"
SYSTEM_PROMPT = SYSTEM_PROMPT_BEGIN + SYSTEM_PROMPT_TEXT + PROMPT_END

# Additional prompt for voice-initiated queries
ASR_DYNAMIC_SUB_PROMPT = """ Answer the user query in less than 30 words. Keep the chat fun and engaging."""

# Message formatting constants
USER_PROMPT_BEGIN = "<|start_header_id|>user<|end_header_id|>"
ASSISTANT_RESPONSE_BEGIN = "<|start_header_id|>assistant<|end_header_id|>"
MAX_TOKEN_CONTEXT = 2000

# Mapping from phoneme characters to integer IDs for the model
PHONEME_TO_ID = {
    ";": 1,
    ":": 2,
    ",": 3,
    ".": 4,
    "!": 5,
    "?": 6,
    "—": 9,
    "…": 10,
    "\"": 11,
    "(": 12,
    ")": 13,
    """: 14,
    """: 15,
    " ": 16,
    "\u0303": 17,
    "ʣ": 18,
    "ʥ": 19,
    "ʦ": 20,
    "ʨ": 21,
    "ᵝ": 22,
    "\uAB67": 23,
    "A": 24,
    "I": 25,
    "O": 31,
    "Q": 33,
    "S": 35,
    "T": 36,
    "W": 39,
    "Y": 41,
    "ᵊ": 42,
    "a": 43,
    "b": 44,
    "c": 45,
    "d": 46,
    "e": 47,
    "f": 48,
    "h": 50,
    "i": 51,
    "j": 52,
    "k": 53,
    "l": 54,
    "m": 55,
    "n": 56,
    "o": 57,
    "p": 58,
    "q": 59,
    "r": 60,
    "s": 61,
    "t": 62,
    "u": 63,
    "v": 64,
    "w": 65,
    "x": 66,
    "y": 67,
    "z": 68,
    "ɑ": 69,
    "ɐ": 70,
    "ɒ": 71,
    "æ": 72,
    "β": 75,
    "ɔ": 76,
    "ɕ": 77,
    "ç": 78,
    "ɖ": 80,
    "ð": 81,
    "ʤ": 82,
    "ə": 83,
    "ɚ": 85,
    "ɛ": 86,
    "ɜ": 87,
    "ɟ": 90,
    "ɡ": 92,
    "ɥ": 99,
    "ɨ": 101,
    "ɪ": 102,
    "ʝ": 103,
    "ɯ": 110,
    "ɰ": 111,
    "ŋ": 112,
    "ɳ": 113,
    "ɲ": 114,
    "ɴ": 115,
    "ø": 116,
    "ɸ": 118,
    "θ": 119,
    "œ": 120,
    "ɹ": 123,
    "ɾ": 125,
    "ɻ": 126,
    "ʁ": 128,
    "ɽ": 129,
    "ʂ": 130,
    "ʃ": 131,
    "ʈ": 132,
    "ʧ": 133,
    "ʊ": 135,
    "ʋ": 136,
    "ʌ": 138,
    "ɣ": 139,
    "ɤ": 140,
    "χ": 142,
    "ʎ": 143,
    "ʒ": 147,
    "ʔ": 148,
    "ˈ": 156,
    "ˌ": 157,
    "ː": 158,
    "ʰ": 162,
    "ʲ": 164,
    "↓": 169,
    "→": 171,
    "↗": 172,
    "↘": 173,
    "ᵻ": 177
  }
# PUNCT_TAGS = [".",",","-LRB-","-RRB-","``",'""',"''",":","$","#",'NFP']
# PUNCT_TAG_PHONEMES = {'-LRB-':'(', '-RRB-':')', '``':'"', '""':'"', "''":'"'}

print(PRIMARY_STRESS, SECONDARY_STRESS, VOWELS, PUNCTS, NON_QUOTE_PUNCTS)

CURRENCIES = {
    '$': ('dollar', 'cent'),
    '£': ('pound', 'pence'),
    '€': ('euro', 'cent'),
}
# Create regex character classes from currency symbols and punctuation
currency_symbols = r'[' + r''.join([re.escape(symbol) for symbol in CURRENCIES.keys()]) + r']'
punct_symbols = r'[' + ''.join([re.escape(p) for p in PUNCTS]) + ']'
LINK_REGEX = r"\[([^\]]+)\]\(([^\)]*)\)"

# Helper functions for the tokenizer implementation

def all(iterable):
    for item in iterable:
        if not item:
            return False
    return True


def any(iterable):
    for item in iterable:
        if item:
            return True
    return False


def replace(text, original, replacement):
    matches = [m for m in re.finditer(re.escape(original), text)]
    for match in matches[::-1]:
        text = text[:match.start()]+replacement+text[match.end():]
    return text


def split(text, delimiter, is_regex):
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
    return all([c in [' ', '\t', '\n', '\r'] for c in input])

class Token:
    
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
    Merge multiple Token objects into a single Token, combining their phonemes.
    
    Args:
        tokens (list): List of Token objects to merge
        unk (str): Placeholder for unknown tokens (not currently used)
        
    Returns:
        Token: A single merged Token object
        
    Logic:
        1. Collect stress information from all tokens
        2. Combine phonemes with appropriate spacing
        3. Remove leading spaces
        4. Inherit properties from component tokens
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
    print("final phonemes", phonemes)
    if isspace(phonemes[0]):
        print("deleting space")
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
        ps: The phoneme string
        stress: Stress level - can be numeric (1 for primary, 0.5/0 for secondary, -1 for no stress)
                or None to determine automatically based on content words
    
    Returns:
        Phonemes with appropriate stress markers
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
    """Check if a word is a function word (articles, prepositions, conjunctions, etc.)"""
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
    """Check if string contains only alphabetic characters."""
    if not text:  # Handle empty string
        return False
    return bool(re.match(r'^[a-zA-Z]+$', text))


def is_content_word(word):
    """Check if a word is a content word (nouns, verbs, adjectives, adverbs)"""
    return not is_function_word(word) and len(word) > 2 and isalpha_regex(word)


def resolve_tokens(tokens):
    """
    Apply stress patterns and format the phoneme string according to G2P conventions.
    
    Args:
        tokens (list): List of Token objects with phoneme information
        
    Returns:
        str: Final phoneme string with appropriate stress markers and spacing
        
    Logic:
        1. Apply phoneme corrections to standardize representation
        2. Handle special word cases
        3. Apply appropriate stress to content words and multi-word expressions
        4. Add proper spaces between tokens
        5. Handle punctuation correctly
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
    Convert time expressions to their spoken form.
    
    Args:
        text (str): Text containing time expressions (e.g., "2:30")
        
    Returns:
        str: Text with time expressions marked with their spoken forms
        
    Logic:
        1. Find time expressions using regex (hours:minutes format)
        2. Handle special cases:
           - Whole hours as "X o'clock"
           - Minutes less than 10 as "oh X" (e.g., 2:05 → "2 oh 5")
           - Regular times as "hour minute" (e.g., 2:30 → "2 30")
        3. Add annotation in [original](spoken form) format
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
    # # Convert number to words (simplified implementation)
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
    Convert currency expressions to their spoken form with formatting annotations.
    
    Args:
        text (str): Text containing currency expressions
        
    Returns:
        str: Text with currency expressions marked with their spoken forms
        
    Logic:
        1. Find currency expressions using regex
        2. Extract currency symbol and value
        3. Format currency as words based on amount (singular/plural)
        4. Add annotation in [original](spoken form) format
    """
    # Handle case when match_obj is a re.Match object
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
    Convert decimal numbers to their spoken form with "point" notation.
    
    Args:
        text (str): Text containing decimal numbers
        
    Returns:
        str: Text with decimal numbers marked with their spoken forms
        
    Logic:
        1. Find decimal numbers using regex
        2. Split number at decimal point
        3. Format as "[number] point [digits]" where digits are read individually
        4. Add annotation in [original](spoken form) format
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
    Prepare text for phonemization by handling special cases like numbers, ranges, and currency.
    
    Args:
        text (str): Raw input text
        
    Returns:
        tuple: (
            str: Preprocessed text,
            list: Tokens extracted from text, 
            dict: Features for special handling indexed by token position,
            list: Indices of non-string features
        )
        
    Logic:
        1. Remove commas between digits (e.g., 1,000 → 1000)
        2. Convert ranges (e.g., 5-10 → 5 to 10)
        3. Format currency expressions (e.g., $5.25 → 5 dollars and 25 cents)
        4. Process time expressions (e.g., 2:30 → 2 30)
        5. Format decimal numbers (e.g., 3.14 → 3 point 1 4)
        6. Process explicit phonetic specifications in [text](/phoneme/) format
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
    Split text by punctuation marks while preserving the punctuation.
    
    Args:
        text (str): Input text to be split
        
    Returns:
        list: Text split into tokens with punctuation preserved as separate tokens
        
    Logic:
        Iterates through each punctuation mark and splits the text whenever the 
        punctuation mark is found, keeping the punctuation as a separate token.
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
    Convert preprocessed text tokens into Token objects with phonetic information.
    
    Args:
        tokens (list): List of string tokens from preprocessing
        features (dict): Special features keyed by token position
        nonStringFeatureIndexList (list): Indices of tokens with non-string features
        
    Returns:
        list: List of Token objects with phonetic information
        
    Logic:
        1. Skip junk tokens
        2. Handle explicit phoneme specifications
        3. Process tokens with aliases/replacements
        4. Split tokens that contain punctuation
        5. Generate phonemes using lexicon or text-to-phoneme conversion
        6. Merge multi-word tokens
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
                    print("found tok", tok, LEXICON[tok])
                    phoneme = LEXICON[tok]
                else:
                    tok_lower = tok.lower()
                    if LEXICON is not None and tok_lower in LEXICON:
                        print("found tok", tok_lower, LEXICON[tok_lower])
                        phoneme = LEXICON[tok_lower]
                    else:
                        print("not found tok lower:"+ tok_lower+ "tok:"+tok)
                        phoneme = nm.convertTextToPhonemes(tok_lower)
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
            print("word_tokens", [t.phonemes for t in word_tokens])
            mutable_tokens.append(merge_tokens(word_tokens, " "))
    return mutable_tokens


def phonemize(text):
    """
    Convert text to phonemes using the preprocessing, tokenization, and stress resolution pipeline.
    
    Args:
        text (str): The input text to convert to phonemes
    
    Returns:
        dict: Dictionary containing:
            - ps (str): The final phonetic string with proper stress markers
            - tokens (list): The token objects used for phonemization
            
    Logic:
        1. Preprocess text to handle special cases (numbers, currency, etc.)
        2. Convert preprocessed tokens to phonemes
        3. Apply appropriate stress patterns and formatting
    """
    _, tokens, features, nonStringFeatureIndexList = preprocess(text)
    print("tokens", [t for t in tokens], "features", features, nonStringFeatureIndexList)
    tokens = tokenize(tokens, features, nonStringFeatureIndexList)   
    print("tokens after tokenize", [t.phonemes for t in tokens]) 
    result = resolve_tokens(tokens)    
    return {"ps": result, "tokens": tokens}


def run_text_to_speech_model(input):
    """
    Convert text to speech using the Kokoro model called from Kotlin/iOS applications.
    
    Args:
        input (dict): A dictionary containing:
            - text (str): The text to convert to speech
    
    Returns:
        dict: A dictionary containing:
            - audio (bytes): The generated audio data
    
    Process:
    1. Convert input text to phonemes
    2. Map phonemes to token IDs
    3. Run the Kokoro model to generate audio
    4. Return the audio data
    """
    # text = "How could I know? It's an unanswerable question. Like asking an unborn child if they'll lead a good life. They haven't even been born."
    phonemes = phonemize(input["text"])["ps"]
    tokens = []
    for p in phonemes:
        if PHONEME_TO_ID.has_key(p):
            tokens.append(PHONEME_TO_ID[p])
        else:
            tokens.append(0)
    tokens = [0] + tokens + [0]
    if len(tokens) > 510:
        tokens = tokens[:510]

    input_ids = nm.tensor([[0] + tokens + [0]], "int64")
    speed = nm.tensor([1.0], "float")
    audio = kokoro_model.run(input_ids, speed)
    return {"audio": audio[0][0]}

def init(input):
    """
    Initialize the model with a custom lexicon for phoneme mapping called from Kotlin/iOS applications.
    
    Args:
        input (dict): A dictionary containing:
            - lexicon (dict): A dictionary mapping words to their phonetic representations
    
    Returns:
        dict: An empty dictionary
    
    Side effects:
        Sets the global LEXICON variable used for phoneme lookups
    """
    LEXICON = input["lexicon"]
    print(LEXICON["'Merica"])
    return {}

def estimate_tokens_by_words(prompt):
    """
    Estimate the number of tokens in a text based on word count.
    
    Args:
        prompt (str): Text to estimate token count for
        
    Returns:
        int: Estimated number of tokens
        
    Logic:
        1. Count words using regex
        2. Apply a multiplier (1.33) to convert words to estimated tokens
        3. Convert to integer
    """
    # Count words in the prompt
    word_count = len(re.findall(r"\b\w+\b", prompt))
    # Estimate tokens (1 word ~ 1.33 tokens)
    estimated_tokens = int(word_count * 1.33)
    return estimated_tokens


SYSTEM_PROMPT_TOKENS = estimate_tokens_by_words(SYSTEM_PROMPT)
messages = []

def llm_cancel(inp):
    """
    Cancel the current LLM generation process.
    
    Args:
        inp (dict): Input dictionary (not used)
        
    Returns:
        dict: Empty dictionary
        
    Side effects:
        - Cancels the current LLM generation
        - Resets the text_stream to None
    """
    llm.cancel()
    text_stream = None
    return {}

def clear_prompt(inp):
    """
    Clear the LLM context and reset state variables.
    
    Args:
        inp (dict): Input dictionary (not used)
        
    Returns:
        dict: Empty dictionary
        
    Side effects:
        - Clears the LLM context
        - Resets the text_stream to None
        - Resets the IS_LLM_INITIATED flag
    """
    llm.clear_context()
    text_stream = None
    IS_LLM_INITIATED = False
    return {}


def prompt_llm(inp):
    """
    Process a user query and generate an LLM response called from Kotlin/iOS applications.
    
    Args:
        inp (dict): A dictionary containing:
            - query (str): The user's text query
            - is_voice_initiated (bool): Whether the query came from voice input
    
    Returns:
        dict: An empty dictionary (responses are retrieved via get_next_str)
    
    Side effects:
        - Sets up the prompt with appropriate system context
        - Initiates LLM generation that can be retrieved via get_next_str
        - Updates the global text_stream variable
    
    Note:
        For voice-initiated queries, the system adds a special prompt to keep
        responses shorter and more engaging.
    """
    # Re-init variables
    text_stream = None
    query = inp["query"]
    sys_prompt = ""
    is_voice_initiated  = inp["is_voice_initiated"]
    if not IS_LLM_INITIATED:
        sys_prompt = SYSTEM_PROMPT_TEXT
        IS_LLM_INITIATED = True
        print("sys_prompt", sys_prompt)
    if is_voice_initiated:
        sys_prompt = SYSTEM_PROMPT_BEGIN + sys_prompt + ASR_DYNAMIC_SUB_PROMPT + PROMPT_END
    
    final_prompt = sys_prompt + USER_PROMPT_BEGIN + query + PROMPT_END + ASSISTANT_RESPONSE_BEGIN
    print("final_prompt", final_prompt)
    text_stream = llm.prompt(final_prompt)
    return {}


current_response = ""


def get_next_str(inp):
    """
    Retrieve the next chunk of text from the LLM's response stream called from Kotlin/iOS applications.
    
    Args:
        inp (dict): An empty dictionary (no input needed)
    
    Returns:
        dict: A dictionary containing:
            - str (str): The next chunk of generated text
            - finished (bool): True if the generation is complete, False otherwise
    
    Side effects:
        - Updates the global current_response variable to accumulate the full response
        - Resets current_response to empty when generation is finished
    
    Note:
        The client application should call this repeatedly until finished=True to
        get the complete response in a streaming manner.
    """
    if text_stream is None:
        return {"finished": True, "str":""}
    
    if text_stream is not None and text_stream.finished():
        strOut = text_stream.next()
        current_response = current_response + strOut
        # print("Text stream finished")
        current_response = ""
        return {"finished": True, "str":strOut}

    strOut = text_stream.next()
    current_response = current_response + strOut
    return {"str": strOut}


def set_context(inp):
    """
    Set the conversation history context for the LLM called from Kotlin/iOS applications.       
    Args:
        inp (dict): A dictionary containing:
            - context (list): A list of message dictionaries, each with:
                - type (str): Either "user" or "assistant"
                - message (str): The content of the message
    
    Returns:
        dict: An empty dictionary
    
    Side effects:
        - Clears the current LLM context
        - Builds a properly formatted prompt from the conversation history
        - Sets this as the new context for the LLM
    
    Note:
        The context is processed in reverse order (newest first) and truncated if
        it exceeds MAX_TOKEN_CONTEXT to prevent exceeding model context limits.
    """
    llm.clear_context()
    final_prompt = ""
    for message_dict in inp["context"][::-1]:
        type = message_dict["type"]
        message = message_dict["message"]
        if len(final_prompt) + len(message) > MAX_TOKEN_CONTEXT:
            substring_len = MAX_TOKEN_CONTEXT - len(final_prompt)
            start_idx = len(message) - substring_len
            final_prompt = USER_PROMPT_BEGIN +  message[start_idx:] + PROMPT_END +  final_prompt
            break
        if type == "user":
            final_prompt = USER_PROMPT_BEGIN + message + PROMPT_END + final_prompt
        elif type == "assistant":
            final_prompt = ASSISTANT_RESPONSE_BEGIN + message + PROMPT_END + final_prompt
        else:
            raise "type should be user or assistant to set context"
    
    final_prompt = SYSTEM_PROMPT + final_prompt
    print("final_prompt set context", final_prompt)
    llm.add_context(final_prompt)
    return {}