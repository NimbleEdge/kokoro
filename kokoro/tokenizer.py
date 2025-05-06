#!/usr/bin/env python3
"""
Python implementation of the tokenizer for ONNX models.
Converted from JavaScript to be compatible with Python-based ONNX implementations.
"""
import re
from nimbleedge import nimblenet as nm

STRESSES = 'ˌˈ'
PRIMARY_STRESS = STRESSES[1]
SECONDARY_STRESS = STRESSES[0]
VOWELS = ['A', 'I', 'O', 'Q', 'W', 'Y', 'a', 'i', 'u', 'æ', 'ɑ', 'ɒ', 'ɔ', 'ə', 'ɛ', 'ɜ', 'ɪ', 'ʊ', 'ʌ', 'ᵻ']

SUBTOKEN_JUNKS = "',-._'’/' "
PUNCTS = ['?', ',', ';', '“', '—', ':', '!', '.', '…', '"', '”']
NON_QUOTE_PUNCTS = ['?', ',', '—', '.', ':', '!', ';', '…']
LEXICON = None

CURRENCIES = {
    '$': ('dollar', 'cent'),
    '£': ('pound', 'pence'),
    '€': ('euro', 'cent'),
}
# Create regex character classes from currency symbols and punctuation
currency_symbols = r'[' + r''.join([re.escape(symbol) for symbol in CURRENCIES.keys()]) + r']'
punct_symbols = r'[' + ''.join([re.escape(p) for p in PUNCTS]) + ']'
LINK_REGEX = r"\[([^\]]+)\]\(([^\)]*)\)"

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
    Apply stress and formatting to match G2P output format.
    G2P places primary stress markers directly before vowels, not at the beginning of words.
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
    Helper function to split numbers into phonetic equivalents.
    
    Args:
        match_obj: The regex match object or the matched string.
    
    Returns:
        The original text with phonetic equivalent in square brackets.
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
    Helper function to format monetary values.
    
    Args:
        match_obj: The regex match object or the matched string.
    
    Returns:
        The original text with formatted currency in square brackets.
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
    Helper function to process decimal numbers.
    
    Args:
        match_obj: The regex match object or the matched string.
    
    Returns:
        The original text with formatted decimal in square brackets.
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
            mutable_tokens.append(merge_tokens(word_tokens, " "))
    return mutable_tokens

def phonemize(text):
    """
    Convert text to phonemes.
    
    Args:
        text: The input text to convert to phonemes.
        language: The language code to use for phonemization.
    
    Returns:
        The phonemized text and the tokens used for phonemization.
    """
    _, tokens, features, nonStringFeatureIndexList = preprocess(text)
    print("tokens", [t for t in tokens], "features", features, nonStringFeatureIndexList)
    tokens = tokenize(tokens, features, nonStringFeatureIndexList)   
    print("tokens after tokenize", [t.phonemes for t in tokens]) 
    result = resolve_tokens(tokens)    
    return {"ps": result, "tokens": tokens}


def init(input):
    LEXICON = input["lexicon"]
    print(LEXICON["'Merica"])
    return {}
