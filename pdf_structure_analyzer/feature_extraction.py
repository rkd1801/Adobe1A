# pdf_structure_analyzer/feature_extraction.py
import re
import numpy as np
from collections import Counter
import difflib
from difflib import get_close_matches
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
import spacy
from .constants import COMMON_HEADINGS, NON_HEADING_KEYWORDS

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Load spaCy model for advanced NLP
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install spaCy English model: python -m spacy download en_core_web_sm")
    nlp = None

def clean_text(t):
    """Cleans text by stripping whitespace and replacing specific unicode characters."""
    if not t:
        return ""
    return t.strip().replace('\u2013', '-').replace('\u2014', '-').replace('\u00a0', ' ').replace('\n', ' ').replace('\r', ' ')

def correct_heading(text, reference_list=COMMON_HEADINGS):
    cleaned = clean_text(text)
    matches = get_close_matches(cleaned, [clean_text(ref) for ref in reference_list], n=1, cutoff=0.6)
    if matches:
        for ref in reference_list:
            if clean_text(ref) == matches[0]:
                return ref
    return text.strip()

def get_document_statistics(doc):
    """Analyzes document-wide statistics for adaptive feature extraction."""
    all_font_sizes = []
    all_font_names = []
    all_texts = []
    all_positions = []
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    text = clean_text(span["text"])
                    if text:
                        all_font_sizes.append(span["size"])
                        all_font_names.append(span["font"])
                        all_texts.append(text)
                        all_positions.append(span["bbox"])
    
    if not all_font_sizes:
        return {
            "common_font_size": 12.0,
            "common_font_name": "Times-Roman",
            "font_size_std": 1.0,
            "unique_font_sizes": [12.0],
            "page_width": 595.0,
            "avg_text_length": 50
        }
    
    font_size_counts = Counter(all_font_sizes)
    font_name_counts = Counter(all_font_names)
    
    return {
        "common_font_size": font_size_counts.most_common(1)[0][0],
        "common_font_name": font_name_counts.most_common(1)[0][0],
        "font_size_std": np.std(all_font_sizes),
        "unique_font_sizes": sorted(list(set(all_font_sizes)), reverse=True),
        "page_width": doc[0].rect.width if doc else 595.0,
        "avg_text_length": np.mean([len(text) for text in all_texts]) if all_texts else 50
    }

def extract_nlp_features(text):
    """Extracts NLP-based features to identify headings regardless of visual formatting."""
    if not text or len(text.strip()) < 2:
        return {}
    
    # Basic linguistic features
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    
    # Remove stopwords for content analysis
    stop_words = set(stopwords.words('english'))
    content_words = [word for word in words if word not in stop_words and word.isalpha()]
    
    # POS tagging to identify grammatical structure
    pos_tags = pos_tag(word_tokenize(text))
    
    # Heading indicators based on linguistic patterns
    heading_keywords = {
    'introduction', 'overview', 'background', 'methodology', 'methods', 'results', 
    'conclusion', 'discussion', 'references', 'bibliography', 'appendix', 'summary',
    'abstract', 'acknowledgments', 'acknowledgements', 'contents', 'index', 'glossary',
    'objective', 'objectives', 'material', 'required', 'construction', 'demonstration',
    'observation', 'application', 'procedure', 'apparatus', 'experiment', 'activity'
}
    
    # Check for heading-like patterns
    is_heading_keyword = any(word.lower() in heading_keywords for word in content_words)
    
    # Check for numbered sections (1., 1.1, A., etc.)
    has_numbering = bool(re.match(r'^\s*(\d+(\.\d+)*|[A-Z]|[IVX]+)\.?\s+', text))
    
    # Check for question-like structure (often section headers)
    is_question = text.strip().endswith('?')
    
    # Check for title case pattern
    is_title_case = text.istitle()
    
    # Check for short, concise text (typical of headings)
    is_concise = len(words) <= 10 and len(sentences) <= 1
    
    # Check for imperative mood (common in headings)
    has_imperative = any(tag in ['VB', 'VBP'] for word, tag in pos_tags[:2])
    
    # Check for proper nouns (often in titles/headings)
    proper_noun_ratio = sum(1 for word, tag in pos_tags if tag == 'NNP') / len(pos_tags) if pos_tags else 0
    
    # Advanced NLP features using spaCy if available
    semantic_features = {}
    if nlp:
        doc_nlp = nlp(text)
        
        # Named entity recognition
        entities = [(ent.text, ent.label_) for ent in doc_nlp.ents]
        has_entities = len(entities) > 0
        
        # Dependency parsing for grammatical structure
        root_deps = [token.dep_ for token in doc_nlp if token.dep_ == 'ROOT']
        
        semantic_features = {
            'has_entities': int(has_entities),
            'entity_density': len(entities) / len(words) if words else 0,
            'has_root_verb': int(any(token.pos_ == 'VERB' and token.dep_ == 'ROOT' for token in doc_nlp))
        }
    
    # Content-based features
    content_features = {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'content_word_ratio': len(content_words) / len(words) if words else 0,
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        'is_heading_keyword': int(is_heading_keyword),
        'has_numbering': int(has_numbering),
        'is_question': int(is_question),
        'is_title_case': int(is_title_case),
        'is_concise': int(is_concise),
        'has_imperative': int(has_imperative),
        'proper_noun_ratio': proper_noun_ratio,
        'contains_colon': int(':' in text),
        'starts_with_capital': int(text[0].isupper() if text else 0),
        'ends_with_period': int(text.strip().endswith('.'))
    }
    
    # Combine all features
    all_features = {**content_features, **semantic_features}
    return all_features

def get_comprehensive_span_features(span, page, doc_stats, page_num):
    text = clean_text(span["text"])
    if not text:
        return None
    font_size = span["size"]
    relative_font_size = font_size / doc_stats["common_font_size"]
    is_larger_font = font_size > doc_stats["common_font_size"]
    font_size_percentile = sum(1 for size in doc_stats["unique_font_sizes"] if size <= font_size) / len(doc_stats["unique_font_sizes"])
    bbox = span["bbox"]
    page_width = doc_stats["page_width"]
    page_height = page.rect.height
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2
    is_centered = abs(x_center - (page_width / 2)) < (page_width * 0.15)
    is_left_aligned = bbox[0] < (page_width * 0.1)
    is_right_aligned = bbox[2] > (page_width * 0.9)
    is_top_of_page = bbox[1] < (page_height * 0.1)
    is_bottom_of_page = bbox[3] > (page_height * 0.9)
    relative_y_position = bbox[1] / page_height
    font_name = span["font"]
    is_bold = "Bold" in font_name
    is_italic = "Italic" in font_name
    is_different_font = font_name != doc_stats["common_font_name"]
    is_all_caps = text.isupper() and len(text) > 1
    is_mixed_case = any(c.isupper() for c in text) and any(c.islower() for c in text)

    text_lower = text.lower().strip()
    starts_with_numbering = int(bool(re.match(r'^\s*(\d+(\.\d+)*)', text)))
    has_known_heading_keyword = int(any(k in text_lower for k in [
        "revision history", "introduction", "overview", "references", "acknowledgement", "appendix", "access"
    ]))
    is_short_heading_like = int(len(text) < 100 and is_top_of_page and is_larger_font)

    nlp_features = extract_nlp_features(text)

    visual_features = {
        "text": text,
        "font_size": font_size,
        "relative_font_size": relative_font_size,
        "is_larger_font": int(is_larger_font),
        "font_size_percentile": font_size_percentile,
        "is_bold": int(is_bold),
        "is_italic": int(is_italic),
        "is_different_font": int(is_different_font),
        "is_all_caps": int(is_all_caps),
        "is_mixed_case": int(is_mixed_case),
        "is_centered": int(is_centered),
        "is_left_aligned": int(is_left_aligned),
        "is_right_aligned": int(is_right_aligned),
        "is_top_of_page": int(is_top_of_page),
        "is_bottom_of_page": int(is_bottom_of_page),
        "relative_y_position": relative_y_position,
        "text_length": len(text),
        'starts_with_number': int(bool(re.match(r'^\s*\d+[\.\)]\s*', text))),
        'ends_with_colon': int(text.strip().endswith(':')),
        'has_phase_keyword': int('phase' in text.lower()),
        "page_num": page_num,
        "is_first_page": int(page_num == 0),
        "is_last_page": int(page_num == doc_stats.get("total_pages", 1) - 1),
        "starts_with_numbering": starts_with_numbering,
        "has_known_heading_keyword": has_known_heading_keyword,
        "is_short_heading_like": is_short_heading_like
    }

    all_features = {**visual_features, **nlp_features}
    return all_features

def extract_all_json_elements(gt_json):
    """Extracts all structural elements from the JSON file."""
    elements = []
    
    # Extract title
    title_text = clean_text(gt_json.get("title", ""))
    if title_text:
        elements.append({
            "text": title_text,
            "page": 1,
            "label": "TITLE"
        })
    
    # Extract outline elements (main structure)
    outline = gt_json.get("outline", [])
    for item in outline:
        if item.get("level") and item.get("text"):
            clean_heading_text = clean_text(item["text"])
            if clean_heading_text:
                elements.append({
                    "text": clean_heading_text,
                    "page": item.get("page", 1),
                    "label": item.get("level", "H1")
                })
    
    # Extract other sections
    for key, value in gt_json.items():
        if key not in ["title", "outline"]:
            if isinstance(value, str) and value.strip():
                elements.append({
                    "text": clean_text(value),
                    "page": 1,
                    "label": f"SECTION_{key.upper()}"
                })
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str) and item.strip():
                        elements.append({
                            "text": clean_text(item),
                            "page": 1,
                            "label": f"{key.upper()}_ITEM"
                        })
                    elif isinstance(item, dict):
                        item_text = item.get("text", "")
                        if item_text:
                            elements.append({
                                "text": clean_text(item_text),
                                "page": item.get("page", 1),
                                "label": item.get("level", f"{key.upper()}_ELEMENT")
                            })
    
    return elements

def merge_adjacent_spans(all_spans, y_threshold=3):
    """Merge spans that are on the same line and likely part of the same text element."""
    if not all_spans:
        return all_spans

    # Sort spans by page and vertical position
    sorted_spans = sorted(all_spans, key=lambda x: (x['page_num'], x.get('relative_y_position', 0)))
    
    merged_spans = []
    current_group = [sorted_spans[0]]

    def is_heading_like(span):
        text = span.get("text", "").strip()
        word_count = len(text.split())
        return (
            text.endswith(":") or
            word_count <= 5 or
            span.get("is_bold", False) or
            span.get("is_centered", False) or
            span.get("font_size", 0) >= 1.1 * span.get("common_font_size", 12)
        )

    for span in sorted_spans[1:]:
        last_span = current_group[-1]

        # Check similarity
        same_page = span['page_num'] == last_span['page_num']
        similar_y = abs(span.get('relative_y_position', 0) - last_span.get('relative_y_position', 0)) < (y_threshold / 100)
        similar_font = abs(span.get('font_size', 12) - last_span.get('font_size', 12)) < 1
        close_horizontally = True  # Optional: add x-distance if needed

        # Prevent merging headings or standalones
        if same_page and similar_y and similar_font and close_horizontally:
            if is_heading_like(last_span) or is_heading_like(span):
                # Don't merge if either looks like a heading
                merged_spans.extend(current_group)
                current_group = [span]
            else:
                current_group.append(span)
        else:
            if len(current_group) > 1:
                merged_span = merge_span_group(current_group)
                merged_spans.append(merged_span)
            else:
                merged_spans.extend(current_group)
            current_group = [span]

    # Final group
    if len(current_group) > 1:
        merged_span = merge_span_group(current_group)
        merged_spans.append(merged_span)
    else:
        merged_spans.extend(current_group)

    return merged_spans

def merge_span_group(span_group):
    """Merge a group of spans into a single span."""
    if not span_group:
        return None
    
    # Use the first span as base
    merged_span = span_group[0].copy()
    
    # Merge text with proper spacing
    texts = [s['text'].strip() for s in span_group if s['text'].strip()]
    merged_text = ' '.join(texts)
    
    # Clean up extra spaces
    merged_text = ' '.join(merged_text.split())
    merged_span['text'] = merged_text
    
    # Update text length
    merged_span['text_length'] = len(merged_text)
    
    # Take average of numeric features
    numeric_features = ['font_size', 'relative_font_size', 'font_size_percentile', 'relative_y_position']
    for feature in numeric_features:
        if feature in merged_span:
            values = [s.get(feature, merged_span[feature]) for s in span_group]
            merged_span[feature] = sum(values) / len(values)
    
    # Take max of boolean features (if any span has the property, merged span has it)
    boolean_features = ['is_larger_font', 'is_bold', 'is_italic', 'is_different_font', 
                       'is_all_caps', 'is_centered', 'is_left_aligned', 'is_top_of_page']
    for feature in boolean_features:
        if feature in merged_span:
            values = [s.get(feature, 0) for s in span_group]
            merged_span[feature] = max(values)
    
    return merged_span

def advanced_text_matching(text1, text2, threshold=0.6):
    """Advanced text matching using multiple strategies."""
    # Strategy 1: Direct similarity
    ratio1 = difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    # Strategy 2: Word-based similarity
    words1 = set(word_tokenize(text1.lower()))
    words2 = set(word_tokenize(text2.lower()))
    
    if words1 and words2:
        word_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
    else:
        word_overlap = 0
    
    # Strategy 3: Substring matching
    substring_ratio = 0
    if len(text1) > 3 and len(text2) > 3:
        # Check if one is a substring of the other
        if text1.lower() in text2.lower() or text2.lower() in text1.lower():
            substring_ratio = 0.8
    
    # Strategy 4: Remove common prefixes/suffixes and match core content
    # Remove numbering from both texts
    core_text1 = re.sub(r'^\s*(\d+(\.\d+)*|[A-Z]|[IVX]+)\.?\s+', '', text1).strip()
    core_text2 = re.sub(r'^\s*(\d+(\.\d+)*|[A-Z]|[IVX]+)\.?\s+', '', text2).strip()
    
    core_ratio = difflib.SequenceMatcher(None, core_text1.lower(), core_text2.lower()).ratio()
    
    # Strategy 5: Handle partial word matches (e.g., "ATERIAL" should match "Material Required")
    partial_ratio = 0
    if len(text1) >= 4 and len(text2) >= 4:
        # Check if shorter text is contained in longer text
        shorter, longer = (text1, text2) if len(text1) < len(text2) else (text2, text1)
        if shorter.lower() in longer.lower():
            partial_ratio = 0.85
    
    # Strategy 6: Fuzzy matching for common heading words
    common_words1 = set(['objective', 'material', 'method', 'construction', 'demonstration', 'observation', 'application'])
    text1_words = set(word_tokenize(text1.lower()))
    text2_words = set(word_tokenize(text2.lower()))
    
    heading_word_match = 0
    if text1_words.intersection(common_words1) and text2_words.intersection(common_words1):
        if text1_words.intersection(text2_words):
            heading_word_match = 0.9
    
    # Combined score
    final_score = max(ratio1, word_overlap, substring_ratio, core_ratio, partial_ratio, heading_word_match)
    
    return final_score >= threshold, final_score