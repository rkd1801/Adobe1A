# pdf_structure_analyzer/utils.py
import fitz
import json
from .constants import COMMON_HEADINGS
from .feature_extraction import get_document_statistics,get_comprehensive_span_features,merge_adjacent_spans,extract_all_json_elements,advanced_text_matching

def extract_training_data(pdf_path, json_path):
    """Extracts features and ground-truth labels from a PDF using improved span matching."""
    try:
        doc = fitz.open(pdf_path)
        with open(json_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
    except Exception as e:
        print(f"Error opening {pdf_path} or {json_path}: {e}")
        return []

    doc_stats = get_document_statistics(doc)
    doc_stats["total_pages"] = len(doc)

    all_spans = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    features = get_comprehensive_span_features(span, page, doc_stats, page_num)
                    if features and len(features["text"].strip()) > 1:
                        features["label"] = "CONTENT"
                        all_spans.append(features)

    print(f"📊 Before merging: {len(all_spans)} spans")
    all_spans = merge_adjacent_spans(all_spans)
    print(f"📊 After merging: {len(all_spans)} spans")

    gt_elements = extract_all_json_elements(gt)
    matched_count = 0

    for gt_elem in gt_elements:
        gt_text = gt_elem["text"]
        gt_page = max(0, gt_elem["page"] - 1)
        gt_label = gt_elem["label"]

        best_match_idx = None
        best_score = 0.0

        for i, span in enumerate(all_spans):
            if span["label"] != "CONTENT":
                continue
            if abs(span["page_num"] - gt_page) > 1:
                continue

            is_match, score = advanced_text_matching(span["text"], gt_text)
            if is_match and score > best_score:
                best_match_idx = i
                best_score = score

        if best_match_idx is not None:
            all_spans[best_match_idx]["label"] = gt_label
            matched_count += 1

    print(f"📘 Matched {matched_count}/{len(gt_elements)} ground truth elements")

    return all_spans