# pdf_structure_analyzer/prediction.py
import os
import json
import joblib
import pandas as pd
import numpy as np
import fitz
from difflib import get_close_matches
from .constants import COMMON_HEADINGS, NON_HEADING_KEYWORDS
from .feature_extraction import (
    get_document_statistics,
    get_comprehensive_span_features,
    merge_adjacent_spans,
    clean_text
)

def predict_structure_universal(pdf_path, model_path):
    """Universal structure prediction for any PDF, with H4 support and structured output."""
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return

    print(f"\n🔮 Analyzing PDF: {os.path.basename(pdf_path)}")

    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_columns = model_data['feature_columns']
    label_encoder = model_data.get('label_encoder')

    doc = fitz.open(pdf_path)
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
                        all_spans.append(features)

    if not all_spans:
        print("❌ No text features extracted")
        return

    print(f"📊 Before merging: {len(all_spans)} spans")
    all_spans = merge_adjacent_spans(all_spans)
    print(f"📊 After merging: {len(all_spans)} spans")

    X_df = pd.DataFrame(all_spans)
    X_inference = X_df[feature_columns].fillna(0)

    predictions_numeric = model.predict(X_inference)
    probabilities = model.predict_proba(X_inference)

    if label_encoder:
        predictions = label_encoder.inverse_transform(predictions_numeric)
        print(f"🏷️ Using label encoder to convert predictions")
    else:
        predictions = predictions_numeric
        print(f"⚠️ No label encoder found - using raw predictions")

    X_df['predicted_label'] = predictions
    X_df['confidence'] = np.max(probabilities, axis=1)

    X_df_sorted = X_df.sort_values(['page_num', 'relative_y_position'])
    X_df_sorted = X_df_sorted.loc[:, ~X_df_sorted.columns.duplicated()]

    debug_output_path = "debug_predictions_with_features.json"
    debug_cols = ['text', 'page_num', 'predicted_label', 'confidence'] + feature_columns
    debug_cols = list(dict.fromkeys(debug_cols))
    X_df_sorted[debug_cols].to_json(debug_output_path, orient='records', indent=2)
    print(f"🐛 Full prediction feature log saved to {debug_output_path}")

    output = {"title": "", "outline": [], "other_elements": {}}

    # --- Title detection ---
    title_candidates = X_df_sorted[X_df_sorted['predicted_label'] == 'TITLE']
    if not title_candidates.empty:
        best_title = title_candidates.loc[title_candidates['confidence'].idxmax()]
        output["title"] = best_title['text']

    if not output["title"]:
        candidates = X_df_sorted[
            (X_df_sorted['page_num'] == 0) &
            (X_df_sorted['relative_y_position'] < 0.25) &
            (X_df_sorted['font_size'] > doc_stats['common_font_size']) &
            (X_df_sorted['text_length'] > 15)
        ]
        if not candidates.empty:
            top_title = candidates.sort_values(
                ['relative_font_size', 'is_bold', 'is_centered'],
                ascending=[False, False, False]
            ).iloc[0]
            output["title"] = top_title['text']

    # --- Heading normalization based on font size ---
    def normalize_heading_level(level, font_size):
        ratio = font_size / doc_stats["common_font_size"]
        if ratio >= 1.7:
            return "H1"
        elif ratio >= 1.4:
            return "H2"
        elif ratio >= 1.2:
            return "H3"
        elif ratio >= 1.05 and font_size >= 11:
            return "H4"
        else:
            return "CONTENT"

    # --- Outline detection ---
    heading_rows = X_df_sorted[
        (X_df_sorted['predicted_label'].str.match(r'^H\d$', na=False)) &
        (X_df_sorted['confidence'] > 0.45) &
        (X_df_sorted['text'].str.split().str.len() < 25)
    ].sort_values(['page_num', 'relative_y_position'])

    outline_ordered = []
    added_headings = []

    for _, row in heading_rows.iterrows():
        raw_text = row['text'].strip()
        raw_upper = raw_text.upper()

        # Fuzzy match fix
        for known in COMMON_HEADINGS:
            if get_close_matches(raw_upper, [known.upper()], n=1, cutoff=0.7):
                raw_text = known
                break

        # Trim long colon lines
        if ":" in raw_text and len(raw_text.split()) > 15:
            raw_text = raw_text.split(":")[0].strip() + ":"

        cleaned_text = raw_text.lower().rstrip(":").strip()
        if cleaned_text in NON_HEADING_KEYWORDS:
            continue
        if get_close_matches(cleaned_text, added_headings, n=1, cutoff=0.9):
            continue

        normalized_level = normalize_heading_level(row['predicted_label'], row['font_size'])

        outline_ordered.append({
            "level": normalized_level,
            "text": raw_text,
            "page": row['page_num'] + 1,
            "confidence": row['confidence']
        })
        added_headings.append(cleaned_text)

    output["outline"] = outline_ordered

    # --- Other elements ---
    other_elements = X_df_sorted[
        (~X_df_sorted['predicted_label'].str.startswith('H', na=False)) &
        (X_df_sorted['predicted_label'] != 'TITLE') &
        (X_df_sorted['predicted_label'] != 'CONTENT')
    ]

    for _, row in other_elements.iterrows():
        label = row['predicted_label']
        if label not in output["other_elements"]:
            output["other_elements"][label] = []
        output["other_elements"][label].append({
            "text": row['text'],
            "page": row['page_num'] + 1,
            "confidence": row['confidence']
        })

    output_path = "universal_prediction_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Analysis complete! Output saved to {output_path}")
    print(f"📋 Title: {output['title'][:50]}...")
    print(f"📑 Outline items: {len(output['outline'])}")
    print(f"🔍 Other elements: {sum(len(v) for v in output['other_elements'].values())}")

    pred_stats = X_df_sorted['predicted_label'].value_counts()
    print(f"\n📊 Prediction Statistics:")
    for label, count in pred_stats.items():
        avg_conf = X_df_sorted[X_df_sorted['predicted_label'] == label]['confidence'].mean()
        print(f"  {label}: {count} items (avg confidence: {avg_conf:.3f})")

    return output
              

