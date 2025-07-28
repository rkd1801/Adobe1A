# pdf_structure_analyzer/model_training.py
import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import xgboost as xgb
from .utils import extract_training_data

def train_universal_model(data_dir):
    """Trains a universal model that can handle any PDF type."""
    pdf_dir = os.path.join(data_dir, "input")
    json_dir = os.path.join(data_dir, "output")

    all_data = []
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"❌ No PDFs found in {pdf_dir}")
        return None

    print(f"🔍 Found {len(pdf_files)} PDF files to process")

    for pdf_file in pdf_files:
        base_name = os.path.splitext(pdf_file)[0]
        pdf_path = os.path.join(pdf_dir, pdf_file)
        json_path = os.path.join(json_dir, f"{base_name}.json")

        if os.path.exists(json_path):
            print(f"\n📄 Processing: {pdf_file}")
            data = extract_training_data(pdf_path, json_path)
            all_data.extend(data)
        else:
            print(f"⚠️ Skipping {pdf_file}: No corresponding JSON file")

    if not all_data:
        print("❌ No training data generated")
        return None

    df = pd.DataFrame(all_data)
    print(f"\n✅ Training data generated: {len(df)} samples")
    print("\n📊 Label distribution:")
    print(df["label"].value_counts())

    label_counts = df['label'].value_counts()
    print("\n📊 All label counts before filtering:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")

    important_labels = {"TITLE", "H1", "H2", "H3", "H4"}
    labels_to_keep = list(set(label_counts[label_counts >= 2].index).union(important_labels))

    print(f"\n🎯 Labels to be used for training: {sorted(labels_to_keep)}")
    df_filtered = df[df['label'].isin(labels_to_keep)]

    if len(df_filtered["label"].unique()) <= 1:
        print("❌ Insufficient labeled data for training")
        return None

    feature_cols = [col for col in df_filtered.columns
                    if col not in ["text", "label", "font_name"] and
                    df_filtered[col].dtype in ['int64', 'float64', 'bool']]

    print(f"\n🔧 Using {len(feature_cols)} features for training")

    X = df_filtered[feature_cols].fillna(0)
    y = df_filtered["label"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"\n🏷️ Label encoding mapping:")
    for i, label in enumerate(le.classes_):
        print(f"  {i}: {label}")

    clf = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        tree_method='hist'
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred_labels = le.inverse_transform(y_pred)
    y_test_labels = le.inverse_transform(y_test)

    print(f"\n📊 XGBoost Model Performance:")
    print(classification_report(y_test_labels, y_pred_labels, zero_division=0))

    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.feature_importances_,
    }).sort_values('importance', ascending=False)

    print(f"\n🌟 Top 10 Most Important Features:")
    print(feature_importance.head(10))

    model_path = "universal_document_structure_xgboost_model.pkl"
    model_data = {
        'model': clf,
        'feature_columns': feature_cols,
        'label_encoder': le
    }
    joblib.dump(model_data, model_path)
    print(f"\n✅ Model saved to {model_path}")

    return model_path