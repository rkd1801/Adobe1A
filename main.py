# main.py
import time
import os
from pdf_structure_analyzer import train_universal_model, predict_structure_universal

INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"
MODEL_PATH = "/app/universal_document_structure_xgboost_model.pkl"

def main():
    # Ensure input/output directories exist
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input directory '{INPUT_DIR}' not found.")
        return
    if not os.path.exists(OUTPUT_DIR):
        print(f"❌ Output directory '{OUTPUT_DIR}' not found. Creating it.")
        os.makedirs(OUTPUT_DIR)

    # Train model only if model.pkl doesn't exist
    if not os.path.exists(MODEL_PATH):
        print("🚀 Model not found. Starting training...")
        model_path = train_universal_model(input_dir=INPUT_DIR, model_output_path=MODEL_PATH)
        if model_path and os.path.exists(model_path):
            print(f"✅ Model trained and saved at {model_path}")
        else:
            print("❌ Model training failed.")
            return
    else:
        print(f"📦 Using existing model: {MODEL_PATH}")

    # Process each PDF in input directory
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".pdf"):
            input_path = os.path.join(INPUT_DIR, filename)
            print(f"\n🔍 Processing: {filename}")

            # Start timer
            start = time.time()

            # Predict structure
            result_json = predict_structure_universal(input_path, MODEL_PATH)

            # Save output
            output_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result_json if isinstance(result_json, str) else str(result_json))

            print(f"✅ Done: {output_filename} in {time.time() - start:.2f}s")

if __name__ == "__main__":
    main()
