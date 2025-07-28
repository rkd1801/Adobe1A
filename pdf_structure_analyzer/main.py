# main.py
import time
import os
from pdf_structure_analyzer import train_universal_model, predict_structure_universal

if __name__ == '__main__':
    data_directory = "dataset"

    if not os.path.exists(data_directory):
        print(f"❌ Directory '{data_directory}' not found")
        print("Please create it with 'input' and 'output' subfolders")
    else:
        print("🚀 Starting Universal PDF Structure Analysis Training")

        # Train the universal model
        model_path = train_universal_model(data_directory)

        if model_path:
            print(f"\n🎯 Model training completed successfully!")

            # Test on a specific file
            test_pdf = os.path.join(data_directory, "input", "file05.pdf")
            if os.path.exists(test_pdf):
                print(f"\n🔍 Testing model on: {os.path.basename(test_pdf)}")

                # ⏱ Measure execution time
                start_time = time.time()
                predict_structure_universal(test_pdf, model_path)
                end_time = time.time()

                print(f"\n⏱ Total execution time: {end_time - start_time:.2f} seconds")
            else:
                print(f"\n⚠️ Test file not found: {test_pdf}")
                print("You can test the model on any PDF using:")
                print(f"predict_structure_universal('your_pdf_path.pdf', '{model_path}')")
        else:
            print("❌ Model training failed")