from src.utils import load_object
import os

model_path = os.path.join("artifacts", "model.pkl")
preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

model = load_object(model_path)
preprocessor = load_object(preprocessor_path)

print("Model and Preprocessor Loaded Successfully!")
