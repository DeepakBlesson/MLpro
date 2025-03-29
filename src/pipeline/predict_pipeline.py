# import sys
# import pandas as pd
# from src.exception import CustomException
# from src.utils import load_object

# class PredictPipeline:
#     def __init__(self):
#         pass
#     def predict(self,features):
#         try:
#             model_path=("artifacts","model.pkl")
#             preprocessor_path=("artifacts","preprocessor.pkl")
#             model=load_object(file_path=model_path)
#             preprocessor=load_object(file_path=preprocessor_path)
#             data_scaled=preprocessor.transform(features)
#             preds=model.predict(data_scaled)
#             return preds
#         except Exception as e:
#             raise CustomException(e,sys) 
# #Customdata is used for mapping inputs from html to backend
# class CustomData:
#     def __init__(self,
#                  gender:str,
#                  race_ethnicity: str,
#                  parental_level_of_education,
#                  lunch: str,
#                  test_preparation_course: str,
#                  reading_score: int,
#                  writing_score: int ):
#         self.gender=gender
#         self.race_ethnicity=race_ethnicity
#         self.parental_level_of_education=parental_level_of_education
#         self.lunch=lunch
#         self.test_preparation_course=test_preparation_course
#         self.writing_score=writing_score
#         self.reading_score=reading_score
    
#     def get_data_as_data_frame(self):
#         try:
#             custom_data_input_dict={
#                 "gender":[self.gender],
#                 "race_ethnicity":[self.race_ethnicity],
#                 "parental_level_of_education":[self.parental_level_of_education],
#                 "lunch":[self.lunch],
#                 "test_preparation_course":[self.test_preparation_course],
#                 "writing_score":[self.writing_score],
#                 "reading_score": [self.reading_score]
#             }
#             return pd.DataFrame(custom_data_input_dict)
        
#         except Exception as e:
#             raise CustomException(e, sys)
import sys
import os
import pandas as pd
from src.exception import CustomException
from sklearn.pipeline import Pipeline
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print(f"Checking model path: {model_path}")
            print(f"Checking preprocessor path: {preprocessor_path}")

            # Ensure model and preprocessor files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")

            print("Before Loading Model and Preprocessor...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("Model and Preprocessor Loaded Successfully!")

            # Debug input data
            print("Raw Input Features:\n", features)
            print("Feature Data Types:\n", features.dtypes)

            # Preprocess input data
            data_scaled = preprocessor.transform(features)

            # Debug preprocessed data
            print(f"Features Shape: {features.shape}")
            print(f"Scaled Features Shape: {data_scaled.shape}")
            print(f"Expected Model Input Shape: {getattr(model, 'n_features_in_', 'Unknown')}")

            # **TRY-CATCH BLOCK FOR PREDICTION**
            try:
                preds = model.predict(data_scaled)
                print("Prediction Successful! Output:", preds)
            except Exception as e:
                print(f"Prediction Error: {e}")
                raise CustomException(e, sys)

            return preds

        except Exception as e:
            print(f"Overall Pipeline Error: {e}")
            raise CustomException(e, sys)


class CustomData:
    def __init__(  self,
        gender: str,
        logical_reasoning: str,
        learning_style,
        stress_level: str,
        tutions: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender

        self.logical_reasoning = logical_reasoning

        self.learning_style  = learning_style

        self.stress_level = stress_level

        self.tutions = tutions

        try:
            self.reading_score = float(reading_score)
            self.writing_score = float(writing_score)
        except ValueError:
            raise ValueError("Reading and Writing scores must be numeric!")


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "logical reasoning": [self.logical_reasoning],
                "learning style": [self.learning_style],
                "stress level": [self.stress_level],
                "tutions": [self.tutions],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)