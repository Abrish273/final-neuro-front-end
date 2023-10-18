import pandas as pd
import joblib
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
import google.generativeai as palm
import os
from dotenv import load_dotenv




# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
palm_api_key = os.getenv("PALM_API_KEY")

# Configure with your API key
palm.configure(api_key=palm_api_key)

# Retrieve and select a generative AI model that supports 'generateText'
models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
if models:
    model = models[0].name
else:
    model = None

app = Flask(__name__)

# Load the pre-trained logistic regression model
logistic_regression = joblib.load('logistic_regression_model.pkl')

def predict_stroke_risk(input_data):
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("Input data should be a pandas DataFrame.")

    input_data['bmi'].fillna(input_data['bmi'].mean(), inplace=True)
    input_data['smoking_status'].fillna(input_data['smoking_status'].mode()[0], inplace=True)
    X_input_preprocessed = preprocessing_pipeline.transform(input_data)

    logistic_regression_prob = logistic_regression.predict_proba(X_input_preprocessed)[0][1]

    return {
        'Logistic Regression Probability': logistic_regression_prob
    }

def generate_stroke_advice(stroke_probability):
    prompt = f"My stroke risk probability is {stroke_probability:.2%}. What should I do?"
    response = palm.generate_text(
        model=model,
        prompt=prompt,
        max_output_tokens=800,
    ).result
    return response
@app.route('/predict_stroke_risk', methods=['POST'])
def predict_stroke():
    try:
        input_data = request.get_json()

        if not input_data:
            return jsonify({"error": "No input data provided"}), 400

        input_df = pd.DataFrame(input_data['data'])
        prediction_result = predict_stroke_risk(input_df)

        # Interpret the probability
        probability = prediction_result['Logistic Regression Probability']
        interpretation = interpret_probability(probability)

        # Generate advice based on the data using PALM generative text
        advice = generate_advice(input_df, interpretation)

        response = {
            "Logistic Regression Probability": probability,
            "Interpretation": interpretation,
            "Advice": advice
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def interpret_probability(probability):
    if probability < 0.2:
        return "Low risk of stroke."
    elif 0.2 <= probability < 0.5:
        return "Moderate risk of stroke."
    elif 0.5 <= probability < 0.8:
        return "High risk of stroke. Please consult a healthcare professional."
    else:
        return "Very high risk of stroke. Urgently consult a healthcare professional."

def generate_advice(input_data, interpretation):
    # Use PALM generative text to generate advice based on the interpretation and input data
    prompt = f"My stroke risk probability is {interpretation}. What should I do given my data: {input_data.to_dict(orient='records')[0]}?"
    
    response = palm.generate_text(
        model=model,
        prompt=prompt,
        max_output_tokens=800,
    ).result
    
    return response


@app.route('/get_stroke_recommendations', methods=['POST'])
def get_stroke_recommendations():
    if request.method == 'POST':
        # Extract data from the request
        data = request.get_json()

        exposure_percent = data.get('exposure_percent', 40)
        weight = data.get('weight', 150)
        height = data.get('height', 1.70)
        history_of_stroke = data.get('history_of_stroke', "yes")
        family_history_of_stroke = data.get('family_history_of_stroke', "yes")
        physical_activity_level = data.get('physical_activity_level', "sedentary")
        diet = data.get('diet', "balanced")
        systolic_blood_pressure = data.get('systolic_blood_pressure', 50)
        diastolic_blood_pressure = data.get('diastolic_blood_pressure', 60)

        # Extract user data, handling missing or empty data['data']
        user_data = data.get('data', [{}])[0]

        # Extract specific user data fields for the prompt
        age = user_data.get('age', 'N/A')
        hypertension = user_data.get('hypertension', 'N/A')
        heart_disease = user_data.get('heart_disease', 'N/A')
        ever_married = user_data.get('ever_married', 'N/A')
        work_type = user_data.get('work_type', 'N/A')
        residence_type = user_data.get('Residence_type', 'N/A')
        avg_glucose_level = user_data.get('avg_glucose_level', 'N/A')
        bmi = user_data.get('bmi', 'N/A')
        smoking_status = user_data.get('smoking_status', 'N/A')
        gender = user_data.get('gender', 'N/A')

        # Construct the prompt with user data
        medical_prompt =  f"""
As an expert in stroke disease prevention, you play a crucial role in advising and developing
personalized diet and exercise plans for patients based on their unique profiles. Your insights 
are backed by extensive data analysis and a powerful model that calculates the risk of stroke.

User Profile:
Weight: {weight} kg
Height: {height} meters
Age: {age} years
Risk of a Stroke: {exposure_percent}%
History of Stroke: {history_of_stroke}
Family History of Stroke: {family_history_of_stroke}
Physical Activity Level: {physical_activity_level}
Diet: {diet}
Systolic Blood Pressure: {systolic_blood_pressure} mmHg
Diastolic Blood Pressure: {diastolic_blood_pressure} mmHg
Hypertension: {hypertension}
Heart Disease: {heart_disease}
Ever Married: {ever_married}
Work Type: {work_type}
Residence Type: {residence_type}
Average Glucose Level: {avg_glucose_level} mg/dL
Gender: {gender}
"""

        response = palm.generate_text(
            model=model,
            prompt=medical_prompt,
            max_output_tokens=800,
        ).result

        return jsonify({'recommendations': response})


    
# Initialize a dictionary to store previous chats
previous_chats = {}

@app.route('/medical', methods=['POST'])
def medical_question():
    try:
        input_data = request.get_json()

        if not input_data:
            return jsonify({"error": "No input data provided"}), 400

        question = input_data.get('question', '')

        if not model:
            return jsonify({"error": "No suitable AI model found"}), 500

        # Check if the question is a greeting or unrelated to health
        if "greeting" in question.lower() or "who are you" in question.lower():
            response = "NuroGen is a health assistant and can only answer health-related questions."
        else:
            # Check if there are previous chats related to this question
            related_chat = previous_chats.get(question.lower())
            if related_chat:
                response = related_chat
            else:
                medical_prompt = f"""
         You are  NuroGen a health assistant for patients, especially on stroke.
         You will be given a question below, and you are not allowed to answer a question that is not related to health and medicine
         if the question is greeting you are alowed to answer 
         if you are asked who you are or what you say that you are  NuroGen a health assistant
         just tell the user that you cannot answer a question not related to health . 
         If the question is related to health, give a response.
         The question:{question}
        """
                response = palm.generate_text(
                    model=model,
                    prompt=medical_prompt,
                    max_output_tokens=800,
                ).result

                # Store the response in the previous chats dictionary
                previous_chats[question.lower()] = response

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
    # Load the dataset and perform preprocessing
    data = pd.read_csv("healthcare-dataset-stroke-data.csv")

    data['bmi'].fillna(data['bmi'].mean(), inplace=True)
    data['smoking_status'].fillna(data['smoking_status'].mode()[0], inplace=True)

    X = data.drop(['id', 'stroke'], axis=1)
    y = data['stroke']

    numeric_features = ['age', 'avg_glucose_level', 'bmi']
    categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)],
        remainder='passthrough')  

    preprocessing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    X_preprocessed = preprocessing_pipeline.fit_transform(X)

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    logistic_regression = LogisticRegression(class_weight='balanced', random_state=42)
    logistic_regression.fit(X_train, y_train)

    joblib.dump(logistic_regression, 'logistic_regression_model.pkl')

    metrics = ['precision', 'recall', 'roc_auc']

    print("Model: Logistic Regression")
    y_pred = logistic_regression.predict(X_test)
    y_proba = logistic_regression.predict_proba(X_test)[:, 1]

    for metric in metrics:
        if metric == 'roc_auc':
            score = roc_auc_score(y_test, y_proba)
        else:
            precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred)
            if metric == 'precision':
                score = precision[1]
            elif metric == 'recall':
                score = recall[1]
        print(f"{metric.capitalize()}: {score:.2f}")

    app.run(debug=True, port=4000)   # Change the port to 4000
# ##############################second level#############################

# import pandas as pd
# import joblib
# from flask import Flask, request, jsonify
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
# from sklearn.linear_model import LogisticRegression
# import google.generativeai as palm
# import os
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # Get the API key from the environment variable
# palm_api_key = os.getenv("PALM_API_KEY")

# # Configure with your API key
# palm.configure(api_key=palm_api_key)

# # Retrieve and select a generative AI model that supports 'generateText'
# models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
# model = models[0].name

# app = Flask(__name__)

# # Load the pre-trained logistic regression model
# logistic_regression = joblib.load('model/logistic_regression_model.pkl')

# def predict_stroke_risk(input_data):
#     if not isinstance(input_data, pd.DataFrame):
#         raise ValueError("Input data should be a pandas DataFrame.")

#     input_data['bmi'].fillna(input_data['bmi'].mean(), inplace=True)
#     input_data['smoking_status'].fillna(input_data['smoking_status'].mode()[0], inplace=True)
#     X_input_preprocessed = preprocessing_pipeline.transform(input_data)

#     logistic_regression_prob = logistic_regression.predict_proba(X_input_preprocessed)[0][1]

#     return {
#         'Logistic Regression Probability': logistic_regression_prob
#     }

# def generate_stroke_advice(stroke_probability):
#     prompt = f"My stroke risk probability is {stroke_probability:.2%}. What should I do?"
#     response = palm.generate_text(
#         model=model,
#         prompt=prompt,
#         max_output_tokens=800,
#     ).result
#     return response
# @app.route('/predict_stroke_risk', methods=['POST'])
# def predict_stroke():
#     try:
#         input_data = request.get_json()

#         if not input_data:
#             return jsonify({"error": "No input data provided"}), 400

#         input_df = pd.DataFrame(input_data['data'])
#         prediction_result = predict_stroke_risk(input_df)

#         # Interpret the probability
#         probability = prediction_result['Logistic Regression Probability']
#         interpretation = interpret_probability(probability)

#         # Generate advice based on the data using PALM generative text
#         advice = generate_advice(input_df, interpretation)

#         response = {
#             "Logistic Regression Probability": probability,
#             "Interpretation": interpretation,
#             "Advice": advice
#         }

#         return jsonify(response)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# def interpret_probability(probability):
#     if probability < 0.2:
#         return "Low risk of stroke."
#     elif 0.2 <= probability < 0.5:
#         return "Moderate risk of stroke."
#     elif 0.5 <= probability < 0.8:
#         return "High risk of stroke. Please consult a healthcare professional."
#     else:
#         return "Very high risk of stroke. Urgently consult a healthcare professional."

# def generate_advice(input_data, interpretation):
#     # Use PALM generative text to generate advice based on the interpretation and input data
#     prompt = f"My stroke risk probability is {interpretation}. What should I do given my data: {input_data.to_dict(orient='records')[0]}?"
    
#     response = palm.generate_text(
#         model=model,
#         prompt=prompt,
#         max_output_tokens=800,
#     ).result
    
#     return response


# # Define the path to the .txt file containing medical keywords
# keywords_file_path = os.path.join('model', 'medical_keywords.txt')


# # Read medical keywords from a .txt file
# def read_medical_keywords_from_file(file_path):
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             keywords = [line.strip() for line in file]
#         return keywords
#     except Exception as e:
#         print(f"Error reading file '{file_path}': {str(e)}")
#         return []


# # Specify the path to the .txt file containing medical keywords
# # keywords_file_path = 'medical_keywords.txt'

# # Read medical keywords from the file
# medical_keywords = read_medical_keywords_from_file(keywords_file_path)

# @app.route('/medical', methods=['POST'])
# def medical_question():
#     try:
#         input_data = request.get_json()

#         if not input_data:
#             return jsonify({"error": "No input data provided"}), 400

#         question = input_data.get('question', '')

#         # Check if the question contains any medical-related keywords
#         if any(keyword in question.lower() for keyword in medical_keywords):
#             # Use the PALM generative AI model to generate a response
#             response = palm.generate_text(
#                 model=model,
#                 prompt=question,
#                 max_output_tokens=800,
#             ).result
#         else:
#             # Inform the user that only medical-related questions are answered
#             response = "I can only answer medical-related questions. Please ask a medical-related question."

#         return jsonify({"response": response})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# if __name__ == '__main__':
#     # Load the dataset and perform preprocessing
#     data = pd.read_csv("data/healthcare-dataset-stroke-data.csv")

#     data['bmi'].fillna(data['bmi'].mean(), inplace=True)
#     data['smoking_status'].fillna(data['smoking_status'].mode()[0], inplace=True)

#     X = data.drop(['id', 'stroke'], axis=1)
#     y = data['stroke']

#     numeric_features = ['age', 'avg_glucose_level', 'bmi']
#     categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

#     numeric_transformer = Pipeline(steps=[
#         ('scaler', StandardScaler())])

#     categorical_transformer = Pipeline(steps=[
#         ('onehot', OneHotEncoder(handle_unknown='ignore'))])

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numeric_transformer, numeric_features),
#             ('cat', categorical_transformer, categorical_features)],
#         remainder='passthrough')  

#     preprocessing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

#     X_preprocessed = preprocessing_pipeline.fit_transform(X)

#     smote = SMOTE(sampling_strategy='auto', random_state=42)
#     X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)

#     X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

#     logistic_regression = LogisticRegression(class_weight='balanced', random_state=42)
#     logistic_regression.fit(X_train, y_train)

#     joblib.dump(logistic_regression, 'logistic_regression_model.pkl')

#     metrics = ['precision', 'recall', 'roc_auc']

#     print("Model: Logistic Regression")
#     y_pred = logistic_regression.predict(X_test)
#     y_proba = logistic_regression.predict_proba(X_test)[:, 1]

#     for metric in metrics:
#         if metric == 'roc_auc':
#             score = roc_auc_score(y_test, y_proba)
#         else:
#             precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred)
#             if metric == 'precision':
#                 score = precision[1]
#             elif metric == 'recall':
#                 score = recall[1]
#         print(f"{metric.capitalize()}: {score:.2f}")

#     app.run(debug=True, port=4000)


