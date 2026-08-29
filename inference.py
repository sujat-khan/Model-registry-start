# inference

import mlflow.pyfunc
import pandas as pd

# Define input features matching the model signature
columns = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]
data = pd.DataFrame([[1, 85, 66, 29, 0, 26.6, 0.351, 31]], columns=columns)

model_name = "diabetes-rf"
model_version = 3

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

print(model.predict(data))

