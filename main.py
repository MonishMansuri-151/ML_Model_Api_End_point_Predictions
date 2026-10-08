
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()

# Load trained model
with open("covid_model.pkl", "rb") as file:
    model = pickle.load(file)


class CovidInput(BaseModel):
    age: int
    gender: str
    cough: str
    fever: float
    city: str


@app.get("/")
def home():
    return {"message": "COVID Prediction API is running"}


@app.post("/predict")
def predict(data: CovidInput):

    input_data = pd.DataFrame([{
        "age": data.age,
        "gender": data.gender,
        "cough": data.cough,
        "fever": data.fever,
        "city": data.city
    }])

    prediction = model.predict(input_data)[0]

    return {
        "prediction": prediction
    }