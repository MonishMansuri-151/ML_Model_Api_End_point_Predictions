import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
df = pd.read_csv("/Users/ss/Desktop/supervise_test.py/covid_toy - covid_toy (1).csv")

print(df.head())
print(df.dtypes)


# X and y
X = df.drop("has_covid", axis=1)
y = df["has_covid"]


# Categorical columns
categorical_columns = [
    "gender",
    "cough",
    "city"
]


# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# Complete pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train
model.fit(X_train, y_train)


# Test
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Save model
with open("covid_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved as covid_model.pkl")