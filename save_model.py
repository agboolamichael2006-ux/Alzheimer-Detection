import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# temporary model
X = np.array([
[74],
[55],
[73],
[60]
])

y = [
"NonDemented",
"NonDemented",
"Demented",
"Demented"
]

model = RandomForestClassifier()

model.fit(
X,
y
)

joblib.dump(
model,
"models/alzheimer_model.pkl"
)

print(
"MODEL SAVED"
)