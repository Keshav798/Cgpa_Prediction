from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

# Attempt to load your trained model.
try:
    model = tf.keras.models.load_model("./DTU_CGPA_Prediction_2.0_MAE_0.060.keras")
except Exception as e:
    raise RuntimeError("Failed to load model: " + str(e))

app = FastAPI()

# Define the expected input JSON structure
class InputData(BaseModel):
    static_features: list[list[float]]
    sequence_features: list[list[list[float]]]
    extra_credit: list[float]

@app.post("/predict")
def predict(data: InputData):
    try:
        static_features_list = data.static_features
        sequence_features_list = data.sequence_features
        extra_credit_list = data.extra_credit

        # Preprocessing (example based on your code)
        seq_array = np.array(sequence_features_list)  # shape: (num_samples, max_history, 2)
        extra_credit_array = np.array(extra_credit_list).reshape(-1, 1)

        # Normalize admissionYear in static features
        year_min = 2021
        year_max = 2023
        static_features_list[0][0] = (static_features_list[0][0] - year_min) / (year_max - year_min)
        static_array = np.array(static_features_list)

        # Normalize sequence features
        credits_min = -1.0
        credits_max = 35.0
        seq_array_norm = np.copy(seq_array).astype(np.float64)
        valid_mask = ~np.all(seq_array == [-1.0, -1.0], axis=-1)
        seq_array_norm[:, :, 0] = np.where(valid_mask, seq_array_norm[:, :, 0] / 10.0, -1.0)
        seq_array_norm[:, :, 1] = np.where(
            valid_mask,
            (seq_array_norm[:, :, 1] - credits_min) / (credits_max - credits_min),
            -1.0
        )
        extra_credit_array_norm = (extra_credit_array - credits_min) / (credits_max - credits_min)

        X_seq = seq_array_norm
        X_static = static_array
        X_extra = extra_credit_array_norm

        y_pred = model.predict([X_seq, X_static, X_extra])
        y_pred = y_pred * 10

        return {"predicted_cgpa": float(y_pred[0][0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
