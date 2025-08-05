from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort

# Initialize FastAPI app
app = FastAPI(title="ONNX Model Serving with FastAPI")

# Load the ONNX model
session = ort.InferenceSession("model.onnx")

# Define the input schema using Pydantic
class InputData(BaseModel):
    features: list[float]

@app.get("/")
def read_root():
    return {"message": "ONNX model is ready to serve!"}

@app.post("/predict")
def predict(data: InputData):
    try:
        # Prepare input for the ONNX model
        input_name = session.get_inputs()[0].name
        input_array = np.array(data.features, dtype=np.float32).reshape(1, -1)

        # Perform inference
        prediction = session.run(None, {input_name: input_array})

        return {"prediction": prediction[0].tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
