from fastapi import FastAPI
import gradio as gr
import numpy as np
import pickle
from sklearn.svm import SVC

# Load the trained model
try:
    svc = pickle.load(open('models/svc.pickle', 'rb'))
except FileNotFoundError:
    print("Error: 'model/svc.pickle' not found. Make sure the model file exists.")
    exit()

app = FastAPI()

def predict_quality(size: float, weight: float, sweetness: float, crunchiness: float, juiciness: float, ripeness: float, acidity: float):
    """Predicts the quality of an apple based on the provided features."""
    input_features = np.array([[size, weight, sweetness, crunchiness, juiciness, ripeness, acidity]])
    prediction = svc.predict(input_features)[0]
    return "Good" if prediction > 0.5 else "Bad"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_quality,
    inputs=[
        gr.Number(label="Size"),
        gr.Number(label="Weight"),
        gr.Number(label="Sweetness"),
        gr.Number(label="Crunchiness"),
        gr.Number(label="Juiciness"),
        gr.Number(label="Ripeness"),
        gr.Number(label="Acidity")
    ],
    outputs=gr.Textbox(label="Predicted Quality"),
    title="Apple Quality Prediction",
    description="Enter the apple features to predict its quality (Good or Bad)."
)

# Mount the Gradio app to the FastAPI app
gradio_app = gr.mount_gradio_app(app, iface, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)