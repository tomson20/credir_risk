from flask import Flask, request, jsonify
from prometheus_client import make_wsgi_app, Counter, Histogram
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import joblib
import pandas as pd

app = Flask(__name__)

# Prometheus metrics
predict_counter = Counter('model_predictions_total', 'Total number of predictions')
predict_latency = Histogram('model_prediction_latency_seconds', 'Prediction latency in seconds')

# WSGI middleware for Prometheus
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

# Load model
try:
    model = joblib.load('models/best_model.pkl')
except Exception as e:
    raise FileNotFoundError("Model file not found. Please ensure best_model.pkl is in the models folder.")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])

    @predict_latency.time()
    def get_prediction():
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0].tolist()
        return prediction, probability

    prediction, probability = get_prediction()
    predict_counter.inc()

    return jsonify({
        'prediction': int(prediction),
        'probability': {
            'non_default': probability[0],
            'default': probability[1]
        }
    })