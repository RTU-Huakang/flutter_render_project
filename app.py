from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load trained model
model = joblib.load("fault_gbt_model.pkl")

# Predict API
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    try:
        # Fetch JSON data
        data = request.get_json()
        print("Received data:", data)  # Print sensor value

        # Convert to DataFrame
        input_data = pd.DataFrame([data])
        # Print DataFrame
        print("Input DataFrame:\n", input_data.to_string(index=False))

        # Prediction
        prediction = model.predict(input_data)
        print("Prediction result:", prediction)  # Print prediction result

        # Reture prediction result
        return jsonify({"prediction_level": int(prediction[0])})

    except Exception as e:
        print("Error:", e)  # Print Error
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    import logging
    log = logging.getLogger('werkzeug')
    # log.setLevel(logging.ERROR) 
    app.run(host="0.0.0.0", port=5000, debug=True)
