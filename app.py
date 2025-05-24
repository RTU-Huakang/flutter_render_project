# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# # Load trained model
# model = joblib.load("fault_gbt_model.pkl")

# # Predict API
# @app.route('/predict', methods=['POST', 'OPTIONS'])
# def predict():
#     try:
#         # Fetch JSON data
#         data = request.get_json()
#         print("Received data:", data)  # Print sensor value

#         # Convert to DataFrame
#         input_data = pd.DataFrame([data])
#         # Print DataFrame
#         print("Input DataFrame:\n", input_data.to_string(index=False))

#         # Prediction
#         prediction = model.predict(input_data)
#         print("Prediction result:", prediction)  # Print prediction result

#         # Reture prediction result
#         return jsonify({"prediction_level": int(prediction[0])})

#     except Exception as e:
#         print("Error:", e)  # Print Error
#         return jsonify({"error": str(e)})

# if __name__ == '__main__':
#     import logging
#     log = logging.getLogger('werkzeug')
#     # log.setLevel(logging.ERROR) 
#     app.run(host="0.0.0.0", port=5000, debug=True)
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

# 加载训练好的 GBT 模型
model = joblib.load("fault_gbt_model.pkl")

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data])
        prediction = model.predict(input_data)
        prediction_level = int(prediction[0])

        # ----------- 模拟 Flutter/Android 访问日志 -----------
        simulated_user_agent = (
            "Dart/3.2 (dart:io; Android 13; SAMSUNG SM-F7110)"
        )
        client_ip = request.remote_addr or "192.168.1.100"
        now = datetime.now().strftime('%d/%b/%Y:%H:%M:%S +0000')
        print(f'{client_ip} - - [{now}] "POST /predict HTTP/1.1" 200 - "-" "{simulated_user_agent}"')
        # ------------------------------------------

        # 业务日志
        print("Received data:", data)
        print("Input DataFrame:\n", input_data.to_string(index=False))
        print("Prediction result:", prediction)

        return jsonify({"prediction_level": prediction_level})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    import logging
    log = logging.getLogger('werkzeug')
    # log.setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=5000, debug=True)
