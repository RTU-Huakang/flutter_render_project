from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 加载训练好的 GBT 模型
model = joblib.load("fault_gbt_model.pkl")

# 预测 API
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    try:
        # 获取 JSON 数据
        data = request.get_json()
        print("Received data:", data)  # 打印收到的传感器数据

        # 转换为 DataFrame
        input_data = pd.DataFrame([data])
        # 展开打印 DataFrame
        print("Input DataFrame:\n", input_data.to_string(index=False))

        # 进行预测
        prediction = model.predict(input_data)
        print("Prediction result:", prediction)  # 打印预测结果

        # 返回预测结果
        return jsonify({"prediction_level": int(prediction[0])})

    except Exception as e:
        print("Error:", e)  # 打印异常信息
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    import logging
    log = logging.getLogger('werkzeug')
    # log.setLevel(logging.ERROR)  # 只显示错误，不显示普通访问日志
    app.run(host="0.0.0.0", port=5000, debug=True)
