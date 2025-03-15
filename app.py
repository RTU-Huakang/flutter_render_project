# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd

# # 初始化 Flask
# app = Flask(__name__)

# # 加载训练好的 GBT 模型
# model = joblib.load("fault_gbt_model.pkl")

# # 预测 API
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # 获取 JSON 数据
#         data = request.get_json()
        
#         # 转换为 DataFrame
#         input_data = pd.DataFrame([data])
        
#         # 进行预测
#         prediction = model.predict(input_data)
        
#         # 返回预测结果
#         return jsonify({"fault_level": int(prediction[0])})
    
#     except Exception as e:
#         return jsonify({"error": str(e)})

# # 运行 Flask 服务器（本地调试用）
# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000, debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
# 配置 CORS，允许所有源，允许 Content-Type 头
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "allow_headers": ["Content-Type"],
        "methods": ["GET", "POST", "OPTIONS"]
    }
})

# 加载模型
try:
    model = joblib.load('fault_gbt_model.pkl')
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    model = None

# 定义传感器名称和顺序
SENSOR_NAMES = [
    'Engine_load',
    'MAP',
    'Engine_RPM',
    'MAF',
    'Catalyst_temp',
    'Intake_air_temp',
    'Throttle_pos',
    'Coolant_temp'
]

# 数据验证函数
def validate_sensor_data(data):
    """验证传感器数据的完整性和范围"""
    for sensor in SENSOR_NAMES:
        if sensor not in data:
            return False, f"Missing sensor data: {sensor}"
        
        value = data[sensor]
        if not isinstance(value, (int, float)):
            return False, f"Invalid value type for {sensor}"
            
    return True, "Data valid"

# 数据预处理函数
def preprocess_data(data):
    """将传感器数据转换为模型输入格式"""
    try:
        # 按照定义的顺序创建特征数组
        features = np.array([[data[sensor] for sensor in SENSOR_NAMES]])
        return features
    except Exception as e:
        logging.error(f"Error in data preprocessing: {str(e)}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """预测端点"""
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        # 获取请求数据
        data = request.json
        logging.info(f"Received prediction request with data: {data}")

        # 验证数据
        is_valid, message = validate_sensor_data(data)
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': message
            }), 400

        # 预处理数据
        features = preprocess_data(data)
        if features is None:
            return jsonify({
                'status': 'error',
                'message': 'Error preprocessing data'
            }), 500

        # 进行预测
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]

        # 记录预测结果
        logging.info(f"Prediction: {prediction}, Probabilities: {prediction_proba}")

        # 返回预测结果
        response = jsonify({
            'status': 'success',
            'prediction_level': int(prediction),
            'confidence_scores': prediction_proba.tolist(),
            'timestamp': datetime.now().isoformat()
        })
        
        return response

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500

@app.route('/sensors/thresholds', methods=['GET'])
def get_sensor_thresholds():
    """获取传感器阈值端点"""
    thresholds = {
        'Engine_load': {
            'normal': (20, 80),
            'level1': [(0, 20), (80, 100)],
            'level2': [(0, 10), (90, 100)],
            'level3': [(0, 5), (95, 100)]
        },
        'MAP': {
            'normal': (20, 40),
            'level1': [(10, 20), (40, 50)],
            'level2': [(5, 10), (50, 60)],
            'level3': [(0, 5), (60, 100)]
        },
        'Engine_RPM': {
            'normal': (800, 4000),
            'level1': [(500, 800), (4000, 5000)],
            'level2': [(300, 500), (5000, 6000)],
            'level3': [(0, 300), (6000, 10000)]
        },
        'MAF': {
            'normal': (5, 30),
            'level1': [(0, 5), (30, 40)],
            'level2': [(0, 2), (40, 50)],
            'level3': [(0, 0), (50, 100)]
        },
        'Catalyst_temp': {
            'normal': (400, 700),
            'level1': [(300, 400), (700, 800)],
            'level2': [(200, 300), (800, 900)],
            'level3': [(0, 200), (900, 1000)]
        },
        'Intake_air_temp': {
            'normal': (-20, 50),
            'level1': [(-40, -20), (50, 60)],
            'level2': [(-60, -40), (60, 80)],
            'level3': [(-100, -60), (80, 100)]
        },
        'Throttle_pos': {
            'normal': (10, 90),
            'level1': [(0, 10), (90, 100)],
            'level2': [(0, 5), (95, 100)],
            'level3': [(0, 0), (100, 100)]
        },
        'Coolant_temp': {
            'normal': (75, 105),
            'level1': [(70, 75), (105, 110)],
            'level2': [(60, 70), (110, 120)],
            'level3': [(0, 60), (120, 150)]
        }
    }
    
    return jsonify(thresholds)

@app.after_request
def after_request(response):
    """添加 CORS 头到所有响应"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
