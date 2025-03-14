from flask import Flask, request, jsonify
import joblib
import pandas as pd

# 初始化 Flask
app = Flask(__name__)

# 加载训练好的 GBT 模型
model = joblib.load("fault_gbt_model.pkl")

# 预测 API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取 JSON 数据
        data = request.get_json()
        
        # 转换为 DataFrame
        input_data = pd.DataFrame([data])
        
        # 进行预测
        prediction = model.predict(input_data)
        
        # 返回预测结果
        return jsonify({"fault_level": int(prediction[0])})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# 运行 Flask 服务器（本地调试用）
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
