from flask import Flask, jsonify
from flask_cors import CORS
import json
import os

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 获取数据文件的绝对路径（避免部署后路径错误）
DATA_DIR = os.path.join(app.root_path, 'data')
EMOTION_JSON = os.path.join(DATA_DIR, 'emotion.json')

@app.route('/')
def home():
    return "心言伴后端服务运行中"

@app.route('/api/emotion')
def get_emotion_data():
    try:
        with open(EMOTION_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)