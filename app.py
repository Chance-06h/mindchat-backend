import json
import requests
from flask_cors import CORS
from flask import Flask, request, jsonify
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)
CORS(app)

OLLAMA_API_URL = "http://127.0.0.1:11434/v1/chat/completions"
MODEL_NAME = "deepseek-r1:1.5b"

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"message": "Backend is running!"})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_input = data.get("input", "")
        if not user_input:
            logging.warning("用户输入为空")
            return jsonify({"error": "输入内容为空"}), 400

        # 【关键】移除字数限制，强化共情+疏导的结构要求
        messages = [
            {"role": "system", "content": "你是温暖的心理陪伴助手，用户分享情绪时，需先共情认可（例如：我理解你的感受/我能体会你的心情），再提供疏导建议。直接回复，无需任何思考过程。"},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": ""}
        ]

        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.7,  # 平衡多样性和逻辑性（取消字数限制后可适当提高）
            "max_tokens": 200,   # 设为200（轻量模型合理范围），允许更长回复
            "stream": False
        }

        response = requests.post(OLLAMA_API_URL, json=payload)
        assistant_reply = response.json()["choices"][0]["message"]["content"]
        
        # 【保留】过滤模型生成的思考过程（如 "</think>" 或多余格式）
        if "</think>" in assistant_reply:
            assistant_reply = assistant_reply.split("</think>")[-1].strip()
        # 额外过滤首尾无关符号（如换行、空格）
        assistant_reply = assistant_reply.strip().replace("\n", " ")
        
        return jsonify({"reply": assistant_reply})
    
    except Exception as e:
        logging.error(f"聊天接口错误: {str(e)}")
        return jsonify({"error": f"服务异常，请稍后再试"}), 500

# 数据文件路径，使用原始字符串表示路径
DATA_FILE = r"D:\心言伴\backend\data\emotion.json"

# 保存情绪数据的接口
@app.route("/save_emotion", methods=["POST"])
def save_emotion():
    try:
        data = request.json
        # 验证必要字段
        if "date" not in data or "score" not in data or "chat_history" not in data:
            logging.warning("保存情绪数据时缺少必要字段")
            return jsonify({"error": "缺少必要字段，需要date、score和chat_history"}), 400
        date = data["date"]
        score = data["score"]
        diary = data.get("diary", "")
        chat_history = data["chat_history"]

        # 验证分数范围
        if not isinstance(score, int) or score < 1 or score > 10:
            logging.warning("情绪分数不在有效范围内")
            return jsonify({"error": "情绪分数必须是1 - 10之间的整数"}), 400

        # 读取现有的情绪数据
        try:
            with open(DATA_FILE, "r", encoding='utf-8') as f:
                emotions = json.load(f)
        except FileNotFoundError:
            emotions = []

        # 添加新的情绪记录
        new_record = {
            "date": date,
            "score": score,
            "diary": diary,
            "chat_history": chat_history
        }
        emotions.append(new_record)

        # 将更新后的数据写回文件
        with open(DATA_FILE, "w", encoding='utf-8') as f:
            json.dump(emotions, f, indent=2)

        return jsonify({"status": "success"})
    except Exception as e:
        logging.error(f"保存情绪数据失败: {str(e)}")
        return jsonify({"error": f"保存情绪数据失败: {str(e)}"}), 500

# 获取情绪数据的接口
@app.route("/get_emotions", methods=["GET"])
def get_emotions():
    try:
        with open(DATA_FILE, "r", encoding='utf-8') as f:
            emotions = json.load(f)
        return jsonify(emotions)
    except FileNotFoundError:
        logging.info("情绪数据文件未找到，返回空列表")
        return jsonify([]), 200
    except Exception as e:
        logging.error(f"获取情绪数据时出现未知错误: {str(e)}")
        return jsonify({"error": f"获取情绪数据时出现未知错误: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)    