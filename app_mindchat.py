import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# ------------------- 模型配置 -------------------
# 模型路径（指向包含safetensors文件的目录）
MODEL_PATH = r"D:\心言伴\MindChat-Qwen-1_8B"  
# 加载模型和分词器（使用FP16混合精度减少内存占用）
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# 设置聊天模板
tokenizer.chat_template = "{|SystemBegin|}{{system}}{|SystemEnd|}{% for message in messages %}{% if message['role'] == 'user' %}{|UserBegin|}{{message['content']}}{|UserEnd|}{% elif message['role'] == 'assistant' %}{|AssistantBegin|}{{message['content']}}{|AssistantEnd|}{% endif %}{% endfor %}"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # 使用半精度降低内存消耗
    device_map="cpu"  # 根据设备调整，若有GPU可改为"cuda"
)
model.eval()  # 切换为评估模式

# ------------------- 原有接口 -------------------
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

        # 构建对话格式（Qwen支持标准ChatML格式）
        messages = [
            {"role": "system", "content": "你是温暖的心理陪伴助手，用户分享情绪时，需先共情认可（例如：我理解你的感受/我能体会你的心情），再提供疏导建议。直接回复，无需任何思考过程。"},
            {"role": "user", "content": user_input}
        ]

        # 将ChatML格式转换为模型输入文本（Qwen支持直接传入messages）
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
        
        # 生成参数
        generation_params = {
            "max_new_tokens": 200,       # 控制回复长度
            "temperature": 0.7,          # 控制创造性
            "top_p": 0.9,                # 核采样参数
            "do_sample": True,           # 启用随机采样
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id
        }

        # 生成回复
        with torch.no_grad():
            output_ids = model.generate(input_ids, **generation_params)
        
        # 解码回复并处理格式
        assistant_reply = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        assistant_reply = assistant_reply.strip().replace("\n", " ")  # 清理空白字符

        return jsonify({"reply": assistant_reply})
    
    except Exception as e:
        logging.error(f"聊天接口错误: {str(e)}")
        return jsonify({"error": f"服务异常，请稍后再试"}), 500

# ------------------- 情绪数据接口（保持不变） -------------------
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