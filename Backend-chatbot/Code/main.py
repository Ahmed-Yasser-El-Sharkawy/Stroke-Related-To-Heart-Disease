from flask import Flask, request, jsonify
import torch
from unsloth import FastLanguageModel

max_seq_length = 4096
dtype = torch.float16
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Ahmed-El-Sharkawy/Stroke-medical-model-finetuned",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model)

# Define Medical prompt
Medical_prompt = """you are sahha chatbot ,You are a knowledgeable medical expert. Analyze the provided medical input and generate a comprehensive, informative response that addresses the patient's query or medical scenario.
### Input:
{input_text}
### Response:
"""

def generate_response(input_text):
    prompt = Medical_prompt.format(input_text=input_text)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        use_cache=True
    )

    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Extract only the part after "### Response:"
    if "### Response:" in decoded_output:
        response = decoded_output.split("### Response:")[-1].strip()
    else:
        response = decoded_output.strip()

    return response

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "✅ Sahha chatbot Health is Running", 200

@app.route('/Q&A', methods=['POST'])
def QandAnswer():
    try:
        user_input = request.json.get('input_text')
        if not user_input:
            return jsonify({'error': 'Missing input_text'}), 400
        
        response = generate_response(user_input)
        return jsonify({'Answers': response})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=8080, debug=True)