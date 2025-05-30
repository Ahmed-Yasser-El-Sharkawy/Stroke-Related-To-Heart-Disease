from flask import Flask, request, jsonify



app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "✅ Sahha chatbot Health is Running", 200





if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)