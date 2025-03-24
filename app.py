import translator
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

translator_istance = translator.Translator(fine_tuned_model_dir = "./fine_tuned_model")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    prompt = data.get("prompt", "")
    response_text = translator_istance.executeInference(prompt)
    response_text = response_text[0].upper() + response_text[1:]
    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)
