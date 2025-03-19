import translator
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

translator_istance = translator.Translator(
    pretrained_model_name = "Helsinki-NLP/opus-mt-it-de", 
    pretrained_download = True, 
    pretrained_local_dir_download = './pretrained_model', 
    
    dataset_training_csv = "training_dataset.csv", 
    dataset_evaluation_csv = "evaluation_dataset.csv", 
    dataset_csv_delimiter = "#", 
    dataset_source_csv_column_header = "source", 
    dataset_target_csv_column_header = "target", 
    use_metric_as_trainer_callback = True,

    optimizer_learning_rate = [1e-5, 5e-4],
    optimizer_per_device_train_batch_size = [6, 8, 32],
    optimizer_num_train_epochs = [4, 6],
    optimizer_weight_decay = [1e-4, 1e-2],

    fine_tuned_model_dir = "./fine_tuned_model",
    fine_tuning_learning_rate=8e-5,
    fine_tuning_per_device_train_batch_size=8,
    fine_tuning_per_device_eval_batch_size=8,
    fine_tuning_num_train_epochs=8,
    fine_tuning_weight_decay=0.01
)

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
