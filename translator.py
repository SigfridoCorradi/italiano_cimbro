from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments 
from transformers import TrainerCallback 
from datasets import load_dataset 
import optuna 
import os
import torch 
import evaluate 

'''
This is a custom callback class that:
 - Tracks the BLEU score (a standard metric for evaluating translation quality) during training
 - Calculates BLEU scores at the end of each epoch
 - Stop training when the BLEU score stops improving
'''
class BLEUCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, model, dataset_source_csv_column_header = "source", dataset_target_csv_column_header = "target"):
        self.best_bleu = 0.0
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.dataset_source_csv_column_header = dataset_source_csv_column_header
        self.dataset_target_csv_column_header = dataset_target_csv_column_header

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    '''
    Method that runs the BLEU metric at the end of each training epoch.
    '''
    def on_epoch_end(self, args, state, control, **kwargs):
        bleu_metric = evaluate.load("bleu")
        predictions = []
        references = []

        for sample in self.eval_dataset:
            input_text = sample[self.dataset_source_csv_column_header]
            reference_text = sample[self.dataset_target_csv_column_header]

            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
            inputs = {key: val.to(self.device) for key, val in inputs.items()} 

            with torch.no_grad(): #Disabled gradient calculation to save memory during inference.
              output_ids = self.model.generate(**inputs)

            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            predictions.append(generated_text)
            references.append([reference_text])

        bleu_score = bleu_metric.compute(predictions=predictions, references=references)["bleu"]
        print(f"New BLEU score [epoch {state.epoch}]: {bleu_score:.4f}")

        if bleu_score > 0 and bleu_score <= self.best_bleu:
            print("BLEU: Should stop the training!")
            control.should_training_stop = True
        else:
            self.best_bleu = bleu_score

#This is the main class that handles the entire translation pipeline
class Translator:
    def __init__(self, pretrained_model_name = "Helsinki-NLP/opus-mt-it-de", pretrained_download = True, pretrained_local_dir_download = './pretrained_model', dataset_training_csv = "training_dataset.csv", dataset_evaluation_csv = "evaluation_dataset.csv", dataset_csv_delimiter = "#", dataset_source_csv_column_header = "source", dataset_target_csv_column_header = "target", use_metric_as_trainer_callback = True, optimizer_learning_rate = [1e-5, 5e-4], optimizer_per_device_train_batch_size = [6, 8, 32], optimizer_num_train_epochs = [4, 6], optimizer_weight_decay = [1e-4, 1e-2], fine_tuned_model_dir = "./fine_tuned_model", fine_tuning_learning_rate=8e-5, fine_tuning_per_device_train_batch_size=8, fine_tuning_per_device_eval_batch_size=8, fine_tuning_num_train_epochs=8, fine_tuning_weight_decay=0.01):

        if not (isinstance(pretrained_download, bool)):
            raise ValueError(f"pretrained_model_name expected, got {pretrained_download}")
        self.pretrained_download = pretrained_download

        if not (isinstance(dataset_training_csv, str) and len(dataset_training_csv) > 0):
            raise ValueError(f"dataset_training_csv expected, got {dataset_training_csv}")
        self.dataset_training_csv = dataset_training_csv

        if not (isinstance(dataset_evaluation_csv, str) and len(dataset_evaluation_csv) > 0):
            raise ValueError(f"dataset_evaluation_csv expected, got {dataset_evaluation_csv}")
        self.dataset_evaluation_csv = dataset_evaluation_csv

        if not (isinstance(dataset_csv_delimiter, str) and len(dataset_csv_delimiter) > 0):
            raise ValueError(f"dataset_csv_delimiter expected, got {dataset_csv_delimiter}")
        self.dataset_csv_delimiter = dataset_csv_delimiter

        if not (isinstance(pretrained_model_name, str) and len(pretrained_model_name) > 0):
            raise ValueError(f"pretrained_model_name expected, got {pretrained_model_name}")
        self.pretrained_model_name = pretrained_model_name

        if not (isinstance(dataset_source_csv_column_header, str) and len(dataset_source_csv_column_header) > 0):
            raise ValueError(f"dataset_source_csv_column_header expected, got {dataset_source_csv_column_header}")
        self.dataset_source_csv_column_header = dataset_source_csv_column_header

        if not (isinstance(dataset_target_csv_column_header, str) and len(dataset_target_csv_column_header) > 0):
            raise ValueError(f"dataset_target_csv_column_header expected, got {dataset_target_csv_column_header}")
        self.dataset_target_csv_column_header = dataset_target_csv_column_header

        if not (isinstance(pretrained_local_dir_download, str)):
            raise ValueError(f"pretrained_local_dir_download expected, got {pretrained_local_dir_download}")
        self.pretrained_local_dir_download = pretrained_local_dir_download

        if not (isinstance(fine_tuned_model_dir, str)):
            raise ValueError(f"fine_tuned_model_dir expected, got {fine_tuned_model_dir}")
        self.fine_tuned_model_dir = fine_tuned_model_dir

        if not (isinstance(use_metric_as_trainer_callback, bool)):
            raise ValueError(f"use_metric_as_trainer_callback expected, got {use_metric_as_trainer_callback}")
        self.use_metric_as_trainer_callback = use_metric_as_trainer_callback

        if not (isinstance(fine_tuning_learning_rate, (int, float))):
            raise ValueError(f"fine_tuning_learning_rate expected, got {fine_tuning_learning_rate}")
        self.fine_tuning_learning_rate = fine_tuning_learning_rate

        if not (isinstance(fine_tuning_per_device_train_batch_size, (int, float))):
            raise ValueError(f"fine_tuning_per_device_train_batch_size expected, got {fine_tuning_per_device_train_batch_size}")
        self.fine_tuning_per_device_train_batch_size = fine_tuning_per_device_train_batch_size

        if not (isinstance(fine_tuning_per_device_eval_batch_size, (int, float))):
            raise ValueError(f"fine_tuning_per_device_eval_batch_size expected, got {fine_tuning_per_device_eval_batch_size}")
        self.fine_tuning_per_device_eval_batch_size = fine_tuning_per_device_eval_batch_size

        if not (isinstance(fine_tuning_num_train_epochs, (int, float))):
            raise ValueError(f"fine_tuning_num_train_epochs expected, got {fine_tuning_num_train_epochs}")
        self.fine_tuning_num_train_epochs = fine_tuning_num_train_epochs

        if not (isinstance(fine_tuning_weight_decay, (int, float))):
            raise ValueError(f"fine_tuning_weight_decay expected, got {fine_tuning_weight_decay}")
        self.fine_tuning_weight_decay = fine_tuning_weight_decay

        if not (isinstance(optimizer_learning_rate, (list, dict, tuple))):
            raise ValueError(f"optimizer_learning_rate expected, got {optimizer_learning_rate}")
        self.optimizer_learning_rate = optimizer_learning_rate

        if not (isinstance(optimizer_per_device_train_batch_size, (list, dict, tuple))):
            raise ValueError(f"optimizer_per_device_train_batch_size expected, got {optimizer_per_device_train_batch_size}")
        self.optimizer_per_device_train_batch_size = optimizer_per_device_train_batch_size

        if not (isinstance(optimizer_num_train_epochs, (list, dict, tuple))):
            raise ValueError(f"optimizer_num_train_epochs expected, got {optimizer_num_train_epochs}")
        self.optimizer_num_train_epochs = optimizer_num_train_epochs

        if not (isinstance(optimizer_weight_decay, (list, dict, tuple))):
            raise ValueError(f"optimizer_weight_decay expected, got {optimizer_weight_decay}")
        self.optimizer_weight_decay = optimizer_weight_decay

    '''
    prepareDataset method:
        - Loads training and evaluation CSV datasets
        - Handles model and tokenizer initialization (either downloading or loading locally)
        - Preprocesses the text data by:
            - Converting inputs and targets to strings
            - Tokenizing with appropriate padding and truncation
            - Preparing model inputs with labels
        - Returns tokenized datasets and the tokenizer
    '''
    def prepareDataset(self):
        dataset = load_dataset("csv", data_files={"train": self.dataset_training_csv, "eval": self.dataset_evaluation_csv}, delimiter=self.dataset_csv_delimiter)

        if self.pretrained_download:
            if os.path.exists(self.pretrained_local_dir_download):
                model = AutoModelForSeq2SeqLM.from_pretrained(self.pretrained_local_dir_download)
                tokenizer = AutoTokenizer.from_pretrained(self.pretrained_local_dir_download)
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(self.pretrained_model_name, ignore_mismatched_sizes=True)
                tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
                model.save_pretrained(self.pretrained_local_dir_download)
                tokenizer.save_pretrained(self.pretrained_local_dir_download)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.pretrained_model_name, ignore_mismatched_sizes=True)
            tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)

        def preprocess_function(examples):
            inputs = [str(x) for x in examples[self.dataset_source_csv_column_header]]
            targets = [str(x) for x in examples[self.dataset_target_csv_column_header]]
            model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_datasets = dataset.map(preprocess_function, batched=True)
        return tokenized_datasets, tokenizer

    '''
    executeFineTuning method:
        - Prepares datasets and loads the model
        - Implements two paths:
            - Hyperparameter optimization (when get_optimized_hyperparameter=True):
                - Uses Optuna to search for optimal hyperparameters
                - Defines an objective function that trains with different parameter configurations
                - Optimizes for minimizing evaluation loss
            - Standard fine-tuning:
                - Sets up training arguments with specified hyperparameters
                - Creates a trainer with the model, datasets, and the BLEU callback
                - Runs training and saves the final model and tokenizer
    '''
    def executeFineTuning(self, get_optimized_hyperparameter = False):
        tokenized_datasets, tokenizer = self.prepareDataset()

        if self.pretrained_download:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.pretrained_local_dir_download)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.pretrained_model_name, ignore_mismatched_sizes=True)

        if get_optimized_hyperparameter:
            def objective(trial):
                learning_rate = trial.suggest_float('learning_rate', self.optimizer_learning_rate[0], self.optimizer_learning_rate[1])
                per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', self.optimizer_per_device_train_batch_size)
                num_train_epochs = trial.suggest_int('num_train_epochs', self.optimizer_num_train_epochs[0], self.optimizer_num_train_epochs[1])
                weight_decay = trial.suggest_float("weight_decay", self.optimizer_weight_decay[0], self.optimizer_weight_decay[1], log=True)

                training_args = Seq2SeqTrainingArguments(
                    output_dir="./results_optm",
                    learning_rate=learning_rate,
                    per_device_train_batch_size=per_device_train_batch_size,
                    num_train_epochs=num_train_epochs,
                    weight_decay=weight_decay,
                    report_to="none"
                )

                trainer = Seq2SeqTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=tokenized_datasets["train"],
                    eval_dataset=tokenized_datasets["eval"],
                    tokenizer=tokenizer,
                )

                trainer.train()

                eval_result = trainer.evaluate()
                return eval_result['eval_loss']

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=20)

            print("Hyperparameterst from Optuna: ", study.best_params)

            return

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.fine_tuned_model_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.fine_tuning_learning_rate,
            per_device_train_batch_size=self.fine_tuning_per_device_train_batch_size,
            per_device_eval_batch_size=self.fine_tuning_per_device_eval_batch_size,
            num_train_epochs=self.fine_tuning_num_train_epochs,
            weight_decay=self.fine_tuning_weight_decay,
            save_total_limit=2, #Limit the number of saved checkpoints
            predict_with_generate=True,
            logging_dir="./logs",
            logging_steps=500,
            load_best_model_at_end=True,
            report_to="none",
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["eval"],
            tokenizer=tokenizer,
            callbacks=[
                BLEUCallback(
                    tokenizer = tokenizer, 
                    eval_dataset = tokenized_datasets["eval"], 
                    model = model, 
                    dataset_source_csv_column_header = self.dataset_source_csv_column_header, 
                    dataset_target_csv_column_header = self.dataset_target_csv_column_header
                    )
                ] if self.use_metric_as_trainer_callback else None
        )

        trainer.train()

        trainer.save_model(self.fine_tuned_model_dir)
        tokenizer.save_pretrained(self.fine_tuned_model_dir)

    '''
    executeInference method:
        - Loads the fine-tuned model and tokenizer
        - Takes an input text, tokenizes it
        - Generates a translation using beam search with some controlled randomness (temperature set to 0.7)
        - Returns the decoded translation
    '''
    def executeInference(self, input_text):
        tokenizer = AutoTokenizer.from_pretrained(self.fine_tuned_model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.fine_tuned_model_dir)

        inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)

        outputs = model.generate(
            inputs["input_ids"],
            max_length=128,
            num_beams=5,
            early_stopping=True,
            temperature=0.7,
            do_sample = True
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)
