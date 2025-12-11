import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import evaluate
import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer

# --- CONFIGURAZIONE ---
EXECUTE_FINE_TUNING = False
TEST_MAX_TOKEN_DATASET = False

MODEL_CHECKPOINT = "Helsinki-NLP/opus-mt-it-de"
OUTPUT_DIR = "./cimbro_model_v2"
DATA_FILE = "dataset.json"

MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 512

LEARNING_RATE = 2e-5
BATCH_SIZE = 16
NUM_EPOCHS = 20
FP16 = True

def check_token_lengths():
    json_file = "dataset.json"
    model_checkpoint = "Helsinki-NLP/opus-mt-it-de"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: missing file '{json_file}'")
        return

    src_lengths = []
    tgt_lengths = []

    print(f"Analysis of {len(dataset)} sentences...")

    for entry in dataset:
        instruction = entry.get("instruction", "")
        input_ctxt = entry.get("input", "")

        if input_ctxt and input_ctxt.strip():
            source_text = f"{instruction} {input_ctxt}"
        else:
            source_text = instruction

        target_text = entry.get("output", "")

        src_token_ids = tokenizer.encode(source_text, truncation=False)
        tgt_token_ids = tokenizer.encode(target_text, truncation=False)

        src_lengths.append(len(src_token_ids))
        tgt_lengths.append(len(tgt_token_ids))

    max_src = max(src_lengths)
    max_tgt = max(tgt_lengths)
    avg_src = int(np.mean(src_lengths))

    print("\n" + "="*30)
    print("      TOKEN LENGTH REPORT")
    print("="*30)
    print(f"Longest ITALIAN sentence: {max_src} token")
    print(f"Longest CIMBRO sentence:   {max_tgt} token")
    print(f"Medium length Italian: {avg_src} token")
    print("-" * 30)

    # 4. Verdetto
    limit = MAX_SOURCE_LENGTH
    if max_src <= limit and max_tgt <= limit:
        print(f"No sentences will be cut.")
    else:
        needed = max(max_src, max_tgt)
        print(f"WARNING: Some sentences exceed the limit of {MAX_SOURCE_LENGTH}.")
        print(f"They will be cut during training.")
        print(f"Tip: Set MAX_SOURCE/TARGET_LENGTH to at least {needed}.")

def run_finetuning():
    print(f"Loading model: {MODEL_CHECKPOINT}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    model.resize_token_embeddings(len(tokenizer))

    # 2. Loading the Dataset (Alpaca Format)
    print("Loading dataset...")
    data_files = {"train": DATA_FILE}
    dataset = load_dataset("json", data_files=data_files, split="train")

    # Split Train/Test (90% train, 10% test)
    dataset = dataset.train_test_split(test_size=0.1)

    #3. Preprocessing (The crucial part for Alpaca -> Translation)
    def preprocess_function(examples):
        inputs = []
        targets = []

        for instruction, inp, output in zip(examples["instruction"], examples["input"], examples["output"]):
            if inp and inp.strip():
                source_text = f"{instruction} {inp}"
            else:
                source_text = instruction

            inputs.append(source_text)
            targets.append(output)

        # Tokenization
        model_inputs = tokenizer(
            inputs,
            max_length=MAX_SOURCE_LENGTH,
            truncation=True,
            padding=False # Dynamic padding is done afterwards by the DataCollator.
        )

        # Tokenization of labels (target)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=MAX_TARGET_LENGTH,
                truncation=True,
                padding=False
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

    #4. Evaluation Metric (SacreBLEU)
    metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Decoding predictions
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in labels (used to ignore padding)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    #5. Data Collator (Handles dynamic padding)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    #6. Training Topics (Optimized)
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",      # Currency at the end of each era
        save_strategy="epoch",            # Save checkpoint every epoch
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,                # Regularization to avoid overfitting on limited data - Essential for languages
                                          # with limited data (such as Cimbrian) to prevent the model from memorizing
                                          # phrases instead of learning grammar.
        save_total_limit=2,               # Keep only the last 2 models to save space
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,       # Essential for calculating BLEU during training
        fp16=FP16,                        # Speed and memory
        push_to_hub=False,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        load_best_model_at_end=True,      # Finally, load the model with the best BLEU (or lowest loss): if epoch 10
                                          # the model performs worse than epoch 8 (overfitting), the script automatically 
                                          # reloads the weights from epoch 8.
        metric_for_best_model="eval_loss", # Or “bleu” if you prefer
        greater_is_better=False,          # False if you use loss, True if you use bleu
        label_smoothing_factor=0.1        # prevents the model from being “too confident” in its predictions, improving generalization and often raising the BLEU score.
    )

    # 7. Initialization Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print(f"Save model in {OUTPUT_DIR}/final_model")
    trainer.save_model(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")

if __name__ == "__main__":
    if TEST_MAX_TOKEN_DATASET:
        check_token_lengths()
    elif EXECUTE_FINE_TUNING:
        run_finetuning()
    else:
        model_path = "./cimbro_model_v2/final_model"
        translator = pipeline("translation", model=model_path, tokenizer=model_path)

        res = translator("Il mio vecchio cane")
        print(res[0]['translation_text'])