"""Fine-tuning Marian IT->cimbro, valutazione e traduzione senza troncamenti."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sacrebleu.metrics import BLEU, CHRF
from transformers import (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, EarlyStoppingCallback,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'cimbro_model_v2_final_model'
BASE = 'Helsinki-NLP/opus-mt-it-de'
OUT = ROOT / 'cimbro_model_v3_from_base'


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def load_data():
    data = {s:json.loads((ROOT/'dataset_v3'/f'{s}.json').read_text()) for s in ('train','validation','test')}
    from prepara_dataset import key
    for side in ('source','target'):
        sets = {s:{key(r[side]) for r in rows} for s,rows in data.items()}
        assert not (sets['train'] & sets['test'] or sets['train'] & sets['validation'] or sets['test'] & sets['validation']), f'Leakage {side}'
    return data


def encode(rows, tokenizer, limit):
    x = tokenizer([r['source'] for r in rows], truncation=False)
    x['labels'] = tokenizer(text_target=[r['target'] for r in rows], truncation=False)['input_ids']
    for i,(src,tgt) in enumerate(zip(x['input_ids'],x['labels'])):
        if max(len(src),len(tgt)) > limit:
            raise ValueError(f'Coppia {rows[i]["id"]} supera {limit} token')
        if tokenizer.unk_token_id in src+tgt:
            raise ValueError(f'Token sconosciuto: {rows[i]["id"]}')
    return Dataset.from_dict(x)


def translate(model, tokenizer, texts, beams=4, batch_size=8):
    limit = model.config.max_position_embeddings
    result, capped = [], 0
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(texts),batch_size):
            batch = texts[start:start+batch_size]
            ids = tokenizer(batch, truncation=False)['input_ids']
            if any(len(x)>limit for x in ids):
                raise ValueError(f'Input oltre il limite di {limit} token (EOS incluso). Dividere il testo in frasi.')
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=False).to(model.device)
            preds = model.generate(**inputs, num_beams=beams, max_length=limit, use_cache=True, early_stopping=True)
            # Marian forza EOS al limite: EOS finale non prova che la frase sia completa.
            capped += sum(
                int(len(p) - int((p == tokenizer.pad_token_id).sum()) + 1 >= limit) for p in preds)
            result.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))
    return result, capped


def evaluate_model(model, tokenizer, rows, beams=4):
    # Stessa sorgente, più traduzioni legittime: riferimenti multipli.
    groups = {}
    for r in rows:
        groups.setdefault(r['source'],[]).append(r['target'])
    sources = list(groups)
    refs = list(groups.values())
    predictions,capped = translate(model,tokenizer,sources,beams)
    refs_stream = [[r[min(i,len(r)-1)] for r in refs] for i in range(max(map(len,refs)))]
    bleu, chrf = BLEU(), CHRF(word_order=2)
    scores = dict(bleu=bleu.corpus_score(predictions,refs_stream).score,
                  chrf_plus=chrf.corpus_score(predictions,refs_stream).score,
                  sources=len(sources), generation_capped=capped,
                  bleu_signature=str(bleu.get_signature()), chrf_signature=str(chrf.get_signature()))
    return scores,[dict(source=s,references=r,prediction=p) for s,r,p in zip(sources,refs,predictions)]


def evaluate_all(model, tokenizer, rows):
    from prepara_dataset import key
    scores, predictions = evaluate_model(model,tokenizer,rows)
    oldrows = json.loads((ROOT/'dataset.json').read_text())
    oldkeys = {key(r['instruction']+' '+r.get('input','')) for r in oldrows}
    for name,is_old in [('historical',True),('new_odt',False)]:
        subset = [r for r in predictions if (key(r['source']) in oldkeys)==is_old]
        refs = [r['references'] for r in subset]
        streams = [[r[min(i,len(r)-1)] for r in refs] for i in range(max(map(len,refs)))]
        preds = [r['prediction'] for r in subset]
        scores[name] = dict(sources=len(subset),bleu=BLEU().corpus_score(preds,streams).score,
                            chrf_plus=CHRF(word_order=2).corpus_score(preds,streams).score)
    return scores,predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','train','evaluate','translate','benchmark'])
    p.add_argument('--model',default=BASE)
    p.add_argument('--output',default=str(OUT))
    p.add_argument('--epochs',type=float,default=25)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--learning-rate',type=float,default=3e-5)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--text')
    p.add_argument('--resume',default=None)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    set_seed(42)
    output = Path(a.output)
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    config = AutoConfig.from_pretrained(a.model)
    data = load_data()
    tokenized = {s:encode(rows,tokenizer,config.max_position_embeddings) for s,rows in data.items()}
    if a.command == 'check':
        print(json.dumps(dict(limit=config.max_position_embeddings,splits={s:len(r) for s,r in data.items()},
                              max_source=max(len(x) for d in tokenized.values() for x in d['input_ids']),
                              max_target=max(len(x) for d in tokenized.values() for x in d['labels'])),indent=2))
        return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model).to(device)
    model.generation_config.max_length = config.max_position_embeddings
    model.generation_config.num_beams = 4
    model.generation_config.use_cache = True
    collator = DataCollatorForSeq2Seq(tokenizer,model=model)
    if a.command == 'translate':
        if not a.text:
            p.error('--text richiesto')
        print(translate(model,tokenizer,[a.text])[0][0])
        return
    if a.command == 'benchmark':
        model.train()
        optim = torch.optim.AdamW(model.parameters(),lr=a.learning_rate)
        start = time.monotonic()
        for i in range(3):
            batch = collator([tokenized['train'][j] for j in range(i*a.batch_size,(i+1)*a.batch_size)])
            batch = {k:v.to(device) for k,v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            print('step',i+1,'loss',loss.item(),'elapsed',time.monotonic()-start,flush=True)
        return
    if a.command == 'train' and (output/'trainer_state.json').exists() and not a.resume:
        raise ValueError('Output contiene già un training: scegliere --output diverso oppure --resume CHECKPOINT')
    output.mkdir(exist_ok=True,parents=True)
    if a.command == 'evaluate':
        scores, predictions = evaluate_all(model,tokenizer,data['test'])
        dump(output/'evaluation.json',scores)
        dump(output/'predictions.json',predictions)
        print(scores)
        return
    # La validation loss evita migliaia di traduzioni autoregressive su CPU ad ogni epoca.
    # BLEU/chrF++ sono misurati sul test indipendente solo alla fine.
    args = Seq2SeqTrainingArguments(
        output_dir=str(output), eval_strategy='epoch',save_strategy='epoch',
        learning_rate=a.learning_rate,per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size, num_train_epochs=a.epochs,
        weight_decay=.01,warmup_ratio=.05,label_smoothing_factor=.05,
        fp16=device=='cuda',use_cpu=device=='cpu',dataloader_pin_memory=device=='cuda',
        save_total_limit=2,load_best_model_at_end=True,metric_for_best_model='eval_loss',
        greater_is_better=False, logging_steps=25,report_to='none',seed=42,
        group_by_length=True, optim='adamw_torch',save_safetensors=True,
        predict_with_generate=False, disable_tqdm=True)
    trainer = Seq2SeqTrainer(model=model,args=args,train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],data_collator=collator,processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3,early_stopping_threshold=.001)])
    dump(output/'run_config.json',dict(arguments=vars(a),device=device,base_model=str(a.model),
        limit=config.max_position_embeddings,torch=torch.__version__,
        dataset_sha256={s:hashlib.sha256((ROOT/'dataset_v3'/f'{s}.json').read_bytes()).hexdigest() for s in data},
        dataset_report=json.loads((ROOT/'dataset_v3/report.json').read_text())))
    baseline = trainer.evaluate()
    dump(output/'baseline_validation.json',baseline)
    print('Baseline validation',baseline,flush=True)
    result = trainer.train(resume_from_checkpoint=a.resume)
    final = output/'final_model'
    trainer.save_model(str(final))
    tokenizer.save_pretrained(final)
    trainer.save_state()
    dump(output/'train_metrics.json',result.metrics)
    scores,predictions = evaluate_all(trainer.model,tokenizer,data['test'])
    dump(output/'evaluation.json',scores)
    dump(output/'predictions.json',predictions)
    print('TEST',scores,flush=True)


if __name__ == '__main__':
    main()
