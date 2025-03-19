# Modello Linguistico per tradurre dall'Italiano al Cimbro

Questo progetto consiste in un modello di traduzione linguistica  dall'Italiano al Cimbro sviluppato mediante fine-tuning del modello [Helsinki-NLP/opus-mt-it-de](https://huggingface.co/Helsinki-NLP/opus-mt-en-de). Il progetto include i seguenti file:
1. **exec_finetuning.py**: permette di eseguire il fine-tuning del modello mediante la chiamata al metodo `executeFineTuning` della classe `Translator`: `translator_instance.executeFineTuning(get_optimized_hyperparameter = False)`. Il parametro `get_optimized_hyperparameter` se posto a `True` permette di eseguire l'ottimizzatore di iperparametri [Optuna](https://optuna.org/).
2. **app.py**: avvia un'applicazione web in Flask per eseguire l'inferenza sul modello ed ottenere quindi la traduzione. Viene usato il metodo `executeInference` della classe `Translator`. Questo metodo riceve un solo parametro: il testo da tradurre. Il parametero `temperature` è impostato a `0.7`, eventualmente si potrebbe aggiungere come parametro al metodo `executeInference` (promemoria: valori tra 0.2 - 0.7 permettono la generazione di risposte più precise e coerenti al dataset di training mentre valori tra 0.8 - 1.5 permettono la generazione di risposte più creative).
3. **translator.py**: classe `Translator` con i due metodi principali `executeFineTuning` ed `executeInference`. La configurazione di tutti i parametri per il fine-tuning, il salvataggio del modello pre-addestrato, la struttura dei file csv, ecc. è definita nel costruttore di classe.
4. **templates**: cartella con il template html `index.html` per lpinterfaccia web di inferenza.

## Note sul modello Helsinki-NLP/opus-mt-it-de

Il modello **Helsinki-NLP/opus-mt-it-de** è un modello di traduzione automatica pre-addestrato, sviluppato dal team Helsinki-NLP.

Il modello utilizza una rete neurale di tipo Transformer, e si basa sul dataset OPUS (Open Parallel Corpus): un ampio corpus di traduzioni parallele tra diverse lingue. L'architettura Transformer è particolarmente adatta per compiti di traduzione, poiché permette di modellare dipendenze a lungo termine tra le parole, migliorando la qualità delle traduzioni.

In questo caso, ho utilizzato questo modello come punto di partenza ed abbiamo eseguito il fine-tuning sui coppie di traduzione dall'italiano al cimbro.

Per maggiori informazioni sul modello Helsinki-NLP/opus-mt-it-de, visita il [repository ufficiale su Hugging Face](https://huggingface.co/Helsinki-NLP/opus-mt-it-de).

## Modello disponibile su Huggin Face

Il modello fine-tuned è disponibile nel formato Safetensors su [Huggin Face - italiano_cimbro](https://huggingface.co/sigfrido-corradi/italiano_cimbro)

## Note sui Cimbri

I Cimbri dei XIII Comuni discendono da migrazioni medievali di coloni tedeschi, giunti nel Veronese prima del 1287 dall'Alta Valle del Chiampo. Il 5 febbraio 1287, il vescovo di Verona concesse loro un'area semi-spopolata nei Monti Lessini per l'insediamento. I Cimbri si dedicarono al disboscamento, all'allevamento della pecora Brogna e alla produzione di lana pregiata.

Espandendosi, formarono XIII comunità in varie località della Lessinia. Dopo la caduta degli Scaligeri, i Cimbri ottennero conferme dei loro privilegi dai Visconti e, sotto la Repubblica di Venezia, divennero proprietari di terre. Nel '600, furono incaricati della difesa dei confini e svilupparono l'uso dei "Trombini", archibugi usati ancora oggi nelle feste locali.

La peste del 1630 e le carestie seguenti portarono alla crisi delle comunità, spingendo all’emigrazione e alla diffusione di nuove colture americane come mais e fagioli. La transumanza offrì nuove opportunità lavorative, ma la lingua cimbra si ridusse progressivamente.

## Note sul dialetto germanico cimbro

Il cimbro dei XIII Comuni, chiamato Tauc a [Giazza](https://it.wikipedia.org/wiki/Giazza), è un dialetto germanico arrivato in Lessinia con i coloni tedeschi dal XII secolo. Derivato dal medio alto tedesco parlato in Tirolo e Baviera, si diffuse nell'altopiano, lasciando tracce nella toponomastica e raggiungendo la massima espansione nel XVII secolo.

Da allora, la lingua iniziò a scomparire nei vari comuni cimbri, resistendo fino alla fine dell’Ottocento solo in Velo, Selva di Progno e San Bortolo. L'influenza delle lingue romanze (italiano, veneto e trentino) portò numerosi prestiti linguistici, specialmente per termini legati a nuovi alimenti, utensili e tecnologia.

Oggi, il cimbro è parlato solo a Giazza e viene tutelato da due associazioni culturali, il Curatorium Cimbricum Veronense e De Zimbar 'un Ljetzan, che promuovono corsi e iniziative per mantenerlo vivo.

## Installazione

1. **Clonare il repository**:

    ```bash
    git clone https://github.com/SigfridoCorradi/italiano_cimbro
    cd italiano_cimbro
    ```

2. **Creare un ambiente virtuale** (opzionale ma **fortemente** consigliato):

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. **Installare le dipendenze**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Eseguire il fine-tuning del modello**:

   Per eseguire il fine-tuning è necessario aver preparato due file csv. Uno con il dataset di trainig chiamato `training_dataset.csv` nella forma:

   | source | target |
   |-----------|-----------|
   | Cari amici, uomini e donne, buon giorno    | Liabe gaseljan, manne un baibar, gùatan tak    |
   | Grazie per essere venuti a trovarci!   | Borkant for sain kent tze vinganus!    |
   | Provincia di Verona    | Prvìnz vòme Bèarn    |
   | ...       | ...       |

   Ed un secondo file csv con il dataset utilizzato per la valutazione durante l'addestramento, nella stessa forma del file csv con il dataset di training e chiamato `evaluation_dataset.csv`. Una volta preparati i due file è possibile avviare il fine-tuning:

    ```bash
    python exec_finetuning.py
    ```

6. **Avviare l'applicazione Flask**:

    Una volta completato il fine-tuning, è sufficiente avviare l'applicazione web per l'inferenza e posizionarsi all'indirizzo `http://127.0.0.1:8080`

    ```bash
    python app.py
    ```
   
