# Language Model for Translating from Italian to Cimbro (adaptable to any other language pair)

This project consists of a language translation model from Italian to Cimbro developed by fine-tuning the [Helsinki-NLP/opus-mt-it-de](https://huggingface.co/Helsinki-NLP/opus-mt-en-de) model. The project includes the following files:
1. **exec_finetuning.py**: allows fine-tuning of the model by calling the `executeFineTuning` method of the `Translator` class: `translator_instance.executeFineTuning(get_optimized_hyperparameter = False)`. If `get_optimized_hyperparameter` is set to `True` it executes the hyperparameter optimizer [Optuna](https://optuna.org/).
2. **app.py**: starts a web application in Flask to perform inference and obtain the translation, using the `executeInference` method of the `Translator` class. This method receives only one parameter: the text to be translated. The `temperature` parameter is set to `0.7`, possibly it could be added in the future as a parameter to the `executeInference` method (reminder: values between 0.2 - 0.7 allow the generation of more accurate and consistent answers to the training dataset while values between 0.8 - 1.5 allow the generation of more creative answers).
3. **translator.py**: `Translator` class with the two main methods `executeFineTuning` and `executeInference`. The configuration of all parameters for fine-tuning, saving the pre-trained model, csv file structure, etc. is defined in the class constructor.
4. **templates**: folder with the html template `index.html` for the inference web interface.

## Notes on the Helsinki-NLP/opus-mt-it-de model

The **Helsinki-NLP/opus-mt-it-de** model is a pre-trained machine translation model developed by the [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) team.

The model uses a Transformer-type neural network, and is based on the OPUS (Open Parallel Corpus) dataset: a large corpus of parallel translations between different languages. I used this model as a starting point to perform fine-tuning on translation pairs from Italian to Cimbro.

For more information on the Helsinki-NLP/opus-mt-it-de model, visit the [official repository on Hugging Face](https://huggingface.co/Helsinki-NLP/opus-mt-it-de).

By choosing a different pre-trained model, it is then possible to use the `Translator` class to perform fine-tuning on any other dialect.

## Model available on Huggin Face

The fine-tuned model on cimbro is available in Safetensors format at [Huggin Face - Italian_cimbro](https://huggingface.co/sigfrido-corradi/italiano_cimbro).

It can also be tested at: [https://www.italianocimbro.it/](https://www.italianocimbro.it/).

## Notes on the "**Cimbri**"

The Cimbri of the "XIII Comuni" are descended from medieval migrations of German settlers who came to the Verona area before 1287 from the "Alta Valle del Chiampo". On February 5, 1287, the bishop of Verona granted them a semi-populated area in the Lessinia Mountains for settlement. The Cimbri engaged in logging, Brogna sheep breeding and the production of fine wool.

Expanding, they formed XIII communities in various localities of Lessinia. After the fall of the Scaligeri, the Cimbri obtained confirmations of their privileges from the Visconti and, under the Venetian Republic, became landowners. In the 1600s, they were in charge of border defense and developed the use of "Trombini", arquebuses still used today in local festivals.

The plague of 1630 and subsequent famines led to community crisis, prompting emigration and the spread of new American crops such as corn and beans. Transhumance offered new employment opportunities, but the language of the Cimbri gradually declined.

## Notes on the Germanic dialect "**Cimbro**"

The cimbro of "XIII Comuni", called "Tauc" in [Giazza](https://it.wikipedia.org/wiki/Giazza), is a Germanic dialect that arrived in Lessinia with German settlers from the 12th century. Derived from Middle High German spoken in Tyrol and Bavaria, it spread across the plateau, leaving traces in toponymy and reaching its greatest expansion in the 17th century.

From then on, the language began to disappear in the various Cimbrian municipalities, enduring until the end of the 19th century only in Velo, Selva di Progno, and San Bortolo. The influence of the romance languages brought numerous linguistic borrowings, especially for terms related to new foods, utensils, and technology.

Today cimbro is spoken only in [Giazza](https://it.wikipedia.org/wiki/Giazza) and is protected by two cultural associations: the [Curatorium Cimbricum Veronense](https://www.cimbri.it/) and [De Zimbar “UN LJETZAN”](https://www.facebook.com/dezimbarunljetzan), which promote courses and initiatives to keep it alive.

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/SigfridoCorradi/italiano_cimbro
    cd italiano_cimbro
    ```

2. **Create a virtual environment** (optional but **strongly** recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Perform fine-tuning of the model**:

    Note: fine-tuning needed only in case of use on model or dataset different from current project on Cimbro language, otherwise just take model from [Huggin Face](https://huggingface.co/sigfrido-corradi/italiano_cimbro) and place it in `fine_tuned_model` folder.

   To perform fine-tuning, it is necessary to have prepared two csv files. One with the trainig dataset called `training_dataset.csv` structured as follows:

   | source | target |
   |-----------|-----------|
   | Cari amici, uomini e donne, buon giorno    | Liabe gaseljan, manne un baibar, gùatan tak    |
   | Grazie per essere venuti a trovarci!   | Borkant for sain kent tze vinganus!    |
   | Provincia di Verona    | Prvìnz vòme Bèarn    |
   | ...       | ...       |

   And a second csv file with the dataset used for evaluation during training, in the same form as the csv file with the training dataset and named `evaluation_dataset.csv`. Once the two files have been prepared, fine-tuning can be started:

    ```bash
    python exec_finetuning.py
    ```

6. **Start the Flask application**:

    Once fine-tuning is completed (or using the model available at [Huggin Face](https://huggingface.co/sigfrido-corradi/italiano_cimbro)), simply start the web application for inference and go to `http://127.0.0.1:8080`

    ```bash
    python app.py
    ```
   
