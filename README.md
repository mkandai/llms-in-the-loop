# LLMs in the loop: Leveraging Large Language Model annotations for Active Learning in low-resource languages

## Project Structure

Below is the tree structure of the project along with a description of each module:

```

│
├── data # Directory for saving data used in the notebooks
│
├── Experiments # Directory for running and saving experiments
│  ├── active_learning_on_foundation_model # Notebooks related to each language for re-training initial model using activate learning based on uncertainty sampling
│  ├── active_learning_on_ground_truth # Notebooks related to each language for re-training initial model using activate learning based on human annotation
│  ├── foundation_model_selection # Notebooks related to each large language model and experiment related to that model
│  └── simple_training_on_ground_truth # Notebooks related to each language for simple model training without using activate learning
│
├── notebooks # Jupyter notebooks
│  ├── 00-token_counts.ipynb # Notebook for analyzing token counts in the dataset
│  ├── 01-preprocessing.ipynb # Notebook for data preprocessing steps
│  ├── 02-querying.ipynb # Notebook for querying foundation model
│  ├── 04.1-sampling_for_foundation_model_selection.ipynb 
│  └── 04.2-sampling_pipeline.ipynb # Example notebook how to use sampling function
│
├── settings # Contains configuration files for the project
│  └── config.yml # YAML configuration file with overall project settings
│
├── src # Source code for the project
│  ├── data # Scripts related to data handling
│  │  ├── preprocess.py # Python script for preprocessing data
│  │  └── sample.py # Functions for sampling the data for the foundation model selection
│  │
│  ├── models # Scripts for defining models
│  │  └── xlmr_ner.py # Script for an XLM-R model for Named Entity Recognition
│  │
│  ├── query # Scripts for querying foundation models
│  │  ├── annotation_examples.json # Manually created annotation examples
│  │  ├── ner_examples_all_languages.json # Automatically created annotation examples
│  │  ├── generate_annotation_examples.ipynb # Notebook to generate annotation examples
│  │  ├── prompts.py # Prompts for querying foundation models
│  │  └── query_gpt.py # Script for querying foundation models
│  │
│  └── utils # Utility scripts used across the project
│     └── utils.py # General utility functions
│
├── README.md # The file you are currently reading
│
├── .gitignore # Ignore special files from commiting to the git repository
│
└── requirements.txt # Required libraries for the project to run
```

## Usage

### 📒 Notebooks

Folder `notebook` contains the main logic of the project.

* To explore text preprocessing steps, see `notebooks/01-preprocessing.ipynb`


* To explore how we query foundation model, see `notebooks/02-querying.ipynb`


* To explore how to use the sampling function for foundation model selection, see `notebooks/04.1-sampling_pipeline.ipynb`.
For more details, see `04.1-sampling_for_foundation_model_selection.ipynb`.

### 🧪 Experiments

Folder `experiments` contains various experiment runs with different settings.

## ✍️ Contributors

- Nataliia Kholodna - [@nataliyakholodna](https://github.com/nataliyakholodna/)
- Mohammad (MK) Khodadadi - [@mkandai](https://github.com/mkandai/)
- Nurullah Gumus - [@silverdevelopper](https://github.com/silverdevelopper)
