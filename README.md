# README

### dependencies

`Python 3.7`

`torch==1.10.0+cu113`

`transformers==4.28.1`

`pandas==1.3.4`

`numpy==1.21.4`

`tqdm==4.65.0`

`python_docx==0.8.11`

`scikit_learn==1.0.2`

`sentence_transformers==2.2.2`

`zhconv==1.4.3`

### structure of my project

- ./data/ contains all datasets used in training, evaluation and clustering
- ./preprocess/
    - `make_ancient_modern_Chinese_parallel.py` pre-processes Ancient-Modern Chinese dataset
    - `make_traditional_simplified_character_parallel.py` creates Traditional-Simplified Character Parallel Corpus
    - `crawl_get_content.py` crawl parallel sentence groups from ctext
    - `ctext_preprocess.py` pre-processes the parallel sentence groups crawled from ctext
- `cluster.py` uses fine-tuned models to cluster ancient Chinese parallel sentences
- `eval.py` evaluates fine-tuned models on two metrics we defined
- `function.py` defines aiding functions
- `train.py` uses three datasets to fine-tune the model
- `plain_approach.py` implements a plain algorithm to cluster ancient Chinese parallel sentences
- ./finetuned_model/
If you want to use our fine-tuned model, or want to run `eval.py` or `evaluate.py`, download the models from [this](https://disk.pku.edu.cn:443/link/C3D9B769F6119B32762CF4687A3AF7AE) pku disk link and put them in this directory.
