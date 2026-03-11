# Enhancing LLM Robustness to Perturbed Instructions
Official repository for [Enhancing LLM Robustness to Perturbed Instructions: An Empirical Study](https://arxiv.org/abs/2504.02733). 

Our AdvMix dataset is available [here](https://huggingface.co/datasets/aryanagrawal1/advmix).


## Instructions

### Setting up the environment
````
git clone https://github.com/ary4n99/llm-robustness.git
cd llm-robustness
pip install -r requirements.txt

cp example.yaml config.yaml
cp .env.example .env
````

### Running the code
To run attack pipelines:
````
python run_pipelines.py --config ./path/to/config --log-level INFO --seed 0
````

To run semantic integrity analysis:
````
python semantic_integrity.py
````

## Citation

````bibtex
@inproceedings{
      agrawal2025enhancing,
      title={Enhancing {LLM} Robustness to Perturbed Instructions: An Empirical Study},
      author={Aryan Agrawal and Lisa Alazraki and Shahin Honarvar and Thomas Mensink and Marek Rei},
      booktitle={ICLR 2025 Workshop on Building Trust in Language Models and Applications},
      year={2025},
      url={https://openreview.net/forum?id=abllmCsDp8}
}
````
