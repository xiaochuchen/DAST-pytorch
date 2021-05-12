# Domain Adaptive Text Style Transfer (DAST) - Pytorch 

This is an unofficial Pytorch implementation of the paper ["Domain Adpative Text Style Transfer"](https://arxiv.org/pdf/1908.09395.pdf) by Dianqi Li, Yizhe Zhang, Zhe Gan, Yu Cheng, Chris Brockett, Ming-Ting Sun and Bill Dolan, *EMNLP 2019*. The original Tensorflow implementation can be found [here](https://github.com/cookielee77/DAST).

## Data
The data can be found in the original repository. Simply copy the content of "data" folder from the original repository to `/data` folder of this repo.

## Dependencies
python 3.7  
pytorch  
scipy  
click  
nltk  

## Training
Training functions are in the `main.py` module. 
1. Train two evaluation classifiers that will be used for the evaluation of the style transfer model.  
    - Train style classifier on target data set. (yelp)  
`python main.py train-classifier config_classif.json` with `is_domain: false` in the json config file.  
    - Train domain classifier on mixed data set. (imdb, yelp)  
`python main.py train-classifier config_classif.json` with `is_domain: true` in the json config file. 

2. Train the DAST model:  
`python main.py train-dast config_dast.json`




