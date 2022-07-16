[//]: # (https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-2.8.0-cp37-cp37m-manylinux2010_x86_64.whl)

# Active Learning
## Notation

    1. U: unlabeled pool
    2. Q: select queries
    3. S: human annotator
    4. L: labeled training set
    5. G: machine learning model
## General process
<img src="https://s4.ax1x.com/2022/02/17/H5OaBq.png">

Referenced from [here](https://coding-zuo.github.io/2022/02/21/%E4%B8%BB%E5%8A%A8%E5%AD%A6%E4%B9%A0Active-Learning-%E5%9F%BA%E7%A1%80/)

# Methods
    1. membership query synthesis
    2. stream-based - one by one
    3. pool-based - batch input


# Dataset in our Experiment

[Yelp](http://www.ics.uci.edu/~vpsaini)

[Yelp Review Full](https://s3.amazonaws.com/fast-ai-nlp/yelp_review_full_csv.tgz)

[WOS]()


# Query Strategies
    1. Random: random pick unlabelled sampled
    2. Max Entropy: pick samples with the largest margin
    3. Core set
    4. MC Dropout
    5. CMBAL: reproduce target
    6. CMBAL Variant 1
    7. CMBAL Variant 2

# Experiment Settings
    1. for each dataset (train, val, test 70%, 15%, 15%)
    2. Init Samples (1% to 5% as init samples)
    3. Batch Size - Frequence
    4. Budget

# Arxiv
## Split
    Number of rows in training set: 25655
    Number of rows in validation set: 5498
    Number of rows in test set: 5498
## Init Samples
    25655 * 0.01 = 256  === 250
    25655 * 0.02 = 513  === 500
    25655 * 0.03 = 759  === 750
    25655 * 0.04 = 1026 === 1000
    25655 * 0.05 = 1282 === 1250

# Yelp
## Split
    Number of rows in training set: 490000
    Number of rows in validation set: 105000
    Number of rows in test set: 105000
## Init Samples
    490000 * 0.01 = 4900  === 5000
    490000 * 0.02 = 9800  === 10000
    490000 * 0.03 = 14700 === 15000
    490000 * 0.04 = 19600 === 20000
    490000 * 0.05 = 24500 === 25000

# WOS5736
## Split
    Number of rows in training set: 4015
    Number of rows in validation set: 860
    Number of rows in test set: 861
## Init Samples
    4015 * 0.01 = 40
    4015 * 0.02 = 80
    4015 * 0.03 = 120
    4015 * 0.04 = 160
    4015 * 0.05 = 200

# WOS5736 wos5736_epoch_30_init_200_f1_56.68_at_1657676569
## Split
    Number of rows in training set: 4015
    Number of rows in validation set: 860
    Number of rows in test set: 861
## Init Samples
    4015 * 0.01 = 40
    4015 * 0.02 = 80
    4015 * 0.03 = 120
    4015 * 0.04 = 160
    4015 * 0.05 = 200
# WOS11967 wos11967_epoch_20_init_400_f1_42.17_at_1657684728
## Split
    Number of rows in training set: 8376
    Number of rows in validation set: 1796
    Number of rows in test set: 1795
## Init Samples
    8376 * 0.01 = 84  === 80
    8376 * 0.02 = 168 === 150
    8376 * 0.03 = 252 === 250
    8376 * 0.04 = 336 === 350
    8376 * 0.05 = 420 === 400

# WOS46985 wos46985_epoch_20_init_1500_f1_35.71_at_1657686012
    Number of rows in training set: 32889
    Number of rows in validation set: 7048
    Number of rows in test set: 7048

## Init Samples
    32889 * 0.05 = 1644 set 1500


# AG_NEWS ag_news_epoch_20_init_800_f1_78.76_at_1657692198

## Split
    Number of rows in training set: 88333
    Number of rows in validation set: 18928
    Number of rows in test set: 18929

## Init Samples
    88333 * 0.01 = 883 set to 800
    88333 * 0.05 = 4416 set to 4000


# 20NG ng20_epoch_20_init_600_f1_45.1_at_1657688507
## Split
    Number of rows in training set: 13192
    Number of rows in validation set: 2827
    Number of rows in test set: 2827

# Init Samples
    13192 * 0.05 = 659 set: 600


# AAPD aapd_epoch_20_init_1000_f1_25.45_at_1657751815

    Number of rows in training set: 36374
    Number of rows in validation set: 7794
    Number of rows in test set: 7795


# Reuters reuters_epoch_20_init_100_f1_49.34_at_1657716769
    Number of rows in training set: 7469
    Number of rows in validation set: 1600
    Number of rows in test set: 1601
