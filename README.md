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

# Evaluation Metrics
    1. Accuracy
    2. U-L ratio vs Accuracy


```
python main.py --dataset arxiv --config config/arxiv/arxiv_seq150_batch128_init100.json
```


# Arxiv RANDOM 50
(Base) Categorical accuracy on the test set: 42.01%, f1-score: 38.22% and  precision: 55.6% and  recall: 29.14%.
(50) Categorical accuracy on the test set: 46.04%, f1-score: 57.88% and  precision: 56.52% and  recall: 59.34%
(100)Categorical accuracy on the test set: 49.05%, f1-score: 46.08% and  precision: 37.69% and  recall: 59.34%
(150)Categorical accuracy on the test set: 64.59%, f1-score: 46.08% and  precision: 37.68% and  recall: 59.34%
(200)Categorical accuracy on the test set: 61.59%, f1-score: 46.08% and  precision: 37.69% and  recall: 59.34%
(250)Categorical accuracy on the test set: 66.18%, f1-score: 46.08% and  precision: 37.69% and  recall: 59.34%
(300)Categorical accuracy on the test set: 64.48%, f1-score: 46.08% and  precision: 37.69% and  recall: 59.34%
(350)Categorical accuracy on the test set: 64.92%, f1-score: 46.08% and  precision: 37.69% and  recall: 59.34%
(400)Categorical accuracy on the test set: 62.63%, f1-score: 46.1% and  precision: 37.71% and  recall: 59.34%
(450)Categorical accuracy on the test set: 62.96%, f1-score: 46.09% and  precision: 37.69% and  recall: 59.34%
(500)Categorical accuracy on the test set: 64.7%, f1-score: 46.08% and  precision: 37.69% and  recall: 59.34%
(550)Categorical accuracy on the test set: 60.88%, f1-score: 46.11% and  precision: 37.72% and  recall: 59.34%
(600)Categorical accuracy on the test set: 67.98%, f1-score: 46.09% and  precision: 37.72% and  recall: 59.31%
(650)Categorical accuracy on the test set: 62.63%, f1-score: 46.1% and  precision: 37.71% and  recall: 59.34%
(700)Categorical accuracy on the test set: 62.03%, f1-score: 46.13% and  precision: 37.76% and  recall: 59.34%


# WOS5736 init:50 incr:50 CMBAL
(base 50)Categorical accuracy on the test set: 16.72%, f1-score: 0.0% and  precision: 0.0% and  recall: 0.0%.
(100)Categorical accuracy on the test set: 17.42%, f1-score: 0.0% and  precision: 0.0% and  recall: 0.0%
(150)Categorical accuracy on the test set: 26.48%, f1-score: 0.52% and  precision: 33.33% and  recall: 0.26%
(200)Categorical accuracy on the test set: 31.01%, f1-score: 6.99% and  precision: 75.0% and  recall: 3.68%
(250)Categorical accuracy on the test set: 44.25%, f1-score: 28.85% and  precision: 78.47% and  recall: 18.24%
(300)Categorical accuracy on the test set: 60.63%, f1-score: 47.61% and  precision: 79.88% and  recall: 34.26%
(350)Categorical accuracy on the test set: 63.41%, f1-score: 55.75% and  precision: 88.18% and  recall: 40.84%
(400)Categorical accuracy on the test set: 65.16%, f1-score: 59.22% and  precision: 81.38% and  recall: 46.63%
(450)Categorical accuracy on the test set: 71.43%, f1-score: 62.02% and  precision: 85.75% and  recall: 48.71% 
(500)Categorical accuracy on the test set: 66.55%, f1-score: 63.8% and  precision: 81.54% and  recall: 52.43% 
(550)Categorical accuracy on the test set: 68.64%, f1-score: 69.49% and  precision: 83.08% and  recall: 59.75%
(600)Categorical accuracy on the test set: 72.13%, f1-score: 72.42% and  precision: 82.81% and  recall: 64.54%
(650)Categorical accuracy on the test set: 77.0%, f1-score: 73.96% and  precision: 84.17% and  recall: 66.14%
(700)Categorical accuracy on the test set: 74.22%, f1-score: 69.93% and  precision: 78.53% and  recall: 63.07%
(750)Categorical accuracy on the test set: 67.6%, f1-score: 70.91% and  precision: 81.29% and  recall: 62.91%
(800)Categorical accuracy on the test set: 75.61%, f1-score: 73.71% and  precision: 86.54% and  recall: 64.21%
(850)Categorical accuracy on the test set: 70.03%, f1-score: 66.93% and  precision: 85.12% and  recall: 55.29%
(900)Categorical accuracy on the test set: 65.85%, f1-score: 54.24% and  precision: 81.57% and  recall: 40.8%
(950) Categorical accuracy on the test set: 66.2%, f1-score: 59.17% and  precision: 81.28% and  recall: 46.86%
(1000) Categorical accuracy on the test set: 60.28%, f1-score: 47.54% and  precision: 76.22% and  recall: 34.71%
(1050 )Categorical accuracy on the test set: 60.98%, f1-score: 56.71% and  precision: 83.77% and  recall: 42.88% 

# WOS5736 init:50 incr:10 RANDOM Epochs: 20
(Base)Categorical accuracy on the test set: 20.21%, f1-score: 0.0% and  precision: 0.0% and  recall: 0.0%.
(10)Categorical accuracy on the test set: 15.68%, f1-score: 16.29% and  precision: 11.01% and  recall: 31.29%
(20)Categorical accuracy on the test set: 14.29%, f1-score: 15.78% and  precision: 10.52% and  recall: 31.55%
(30)Categorical accuracy on the test set: 10.45%, f1-score: 15.78% and  precision: 10.52% and  recall: 31.55%
(40)Categorical accuracy on the test set: 14.29%, f1-score: 15.78% and  precision: 10.52% and  recall: 31.55%
(50)Categorical accuracy on the test set: 13.24%, f1-score: 15.76% and  precision: 10.53% and  recall: 31.29%
(60)Categorical accuracy on the test set: 11.15%, f1-score: 15.78% and  precision: 10.52% and  recall: 31.55%
(70)Categorical accuracy on the test set: 14.98%, f1-score: 15.78% and  precision: 10.52% and  recall: 31.55%
(80)Categorical accuracy on the test set: 13.24%, f1-score: 15.78% and  precision: 10.52% and  recall: 31.55%
(90)Categorical accuracy on the test set: 13.59%, f1-score: 15.78% and  precision: 10.52% and  recall: 31.55%
(100)Categorical accuracy on the test set: 15.68%, f1-score: 15.78% and  precision: 10.52% and  recall: 31.55% 

# WOS5736 init:50 incr:50 RANDOM Epochs: 20
(Base)Categorical accuracy on the test set: 10.8%, f1-score: 0.0% and  precision: 0.0% and  recall: 0.0%.
(50)Categorical accuracy on the test set: 13.94%, f1-score: 17.86% and  precision: 13.13% and  recall: 27.91%
(100)Categorical accuracy on the test set: 13.59%, f1-score: 17.27% and  precision: 11.55% and  recall: 34.22%
(150)Categorical accuracy on the test set: 15.33%, f1-score: 17.18% and  precision: 11.47% and  recall: 34.22%
(200)Categorical accuracy on the test set: 13.94%, f1-score: 17.27% and  precision: 11.55% and  recall: 34.22%
(250)Categorical accuracy on the test set: 13.59%, f1-score: 17.18% and  precision: 11.47% and  recall: 34.22%
(300)Categorical accuracy on the test set: 16.38%, f1-score: 17.12% and  precision: 11.42% and  recall: 34.22%
(350)Categorical accuracy on the test set: 13.94%, f1-score: 17.13% and  precision: 11.43% and  recall: 34.22%
(400)Categorical accuracy on the test set: 16.38%, f1-score: 17.46% and  precision: 11.69% and  recall: 34.48% 



# WOS5736 init:100 incr:100 RANDOM Epochs: 20
(Base)Categorical accuracy on the test set: 8.36%, f1-score: 1.02% and  precision: 22.22% and  recall: 0.52%.
(100)Categorical accuracy on the test set: 6.97%, f1-score: 12.47% and  precision: 8.7% and  recall: 22.04%
(200)Categorical accuracy on the test set: 20.56%, f1-score: 14.12% and  precision: 9.52% and  recall: 27.32% 
(300)Categorical accuracy on the test set: 20.91%, f1-score: 13.77% and  precision: 9.23% and  recall: 27.06%
(400)Categorical accuracy on the test set: 19.86%, f1-score: 12.94% and  precision: 8.65% and  recall: 25.72%
(500)Categorical accuracy on the test set: 20.21%, f1-score: 12.09% and  precision: 8.07% and  recall: 24.13% 
(600)Categorical accuracy on the test set: 19.86%, f1-score: 12.63% and  precision: 8.42% and  recall: 25.2%
(700)Categorical accuracy on the test set: 14.98%, f1-score: 12.06% and  precision: 8.04% and  recall: 24.13%
(800)Categorical accuracy on the test set: 19.16%, f1-score: 12.57% and  precision: 8.38% and  recall: 25.2%
(900)Categorical accuracy on the test set: 10.8%, f1-score: 12.6% and  precision: 8.4% and  recall: 25.2%
(1000)Categorical accuracy on the test set: 11.15%, f1-score: 11.82% and  precision: 7.89% and  recall: 23.61% 

# WOS5736 init:100 incr:10 RANDOM Epochs: 20
(Base)Categorical accuracy on the test set: 15.33%, f1-score: 0.0% and  precision: 0.0% and  recall: 0.0%.
(10)Categorical accuracy on the test set: 13.24%, f1-score: 17.34% and  precision: 12.12% and  recall: 30.51%
(20)Categorical accuracy on the test set: 14.29%, f1-score: 17.42% and  precision: 11.69% and  recall: 34.19%
(30)Categorical accuracy on the test set: 17.77%, f1-score: 16.72% and  precision: 11.21% and  recall: 32.85%
(40)Categorical accuracy on the test set: 17.42%, f1-score: 17.23% and  precision: 11.52% and  recall: 34.19%
(50)Categorical accuracy on the test set: 19.16%, f1-score: 16.51% and  precision: 11.02% and  recall: 32.85%
(60)Categorical accuracy on the test set: 14.29%, f1-score: 16.41% and  precision: 10.93% and  recall: 32.85%
(70)Categorical accuracy on the test set: 16.03%, f1-score: 16.29% and  precision: 10.86% and  recall: 32.59%
(80)Categorical accuracy on the test set: 11.85%, f1-score: 16.37% and  precision: 10.9% and  recall: 32.85%
(90)Categorical accuracy on the test set: 12.54%, f1-score: 16.14% and  precision: 10.76% and  recall: 32.33%
(100)Categorical accuracy on the test set: 12.2%, f1-score: 15.89% and  precision: 10.59% and  recall: 31.81% 


# WOS5736 init:100 incr:100 RANDOM Epochs: 50
(Base)Categorical accuracy on the test set: 42.16%, f1-score: 0.52% and  precision: 33.33% and  recall: 0.26%.
(100)Categorical accuracy on the test set: 27.87%, f1-score: 27.56% and  precision: 19.78% and  recall: 45.62%
(200)Categorical accuracy on the test set: 27.53%, f1-score: 24.13% and  precision: 16.17% and  recall: 47.55%
(300)Categorical accuracy on the test set: 26.13%, f1-score: 22.66% and  precision: 15.13% and  recall: 45.14%
(400)Categorical accuracy on the test set: 26.13%, f1-score: 21.35% and  precision: 14.26% and  recall: 42.5%
(500)Categorical accuracy on the test set: 24.39%, f1-score: 21.21% and  precision: 14.13% and  recall: 42.46%
(600)Categorical accuracy on the test set: 21.95%, f1-score: 21.34% and  precision: 14.22% and  recall: 42.73% 


# WOS5736 init:300 incr:50 RANDOM Epochs: 20
(Base)Categorical accuracy on the test set: 52.26%, f1-score: 16.51% and  precision: 92.86% and  recall: 9.06%.
(50)Categorical accuracy on the test set: 37.28%, f1-score: 40.76% and  precision: 35.19% and  recall: 48.43%
(100)Categorical accuracy on the test set: 39.02%, f1-score: 37.8% and  precision: 27.91% and  recall: 58.54%
(150)Categorical accuracy on the test set: 42.16%, f1-score: 34.13% and  precision: 23.86% and  recall: 59.93%
(200)Categorical accuracy on the test set: 37.28%, f1-score: 34.06% and  precision: 23.38% and  recall: 62.72%
(250)Categorical accuracy on the test set: 42.16%, f1-score: 34.38% and  precision: 23.4% and  recall: 64.81%
(300)Categorical accuracy on the test set: 38.33%, f1-score: 31.23% and  precision: 21.07% and  recall: 60.28%
(350)Categorical accuracy on the test set: 36.93%, f1-score: 31.14% and  precision: 21.0% and  recall: 60.28%
(400)Categorical accuracy on the test set: 37.98%, f1-score: 32.09% and  precision: 21.69% and  recall: 61.67%
(450)Categorical accuracy on the test set: 37.28%, f1-score: 31.1% and  precision: 20.91% and  recall: 60.63%
(500)Categorical accuracy on the test set: 34.15%, f1-score: 30.15% and  precision: 20.26% and  recall: 58.89%
(550)Categorical accuracy on the test set: 37.98%, f1-score: 30.17% and  precision: 20.37% and  recall: 58.19% 

# WOS5736 init:300 incr:50 CMBAL Epochs: 20


# Yelp RANDOM init:200 increase:200 budget:2000
(Base)Categorical accuracy on the test set: 25.45%, f1-score: 1.13% and  precision: 43.43% and  recall: 0.58%.
(200)Categorical accuracy on the test set: 27.84%, f1-score: 32.29% and  precision: 23.03% and  recall: 54.04%
(400)Categorical accuracy on the test set: 31.5%, f1-score: 31.86% and  precision: 22.08% and  recall: 57.2%
(600)Categorical accuracy on the test set: 30.68%, f1-score: 31.43% and  precision: 21.37% and  recall: 59.4%
(800)Categorical accuracy on the test set: 30.84%, f1-score: 31.15% and  precision: 20.91% and  recall: 61.04%
(1000)Categorical accuracy on the test set: 31.75%, f1-score: 30.85% and  precision: 20.67% and  recall: 60.78%
(1200)Categorical accuracy on the test set: 29.42%, f1-score: 30.74% and  precision: 20.57% and  recall: 60.84% 

# Yelp RANDOM init:100 increase:100 budget:1000
(Base)Categorical accuracy on the test set: 21.19%, f1-score: 6.11% and  precision: 32.77% and  recall: 3.45%.
(100) Categorical accuracy on the test set: 20.58%, f1-score: 20.18% and  precision: 20.76% and  recall: 19.66%. 


# Arxiv High Confidence Value 100 100 20
Categorical accuracy on the test set: 59.03%, f1-score: 42.97% and  precision: 60.25% and  recall: 33.54%. 



[distil.active_learning_strategies.core_set](https://decile-team-distil.readthedocs.io/en/latest/_modules/index.html)