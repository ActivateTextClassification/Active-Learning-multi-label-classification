import argparse
import json
import os
import time
from utils.metrics import make_model
from train.active_learning import active_learning_loop
from train.train_inference import train_model, precision_m, recall_m
from utils.data_loader import preprocessing_dataset, load_dataset_by_name
import tensorflow_addons as tfa

"""
## Perform exploratory data analysis

In this section, we first load the dataset into a `pandas` dataframe and then perform
some basic exploratory data analysis (EDA).
"""

# Splitting the test set further into validation
# and new test sets.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=False, default="")
    parser.add_argument('--loop', required=False, default="hloop")
    parser.add_argument('--config', required=True, help="Please give a config.json file with "
                                                        "training/model/data/param details")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    params = config["params"]
    params['loop'] = args.loop
    text_name = params["text"]
    label_name = params["label"]
    split_ratio = params["split_ratio"]
    max_seq_len = params["max_seq_len"]
    batch_size = params["batch_size"]
    padding_token = params["padding_token"]
    num_samples = params["init_samples"]
    epochs = params["epochs"]
    """
    ## 1. Load data
    """
    data_df = load_dataset_by_name(name=params['all_path'], split_ratio=0.3)
    train_df, val_df, test_df = data_df
    print(f"Number of rows in training set: {len(train_df)}")
    print(f"Number of rows in validation set: {len(val_df)}")
    print(f"Number of rows in test set: {len(test_df)}")
    """
    ## 2. Prepare dataset
    """
    zipped_dataset, zipped_df, params, text_vectorizer_base = preprocessing_dataset(data_df, params)
    train_dataset, validation_dataset, test_dataset = zipped_dataset

    """
    ## 3. Create and Train a text classification model
    """
    model_param = config["model"]
    shallow_mlp_model = train_model(train_dataset, validation_dataset, model_param, params)
    """
    ## 4. Evaluate the model
    """
    loss, categorical_acc, f1_score, precision, recall = shallow_mlp_model.evaluate(test_dataset)
    print(f"Categorical accuracy on the test set: {round(categorical_acc * 100, 2)}%,"
          f" f1-score: {round(f1_score * 100, 2)}% and "
          f" precision: {round(precision * 100, 2)}% and "
          f" recall: {round(recall * 100, 2)}%. ")
    init_rst = (f"Budget used {0} - "
                f"Categorical accuracy on the test set: {round(categorical_acc * 100, 2)}%,"
                f" loss: {round(loss, 2)} and "
                f" f1-score: {round(f1_score * 100, 2)}% and "
                f" precision: {round(precision * 100, 2)}% and "
                f" recall: {round(recall * 100, 2)}% \n ")
    params['init_rst'] = init_rst
    """
    ## 5. Active learning loop
    """
    if not os.path.exists("output"):
        os.mkdir("output")
    if not os.path.exists("output/" + params['loop']):
        os.mkdir("output/" + params['loop'])
    exp_folder = "output/" + params['loop'] + '/' + params['name'] + "_epoch_" + str(params['epochs']) + "_init_" + str(
        num_samples) + "_f1_" + str(
        round(f1_score * 100, 2)) + "_at_" + str(round(time.time())) + "/"
    if not os.path.exists(exp_folder):
        os.mkdir(exp_folder)
    # Save base model
    shallow_mlp_model.save_weights(exp_folder + "model_ckpt")
    result_file_name = params['name'] + "_init_" + str(params['init_samples']) \
                       + "_batch_" + str(params['batch_size']) + "_freq_" + str(params['freq']) \
                       + "_budget_" + str(params['budget']) + "_epochs_" + str(params['epochs']) \
                       + "_" + str(round(time.time())) + '.txt'
    strategy_list = ["RANDOM",
                     "LEAST_PROB",
                     "HIGH_ENTROPY",
                     "MC_DROPOUT",
                     "CMBAL"
                     ]
    # if passing strategy, run it
    if args.strategy != "":
        strategy_list = [args.strategy]
    # else run all the strategies
    for strategy in strategy_list:
        params['strategy'] = strategy
        lookup = params['lookup']
        load_model = make_model(int(model_param["dim1"]), int(model_param["dim2"]), lookup=params["lookup"])
        # load saved base model
        load_model.load_weights(exp_folder + "model_ckpt")
        load_model.compile(
            loss="binary_crossentropy", optimizer="adam",
            metrics=["categorical_accuracy", tfa.metrics.F1Score(num_classes=lookup.vocabulary_size(), average='micro'),
                     precision_m, recall_m]
        )
        strategy_folder = exp_folder + strategy + "/"
        if not os.path.exists(strategy_folder):
            os.mkdir(strategy_folder)
        result_file = strategy_folder + result_file_name
        params['result'] = result_file
        active_learning_loop(zipped_df, data_df, params, load_model, text_vectorizer_base)
