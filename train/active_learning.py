from utils.metrics import make_batch_dataset, map_dataset, \
    get_text_vectorizer
from tensorflow import keras
import tensorflow as tf
from utils.strategies import switch_strategy


def active_learning_loop(zipped_df, data_df, params, shallow_mlp_model, text_vectorizer):
    print("Start Active learning loop:")
    budget = params["budget"]
    freq = params["freq"]  # the number of annotations in each AL round
    b = params["init_b"]
    epochs = params["epochs"]
    start_df, init_df, unlabelled_df = zipped_df
    params['init_df'] = init_df
    # we have initialized the model based on init_df, so just go to the active learning part
    result_file_name = params['result']
    with open(result_file_name, "a+") as result:
        result.write(params['init_rst'])
        while b < budget:
            model_for_inference = keras.Sequential([text_vectorizer, shallow_mlp_model])

            intermediate_layer_model = keras.Model(inputs=shallow_mlp_model.input,
                                                   outputs=shallow_mlp_model.layers[1].output)

            # apply active learning strategies
            probability_based_strategy_list = ["RANDOM", "LEAST_PROB", "HIGH_ENTROPY", "MC_DROPOUT"]
            diversity_based_strategy_list = ["CMBAL"]
            if params['strategy'] in diversity_based_strategy_list:
                selected_df = switch_strategy("diversity")(shallow_mlp_model, intermediate_layer_model,
                                                           text_vectorizer, init_df,
                                                           unlabelled_df, params)
            elif params['strategy'] in probability_based_strategy_list:
                selected_df = switch_strategy("probability")(model_for_inference, unlabelled_df, params)
            else:
                # TODO
                break
            # Add newly labeled data
            rows = selected_df.index
            init_df = init_df.append(selected_df, ignore_index=True)
            init_df.reset_index(drop=True, inplace=True)
            unlabelled_df.drop(rows, inplace=True)
            unlabelled_df = unlabelled_df.reset_index(drop=True)
            print(f"unlabelled data size: {len(unlabelled_df)}")
            print(f"init data size: {len(init_df)}")
            # update init df
            params['init_df'] = init_df
            # Retrain the model
            start_df, val_df, test_df = data_df
            train_dataset = make_batch_dataset(init_df, params, is_train=True)
            validation_dataset = make_batch_dataset(val_df, params, is_train=False)
            test_dataset = make_batch_dataset(test_df, params, is_train=False)
            # update vectorizer by vocab size
            vocabulary_size = params['vocabulary_size']
            print("Updated Vocabulary Size:{}".format(vocabulary_size))
            text_vectorizer = get_text_vectorizer(vocabulary_size, ngrams=2)
            # training set.
            with tf.device("/CPU:0"):
                text_vectorizer.adapt(train_dataset.map(lambda text, label: text))
            train_dataset = map_dataset(train_dataset, text_vectorizer)
            validation_dataset = map_dataset(validation_dataset, text_vectorizer)
            test_dataset = map_dataset(test_dataset, text_vectorizer)
            # fit model using updated train dataset
            shallow_mlp_model.fit(train_dataset, validation_data=validation_dataset, batch_size=256, epochs=epochs)
            loss, categorical_acc, f1_score, precision, recall = shallow_mlp_model.evaluate(test_dataset)
            b = b + freq
            print('Budget used:' + str(b))
            print(f"Budget used {b} - "
                  f"Categorical accuracy on the test set: {round(categorical_acc * 100, 2)}%,"
                  f" loss: {round(loss, 2)} and "
                  f" f1-score: {round(f1_score * 100, 2)}% and "
                  f" precision: {round(precision * 100, 2)}% and "
                  f" recall: {round(recall * 100, 2)}% ")

            result.write(f"Budget used {b} - "
                         f"Categorical accuracy on the test set: {round(categorical_acc * 100, 2)}%,"
                         f" loss: {round(loss, 2)} and "
                         f" f1-score: {round(f1_score * 100, 2)}% and "
                         f" precision: {round(precision * 100, 2)}% and "
                         f" recall: {round(recall * 100, 2)}% \n ")
        result.close()
