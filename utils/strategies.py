from scipy.stats import entropy
import pandas as pd
import numpy as np
import tensorflow as tf
from utils.metrics import compute_correlation, compute_diversity, compute_label_sparse, make_batch_dataset, \
    make_test_dataset, map_dataset
from keras import backend as K


def switch_strategy(method):
    switcher = {
        "probability": acquire_probability_based_data_samples,  # AL baseline, randomly sample from unlabeled data.
        "diversity": acquire_diversity_based_data_samples,
    }
    return switcher.get(method, "Invalid Strategy!")


def acquire_diversity_based_data_samples(model_for_inference, intermediate_layer_model,
                                         text_vectorizer, init_df, unlabelled_df, params):
    print("diversity based data acquiring")
    phi_12 = [3, 6, 8, 10000, 2, 234, 5]
    del phi_12[5]
    print(phi_12)
    vocab = params['vocab']
    lookup = params['lookup']
    top_k = params['topK']
    freq_samples = params['freq']
    batch_size = int(params['batch_size'])
    selected_batches = int(freq_samples / batch_size)

    # do not shuffle data if is_train=False
    # unlabelled_dataset = make_batch_dataset(unlabelled_df, params, is_train=False)
    unlabelled_dataset = make_test_dataset(unlabelled_df, params, is_train=False)
    unlabelled_dataset = map_dataset(unlabelled_dataset, text_vectorizer)

    #
    suggest_annotate_index_list = []
    # cor_matrix = compute_correlation(init_df, len(vocab), lookup)
    # label_counts = compute_label_sparse(init_df, len(vocab), lookup)
    # diversity_list = compute_diversity(init_df, unlabelled_df, text_vectorizer,
    #                                    intermediate_layer_model, params)
    for i, (text_batch, label_batch) in enumerate(unlabelled_dataset):
        # the predicted probabilities, which can be used to compute uncertainty
        predicted_probabilities = model_for_inference.eval(text_batch)
        # shallow_mlp_model.evaluate(text_batch)
        # output = shallow_mlp_model.layers[2].get_weights()
        breakpoint()
        # label_indexes = []
        # predicted_labels = []
        # cor_uncertainty = []
        # for prob in predicted_probabilities:
        #     predicted_label = tf.math.argmax(prob).numpy()
        #     predicted_labels.append(predicted_label)
        #     phi_1 = 0
        #     top_indexes = sorted(range(len(prob)), key=lambda k: prob[k])[-top_k:]
        #     for m in range(0, len(top_indexes)):
        #         label_uncertainty = entropy([prob[m], 1 - prob[m]], base=2)
        #         w = 0
        #         for n in range(m, len(top_indexes)):
        #             w += cor_matrix[m][n]
        #         w1 = 1 - w / cor_matrix.shape[0]
        #         w2 = 1 - label_counts[m] / len(init_df)
        #         phi_1 = phi_1 + w1 * w2 * label_uncertainty
        #     phi_1 = phi_1 / len(top_indexes)
        #     cor_uncertainty.append(phi_1)
        #     label_indexes.append(top_indexes)
        # phi_12 = [a * b for a, b in zip(cor_uncertainty, diversity_list)]
        # top_suggestions = sorted(range(len(phi_12)), key=lambda l: phi_12[l])[-selected_batches:]
        # suggest_annotate_index = sorted(top_suggestions)
        # suggest_annotate_index_list += suggest_annotate_index
    return unlabelled_df.iloc[suggest_annotate_index_list]


def acquire_probability_based_data_samples(model_for_inference, unlabelled_df, params):
    print("probability based data acquiring")
    freq_samples = int(params['freq'])
    batch_size = int(params['batch_size'])
    selected_batches = int(freq_samples / batch_size)
    # do not shuffle data if is_train=False
    unlabelled_dataset = make_batch_dataset(unlabelled_df, params, is_train=False)
    unlabelled_df_index = unlabelled_df.index
    selected_idx = []
    strategy = params['strategy']
    df = pd.DataFrame(columns=['strategy', 'idx'])
    print("Applied data strategy:{}".format(strategy))
    for i, (text_batch, label_batch) in enumerate(unlabelled_dataset):
        if text_batch.shape[0] == batch_size:
            predicted_probabilities = model_for_inference.predict(text_batch)
            if strategy == 'RANDOM':
                return unlabelled_df.sample(n=freq_samples)
            elif strategy in ["LEAST_PROB", 'HIGH_PROB']:
                strategy_value = sum(sum(predicted_probabilities)) / batch_size
            elif strategy in ['LEAST_ENTROPY', 'HIGH_ENTROPY']:
                strategy_value = entropy([max(p) for p in predicted_probabilities])
            elif strategy == 'MC_DROPOUT':
                strategy_value = entropy([np.mean(p) for p in predicted_probabilities])
            else:
                return
            idx = list(range(batch_size * i, batch_size * (i + 1)))
            df.loc[i] = [strategy_value, np.array(unlabelled_df_index[idx])]
        else:
            print("last batch, don't consider")
    ascending_flag = True if strategy.__contains__('LEAST') else False
    strategy_value_df = df.sort_values(by=['strategy'], ascending=ascending_flag).head(selected_batches)
    for indexArr in strategy_value_df.idx:
        selected_idx += indexArr.tolist()
    selected_df = unlabelled_df.loc[selected_idx]
    return selected_df


if __name__ == '__main__':
    print("")
