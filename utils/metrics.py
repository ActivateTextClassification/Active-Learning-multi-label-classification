import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy.stats import entropy
from tensorflow import keras
from tensorflow.keras import layers

auto = tf.data.AUTOTUNE


def compute_correlation(start_df, dim, lookup):
    cor_matrix = [[0 for i in range(dim)] for j in range(dim)]
    for label in start_df["labels"].tolist():
        label = lookup(label).numpy().tolist()

        indices = [i for i, e in enumerate(label) if e != 0]
        if len(indices) > 1:
            for m in range(0, len(indices) - 1):
                for n in range(m + 1, len(indices)):
                    row = indices[m]
                    col = indices[n]
                    temp = cor_matrix[row][col]
                    cor_matrix[row][col] = temp + 1
    pre_two_array = np.array(cor_matrix)
    norm = np.linalg.norm(pre_two_array)  # To find the norm of the array
    normalized_matrix = (pre_two_array + 0.001) / (norm + 2 * 0.001)  # Formula used to perform array normalization
    return normalized_matrix


# compute label sparseness
def compute_label_sparse(start_df, dim, lookup):
    label_counts = [0 for i in range(dim)]
    for label in start_df["labels"].tolist():
        label = lookup(label).numpy().tolist()
        for i in range(0, len(label_counts)):
            label_counts[i] = label_counts[i] + label[i]
    return label_counts


# compute data point uncertainty based on top k average entropies
def compute_uncertainty(unlabelled_df, model_for_inference, top_k):
    avg_entropy_list = []
    inference_dataset = make_dataset(unlabelled_df, is_train=False)
    text_batch, label_batch = next(iter(inference_dataset))
    predicted_probabilities = model_for_inference.predict(text_batch)
    # for the top_k probabilities , compute average entropy
    for prob in predicted_probabilities:
        sorted_prob = sorted(prob)
        temp_entropy = 0
        for item in sorted_prob[-top_k:]:
            temp_entropy = temp_entropy + entropy([item, 1 - item], base=2)
        avg_entropy_list.append(temp_entropy / top_k)
    return avg_entropy_list


# compute data point diversity based on average euclidean distance between labeled set and unlabeled data point
def compute_diversity(start_df, unlabelled_df, text_vectorizer, intermediate_layer_model, params):
    avg_diversity_list = []
    len_start = len(start_df)
    len_unlabelled = len(unlabelled_df)
    all_df = pd.concat([start_df, unlabelled_df], axis=0)
    # Create a small dataset just for demoing inference.
    inference_dataset = make_dataset(all_df, params, is_train=False)
    # text_batch, label_batch = next(iter(inference_dataset))
    inference_feature_dataset = inference_dataset.map(
        lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
    ).prefetch(auto)

    # the hidden representations, which can be used to compute diversity in active learning
    hidden_outputs = intermediate_layer_model.predict(inference_feature_dataset)
    features_start = hidden_outputs[0:len_start]
    features_unlabelled = hidden_outputs[-len_unlabelled:]

    for feature in features_unlabelled:
        temp_diversity = 0
        for feature_train in features_start:
            dst = distance.euclidean(feature, feature_train)
            temp_diversity = temp_diversity + dst
        temp_diversity = temp_diversity / len_start
        avg_diversity_list.append(temp_diversity)
    return avg_diversity_list


def make_dataset(dataframe, params, is_train=True):
    lookup = params["lookup"]
    batch_size = params["batch_size"]
    label_name = params["label"]
    text_name = params["text"]
    labels = tf.ragged.constant(dataframe[label_name].values)
    binarized_label = lookup(labels).numpy()
    print("make dataset:", labels.shape, dataframe[text_name].values.shape)
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe[text_name].values, binarized_label)
    )
    dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
    dataset = dataset.map(unify_text_length, num_parallel_calls=auto).cache()
    return dataset.batch(batch_size)


def unify_text_length(text, label, max_seq_len=150):
    # Split the given abstract and calculate its length.
    word_splits = tf.strings.split(text, sep=" ")
    sequence_length = tf.shape(word_splits)[0]

    # Calculate the padding amount.
    padding_amount = max_seq_len - sequence_length

    # Check if we need to pad or truncate.
    if padding_amount > 0:
        unified_text = tf.pad([text], [[0, padding_amount]], constant_values="<pad>")
        unified_text = tf.strings.reduce_join(unified_text, separator="")
    else:
        unified_text = tf.strings.reduce_join(word_splits[:max_seq_len], separator=" ")

    # The expansion is needed for subsequent vectorization.
    return tf.expand_dims(unified_text, -1), label


def map_dataset(df, text_vectorizer):
    return df.map(
        lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
    ).prefetch(auto)


def make_model(dim1, dim2, lookup):
    mlp_model = keras.Sequential(
        [
            layers.Dense(dim1, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(dim2, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(lookup.vocabulary_size(), activation="sigmoid"),
        ]  # More on why "sigmoid" has been used here in a moment.
    )
    return mlp_model


def plot_result(item, history):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


def invert_multi_hot(encoded_labels, vocab):
    """Reverse a single multi-hot encoded label to a tuple of vocab terms."""
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    return np.take(vocab, hot_indices)


def get_text_vectorizer(vocabulary_size, ngrams):
    return layers.TextVectorization(max_tokens=vocabulary_size, ngrams=ngrams, output_mode="tf_idf")
