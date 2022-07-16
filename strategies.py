import tensorflow as tf
from scipy.stats import entropy
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

from utils.metrics import compute_correlation, compute_diversity, compute_label_sparse, make_dataset


class Strategy:
    def __init__(self, dataset, device, model):
        self.dataset = dataset
        self.device = device
        self.model = model


def switch_strategy(method):
    switcher = {
        "RANDOM": RandomSampling,  # AL baseline, randomly sample from unlabeled data.
        "HIGH_ENTROPY": HighEntropy,
        "LEAST_PROB": LeastConfidence,  # uncertainty sampling / least confidence;
        "MC_DROPOUT": MCDropout,
    }
    return switcher.get(method, "Invalid Strategy!")


def RandomSampling(model_for_inference, unlabelled_df, params):
    # Create a small dataset just for demoing inference.
    freq_samples = int(params['freq'])
    loop = params['loop']
    top_k = int(params['topK'])

    random_selected_df = unlabelled_df.sample(n=freq_samples)
    if loop == 'mloop':
        random_selected_dataset = make_dataset(random_selected_df, params, is_train=False)
        predicted_labels = []
        for _, (text_batch, label_batch) in enumerate(random_selected_dataset):
            # the predicted probabilities, which can be used to compute uncertainty
            predicted_probabilities = model_for_inference.predict(text_batch)
            lookup = params['lookup']
            for i in range(len(predicted_probabilities)):
                top_labels = [x for _, x in sorted(
                    zip(predicted_probabilities[i], lookup.get_vocabulary()),
                    key=lambda pair: pair[0],
                    reverse=True,
                )][:top_k]
                predicted_labels.append(top_labels)
        random_selected_df[params['label']] = predicted_labels
    else:
        predicted_labels = random_selected_df[params['label']]
    return random_selected_df, predicted_labels


def HighConfidence(model_for_inference, unlabelled_df, params):
    print("Select Highest Confidence Score")
    freq_samples = int(params['freq'])
    batch_size = int(params['batch_size'])
    top_k = int(params["topK"])
    loop = params['loop']
    selected_batches = int(freq_samples / batch_size)
    unlabelled_dataset = make_dataset(unlabelled_df, params, is_train=False)
    unlabelled_df_index = unlabelled_df.index
    selected_idx = []
    df = pd.DataFrame(columns=['prob', 'idx', 'labels'])
    for i, (text_batch, label_batch) in enumerate(unlabelled_dataset):
        if text_batch.shape[0] == batch_size:
            predicted_labels = []
            predicted_probabilities = model_for_inference.predict(text_batch)
            avg_prob = sum(sum(predicted_probabilities)) / batch_size
            idx = list(range(batch_size * i, batch_size * (i + 1)))
            for j in range(len(predicted_probabilities)):
                top_labels = [x for _, x in sorted(
                    zip(predicted_probabilities[j], params['lookup'].get_vocabulary()),
                    key=lambda pair: pair[0],
                    reverse=True,
                )][:top_k]
                predicted_labels.append(top_labels)
            df.loc[i] = [avg_prob, np.array(unlabelled_df_index[idx]), np.array(predicted_labels)]
        else:
            print("last batch, don't consider")
    high_prob_df = df.sort_values(by=['prob'], ascending=False).head(selected_batches)
    for indexArr in high_prob_df.idx:
        selected_idx += indexArr.tolist()
    # assert len(selected_idx) == selected_batches * batch_size
    selected_df = unlabelled_df.loc[selected_idx]
    if loop == 'mloop':
        selected_labels = []
        for labels in high_prob_df.labels:
            for label in labels:
                selected_labels.append(label)
        selected_df[params['label']] = selected_labels
    else:
        selected_labels = selected_df[params['label']]
    return selected_df, selected_labels


def HighEntropy(model_for_inference, unlabelled_df, params):
    print("Select Highest Entropy Score")
    freq_samples = int(params['freq'])
    batch_size = int(params['batch_size'])
    top_k = int(params["topK"])
    loop = params['loop']
    selected_batches = int(freq_samples / batch_size)
    unlabelled_dataset = make_dataset(unlabelled_df, params, is_train=False)
    unlabelled_df_index = unlabelled_df.index
    selected_idx = []
    df = pd.DataFrame(columns=['entropy', 'idx', 'labels'])
    for i, (text_batch, label_batch) in enumerate(unlabelled_dataset):
        if text_batch.shape[0] == batch_size:
            predicted_labels = []
            predicted_probabilities = model_for_inference.predict(text_batch)
            prob_entropy = entropy([max(p) for p in predicted_probabilities])
            idx = list(range(batch_size * i, batch_size * (i + 1)))
            for j in range(len(predicted_probabilities)):
                top_labels = [x for _, x in sorted(
                    zip(predicted_probabilities[j], params['lookup'].get_vocabulary()),
                    key=lambda pair: pair[0],
                    reverse=True,
                )][:top_k]
                predicted_labels.append(top_labels)
            df.loc[i] = [prob_entropy, np.array(unlabelled_df_index[idx]), np.array(predicted_labels)]
        else:
            print("last batch, don't consider")
    high_entropy_df = df.sort_values(by=['entropy'], ascending=False).head(selected_batches)
    for indexArr in high_entropy_df.idx:
        selected_idx += indexArr.tolist()
    # assert len(selected_idx) == selected_batches * batch_size
    selected_df = unlabelled_df.loc[selected_idx]
    selected_labels = []
    if loop == 'mloop':
        for labels in high_entropy_df.labels:
            for label in labels:
                selected_labels.append(label)
        selected_df[params['label']] = selected_labels
    else:
        selected_labels = selected_df[params['label']]
    return selected_df, selected_labels


def MCDropout(model_for_inference, unlabelled_df, params):
    print("Select MC Dropout Score")
    freq_samples = int(params['freq'])
    batch_size = int(params['batch_size'])
    top_k = int(params["topK"])
    loop = params['loop']
    selected_batches = int(freq_samples / batch_size)
    unlabelled_dataset = make_dataset(unlabelled_df, params, is_train=False)
    unlabelled_df_index = unlabelled_df.index
    selected_idx = []
    df = pd.DataFrame(columns=['entropy', 'idx', 'labels'])
    for i, (text_batch, label_batch) in enumerate(unlabelled_dataset):
        if text_batch.shape[0] == batch_size:
            predicted_labels = []
            predicted_probabilities = model_for_inference.predict(text_batch)
            prob_entropy = entropy([np.mean(p) for p in predicted_probabilities])
            idx = list(range(batch_size * i, batch_size * (i + 1)))
            for j in range(len(predicted_probabilities)):
                top_labels = [x for _, x in sorted(
                    zip(predicted_probabilities[j], params['lookup'].get_vocabulary()),
                    key=lambda pair: pair[0],
                    reverse=True,
                )][:top_k]
                predicted_labels.append(top_labels)
            df.loc[i] = [prob_entropy, np.array(unlabelled_df_index[idx]), np.array(predicted_labels)]
        else:
            print("last batch, don't consider")
    high_entropy_df = df.sort_values(by=['entropy'], ascending=False).head(selected_batches)
    for indexArr in high_entropy_df.idx:
        selected_idx += indexArr.tolist()
    # assert len(selected_idx) == selected_batches * batch_size
    selected_df = unlabelled_df.loc[selected_idx]
    selected_labels = []
    if loop == 'mloop':
        for labels in high_entropy_df.labels:
            for label in labels:
                selected_labels.append(label)
        selected_df[params['label']] = selected_labels
    else:
        selected_labels = selected_df[params['label']]
    return selected_df, selected_labels


def LeastConfidence(model_for_inference, unlabelled_df, params):
    print("Select Least Confidence Score")
    freq_samples = int(params['freq'])
    batch_size = int(params['batch_size'])
    top_k = int(params["topK"])
    loop = params['loop']
    selected_batches = int(freq_samples / batch_size)
    unlabelled_dataset = make_dataset(unlabelled_df, params, is_train=False)
    unlabelled_df_index = unlabelled_df.index
    selected_idx = []
    df = pd.DataFrame(columns=['prob', 'idx', 'labels'])
    for i, (text_batch, label_batch) in enumerate(unlabelled_dataset):
        if text_batch.shape[0] == batch_size:
            predicted_labels = []
            predicted_probabilities = model_for_inference.predict(text_batch)
            avg_prob = sum(sum(predicted_probabilities)) / batch_size
            idx = list(range(batch_size * i, batch_size * (i + 1)))
            for j in range(len(predicted_probabilities)):
                top_labels = [x for _, x in sorted(
                    zip(predicted_probabilities[j], params['lookup'].get_vocabulary()),
                    key=lambda pair: pair[0],
                    reverse=True,
                )][:top_k]
                predicted_labels.append(top_labels)
            df.loc[i] = [avg_prob, np.array(unlabelled_df_index[idx]), np.array(predicted_labels)]
        else:
            print("last batch, don't consider")
    least_prob_df = df.sort_values(by=['prob'], ascending=True).head(selected_batches)
    for indexArr in least_prob_df.idx:
        selected_idx += indexArr.tolist()
    # assert len(selected_idx) == selected_batches * batch_size
    selected_df = unlabelled_df.loc[selected_idx]
    selected_labels = []
    if loop == 'mloop':
        for labels in least_prob_df.labels:
            for label in labels:
                selected_labels.append(label)
        selected_df[params['label']] = selected_labels
    else:
        selected_labels = selected_df[params['label']]
    return selected_df, selected_labels


if __name__ == '__main__':
    print("")
