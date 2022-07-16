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
        "HIGH_PROB": HighConfidence,  # select the top scored instances by the model.
        "HIGH_ENTROPY": HighEntropy,
        "LEAST_PROB": LeastConfidence,  # uncertainty sampling / least confidence;
        "GREEDY": GreedyCoreSet,  # The greedy method from Sener and Savarese 2017
        # "DAL": DiscriminativeRepresentation,  # Discriminative representation sampling;
        # "LEAST_ENTROPY": LeastEntropy,
        "MC_DROPOUT": MCDropout,
        "CMBAL": Cmbal,
        # "CMBAL_V1": CmbalVariant1,
        # "CMBAL_V2": CmbalVariant2
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


def GreedyCoreSet(model_for_inference, unlabelled_df, params):
    print("Select Greedy CoreSet")
    freq_samples = int(params['freq'])
    batch_size = int(params['batch_size'])
    top_k = int(params["topK"])
    selected_batches = int(freq_samples / batch_size)
    init_df = params['init_df']
    embedding_unlabeled = get_embedding(model_for_inference, unlabelled_df, params)
    embedding_labeled = get_embedding(model_for_inference, init_df, params)
    chosen = furthest_first(embedding_unlabeled, embedding_labeled, freq_samples)
    breakpoint()
    print("GreedyCoreSet")
    return [], []


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


def Cmbal(model_for_inference, intermediate_layer_model, text_vectorizer, init_df, unlabelled_df, params):
    phi_12 = [3, 6, 8, 10000, 2, 234, 5]
    del phi_12[5]
    print(phi_12)
    vocab = params['vocab']
    lookup = params['lookup']
    top_k = params['topK']
    freq_samples = params['freq']
    batch_size = int(params['batch_size'])
    selected_batches = int(freq_samples / batch_size)

    cor_matrix = compute_correlation(init_df, len(vocab), lookup)
    label_counts = compute_label_sparse(init_df, len(vocab), lookup)

    # Create a small dataset just for demoing inference.
    unlabelled_dataset = make_dataset(unlabelled_df, params, is_train=False)
    #
    suggest_annotate_index_list = []
    diversity_list = compute_diversity(init_df, unlabelled_df, text_vectorizer,
                                       intermediate_layer_model, params)
    for i, (text_batch, label_batch) in enumerate(unlabelled_dataset):
        # text_batch, label_batch = next(iter(unlabelled_dataset))
        # the predicted probabilities, which can be used to compute uncertainty
        predicted_probabilities = model_for_inference.predict(text_batch)
        label_indexes = []
        predicted_labels = []
        cor_uncertainty = []
        for prob in predicted_probabilities:
            predicted_label = tf.math.argmax(prob).numpy()
            predicted_labels.append(predicted_label)
            phi_1 = 0
            top_indexes = sorted(range(len(prob)), key=lambda k: prob[k])[-top_k:]
            for m in range(0, len(top_indexes)):
                label_uncertainty = entropy([prob[m], 1 - prob[m]], base=2)
                w = 0
                for n in range(m, len(top_indexes)):
                    w += cor_matrix[m][n]
                w1 = 1 - w / cor_matrix.shape[0]
                w2 = 1 - label_counts[m] / len(init_df)
                phi_1 = phi_1 + w1 * w2 * label_uncertainty
            phi_1 = phi_1 / len(top_indexes)
            cor_uncertainty.append(phi_1)
            label_indexes.append(top_indexes)
        phi_12 = [a * b for a, b in zip(cor_uncertainty, diversity_list)]
        top_suggestions = sorted(range(len(phi_12)), key=lambda l: phi_12[l])[-selected_batches:]
        suggest_annotate_index = sorted(top_suggestions)
        suggest_annotate_index_list += suggest_annotate_index
    return unlabelled_df.iloc[suggest_annotate_index_list]


def DiscriminativeRepresentation():
    print("DiscriminativeRepresentation")


def CmbalVariant1():
    print("v1")


def CmbalVariant2():
    print("v2")


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


# def EntropyLeast(model, device, dataloader, num_batch):
#     print("Using entropy of probability")
#     model.eval()
#     selected_idx = []
#     df = pd.DataFrame(columns=['entropy', 'idx'])
#     batch_size = 0
#     with torch.no_grad():
#         for i, (x, y, idx) in enumerate(dataloader):
#             batch_size = x.shape[0] if batch_size == 0 else batch_size
#             if x.shape[0] == batch_size:
#                 x = x.long().to(device)
#                 output = model(x)
#                 prob = F.softmax(output, dim=1)
#                 prob_entropy = entropy([torch.max(p).cpu().item() for p in prob])
#                 df.loc[i] = [prob_entropy, idx.numpy()]
#             else:
#                 print("last batch, don't consider")
#     selected_df = df.sort_values(by=['entropy'], ascending=True).head(int(num_batch))
#     for indexArr in selected_df.idx:
#         selected_idx += indexArr.tolist()
#     assert len(selected_idx) == int(num_batch) * batch_size
#     entropy_score_ds = Subset(dataloader.dataset, selected_idx)
#     return entropy_score_ds, selected_idx

def get_embedding(model, df, params):
    batch_size = params['batch_size']
    dataset = make_dataset(df, params, is_train=False)
    embedding = tf.Variable(tf.zeros([len(df), 256]))
    for i, (text_batch, label_batch) in enumerate(dataset):
        if text_batch.shape[0] == batch_size:
            idx = list(range(batch_size * i, batch_size * (i + 1)))
            output = model.predict(text_batch)
            embedding[idx].assign(output)
        else:
            print("last batch, don't consider")
    return embedding


def furthest_first(unlabeled_embeddings, labeled_embeddings, n):
    m = np.shape(unlabeled_embeddings)[0]
    if np.shape(labeled_embeddings)[0] == 0:
        min_dist = np.tile(float("inf"), m)
    else:
        dist_ctr = pairwise_distances(unlabeled_embeddings, labeled_embeddings)
        min_dist = np.amin(dist_ctr, axis=1)
    indices = []
    for i in range(n):
        idx = min_dist.argmax()
        indices.append(idx)
        dist_new_ctr = pairwise_distances(unlabeled_embeddings, unlabeled_embeddings[[idx], :])
        for j in range(m):
            min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])
    return indices


if __name__ == '__main__':
    print("")
