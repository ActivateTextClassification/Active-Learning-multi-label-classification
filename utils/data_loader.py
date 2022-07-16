from sklearn.model_selection import train_test_split
from ast import literal_eval
import tensorflow as tf
import pandas as pd
from utils.metrics import map_dataset, make_dataset, get_text_vectorizer


def load_dataset_by_name(name, split_ratio=0.1):
    if str(name).__contains__("arxiv"):
        return load_arxiv(split_ratio)
    else:
        return load_yelp_full_review(name, split_ratio)


def load_yelp_full_review(name, split_ratio=0.1):
    # train_df = pd.read_csv('./dataset/yelp/yelp_train.csv')
    test_data = pd.read_csv(name)
    # train_data = pd.concat([train_df, test_df])
    # train_df["labels"] = train_df["labels"].apply(
    #     lambda x: [str(x)]
    # )
    test_data["labels"] = test_data["labels"].apply(
        lambda x: [str(x)]
    )
    train_df, test_df = train_test_split(
        test_data,
        test_size=split_ratio,
        stratify=test_data["labels"].values,
        random_state=1
    )
    val_df = test_df.sample(frac=0.5)
    test_df.drop(val_df.index, inplace=True)
    return train_df, val_df, test_df


def load_arxiv(split_ratio=0.1):
    arxiv_data = pd.read_csv('dataset/arxiv/arxiv_all.csv')
    print(arxiv_data.head())
    """
    Our text features are present in the `summaries` column and their corresponding labels
    are in `terms`. As you can notice, there are multiple categories associated with a
    particular entry.
    """

    print(f"There are {len(arxiv_data)} rows in the dataset.")

    """
    Real-world data is noisy. One of the most commonly observed source of noise is data
    duplication. Here we notice that our initial dataset has got about 13k duplicate entries.
    """

    total_duplicate_titles = sum(arxiv_data["titles"].duplicated())
    print(f"There are {total_duplicate_titles} duplicate titles.")

    """
    Before proceeding further, we drop these entries.
    """

    arxiv_data = arxiv_data[~arxiv_data["titles"].duplicated()]
    print(f"There are {len(arxiv_data)} rows in the deduplicated dataset.")

    # There are some terms with occurrence as low as 1.
    print(sum(arxiv_data["labels"].value_counts() == 1))

    # How many unique terms?
    print(arxiv_data["labels"].nunique())

    # Filtering the rare terms.
    arxiv_data_filtered = arxiv_data.groupby("labels").filter(lambda x: len(x) > 1)

    print(arxiv_data_filtered.shape)
    """
    ## Convert the string labels to lists of strings

    The initial labels are represented as raw strings. Here we make them `List[str]` for a
    more compact representation.
    """

    arxiv_data_filtered["labels"] = arxiv_data_filtered["labels"].apply(
        lambda x: literal_eval(x)
    )
    print(arxiv_data_filtered["labels"].values[:5])

    """
    ## Use stratified splits because of class imbalance
    """
    # arxiv_data_filtered.to_csv("./dataset/arxiv/arxiv_all.csv", index=False)
    # Initial train and test split.
    train_df, test_df = train_test_split(
        arxiv_data_filtered,
        test_size=split_ratio,
        stratify=arxiv_data_filtered["labels"].values,
        random_state=0
    )
    val_df = test_df.sample(frac=0.5)
    test_df.drop(val_df.index, inplace=True)
    return train_df, val_df, test_df


def preprocessing_dataset(data, params):
    """
    ## Multi-label binarization
    """
    train_df, val_df, test_df = data
    terms = tf.ragged.constant(train_df['labels'].values)
    lookup = tf.keras.layers.StringLookup(output_mode="multi_hot")
    lookup.adapt(terms)
    vocab = lookup.get_vocabulary()
    print("Vocabulary:{} and Size:{}".format(vocab, len(vocab)))
    sample_label = train_df["labels"].iloc[0]
    print(f"Original label: {sample_label}")

    label_binarized = lookup([sample_label])
    print(f"Label-binarized representation: {label_binarized}")
    # update params
    params["lookup"] = lookup
    params['vocab'] = vocab
    """
    ## Data preprocessing and `tf.data.Dataset` objects
    We first get percentile estimates of the sequence lengths. The purpose will be clear in a
    moment.
    """
    print(train_df[params["text"]].apply(lambda x: len(x.split(" "))).describe())
    """
    Create train_dataset, validation_dataset and test_dataset
    """
    start_df = train_df.sample(frac=1)
    init_df = start_df[:params["init_samples"]]
    unlabelled_df = start_df[params["init_samples"]:]
    """
    make dataset
    """
    train_dataset = make_dataset(init_df, params, is_train=True)
    validation_dataset = make_dataset(val_df, params, is_train=False)
    test_dataset = make_dataset(test_df, params, is_train=False)
    print("train_dataset:{}".format(train_dataset))
    """
    ## Vectorization
    """
    vocabulary = set()
    init_df[params["text"]].str.lower().str.split().apply(vocabulary.update)
    vocabulary_size = len(vocabulary)
    params['vocabulary_size'] = vocabulary_size
    print("Vocabulary Size:{}".format(vocabulary_size))
    """
    We now create our vectorization layer and `map()` to the `tf.data.Dataset`s created
    earlier.
    """
    text_vectorizer = get_text_vectorizer(params['vocabulary_size'], ngrams=2)

    # `TextVectorization` layer needs to be adapted as per the vocabulary from our
    # training set.
    with tf.device("/CPU:0"):
        text_vectorizer.adapt(train_dataset.map(lambda text, label: text))

    train_dataset = map_dataset(train_dataset, text_vectorizer)
    validation_dataset = map_dataset(validation_dataset, text_vectorizer)
    test_dataset = map_dataset(test_dataset, text_vectorizer)
    print("Validation Dataset:{}".format(validation_dataset))
    print("lookup.vocabulary_size;{}".format(lookup.vocabulary_size()))

    dataset = train_dataset, validation_dataset, test_dataset
    df = start_df, init_df, unlabelled_df
    return dataset, df, params, text_vectorizer
