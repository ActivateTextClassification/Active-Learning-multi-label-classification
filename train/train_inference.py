from sklearn.metrics import f1_score
import tensorflow_addons as tfa
from utils.metrics import make_model, plot_result
from keras import backend as K


def TP(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))


def recall_m(y_true, y_pred):
    true_positives = TP(y_true, y_pred)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = TP(y_true, y_pred)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def micro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def train_model(train_dataset, validation_dataset, model_param, params):
    shallow_mlp_model = make_model(int(model_param["dim1"]), int(model_param["dim2"]), lookup=params["lookup"])
    lookup = params["lookup"]
    shallow_mlp_model.compile(
        loss="binary_crossentropy", optimizer="adam",
        metrics=["categorical_accuracy", tfa.metrics.F1Score(num_classes=lookup.vocabulary_size(), average='micro'),
                 precision_m, recall_m]
    )
    history = shallow_mlp_model.fit(
        train_dataset, validation_data=validation_dataset, epochs=params["epochs"]
    )
    # print("loss===============:{}".format(history.history["loss"]))
    return shallow_mlp_model
