import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from recordlinkage.preprocessing import phonetic
from numpy.random import choice
import collections
from sklearn.model_selection import train_test_split, KFold


def train_model(modeltype, modelparam, train_vectors, train_labels, modeltype_2):
    if modeltype == "svm":  # Support Vector Machine
        model = svm.SVC(C=modelparam, kernel=modeltype_2)
        model.fit(train_vectors, train_labels)
    elif modeltype == "lg":  # Logistic Regression
        if modeltype_2 == "l1":
            model = LogisticRegression(
                C=modelparam,
                penalty=modeltype_2,
                class_weight=None,
                dual=False,
                fit_intercept=True,
                intercept_scaling=1,
                max_iter=5000,
                multi_class="ovr",
                n_jobs=1,
                random_state=None,
                solver="liblinear",
            )

        if modeltype_2 == "l2":
            model = LogisticRegression(
                C=modelparam,
                penalty=modeltype_2,
                class_weight=None,
                dual=False,
                fit_intercept=True,
                intercept_scaling=1,
                max_iter=5000,
                multi_class="ovr",
                n_jobs=1,
                random_state=None,
            )

        model.fit(train_vectors, train_labels)
    elif modeltype == "nb":  # Naive Bayes
        model = GaussianNB()
        model.fit(train_vectors, train_labels)
    elif modeltype == "nn":  # Neural Network
        model = MLPClassifier(
            solver="lbfgs",
            alpha=modelparam,
            hidden_layer_sizes=(256,),
            activation=modeltype_2,
            random_state=None,
            batch_size="auto",
            learning_rate="constant",
            learning_rate_init=0.001,
            power_t=0.5,
            max_iter=30000,
            shuffle=True,
            tol=0.0001,
            verbose=False,
            warm_start=False,
            momentum=0.9,
            nesterovs_momentum=True,
            early_stopping=False,
            validation_fraction=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
        )
        model.fit(train_vectors, train_labels)
    return model


def classify(model, test_vectors):
    result = model.predict(test_vectors)
    return result


def evaluation(test_labels, result):
    true_pos = np.logical_and(test_labels, result)
    count_true_pos = np.sum(true_pos)

    true_neg = np.logical_and(np.logical_not(test_labels), np.logical_not(result))
    count_true_neg = np.sum(true_neg)

    false_pos = np.logical_and(np.logical_not(test_labels), result)
    count_false_pos = np.sum(false_pos)

    false_neg = np.logical_and(test_labels, np.logical_not(result))
    count_false_neg = np.sum(false_neg)

    precision = count_true_pos / (count_true_pos + count_false_pos)
    sensitivity = count_true_pos / (
        count_true_pos + count_false_neg
    )  # sensitivity = recall
    confusion_matrix = [
        count_true_pos,
        count_false_pos,
        count_false_neg,
        count_true_neg,
    ]
    no_links_found = np.count_nonzero(result)
    no_false = count_false_pos + count_false_neg
    Fscore = 2 * precision * sensitivity / (precision + sensitivity)

    metrics_result = {
        "no_false": no_false,
        "confusion_matrix": confusion_matrix,
        "precision": precision,
        "sensitivity": sensitivity,
        "no_links": no_links_found,
        "F-score": Fscore,
        "true_pos": count_true_pos,
        "true_neg": count_true_neg,
        "false_pos": count_false_pos,
        "false_neg": count_false_neg,
    }

    return metrics_result


def get_best_model(df_fscore, df_precision, df_sensitivity, df_nb_false, models):

    # Scores from best model
    df_best = pd.DataFrame(
        columns=[
            "model",
            "modeltype",
            "param",
            "precision",
            "recall",
            "fscore",
            "no_false",
        ]
    )
    index = 0

    for i in models:
        models_filter = [i + "-" + j for j in models[i]]
        col = df_fscore.filter(models_filter)
        score = []

        for c in col:
            score.append(df_fscore[df_fscore[c] == df_fscore[c].max()][c].values[0])

        max_score = np.max(score)
        best_model = col.columns[score.index(max_score)]
        best_param = df_fscore[df_fscore[best_model] == max_score]["param"].values[0]

        precision = df_precision[df_precision["param"] == best_param][
            best_model
        ].values[0]
        recall = df_sensitivity[df_precision["param"] == best_param][best_model].values[
            0
        ]
        no_false = df_nb_false[df_precision["param"] == best_param][best_model].values[
            0
        ]

        df_best.loc[index] = [
            best_model.split("-")[0],
            best_model.split("-")[1],
            best_param,
            precision * 100,
            recall * 100,
            max_score * 100,
            no_false,
        ]
        index += 1

    return df_best


def get_scores(df_fscore, df_precision, df_sensitivity, df_nb_false, models_list):

    df_best = pd.DataFrame(
        columns=[
            "model",
            "modeltype",
            "param",
            "precision",
            "recall",
            "fscore",
            "no_false",
        ]
    )
    index = 0

    for i in models_list:
        score = df_fscore[df_fscore["param"] == i[1]][i[0]].values[0]

        best_model = i[0]
        best_param = i[1]

        precision = df_precision[df_precision["param"] == best_param][
            best_model
        ].values[0]
        recall = df_sensitivity[df_precision["param"] == best_param][best_model].values[
            0
        ]
        no_false = df_nb_false[df_precision["param"] == best_param][best_model].values[
            0
        ]

        df_best.loc[index] = [
            i[0].split("-")[0],
            i[0].split("-")[1],
            i[1],
            precision * 100,
            recall * 100,
            score * 100,
            no_false,
        ]
        index += 1

    return df_best
