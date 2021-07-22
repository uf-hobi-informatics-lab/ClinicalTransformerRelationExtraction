from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from functools import partial


def score_func(y_true, y_pred, rel2id=None):
    return f1_score(y_true, y_pred, labels=[v for k, v in rel2id.items() if v != 0], average="micro")


def expr(tr_dx, tr_dy, ts_dx, ts_dy, rel2id):
    print(tr_dx.shape, ts_dx.shape)

    clf = SVC(kernel='rbf', random_state=13)
    tuned_parameters = {"C": [2 ** i for i in range(-1, 12)] + [10 ** i for i in range(-3, 0)],
                        'gamma': ['scale', 'auto'],
                        'tol': [10 ** i for i in range(-5, 0)]
                        }

    cv_model = RandomizedSearchCV(clf, tuned_parameters,
                                  scoring=make_scorer(partial(score_func, rel2id=rel2id)), n_jobs=8,
                                  cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=13),
                                  n_iter=50, random_state=13)
    cv_model.fit(tr_dx, tr_dy)
    opt_clf = cv_model.best_estimator_
    preds = opt_clf.predict(ts_dx)
    print(score_func(ts_dy, preds, rel2id=rel2id))

    return preds


def main():
    pass


if __name__ == '__main__':
    main()