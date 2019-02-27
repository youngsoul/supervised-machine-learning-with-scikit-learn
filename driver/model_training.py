from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
"""
This file will run through a number of scikit learn models on the the training data
in training.csv.  This training data was collected through running:

python training.py

and 'driving' a car down a windy road and collecting the turning necessary at easy step.



"""

def get_data():
    """
    read training.csv and return the X,y as series
    :return: X - the data representing the road view
             y - what turn value
    """
    df = pd.read_csv('training.csv', header=None)
    # print(df.head())
    X = df.loc[:, 1:]
    y = df.loc[:, 0]
    # print(X.shape)
    # print(y.shape)
    return X, y


def train_model(model, X, y, name=None, param_grid=None):
    if name:
        print(name)

    if param_grid:
        grid = GridSearchCV(model, param_grid, cv=5)
        grid.fit(X, y)
        print(grid.best_score_)
        print(grid.best_params_)
        print(grid.best_estimator_)
    else:
        cv_scores = cross_val_score(model, X, y, cv=5)
        print(cv_scores, cv_scores.mean())


def create_logistic_regression_model():
    logreg = LogisticRegression(multi_class='multinomial')
    return logreg


def create_decision_tree():
    tree = DecisionTreeClassifier()
    return tree


def create_svc():
    svc = SVC(kernel='linear', C=1)
    return svc


def create_mnb():
    mnb = GaussianNB()
    return mnb


def create_knn():
    knn = KNeighborsClassifier()
    return knn


def create_linear():
    lin = LinearRegression()
    return lin


def find_best_model(X, y):
    models = [
        {
            'model': create_logistic_regression_model(),
            'params_grid': dict(penalty=['l2'], C=[10, 1, 0.1, 0.01], solver=['newton-cg', 'sag', 'lbfgs'],
                      max_iter=[100, 200, 300]),
            'name': 'LogisticRegression',
            'skip': True
        },
        {
            'model': create_decision_tree(),
            'params_grid': dict(criterion=['gini', 'entropy'], max_depth=[2, 3, 4, 5], min_samples_split=[2, 3]),
            'name': 'DecisionTree',
            'skip': True
        },
        {
            'model': create_svc(),
            'params_grid': dict(kernel=['linear', 'rbf', 'poly'], gamma=['auto', 'scale']),
            'name': 'SVC',
            'skip': True
        },
        {
            'model': create_mnb(),
            'params_grid': None,
            'name': 'MultinomialNB',
            'skip': True
        },
        {
            'model': create_knn(),
            'params_grid': dict(n_neighbors=list(range(1, 31)), weights=['uniform', 'distance']),
            'name': 'KNN GridSearch',
            'skip': True
        },
        {
            'model': create_knn(),
            'params_grid': None,
            'name': 'KNN Default',
            'skip': True
        },
        {
            'model': create_linear(),
            'params_grid': None,
            'name': 'Linear',
            'skip': True
        },
        {
            'model': MLPClassifier(),
            'params_grid': dict(activation=['relu', 'logistic', ],
                                solver=['sgd', 'adam'],
                                alpha=[1, 0.1, 0.01],
                                hidden_layer_sizes=[(X.shape[1],128,16), (X.shape[1],100) ]),
            'name': 'MLP',
            'skip': False
        }

    ]
    for model in models:
        if not model['skip']:
            train_model(model['model'], X, y, name=model['name'], param_grid=model['params_grid'])


"""
KNN Grid
0.72
{'n_neighbors': 23, 'weights': 'uniform'}
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=23, p=2,
           weights='uniform')


MLP
0.7066666666666667
{'activation': 'relu', 'alpha': 1, 'hidden_layer_sizes': (250, 100), 'solver': 'sgd'}
MLPClassifier(activation='relu', alpha=1, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(250, 100), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=None, shuffle=True, solver='sgd', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)
"""


def save_best_model(X, y):
    knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                               metric_params=None, n_jobs=None, n_neighbors=23, p=2,
                               weights='uniform')
    knn.fit(X, y)

    filename = 'driving_model.sav'
    joblib.dump(knn, filename)


if __name__ == '__main__':
    X, y = get_data()
    find_best_model(X,y)

    # save_best_model(X, y)
