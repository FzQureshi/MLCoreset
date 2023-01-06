import time
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


def train_classifier(dataset, clf_name='', cs_creation_time=0, models_dir=''):
    clf_name = 'mnist_classifier_' + clf_name
    print(f'Classifier: {clf_name}')
    start_time = time.time()
    x_train, x_test, y_train, y_test = train_test_split(x := dataset.drop(columns='label'),
                                                        y := dataset['label'],
                                                        test_size=0.2,
                                                        random_state=42)


    # Normalize values:
    x_train /= 255
    x_test /= 255

    # Basic SVC
    clf = SVC(kernel='linear')
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    clf.fit(x_train, y_train)

    # Make predictions on test set
    y_preds = clf.predict(x_test)

    # Metrics
    acc = round(accuracy_score(y_true=y_test, y_pred=y_preds), 2)
    f1 = round(f1_score(y_true=y_test, y_pred=y_preds, average='macro'), 2)

    #print('Saving model on disk...')
    save_model(clf, models_dir, clf_name)
    time_elapsed = round(time.time() - start_time + cs_creation_time, 2)
    stat = dict(name=clf_name,
                accuracy=acc,
                f1=f1,
                train_shape=x_train.shape,
                test_shape=x_test.shape,
                time_elapsed=time_elapsed)
    return stat


def save_model(model, model_dir, model_name):
    file_path = os.path.join(model_dir, model_name + '.pkl')
    pickle.dump(model, open(file_path, 'wb'))


def load_model(model_dir, model_name):
    file_path = os.path.join(model_dir, model_name + '.pkl')
    return pickle.load(open(file_path, 'r'))
