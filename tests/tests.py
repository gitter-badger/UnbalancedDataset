__author__ = 'fmfnogueira'


def test1():

    from unbalanced_dataset.over_sampling import SVM_SMOTE

    sm = SVM_SMOTE(probability=True, class_weight='auto')
    print(sm)

    from sklearn.datasets import make_classification
    x, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],\
                               n_informative=3, n_redundant=1, flip_y=0,\
                               n_features=20, n_clusters_per_class=1,\
                               n_samples=5000, random_state=10)

    #svmsmote = SVM_SMOTE(random_state=1, class_weight='auto')
    svmx, svmy = sm.fit_transform(x, y)


def test2():

    from unbalanced_dataset.over_sampling import smote

    from sklearn.datasets import make_classification
    x, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],\
                               n_informative=3, n_redundant=1, flip_y=0,\
                               n_features=20, n_clusters_per_class=1,\
                               n_samples=5000, random_state=10)



    sm = smote(kind='regular')
    svmx, svmy = sm.fit_transform(x, y)

    sm = smote(kind='borderline1')
    svmx, svmy = sm.fit_transform(x, y)

    sm = smote(kind='borderline2')
    svmx, svmy = sm.fit_transform(x, y)

    sm = smote(kind='svm')
    svmx, svmy = sm.fit_transform(x, y)


if __name__ == '__main__':
    test2()
