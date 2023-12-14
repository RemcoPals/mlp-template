from sklearn.svm import SVC

class Model:
    svm_model=None

    # parameterized constructor
    def __init__(self):
        self.svm_model = SVC(kernel='rbf', gamma=1, class_weight='balanced')

    def fit_model(self,X,y):
        self.svm_model.fit(X, y)

    def predict(self,X):
        y_pred = self.svm_model.predict(X)
        return y_pred