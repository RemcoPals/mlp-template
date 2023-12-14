from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

class Evaluation:
    y_test=None
    y_pred=None

    # parameterized constructor
    def __init__(self,y_test,y_pred):
        self.y_test = y_test
        self.y_pred = y_pred

    def get_accuracy(self):
        # Evaluate accuracy
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"Accuracy: {accuracy}")
        return accuracy

    def get_f1_score(self):
        f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        print(f"Accuracy: {f1}")
        return f1

    def plot_confusion_matrix(self):
        # Compute the confusion matrix
        conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        # Visualize the confusion matrix as a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix (Test Set)')
        plt.show()
    def get_eval_report(self):
        report = classification_report(self.y_test, self.y_pred)
        print("Classification Report:")
        print(report)
        return report