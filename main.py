
from  project_name.data.preprocesing import Preprocesing
from project_name.models.model import Model
from project_name.features.evaluation import Evaluation


def main():
    preprocesing=Preprocesing()
    model=Model()
    model.fit_model(preprocesing.X_train_tfidf,preprocesing.y_train)
    y_predict=model.predict(preprocesing.X_test_tfidf)
    evaluation=Evaluation(y_predict,preprocesing.y_test)

    evaluation.get_eval_report()
    evaluation.plot_confusion_matrix()



    return "Hello, World!"


if __name__ == '__main__':
    main()
