from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



if __name__ == '__main__':

    y_lable = ["恶性","恶性","恶性","恶性","恶性","恶性","良性","良性","良性","良性",]

    A_model_pred = ["恶性","恶性","恶性", "良性","良性","良性","良性","良性","良性","良性"]
    B_model_pred = ["恶性","恶性","恶性", "恶性","恶性","恶性","恶性","恶性","恶性","良性"]

    lables = ["恶性","良性"]
    dataframe_labels = ["恶性(正例)", "良性(反例)"]

    A_result = confusion_matrix(y_lable, A_model_pred, labels=lables)
    B_result = confusion_matrix(y_lable, B_model_pred, labels=lables)

    print(pd.DataFrame(A_result, index= dataframe_labels, columns= dataframe_labels))
    print(pd.DataFrame(B_result, index= dataframe_labels, columns= dataframe_labels))

    print(precision_score(y_lable, A_model_pred, pos_label="恶性"))
    print(recall_score(y_lable, A_model_pred, pos_label="恶性"))
    print(f1_score(y_lable, A_model_pred, pos_label="恶性"))

    print(precision_score(y_lable, B_model_pred, pos_label="恶性"))
    print(recall_score(y_lable, B_model_pred, pos_label="恶性"))
    print(f1_score(y_lable, B_model_pred, pos_label="恶性"))





