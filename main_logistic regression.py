import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import MinMaxScaler

from preprocessing import filter_by_time_from_first_visit, impute, transform2seq

TASK =  "outcome"

if __name__ == '__main__':
    log = open("log.txt", "w+")
    observation_point_list = [15, 30, 60, 90, 120, 150, 180, 210, 240, 270]
    prediction_point_shift = 1.5

    for observation_point in observation_point_list:
        df = pd.read_csv("data/qlik_aug_final.csv", low_memory=False)
        prediction_point = (observation_point * prediction_point_shift)
        feature_df = filter_by_time_from_first_visit(df, 0, observation_point)
        label_df = filter_by_time_from_first_visit(df, 0, prediction_point)
        x, x_temp, y = transform2seq(feature_df, label_df, df, prediction_point, list(range(0, observation_point + 1, 15)))

        x = np.hstack((x, x_temp.reshape((x_temp.shape[0], x_temp.shape[1]*x_temp.shape[2]))))

        # new_y = np.zeros(y.shape[0])

        # for i in range(y.shape[0]):
        #     if y[i, 1] > 1:
        #         if y[i, 0] == 0:
        #             new_y[i] = 0
        #         if y[i, 0] == 1:
        #             new_y[i] = 1
        #     if y[i, 1] == 0:
        #         if y[i, 0] == 0:
        #             new_y[i] = 2
        #         if y[i, 0] == 1:
        #             new_y[i] = 3
        #     if y[i, 1] == 1:
        #         if y[i, 0] == 0:
        #             new_y[i] = 4
        #         if y[i, 0] == 1:
        #             new_y[i] = 5

        y = y[:,0]
        # # # Normalization
        # scaler = MinMaxScaler()
        # scaler.fit(x)
        # x = scaler.transform(x)


        for train_index, test_index in KFold(shuffle=True, random_state=73).split(x):
            x_train = x[train_index, :]
            x_test = x[test_index, :]
            y_train = y[train_index]
            y_test = y[test_index]

            # Count and print samples by their label
            unique, counts = np.unique(y_train, return_counts=True)
            print(dict(zip(unique, counts)))

            classifier = LogisticRegression()
            classifier.fit(x_train, y_train)
            pred_test = classifier.predict_proba(x_test)[:, 1] #+ classifier.predict_proba(x_test)[:, 3] + classifier.predict_proba(x_test)[:, 5]


            #y_attr_test = y_test %2
            out_str = str(observation_point) + ","
            p, r, f, s = precision_recall_fscore_support(np.where(y_test > 0.5, True, False), np.where(pred_test > 0.5, True, False), average='binary')
            out_str += str(roc_auc_score(y_test, pred_test)) + ","  # AUROC
            out_str += str(average_precision_score(y_test, pred_test)) + ","  # AUPRC
            out_str += str(p) + ","  # Precision
            out_str += str(r) + ","  # Recall
            out_str += str(f)  + "\n"  # F1

            print(out_str)
            log.write(out_str)
    log.flush()
    log.close()
