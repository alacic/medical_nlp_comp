from operator import add
from functools import reduce

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from core.utils import (fix_columns, extract_X, extract_y, MultiLabelOverSampler,
                        export_result, k_fold, TransProbaTransformer)
from core.model import (get_cnn_model, loss_region, loss_type, metric_score_type,
                        PADDING_SIZE, metric_score, EPOCHS, BATCH_SIZE)


if __name__ == "__main__":
    np.random.seed(2021)
    tf.random.set_seed(2021)
    # test_data_file = "./data/testB.csv"
    # train_data_file = "./data/train.csv"
    # train_data_file_round_1 = "./data/track1_round1_train_20210222.csv"
    test_data_file = "/tcdata/testB.csv"
    train_data_file = "/tcdata/train.csv"
    train_data_file_round_1 = "/tcdata/track1_round1_train_20210222.csv"
    result_file = "./result.csv"

    df_train = pd.read_csv(
        train_data_file,
        header=None,
        names=["report_ID", "description", "labelA", "labelB"],
    ).fillna("").applymap(fix_columns)

    df_train_round_1 = pd.read_csv(
        train_data_file_round_1,
        header=None,
        names=["report_ID", "description", "labelA"],
    ).fillna("").applymap(fix_columns)

    df_train_round_1["labelB"] = ""

    df_train = pd.concat([df_train, df_train_round_1])

    df_train["X"] = df_train.description.apply(extract_X)
    df_train["X_len"] = df_train.X.str.len()
    df_train = df_train.loc[df_train.X_len.between(5, 90)]

    condi_value = (df_train.labelB!="")
    df_train_labelB = df_train.loc[condi_value]
    df_train.index = range(len(df_train))
    df_train_labelB.index = range(len(df_train_labelB))


    df_test = pd.read_csv(
        test_data_file,
        header=None,
        names=["report_ID", "description"],
    ).fillna("").applymap(fix_columns)

    print(f"basic info {'=' * 15}")
    print(f"train_size {len(df_train)}")
    print(f"df_train_labelB_size {len(df_train_labelB)}")
    print(f"test_size {len(df_test)}")
    print(f"basic info {'=' * 15}\n\n")

    df_test["X"] = df_test.description.apply(extract_X)

    trainX = sequence.pad_sequences(df_train.X, PADDING_SIZE, padding="post")
    trainX_labelB = sequence.pad_sequences(df_train_labelB.X, PADDING_SIZE, padding="post")

    train_yA = np.array(df_train.labelA.apply(extract_y).tolist())
    train_yB = np.array(df_train_labelB.labelB.apply(extract_y, args=(12, )).tolist())

    testX = sequence.pad_sequences(df_test.X, PADDING_SIZE, padding="post")

    n_splits = 5

    EPOCHS = 20
    BATCH_SIZE = 64
    textCNN_models_labelB = []
    historys = []
    for j, ((trainX_i, valiX_i), (train_yB_i, vali_yB_i)) in enumerate(k_fold(trainX_labelB, train_yB, random_state=0)):
        print(f"{j:=^90d}")
        model_i = get_cnn_model(output_shape=12)
        model_i.compile(
            optimizer="adam", 
            loss=loss_type, 
            metrics=[loss_type]
        )
        
        history_i = model_i.fit(
            trainX_i,
            train_yB_i, 
            batch_size=BATCH_SIZE, 
            epochs=EPOCHS, 
            validation_data=(valiX_i, vali_yB_i),
        )
        
        textCNN_models_labelB.append(model_i)
        historys.append(history_i)
        print(f"{'=' * 90}\n")

    EPOCHS = 15
    BATCH_SIZE = 64
    textCNN_models_labelA = []
    historys = []
    for j, ((trainX_i, valiX_i), (train_yA_i, vali_yA_i)) in enumerate(k_fold(trainX, train_yA, random_state=0)):
        print(f"{j:=^90d}")
        model_i = get_cnn_model(output_shape=17)
        model_i.compile(
            optimizer="adam", 
            loss=loss_region, 
            metrics=[loss_region]
        )
        
        history_i = model_i.fit(
            trainX_i,
            train_yA_i, 
            batch_size=BATCH_SIZE, 
            epochs=EPOCHS, 
            validation_data=(valiX_i, vali_yA_i),
        )
        
        textCNN_models_labelA.append(model_i)
        historys.append(history_i)
        print(f"{'=' * 90}\n")

    results = []
    for m, mB in zip(textCNN_models_labelA, textCNN_models_labelB):
        yhat_A_i = m.predict(testX)
        yhat_B_i = mB.predict(testX)
        results.append(np.c_[yhat_A_i, yhat_B_i])
    
    y_hat = reduce(add, results) / n_splits
    export_result(result_file, y_hat)
