from evaluation.metrics import evaluate
from models.predictor import predict
from tqdm import tqdm
from utils.io import load_dataframe_csv, save_dataframe_csv, load_yaml
from utils.progress import WorkSplitter

import inspect
import numpy as np
import pandas as pd


def hyper_parameter_tuning(train, validation, params, save_path, gpu_on=True):
    progress = WorkSplitter()
    table_path = load_yaml('config/global.yml', key='path')['tables']

    try:
        df = load_dataframe_csv(table_path, save_path)
    except:
        df = pd.DataFrame(columns=['model', 'rank', 'lambda', 'epoch',
                                   'corruption', 'topK'])

    num_user = train.shape[0]

    for algorithm in params['models']:

        for rank in params['rank']:

            for lamb in params['lambda']:

                for corruption in params['corruption']:

                    if ((df['model'] == algorithm) &
                        (df['rank'] == rank) &
                        (df['lambda'] == lamb) &
                        (df['corruption'] == corruption)).any():
                        continue

                    format = "model: {}, rank: {}, lambda: {}, corruption: {}"
                    progress.section(format.format(algorithm, rank, lamb, corruption))
                    RQ, Yt, Bias = params['models'][algorithm](train,
                                                               epoch=params['epoch'],
                                                               lamb=lamb,
                                                               rank=rank,
                                                               corruption=corruption)
                    Y = Yt.T

                    progress.subsection("Prediction")

                    prediction = predict(matrix_U=RQ, matrix_V=Y, bias=Bias,
                                         topK=params['topK'][-1], matrix_Train=train,
                                         gpu=gpu_on)

                    progress.subsection("Evaluation")

                    result = evaluate(prediction, validation, params['metric'], params['topK'])

                    result_dict = {'model': algorithm, 'rank': rank, 'lambda': lamb,
                                   'epoch': params['epoch'], 'corruption': corruption}

                    for name in result.keys():
                        result_dict[name] = [round(result[name][0], 4), round(result[name][1], 4)]

                    df = df.append(result_dict, ignore_index=True)

                    save_dataframe_csv(df, table_path, save_path)
