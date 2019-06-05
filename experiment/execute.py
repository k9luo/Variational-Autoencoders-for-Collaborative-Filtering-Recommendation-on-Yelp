from evaluation.metrics import evaluate
from models.predictor import predict
from tqdm import tqdm
from utils.progress import WorkSplitter

import pandas as pd


def execute(train, test, params, model, gpu_on=True, analytical=False):
    progress = WorkSplitter()

    columns = ['model', 'rank', 'lambda', 'epoch', 'corruption', 'topK']

    progress.section("\n".join([":".join((str(k), str(params[k]))) for k in columns]))

    df = pd.DataFrame(columns=columns)

    progress.subsection("Train")
    RQ, Yt, Bias = model(train,
                            epoch=params['epoch'],
                            lamb=params['lambda'],
                            rank=params['rank'],
                            corruption=params['corruption'])
    Y = Yt.T

    progress.subsection("Prediction")
    prediction = predict(matrix_U=RQ, matrix_V=Y, bias=Bias, topK=params['topK'][-1],
                         matrix_Train=train, gpu=gpu_on)

    progress.subsection("Evaluation")
    result = evaluate(prediction, test, params['metric'], params['topK'], analytical=analytical)

    if analytical:
        return result
    else:
        result_dict = params

        for name in result.keys():
            result_dict[name] = [round(result[name][0], 4), round(result[name][1], 4)]
        df = df.append(result_dict, ignore_index=True)

        return df
