import sys 
from data import Subtask1Dataset
from concurrent.futures import ProcessPoolExecutor





if __name__ == '__main__':
    gold = 'dataset/text/Subtask_1_train.json'
    # pred = 'submission/res/Subtask_1_pred.json'
    
    data = Subtask1Dataset.from_path(gold, [], [])
    # golds = sorted(Subtask1Dataset.from_path(gold, [], []).convs, key=lambda conv: conv.id)
    # preds = sorted(Subtask1Dataset.from_path(pred, [], []).convs, key=lambda conv: conv.id)
    
    # assert len(golds) == len(preds)
    # result = golds.pop(0).eval(preds.pop(0))
    # n = 1
    # for gold, pred in zip(golds, preds):
    #     if len(gold.pairs) == 0 and len(pred.pairs) == 0:
    #         continue
    #     partial = gold.eval(pred)
    #     result = {key: result[key] + partial[key] for key in result.keys()}
    #     n += 1
    # result = {key: float(result[key]/n) for key in result.keys()}
    # print(result)
        