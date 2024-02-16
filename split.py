from data import Subtask1Dataset, Subtask2Dataset
import os, shutil

if __name__ == '__main__':
    # run python3 split.py && python3 evaluate.py dataset/submission .
    folder = 'dataset/submission/'
    
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder + 'ref')
    os.makedirs(folder + 'res')
        
    data = Subtask1Dataset.from_path('dataset/text/Subtask_1_train.json')
    train, val = data.split(0.15)
    val.save(f'{folder}/ref/Subtask_1_gold.json', False)
    val.save(f'{folder}/res/Subtask_1_pred.json', True)
    train.save('dataset/text/Subtask_1_trainset.json', False)
    val.save('dataset/text/Subtask_1_devset.json', False)
    
    data = Subtask2Dataset.from_path('dataset/text/Subtask_2_train.json', 'dataset/')
    train, val = data.split(0.15)
    val.save(f'{folder}/ref/Subtask_2_gold.json', False)
    val.save(f'{folder}/res/Subtask_2_pred.json', True)
    train.save('dataset/text/Subtask_2_trainset.json', False)
    val.save('dataset/text/Subtask_2_devset.json', False)