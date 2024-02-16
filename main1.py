from subtask1 import Subtask1Analyzer
from data import Subtask1Dataset
from utils import Config


if __name__ == '__main__':
    train = Subtask1Dataset.from_path('dataset/text/Subtask_1_trainset.json') 
    dev = Subtask1Dataset.from_path('dataset/text/Subtask_1_devset.json') 
    test = Subtask1Dataset.from_path('dataset/text/Subtask_1_test.json')
    
    text_conf = Config(pretrained='bert-large-cased', finetune=True, device='cuda:0')
    analyzer = Subtask1Analyzer.build(train, text_conf, device='cuda:1', ut_embed_size=400)
    analyzer.train(
        train, dev, test, 'results/bert/',
        batch_size=1500, batch_update=1, lr=1e-5, epochs=500, step_lr=20, last_lr=1e-5, gamma=0.9)

