from subtask2 import Subtask2Analyzer
from data import Subtask2Dataset
from utils import Config

if __name__ == '__main__':
    train = Subtask2Dataset.from_path('../dataset/text/Subtask_2_trainset.json', '../dataset/')
    dev = Subtask2Dataset.from_path('../dataset/text/Subtask_2_devset.json', '../dataset/')
    test = Subtask2Dataset.from_path('../dataset/text/Subtask_2_test.json', '../dataset/')
    
    text_conf = Config(pretrained='bert-large-cased', finetune=True, device='cuda:0')
    # img_conf = Config(pretrained='trpakov/vit-face-expression', finetune=False, device='cuda:0', num_frames=5, embed_size=500)
    img_conf = None
    # audio_conf = Config(pretrained='facebook/wav2vec2-base-960h', finetune=False, device='cuda:1', embed_size=500)
    audio_conf = None
    
    analyzer = Subtask2Analyzer.build(train, text_conf, img_conf, audio_conf, device='cuda:1')
    analyzer.train(train, dev, test, 'results/subtask2/', batch_size=1500, epochs=500)
