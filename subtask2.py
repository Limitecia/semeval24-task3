from subtask2 import Subtask2Analyzer
from data import Subtask2Dataset
from utils import Config
from argparse import ArgumentParser
from configparser import ConfigParser

if __name__ == '__main__':
    parser = ArgumentParser(description='Textual Emotion Cause Analysis')
    parser.add_argument('--conf', type=str, default='config/subtask2.ini', help='Configuration file.')
    parser.add_argument('--path', type=str, default='results/subtask2/', help='Path to load the analyzer or store it.')
    parser.add_argument('--load', action='store_true', help='Wheter to load the analyzer.')
    parser.add_argument('--batch_size', type=int, default=1500, help='Batch size.')
    parser.add_argument('--video_folder', type=str, default='../dataset/', help='Path to stored videos and audios.')
    
    
    modes = parser.add_subparsers(dest='mode')
    train = modes.add_parser('train')
    predict = modes.add_parser('predict')
    eval = modes.add_parser('eval')
    
    # train parser 
    train.add_argument('--train', type=str, help='Path to the train set.')
    train.add_argument('--dev', type=str, help='Path to the dev set.')
    train.add_argument('--test', type=str, help='Path to the test set.')
    train.add_argument('--batch_update', type=int, default=1)
    train.add_argument('--batch_size', type=int, default=1500)
    train.add_argument('--lr', type=float, default=1e-5)
    train.add_argument('--epochs', type=int, default=100)
    train.add_argument('--patience', type=int, default=20)
    
    # predict parser
    predict.add_argument('--input', type=str, help='Path to the input set.')
    predict.add_argument('--output', type=str, help='Path of the output set')
    
    # eval parser
    eval.add_argument('--data', type=str, help='Path to the evaluation set.')
    
    args = parser.parse_args()
    conf = ConfigParser().read(args.conf)

    if args.load: 
        analyzer = Subtask2Analyzer.load(args.path)
    else:
        assert args.mode == 'train', 'The model needs to be trained'
        analyzer = None
    if args.mode == 'train':
        train, dev, test = [Subtask2Dataset.from_path(d, args.video_folder) for d in (args.train, args.dev, args.test)]
        text_conf = Config.from_ini(conf['text'])
        audio_conf = Config.from_ini(conf['audio']) if 'audio' in conf.keys() else None 
        img_conf = Config.from_ini(conf['img']) if 'img' in conf.keys() else None 
        model_conf = Config.from_ini(conf['model'])
        analyzer = Subtask2Analyzer.build(train, text_conf, img_conf, audio_conf, **model_conf())
        analyzer.train(train, dev, test, args.path, args.lr, args.epochs, args.batch_size, args.batch_update, args.patience) 
    elif args.mode == 'predict':
        input = Subtask2Dataset.from_path(args.input, args.video_folder)
        analyzer.predict(input, args.output, args.batch_size)
    elif args.mode == 'eval':
        data = Subtask2Dataset.from_path(args.data, args.video_folder)
        analyzer.eval(data, '/tmp/pred.json', '/tmp/gold.json', args.batch_size)
        