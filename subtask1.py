from subtask1 import Subtask1Analyzer
from data import Subtask1Dataset
from utils import Config
from argparse import ArgumentParser
from configparser import ConfigParser
import torch

if __name__ == '__main__':
    parser = ArgumentParser(description='Textual Emotion Cause Analysis')
    parser.add_argument('--conf', type=str, default='config/subtask1.ini', help='Configuration file.')
    parser.add_argument('--path', type=str, default='results/subtask1/', help='Path to load the analyzer or store it.')
    parser.add_argument('--load', action='store_true', help='Whether to load the analyzer.')
    parser.add_argument('--batch_size', type=int, default=1500, help='Batch size.')
    
    modes = parser.add_subparsers(dest='mode')
    train = modes.add_parser('train')
    predict = modes.add_parser('predict')
    eval = modes.add_parser('eval')
    
    # train parser 
    train.add_argument('--train', type=str, default='dataset/text/Subtask_1_trainset.json', help='Path to the train set.')
    train.add_argument('--dev', type=str, default='dataset/text/Subtask_1_devset.json', help='Path to the dev set.')
    train.add_argument('--test', type=str, default='dataset/text/Subtask_1_test.json', help='Path to the test set.')
    train.add_argument('--lr', type=float, default=1e-5)
    train.add_argument('--epochs', type=int, default=100)
    train.add_argument('--patience', type=int, default=20)
    
    # predict parser
    predict.add_argument('--input', type=str, help='Path to the input set.')
    predict.add_argument('--output', type=str, help='Path of the output set')
    
    # eval parser
    eval.add_argument('--data', type=str, help='Path to the evaluation set.')
    
    args = parser.parse_args()
    conf = ConfigParser()
    conf.read(args.conf)
    
    # 自动选择设备（DataParallel 默认要求模型在 cuda:0）
    default_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.load: 
        analyzer = Subtask1Analyzer.load(args.path)
    else:
        assert args.mode == 'train', 'The model needs to be trained'
        analyzer = None
    if args.mode == 'train':
        train_data, dev_data, test_data = map(Subtask1Dataset.from_path, (args.train, args.dev, args.test))
        text_conf = Config.from_ini(conf['text'])
        model_conf = Config.from_ini(conf['model'])
        analyzer = Subtask1Analyzer.build(train_data, text_conf, **model_conf())
        
        # 如果检测到多块 GPU，则用 DataParallel 包装模型，并转移到默认设备 cuda:0
        if torch.cuda.device_count() > 1:
            print(f"发现 {torch.cuda.device_count()} 块 GPU，使用 DataParallel 进行数据并行。")
            analyzer.model = torch.nn.DataParallel(analyzer.model)
            analyzer.model.to(default_device)
        else:
            analyzer.model.to(default_device)
        
        analyzer.train(train_data, dev_data, test_data, args.path, args.lr, args.epochs, args.batch_size, args.patience) 
    elif args.mode == 'predict':
        input_data = Subtask1Dataset.from_path(args.input)
        analyzer.predict(input_data, args.output, args.batch_size)
    elif args.mode == 'eval':
        eval_data = Subtask1Dataset.from_path(args.data)
        analyzer.eval(eval_data, '/tmp/pred.json', '/tmp/gold.json', args.batch_size)
