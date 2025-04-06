# 导入项目中自定义的模块和类
from subtask1 import Subtask1Analyzer  # 导入用于分析/训练的主类
from data import Subtask1Dataset       # 导入数据集类，用于加载和处理数据
from utils import Config               # 导入配置工具类，用于读取配置参数
from argparse import ArgumentParser    # 导入命令行参数解析器
from configparser import ConfigParser  # 导入配置文件解析器

# 仅当当前脚本作为主程序运行时才执行以下代码
if __name__ == '__main__':
    # 创建命令行参数解析器，描述程序用途
    parser = ArgumentParser(description='Textual Emotion Cause Analysis')
    
    # 添加通用参数：
    # --conf：指定配置文件路径，默认为 config/subtask1.ini
    parser.add_argument('--conf', type=str, default='config/subtask1.ini', help='Configuration file.')
    # --path：指定保存或加载分析器的路径
    parser.add_argument('--path', type=str, default='results/subtask1/', help='Path to load the analyzer or store it.')
    # --load：如果设置该标志，则从指定路径加载预训练或已训练好的分析器
    parser.add_argument('--load', action='store_true', help='Wheter to load the analyzer.')
    # --batch_size：设置批次大小，默认为1500
    parser.add_argument('--batch_size', type=int, default=1500, help='Batch size.')
    
    # 创建子命令解析器，根据不同模式执行不同任务
    modes = parser.add_subparsers(dest='mode')
    # 定义 train 模式（训练）
    train = modes.add_parser('train')
    # 定义 predict 模式（预测）
    predict = modes.add_parser('predict')
    # 定义 eval 模式（评估）
    eval = modes.add_parser('eval')
    
    # 为 train 模式添加特定参数：
    # --train：训练集文件路径，默认为 dataset/text/Subtask_1_trainset.json
    train.add_argument('--train', type=str, default='dataset/text/Subtask_1_trainset.json', help='Path to the train set.')
    # --dev：开发集（验证集）文件路径
    train.add_argument('--dev', type=str, default='dataset/text/Subtask_1_devset.json', help='Path to the dev set.')
    # --test：测试集文件路径
    train.add_argument('--test', type=str, default='dataset/text/Subtask_1_test.json', help='Path to the test set.')
    # --lr：学习率，默认 1e-5
    train.add_argument('--lr', type=float, default=1e-5)
    # --epochs：训练的轮数，默认 100
    train.add_argument('--epochs', type=int, default=100)
    # --patience：早停策略中容忍的轮数，默认 20
    train.add_argument('--patience', type=int, default=20)
    
    # 为 predict 模式添加参数：
    # --input：待预测数据集的路径
    predict.add_argument('--input', type=str, help='Path to the input set.')
    # --output：预测结果输出路径
    predict.add_argument('--output', type=str, help='Path of the output set')
    
    # 为 eval 模式添加参数：
    # --data：评估数据集的路径
    eval.add_argument('--data', type=str, help='Path to the evaluation set.')
    
    # 解析命令行参数
    args = parser.parse_args()
    # 使用 ConfigParser 读取配置文件（ini格式）
    conf = ConfigParser()
    conf.read(args.conf)

    # 判断是否需要加载已有的分析器
    if args.load: 
        # 如果设置了 --load 参数，则从指定路径加载已训练的模型
        analyzer = Subtask1Analyzer.load(args.path)
    else:
        # 如果未设置 --load，则必须是训练模式，因为没有可加载的模型
        assert args.mode == 'train', 'The model needs to be trained'
        analyzer = None

    # 根据不同的模式执行不同操作：
    if args.mode == 'train':
        # 加载训练、开发和测试数据集，使用 Subtask1Dataset.from_path 方法根据文件路径加载数据
        train, dev, test = map(Subtask1Dataset.from_path, (args.train, args.dev, args.test))
        # 从配置文件中读取与文本相关的配置参数
        text_conf = Config.from_ini(conf['text'])
        # 从配置文件中读取与模型相关的配置参数
        model_conf = Config.from_ini(conf['model'])
        # 构造分析器（内部包括模型等组件），传入训练数据、文本配置和模型配置参数
        analyzer = Subtask1Analyzer.build(train, text_conf, **model_conf())
        # 开始训练，传入训练集、开发集、测试集以及其他超参数和保存路径
        analyzer.train(train, dev, test, args.path, args.lr, args.epochs, args.batch_size, args.patience) 
    elif args.mode == 'predict':
        # 如果是预测模式，则加载输入数据集
        input = Subtask1Dataset.from_path(args.input)
        # 调用预测方法，将结果保存到指定输出路径，并指定批次大小
        analyzer.predict(input, args.output, args.batch_size)
    elif args.mode == 'eval':
        # 如果是评估模式，则加载评估数据集
        data = Subtask1Dataset.from_path(args.data)
        # 调用评估方法，默认将预测结果和标签结果分别保存到 /tmp/pred.json 和 /tmp/gold.json，并指定批次大小
        analyzer.eval(data, '/tmp/pred.json', '/tmp/gold.json', args.batch_size)
