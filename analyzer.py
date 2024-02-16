from __future__ import annotations

import os, torch, pickle, shutil
import torch.nn as nn
from typing import List, Tuple
from data import Subtask2Dataset, Conversation
from utils import *
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.optim import AdamW


os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Analyzer:
    OPTIMIZER = AdamW
    MODEL, OPTIMIZER, INPUT_FIELDS, TARGET_FIELDS = None, None, None, None
    
    MODEL_FILE = 'model.pt'
    PARAMS_FILE = 'params.pickle'
    
    def __init__(
        self,
        model: nn.Module,
        input_tkzs: List[Tokenizer],
        target_tkzs: List[Tokenizer]
    ):
        r"""Initializes the Emotion Cause Analyzer.

        Args:
            model (nn.Module): Neural model.
            input_tkzs (List[Tokenizer]): List of input tokenizers.
            target_tkzs (List[Tokenizer]): List of target tokenizers.
        """
        self.model = model
        self.input_tkzs = input_tkzs
        self.target_tkzs = target_tkzs
        self.optimizer = None
        
        for tkz in [*input_tkzs, *target_tkzs]:
            self.__setattr__(tkz.field, tkz)


    def train(
        self,
        train: Subtask2Dataset,
        dev: Subtask2Dataset,
        test: Subtask2Dataset,
        path: str,
        lr: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 100,
        batch_update: int = 1, 
        patience: int = 20
    ):
        r"""_summary_

        Args:
            train (Subtask2Dataset): _description_
            dev (Subtask2Dataset): _description_
            test (Subtask2Dataset): _description_
            path (str): _description_
            lr (float, optional): _description_. Defaults to 1e-4.
            epochs (int, optional): _description_. Defaults to 100.
            batch_size (int, optional): _description_. Defaults to 100.
            batch_update (int, optional): _description_. Defaults to 1.
            patience (int, optional): _description_. Defaults to 20.
        """
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path)
            
        train_dl = DataLoader(train, collate_fn=self.transform, batch_sampler=LengthSampler(train.lens, batch_size))

        self.optimizer = self.OPTIMIZER(self.model.parameters(), lr=lr)

        train_improv, val_improv, best_dev = patience, patience, self.METRIC()
        for epoch in range(epochs):
            train_loss = self.forward(epoch, train_dl, batch_update)
            dev_metric = self.eval(dev, f'{path}/dev_pred.json', f'{path}/dev_gold.json', batch_size)
            
            print(f'Epoch {epoch} [train]: loss={float(train_loss):.3f}, lr={self.lr:.3e}' + (' (improved)' if dev_metric.improves(best_dev) else ''))
            print(f'Epoch {epoch} [dev]: {repr(dev_metric)}')
            
            if dev_metric.improves(best_dev):
                best_dev, val_improv = dev_metric, patience 
                self.save_model(f'{path}/model.pt')
                # predicts with the evaluation set
                self.predict(test, output=f'{path}/test_pred.json', batch_size=batch_size)  
            else:
                val_improv -= 1

            if train_improv == 0:
                print('No more improvement in train set')
                break 
            if val_improv == 0:
                print('No more improvement in val set')
                break
            del train_loss, dev_metric
        
        self.load_model(f'{path}/{self.MODEL_FILE}')
        self.save(path)
        
    def predict(self, data: Dataset, output: str,  batch_size: int) -> Dataset:
        r"""_summary_

        Args:
            data (Dataset): _description_
            output (str): _description_
            batch_size (int): _description_

        Returns:
            Dataset: _description_
        """
        loader = DataLoader(data, collate_fn=self.transform, batch_sampler=LengthSampler(data.lens, batch_size, shuffle=False))
        preds = []
        for inputs, _, masks, convs in tqdm(loader, total=len(loader), desc=output):
            preds += self.pred_step(inputs, masks, convs)
        data.convs = preds 
        data.save(output, submission=True)
        return data
        
    def eval(self, data: Dataset, pred: str, gold: str, batch_size: int) -> Metric:
        self.predict(data, pred, batch_size)
        return self.METRIC(pred, gold)

    def forward(self, epoch: int, loader: DataLoader, batch_update: int = 1) -> torch.Tensor:
        global_loss = 0.0
        batch_loss = 0.0
        with tqdm(total=len(loader)) as bar:
            bar.set_description(f'Epoch {epoch} (train)')
            for i, (inputs, targets, masks, _) in enumerate(loader):
                loss = self.train_step(inputs, targets, masks)
                batch_loss += loss
                if i % batch_update == 0:
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
                    self.optimizer.step()
                    global_loss += loss
                    batch_loss = 0.0
                bar.update(1)
                bar.set_postfix({'loss': round(float(loss), 2)})
                torch.cuda.empty_cache()
            bar.close()
        return global_loss / len(loader)

    def train_step(
            self, 
            inputs: Tuple[torch.Tensor], 
            targets: Tuple[torch.Tensor], 
            masks: Tuple[torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.model(*inputs)
        return self.model.loss(*scores, *targets, *masks)

    @torch.no_grad()
    def pred_step(
            self, 
            inputs: Tuple[torch.Tensor], 
            masks: Tuple[torch.Tensor], 
            convs: List[Conversation]
        ) -> List[Conversation]:
        raise NotImplementedError

    def transform(self, batch) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor], Tuple[torch.Tensor], List[Conversation]]:
        raise NotImplementedError

    
    def save(self, path: str):
        # save tokenizers
        os.makedirs(f'{path}/tkz/')
        for tkz in [*self.input_tkzs, *self.target_tkzs]:
            tkz.save(f'{path}/tkz/{tkz.field}')
            
        # save parameters 
        with open(f'{path}/{self.PARAMS_FILE}', 'wb') as writer:
            params = {param: getattr(self.model, param) for param in self.model.PARAMS}
            pickle.dump(params, writer)
        
        # save model 
        self.save_model(f'{path}/{self.MODEL_FILE}')
        
        
    def save_model(self, path: str):
        torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, path)
        
    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    @classmethod 
    def load(cls, path: str):
        # load tokenizers 
        tkzs = [Tokenizer.load(f'{path}/tkz/{name}') for name in os.listdir(f'{path}/tkz/')]
        tkzs = {tkz.field: tkz for tkz in tkzs}
        input_tkzs = [tkzs[field] for field in Analyzer.INPUT_FIELDS]
        target_tkzs = [tkzs[field] for field in Analyzer.TARGET_FIELDS]
        
        # load params
        with open(f'{path}/{Analyzer.PARAMS_FILE}', 'rb') as reader:
            args = pickle.load(reader)
        model = cls.MODEL(**args())
        
        analyzer = cls(model, input_tkzs, target_tkzs)
        analyzer.load_model(f'{path}/{Analyzer.MODEL_FILE}')
        return analyzer 
        
