from __future__ import annotations

import os, torch, pickle, shutil
import torch.nn as nn
from typing import List, Tuple, Union, Callable
from data import Subtask1Dataset, Conversation, CauseRelation
from utils import *
from torch.utils.data import DataLoader, Dataset
from subtask1.model import Subtask1Model
from tqdm import tqdm
from metric import Subtask1Metric
from torch.optim import AdamW, Optimizer, Adam, RMSprop
from torch.optim.lr_scheduler import StepLR
import numpy as np 

class Subtask1Analyzer:
    def __init__(
        self,
        model: nn.Module,
        input_tkzs: List[Tokenizer],
        target_tkzs: List[Tokenizer]
    ):
        self.model = model
        self.input_tkzs = input_tkzs
        self.target_tkzs = target_tkzs
        self.optimizer = AdamW(self.model.parameters(), lr=1e-4)
        
        for tkz in [*input_tkzs, *target_tkzs]:
            self.__setattr__(tkz.field, tkz)


    def train(
        self,
        train: Subtask1Dataset,
        dev: Subtask1Dataset,
        test: Subtask1Dataset,
        path: str,
        optimizer: Callable = AdamW,
        lr: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 100,
        batch_update: int = 1, 
        patience: int = 50,
        show: bool = True,
        step_lr: int = 10, 
        last_lr: float = 1e-6,
        gamma: float = -1
    ):
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path)
            
        train_dl = DataLoader(train, collate_fn=self.transform, batch_sampler=LengthSampler(train.lens, batch_size))

        # create a folder like the submission format 
        os.makedirs(f'{path}/submission/ref/')
        os.makedirs(f'{path}/submission/res/')
        dev.save(f'{path}/submission/ref/Subtask_1_gold.json', submission=False)

        
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.last_lr = last_lr
        self.scheduler = StepLR(self.optimizer, step_size=step_lr, gamma=(last_lr/lr)**(step_lr/(epochs*len(train_dl)) if gamma==-1 else gamma), last_epoch=-1)

        train_improv, val_improv = patience, patience 
        best_dev = Subtask1Metric()
        for epoch in range(epochs):
            train_loss = self.forward(epoch, train_dl, batch_update, step_lr=step_lr)
            dev_metric = self.eval(dev, f'{path}/submission/', batch_size, show)
            if show:
                print(f'Epoch {epoch} [train]: loss={float(train_loss):.3f}, lr={self.lr:.3e}' + (' (improved)' if dev_metric.improves(best_dev) else ' '))
                print(f'Epoch {epoch} [dev]: {repr(dev_metric)}')
            if dev_metric.improves(best_dev):
                best_dev, val_improv = dev_metric, patience 
                self.save_model(f'{path}/model.pt')
                # prediction with tests         
                self.predict(test, [f'{path}/Subtask_{i}_pred.json' for i in (1, 2)], batch_size=batch_size, show=show)
            else:
                val_improv -= 1

            if train_improv == 0:
                print('No more improvement in train set')
                break 
            if val_improv == 0:
                print('No more improvement in val set')
                break
            del train_loss, dev_metric
        
        
        self.load_model(f'{path}/model.pt')
        self.save(path)
        
    def predict(
        self, 
        data: Subtask1Dataset, 
        output: Optional[str],
        batch_size: int = 10,
        show: bool = True
    ) -> Subtask1Dataset: 
        loader = DataLoader(data, collate_fn=self.transform, batch_sampler=LengthSampler(data.lens, batch_size, shuffle=False))
        preds = []
        for inputs, _, masks, convs in tqdm(loader, total=len(loader), disable=not show, desc='Prediction'):
            preds += self.pred_step(inputs, masks, convs)
        data.convs = preds 
        if output:
            data.save(output, submission=True)
        return data
        
    def eval(
        self, 
        data: Subtask1Dataset,
        path: str, 
        batch_size: int = 500,
        show: bool = True 
    ) -> Subtask1Metric:
        self.predict(data, path, batch_size, show)
        return Subtask1Metric(path)

    def forward(self, epoch: int, loader: DataLoader, batch_update: int = 1, show: bool = True, step_lr: int = 1) -> torch.Tensor:
        global_loss = 0.0
        loss = 0.0
        with tqdm(total=len(loader), disable=not show) as bar:
            bar.set_description(f'Epoch {epoch} (train)')
            for i, (inputs, targets, masks, _) in enumerate(loader):
                ut_loss, em_loss, span_loss = self.train_step(inputs, targets, masks)
                loss += (ut_loss + em_loss)
                if i % batch_update == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
                    self.optimizer.step()
                    if (epoch*len(loader) + i) % step_lr == 0 and (self.lr > self.last_lr):
                        self.scheduler.step()
                    global_loss += loss
                    loss = 0.0
                bar.update(1)
                bar.set_postfix({'ut': round(float(ut_loss), 2), 'em': round(float(em_loss), 2), 'span': round(float(span_loss), 2), 'lr': self.lr})
                torch.cuda.empty_cache()
            bar.close()
        return global_loss / len(loader)

    def train_step(self, inputs: Tuple[torch.Tensor], targets: Tuple[torch.Tensor], masks: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        words, speakers, emotions, graphs, spans, pad_mask, span_mask = *inputs, *targets, *masks
        scores = self.model(words, speakers, pad_mask, graphs)
        return self.model.loss(*scores, emotions, graphs, spans > 0, pad_mask, span_mask)

    @torch.no_grad()
    def pred_step(self, inputs: Tuple[torch.Tensor], masks: Tuple[torch.Tensor], convs: List[Conversation]) -> List[Conversation]:
        words, speakers, pad_mask, span_mask = *inputs, *masks
        lens = pad_mask.sum(-1).tolist()
        ut_preds, em_preds, span_preds = self.model.predict(words, speakers, pad_mask, span_mask)
        em_preds = em_preds[pad_mask].split(lens)
        ut_preds = [ut_pred[:l, :l] for ut_pred, l in zip(ut_preds.unbind(0), lens)]
        span_preds = [span_pred[:l, :l, :(conv.max_len+1)] for span_pred, l, conv in zip(span_preds.unbind(0), lens, convs)]
        preds = [conv.update1(graph, self.EMOTION.batch_decode(em), span) for conv, graph, em, span in zip(convs, ut_preds, em_preds, span_preds)]
        return preds 

    def transform(self, batch) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[Conversation]]:
        inputs, targets, conv = zip(*batch)
        words, speakers = [t.batch_encode(input) for t, input in zip(self.input_tkzs, zip(*inputs))]
        emotions, graphs, spans = [t.batch_encode(target) for t, target in zip(self.target_tkzs, zip(*targets))]
        pad_mask = (speakers != self.SPEAKER.pad_index)
        span_mask = (spans != self.SPAN.pad_index)
        return (words, speakers), (emotions, graphs, spans), (pad_mask, span_mask), conv
    
    @property 
    def lr(self):
        return self.scheduler.get_last_lr()[-1]
    
    def save(self, path: str):
        # save tokenizers
        os.makedirs(f'{path}/tkz/')
        for tkz in [*self.input_tkzs, *self.target_tkzs]:
            tkz.save(f'{path}/tkz/{tkz.field}')
            
        # save parameters 
        with open(f'{path}/params.pickle', 'wb') as writer:
            pickle.dump(self.model.params, writer)
        
        # save model 
        self.save_model(f'{path}/model.pt')
        
        
    def save_model(self, path: str):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
        
        
    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    @classmethod 
    def load(cls, path: str):
        # load tokenizers 
        tkzs = [Tokenizer.load(f'{path}/tkz/{name}') for name in os.listdir(f'{path}/tkz/')]
        tkzs = {tkz.field: tkz for tkz in tkzs}
        input_tkzs = [tkzs[field] for field in ['TEXT', 'SPEAKER']]
        target_tkzs = [tkzs[field] for field in ['EMOTION', 'GRAPH', 'SPAN']]
        
        # load params
        with open(f'{path}/params.pickle', 'rb') as reader:
            args = pickle.load(reader)
        model = Subtask1Model(**args())
        
        analyzer = Subtask1Analyzer(model, input_tkzs, target_tkzs)
        analyzer.load_model(f'{path}/model.pt')
        return analyzer 
        

    @classmethod
    def build(
        cls,
        data: Subtask1Dataset,
        text_conf: Config,
        device: str, 
        ut_embed_size: int = 400,
        spk_embed_size: int = 50,
    ) -> Subtask1Analyzer:
        input_tkzs = [
            TextTokenizer('TEXT', text_conf.pretrained, lower=False, bos=True, eos=False), 
            PositionalTokenizer('SPEAKER', max(map(len, data.convs)))
        ]
        target_tkzs = [
            Tokenizer('EMOTION', lower=True, max_words=None), 
            GraphTokenizer('GRAPH'), 
            SpanTokenizer('SPAN')
        ]

        for tkz in filter(lambda x: x.TRAINABLE, [*input_tkzs, *target_tkzs]):
            tkz.fit(flatten_list([getattr(conv, tkz.field) for conv in data.convs]))

        # construct model
        text_conf.pad_index = input_tkzs[0].pad_index
        spk_conf = Config(vocab_size=len(input_tkzs[1]), embed_size=spk_embed_size, pad_index=input_tkzs[1].pad_index)
        em_conf = Config(vocab_size=len(target_tkzs[0]), pad_index=target_tkzs[0].pad_index, weights=target_tkzs[0].weights)
        model = Subtask1Model(ut_embed_size, device, text_conf, spk_conf, em_conf)

        analyzer = Subtask1Analyzer(model, input_tkzs, target_tkzs)
        return analyzer



