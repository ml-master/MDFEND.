from typing import List

import torch
import yaml
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected

from faknow.evaluate.evaluator import Evaluator
from faknow.model.social_context.upfd import (UPFDSAGE, UPFDGCN, UPFDGAT,
                                              UPFDGCNFN)
from faknow.train.base_gnn_trainer import BaseGNNTrainer
from faknow.utils.util import dict2str

__all__ = ['run_upfd', 'run_upfd_from_yaml']


def run_upfd(root: str,
             name: str,
             feature='bert',
             splits=None,
             base_model='sage',
             batch_size=128,
             epochs=30,
             lr=0.01,
             weight_decay=0.01,
             metrics: List = None,
             device='cpu'):
    r"""
    run UPFD using UPFD dataset, including training, validation and testing.
    If validation and testing data are not provided, only training is performed.

    Args:
        root (str): Root directory where the dataset should be saved
        name (str): The name of the graph set (:obj:`"politifact"`, :obj:`"gossipcop"`)
        feature (str): The node feature type (:obj:`"profile"`, :obj:`"spacy"`, :obj:`"bert"`, :obj:`"content"`)
            If set to :obj:`"profile"`, the 10-dimensional node feature
            is composed of ten Twitter user profile attributes.
            If set to :obj:`"spacy"`, the 300-dimensional node feature is
            composed of Twitter user historical tweets encoded by
            the `spaCy word2vec encoder
            <https://spacy.io/models/en#en_core_web_lg>`_.
            If set to :obj:`"bert"`, the 768-dimensional node feature is
            composed of Twitter user historical tweets encoded by the
            `bert-as-service <https://github.com/hanxiao/bert-as-service>`_.
            If set to :obj:`"content"`, the 310-dimensional node feature is
            composed of a 300-dimensional "spacy" vector plus a
            10-dimensional "profile" vector. default='bert'
        splits (List[str]): dataset split, including 'train', 'val' and 'test'.
            If None, ['train', 'val', 'test'] will be used, default=None
        base_model (str): base model for UPFD, including 'sage', 'gcn', 'gat', 'gcnfn', default='sage'
        batch_size (int): batch size, default=128
        epochs (int): number of epochs, default=30
        lr (float): learning rate, default=0.01
        weight_decay (float): weight decay, default=0.01
        metrics (List): evaluation metrics, if None, ['accuracy', 'precision', 'recall', 'f1'] is used, default=None
        device (str): device, default='cpu'
    """

    assert base_model in [
        'sage', 'gcn', 'gat', 'gcnfn'
    ], "base_model must be in ['sage', 'gcn', 'gat', 'gcnfn']"

    if splits is None:
        splits = ['train', 'val', 'test']

    train_dataset = UPFD(root, name, feature, 'train', ToUndirected())
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    if 'val' in splits:
        val_dataset = UPFD(root, name, feature, 'val', ToUndirected())
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    else:
        val_loader = None

    feature_size = train_dataset.num_features
    if base_model == 'sage':
        model = UPFDSAGE(feature_size)
    elif base_model == 'gcn':
        model = UPFDGCN(feature_size)
    elif base_model == 'gat':
        model = UPFDGAT(feature_size)
    else:
        model = UPFDGCNFN(feature_size)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr,
                                 weight_decay=weight_decay)
    evaluator = Evaluator(metrics)

    trainer = BaseGNNTrainer(model, evaluator, optimizer, device=device)
    trainer.fit(train_loader, epochs, val_loader)

    if 'test' in splits:
        test_dataset = UPFD(root, name, feature, 'test', ToUndirected())
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)
        test_result = trainer.evaluate(test_loader)
        trainer.logger.info(f"test result: {dict2str(test_result)}")


def run_upfd_from_yaml(path: str):
    """
    run UPFD from yaml config file

    Args:
        path (str): yaml config file path
    """

    with open(path, 'r', encoding='utf-8') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_upfd(**_config)
