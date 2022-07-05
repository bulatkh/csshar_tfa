import importlib
import itertools
import os
import shutil
import torch
from models.mlp import ProjectionMLP

from models.simclr import SimCLR
from models.mlp import MLP, MLPDropout
from models.supervised import SupervisedModel
from torchvision import transforms
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from callbacks.log_classifier_metrics import LogClassifierMetrics
from callbacks.log_confusion_matrix import LogConfusionMatrix

from datasets.sensor_torch_datamodule import SensorDataModule
from utils.augmentation_utils import compose_random_augmentations
from utils.experiment_utils import load_yaml_to_dict


def init_transforms(random_augmentations_dict={}):
    train = compose_random_augmentations(random_augmentations_dict)
    # train.append(transforms.ToTensor())
    train = transforms.Compose(train)
    test = transforms.Compose([])
    return train, test

  
def init_datamodule(train_path, val_path, test_path, batch_size,
                    train_transforms={}, test_transforms={},
                    ssl = False, n_views = 2, num_workers = 1, limited_k=None, store_in_ram=True):
    data_module = SensorDataModule(train_path, val_path, test_path, batch_size, train_transforms = train_transforms, test_transforms = test_transforms,
        ssl = ssl, n_views = n_views, num_workers = num_workers, limited_k = limited_k, store_in_ram = store_in_ram)
    return data_module

  
def init_model(model_cfg, metric_scheduler='accuracy', ckpt_path=None):
    module = importlib.import_module(f"models.{model_cfg['from_module']}")
    class_ = getattr(module, model_cfg['class_name'])
    if ckpt_path is None:
        return class_(*model_cfg['args'], **model_cfg['kwargs'], metric_scheduler=metric_scheduler)
    else:
        return class_.load_from_checkpoint(ckpt_path, strict=False)


def init_ssl_pretrained(model_cfg, ckpt_path, fc_size, n_classes, ssl_batch_size=256, temperature=0.1):
    encoder = init_encoder(model_cfg)
    projection = ProjectionMLP(encoder.out_size, fc_size, n_classes)
    class_ = SimCLR(encoder, projection, ssl_batch_size, temperature)
    return class_.load_from_checkpoint(ckpt_path, encoder=encoder, mlp_in_size=encoder.out_size, strict=False)


def init_encoder(model_cfg, ckpt_path=None):
    module = importlib.import_module(f"models.{model_cfg['from_module']}")
    class_ = getattr(module, model_cfg['encoder_class_name'])
    if ckpt_path is None:
        return class_(*model_cfg['args'], **model_cfg['kwargs'])
    else:
        return class_.load_from_checkpoint(ckpt_path)
    
def init_finetuned_ckpt(model_cfg, n_classes, le=False, ckpt_path=None, mlp_do=True):
    encoder = init_encoder(model_cfg)
    if le:
        LinearClassifier(encoder.out_size, n_classes)
    elif mlp_do:
        classifier = MLPDropout(encoder.out_size, n_classes)
    else:
        classifier = MLP(encoder.out_size, n_classes)
    model = SupervisedModel(encoder=encoder, classifier=classifier)
    return model.load_from_checkpoint(ckpt_path, encoder=encoder, classifier=classifier, strict=False)

def parse_splits(dataset_configs):
    return dataset_configs['splits']['train'], dataset_configs['splits']['val'], dataset_configs['splits']['test']


def setup_tb_logger(dir, name):
    return loggers.TensorBoardLogger(dir, name=name)


def setup_wandb_logger(experiment_info, dataset, experiment_id, entity='ssl_har', approach='supervised'):
    return loggers.WandbLogger(config=experiment_info, entity=entity, project=f"{approach}-{dataset}", name=experiment_id, id=experiment_id)


def setup_loggers(logger_names=['tensorboard', 'wandb'], tb_dir=None, experiment_info=None, dataset=None, 
        experiment_id=None, entity='sensor_har', approach='supervised', experiment_config_path=None):
    loggers = []
    loggers_dict = {}
    if 'tensorboard' in logger_names:
        tb_logger = setup_tb_logger(tb_dir, experiment_id)
        loggers.append(tb_logger)
        loggers_dict['tensorboard'] = tb_logger
    if 'wandb' in logger_names:
        wandb_logger = setup_wandb_logger(experiment_info, dataset, experiment_id, entity, approach)
        loggers.append(wandb_logger)
        loggers_dict['wandb'] = wandb_logger
        shutil.copy(experiment_config_path, os.path.join(wandb_logger.experiment.dir, "experiment_config.yaml"))
    return loggers, loggers_dict


def setup_early_stopping_callback(metric, min_delta=0.00, patience=50, mode="min"):
    return EarlyStopping(monitor=metric, min_delta=min_delta, patience=patience, verbose=False, mode=mode)


def setup_confusion_matrix_logger(class_names):
    return LogConfusionMatrix(class_names)


def setup_classifier_metrics_logger(num_classes, metric_names=['accuracy', 'f1-score', 'precision', 'recall'], average='macro'):
    return LogClassifierMetrics(num_classes, metric_names, average=average)


def setup_model_checkpoint_callback(model_weights_path, metric, dataset, model, experiment_id):
    return ModelCheckpoint(
        monitor=metric, 
        dirpath=os.path.join(model_weights_path, f"{dataset}-{model}-{experiment_id}"),
        filename="{epoch}",
        save_top_k=1,
        mode="max",
        save_last=True
    )	


def setup_model_checkpoint_callback_last(model_weights_path, dataset, model, experiment_id):
    return ModelCheckpoint(
        save_last=True,
        dirpath=os.path.join(model_weights_path, f"{dataset}-{model}-{experiment_id}"),
        filename="{epoch}"
    )	


def setup_callbacks(early_stopping_metric, early_stopping_mode, class_names, num_classes, no_ckpt, model_weights_path, metric, dataset, model, experiment_id):
    callbacks = []
    callbacks.append(setup_early_stopping_callback(early_stopping_metric, mode=early_stopping_mode))
    callbacks.append(setup_confusion_matrix_logger(class_names))
    callbacks.append(setup_classifier_metrics_logger(num_classes))
    if not no_ckpt:
        callbacks.append(setup_model_checkpoint_callback(model_weights_path, metric, dataset, model, experiment_id))
    return callbacks


def setup_callbacks_ssl(no_ckpt, model_weights_path, dataset, model, experiment_id, online_eval, online_eval_args):
    callbacks = []
    if not no_ckpt:
        callbacks.append(setup_model_checkpoint_callback_last(model_weights_path, dataset, model, experiment_id))
    return callbacks


def check_sampling_cfg(model_cfg, transform_cfg):
    for i, transform in enumerate(transform_cfg):
        if ('transform_name' in transform
            and transform['transform_name'] == 'sampling'
            and transform_cfg[i]['kwargs']['size'] != model_cfg['kwargs']['sample_length']):
            transform_cfg[i]['kwargs']['size'] = model_cfg['kwargs']['sample_length']
    return model_cfg, transform_cfg


def flat_key_to_dict(flat_key, value):
    """
    Example input: "augmentations_configs.scaling.apply", True
    Example output: {'augmentations_configs': {'scaling': {'apply': True}}}
    """
    tokens = flat_key.split('.')
    if len(tokens) == 1:
        return {flat_key: value}
    
    last_key = tokens[-1]
    last_dict = {last_key: value}
    for inner_key in tokens[-2::-1]:
        last_dict = {inner_key: last_dict}

    return last_dict


def deep_merge_dicts(dict1, dict2):
    """
    Performs a deep merge of dict1 and dict2, with leaf values from dict2 overwriting dict1 if needed.
    """
    for k in set(dict1.keys()).union(dict2.keys()):
        if k in dict1 and k in dict2:
            if isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
                yield (k, dict(deep_merge_dicts(dict1[k], dict2[k])))
            else:
                # If one of the values is not a dict, value from second dict overrides one in first and we move on.
                yield (k, dict2[k])
        elif k in dict1:
            yield (k, dict1[k])
        else:
            yield (k, dict2[k])


def flat_to_nested_dict(flat_dict):
    """
    Example input:
        {
            "augmentations.scaling.apply": True,
            "augmentations.scaling.parameters.min_p": 0.5,
            "augmentations.scaling.parameters.max_p": 0.8
        }
    Example output:
        {
            "augmentations": {
                "scaling": {
                    "apply": True,
                    "parameters": {
                        "min_p": 0.5,
                        "max_p": 0.8
                    }
                }
            }
        }
    """
    nested_dict = {}
    for key in flat_dict:
        nested_inner = flat_key_to_dict(key, flat_dict[key])
        nested_dict = dict(deep_merge_dicts(nested_dict, nested_inner))
    return nested_dict


def nested_to_flat_dict(nested_dict):
    out = {}
    for key, val in nested_dict.items():
        if isinstance(val, dict):
            val = [val]
        if isinstance(val, list):
            for subdict in val:
                deeper = nested_to_flat_dict(subdict).items()
                out.update({key + '.' + key2: val2 for key2, val2 in deeper})
        else:
            out[key] = val
    return out