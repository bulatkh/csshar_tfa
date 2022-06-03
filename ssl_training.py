import argparse

from models.dtw import DTWModule
import os

from pytorch_lightning import Trainer, seed_everything

from models.simclr import SimCLR
from models.mlp import LinearClassifier, MLPDropout, ProjectionMLP, MLP
from models.supervised import SupervisedModel
from utils.experiment_utils import generate_experiment_id, load_yaml_to_dict, dict_to_json
from utils.training_utils import (init_datamodule, init_encoder,
                                  init_transforms, nested_to_flat_dict,
                                  setup_callbacks, setup_callbacks_ssl,
                                  setup_loggers, flat_to_nested_dict, init_ssl_pretrained)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    # configs paths
    parser.add_argument('--experiment_config_path', required=True, help='Path to experiment yaml file')
    parser.add_argument('--dataset_config_path', default='configs/dataset_configs.yaml', help='Path to datasets yaml file')
    parser.add_argument('--augmentations_path', help='Path to augmentations yaml file')

    # data and models
    parser.add_argument('--dataset', required=True, choices=['uci_har', 'mobi_act', 'usc_had'], help='Dataset name')
    parser.add_argument('--framework', default='simclr', choices=['simclr', 'dtw'], help='SSL framework')
    parser.add_argument('--model', required=True, choices=['transformer'], help='Encoder model')
    parser.add_argument('--model_save_path', default='./model_weights', help='Folder for the model weights')

    # used to run only in fine tuning mode
    parser.add_argument('--linear_eval', action='store_true', help='Flag for using linear evaluation protocol')
    parser.add_argument('--supervised', action='store_true', help='Flag for supervised experiments')
    parser.add_argument('--fine_tuning', action='store_true', help='Flag for fine-tuning only mode (pre-training is skipped)')
    parser.add_argument('--fine_tuning_ckpt_path', help='Path to a pretrained encoder. Required if running with --fine_tuning.')

    # other training configs
    parser.add_argument('--no_ckpt', action='store_true', default=False, help='Flag for running experiments without saving model weights')
    parser.add_argument('--num-workers', default=1, type=int, help='Num workers in dataloaders')
    parser.add_argument('--sweep', action='store_true', default=False, help='Set automatically if running in WandB sweep mode. You do not need to set this manually.')

	# cross-subject cross-validation
    parser.add_argument('--cross_subject_cv', action='store_true', default=False, help='Flag for using cross-subject cross-validation')
    parser.add_argument('--num_folds', default=5, help='Number of folds in cross-subject cv')
    parser.add_argument('--fine_tuning_ckpt_paths', nargs='+', help='Path to pre-trained encoders if only fine-tuning is needed for cross-subject cv')

	# semi-supervised learning
    parser.add_argument('--semi_sup', action='store_true', default=False, help='Flag for running semi-supervised learning experiments. Can be combined with --supervised')
    parser.add_argument('--semi_sup_runs', default=10, help='Number of SSL runs')
    parser.add_argument('--semi_sup_results_path', default='./results/semi_sup', help='Semi-sup results path')
    
    return parser.parse_args()


def ssl_pre_training(args, cfg, dataset_cfg, experiment_id, loggers_list, loggers_dict):
	""" Runs SSL pre-training
	
	Parameters
	----------
	args : argparse.Namespace
		arguments parsed from argparse
	cfg : dict
		experiment configs parsed from the input yaml
	dataset_cfg : dict
		dataset configs parsed from the input yaml
	experiment_id : string
		unique experiment name
	loggers_list : list
		list of loggers
	loggers_dict : dict
		dictionary with loggers

	Returns
	-------
	Pre-trained encoder model and altered cfg
	"""
	# seed for pre-training for reproducability
    seed_everything(cfg['experiment']['seed'])

    # initialize transforms: modailty transforms + random transformations for view generation
    num_epochs = cfg['experiment']['num_epochs_ssl']
    augmentations_dict = load_yaml_to_dict(args.augmentations_path)
    flat_augmentations_dict = nested_to_flat_dict({"augmentations": augmentations_dict})

    # config overwriting for sweeps
    if args.sweep:
        _wandb = loggers_dict['wandb'].experiment

        # Take some specific parameters.
        num_epochs = _wandb.config["num_epochs_ssl"]
        
        # Take SSL model kwargs and merge with experiment config.
        ssl_key_values = {key: _wandb.config[key] for key in _wandb.config.keys() if key.startswith('ssl.')}
        ssl_kwargs_dict = flat_to_nested_dict(ssl_key_values)
        if ssl_kwargs_dict != {}:
            cfg['model']['ssl']['kwargs'] = {**cfg['model']['ssl']['kwargs'], **ssl_kwargs_dict['ssl']}

        # Take encoder kwargs and merge with experiment config.
        encoder_key_values = {key: _wandb.config[key] for key in _wandb.config.keys() if key.startswith('model.')}
        encoder_kwargs_dict = flat_to_nested_dict(encoder_key_values)
        if encoder_kwargs_dict != {}:
            cfg['model'][args.model]['kwargs'] = {**cfg['model'][args.model]['kwargs'], **encoder_kwargs_dict['model']}

        # Take augmentation config from sweep and merge with default config.
        augmentation_key_values = {key: _wandb.config[key] for key in _wandb.config.keys() if key.startswith('augmentations.')}
        flat_augmentations_dict = {**flat_augmentations_dict, **augmentation_key_values}
        augmentations_dict = flat_to_nested_dict(flat_augmentations_dict)['augmentations']
    
    # init transforms   
    train_transforms, test_transforms = init_transforms(augmentations_dict)
    
    # init datamodule with ssl flag
    datamodule = init_datamodule(dataset_cfg[args.dataset]['train'], dataset_cfg[args.dataset]['val'], dataset_cfg[args.dataset]['test'], 
        batch_size=cfg['model']['ssl']['kwargs']['ssl_batch_size'], train_transforms=train_transforms, test_transforms=test_transforms, ssl=True, n_views=2, num_workers=args.num_workers)

    # initialize encoder, projection and ssl framework model
    encoder = init_encoder(cfg['model'][args.model])
	projection = ProjectionMLP(encoder.out_size, cfg['model']['ssl']['kwargs']['projection_hidden'], cfg['model']['ssl']['kwargs']['embedding_size'])
    
    if args.framework == 'simclr':
        model = SimCLR(encoder, projection, **cfg['model']['ssl']['kwargs'])
    elif args.framework == 'dtw':
        model = DTWModule(encoder, projection, **cfg['model']['ssl']['kwargs'])

    # init callbacks
    callbacks = setup_callbacks_ssl(
        no_ckpt               = args.no_ckpt,
        model_weights_path    = args.model_save_path, 
        dataset               = args.dataset, 
        model                 = args.model, 
        experiment_id         = experiment_id,
        online_eval           = False,
        online_eval_args      = None 
    )

    # initialize trainer and fit the ssl model
    trainer = Trainer.from_argparse_args(args=args, logger=loggers_list, gpus=1, deterministic=True, max_epochs=num_epochs, default_root_dir='logs',
        callbacks=callbacks, checkpoint_callback=not args.no_ckpt)
    
	# train the model
    trainer.fit(model, datamodule)

    return model.encoder, cfg


def fine_tuning(args, cfg, dataset_cfg, encoder, loggers_list, loggers_dict, experiment_id, limited_k=None, ft=True):
	""" Fine-tunes and tests an output model and freezes the provided encoder. If supervised argument is True, encoder is fine-tuned as well. 
	
	Parameters
	----------
	args : argparse.Namespace
		arguments parsed from argparse
	cfg : dict
		experiment configs parsed from the input yaml
	dataset_cfg : dict
		dataset configs parsed from the input yaml
	encoder : pytorch_lightning.core.lightning.LightningModule OR torch.nn.Module
		pytorch encoder
	loggers_list : list
		list of loggers
	loggers_dict : dict
		dictionary with loggers
	experiment_id : string
		unique experiment name
	limited_k : int
		Only for semi-sup: number of training data instances per class available for training 
	ft : bool
		fine-tuning flag: if ft is True -> encoder is frozen, if ft is False -> encoder is tuned as well

	Returns
	-------
	Dictionary with metrics and their values
	"""

    if not args.semi_sup:
           seed_everything(cfg['experiment']['seed']) # reset seed for consistency in results
    batch_size = cfg['experiment']['batch_size_fine_tuning']
    num_epochs = cfg['experiment']['num_epochs_fine_tuning']

    # if using wandb and performing a sweep, overwrite some config params with the sweep params.
    if args.sweep:
        _wandb = loggers_dict['wandb'].experiment
        batch_size = _wandb.config["batch_size_fine_tuning"]
        num_epochs = _wandb.config["num_epochs_fine_tuning"]

    # initialize classifier and the whole model
    if args.linear_eval:
        classifier = LinearClassifier(encoder.out_size, dataset_cfg[args.dataset]['n_classes'])
    else:
        classifier = MLPDropout(encoder.out_size, dataset_cfg[args.dataset]['n_classes'])
    model = SupervisedModel(encoder, classifier, fine_tuning=ft, metric_scheduler=dataset_cfg[args.dataset]['main_metric'], lr=cfg['model'][args.model]['kwargs']['lr'])

    # setup callbacks
    callbacks = setup_callbacks(
        early_stopping_metric = "val_f1-score",
        early_stopping_mode   = "max",
        class_names           = dataset_cfg[args.dataset]["class_names"],
        num_classes           = len(dataset_cfg[args.dataset]["class_names"]),
        no_ckpt               = args.no_ckpt,
        model_weights_path    = args.model_save_path, 
        metric                = 'val_' + dataset_cfg[args.dataset]['main_metric'], 
        dataset               = args.dataset, 
        model                 = 'ssl_finetuned_' + args.framework + '_' + args.model, 
        experiment_id         = experiment_id
    )

    # init datamodule
    datamodule = init_datamodule(dataset_cfg[args.dataset]['train'], dataset_cfg[args.dataset]['val'], dataset_cfg[args.dataset]['test'],
        batch_size=batch_size, num_workers=args.num_workers, limited_k=limited_k)

    # init trainer, run training (fine-tuning) and test
    trainer = Trainer.from_argparse_args(args=args, logger=loggers_list, gpus=1, deterministic=True, max_epochs=num_epochs, default_root_dir='logs', 
        callbacks=callbacks, checkpoint_callback=not args.no_ckpt)

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule, ckpt_path='best')

    # compile metrics
    metrics = {metric: float(val) for metric, val in trainer.callback_metrics.items()}

    # close wandb experiment logging
    if 'wandb' in loggers_dict:
        loggers_dict['wandb'].experiment.finish()

    return metrics


def init_loggers(args, cfg, experiment_id, fine_tune_only=False, approach='simclr'):
	""" Initialize the loggers based on the experiment configs. Typically creates wandb and tensorboard loggers.

	Parameters
	----------
	args : argparse.Namespace
		arguments parsed from argparse
	cfg : dict
		experiment configs parsed from the input yaml
	experiment_id : string
		unique experiment name
	fine_tuning_only : bool
		flag for fine-tuning without pre-training
	approach : string
		framework 
	Returns
	-------
	Dictionary with metrics and their values
	"""
    experiment_info = { # default values; may be overrided by sweep config
        "dataset": args.dataset,
        "model": cfg['model'][args.model]['encoder_class_name'],
        "seed": cfg['experiment']['seed']
    }
	# overwrite configs for sweeps
    if not fine_tune_only:
        num_epochs = cfg['experiment']['num_epochs_ssl']
        if args.augmentations_path is not None:
            augmentations_dict = load_yaml_to_dict(args.augmentations_path)
            flat_augmentations_dict = nested_to_flat_dict({"augmentations": augmentations_dict}) # need flat structure for wandb sweep to properly overwrite it
        else:
            flat_augmentations_dict = {}
        additional_info = { # default values; may be overrided by sweep config
            "ssl_framework": args.framework,
            "num_epochs_ssl": num_epochs,
            "num_epochs_fine_tuning": cfg['experiment']['num_epochs_fine_tuning'],
            "batch_size_fine_tuning": cfg['experiment']['batch_size_fine_tuning'],
            **flat_augmentations_dict,
        }
        experiment_info = {**experiment_info, **additional_info}
    
    loggers_list, loggers_dict = setup_loggers(tb_dir="tb_logs", experiment_info=experiment_info, dataset=args.dataset, 
        experiment_id=experiment_id, experiment_config_path=args.experiment_config_path, approach=approach)
    return loggers_list, loggers_dict


def run_one_experiment(args, cfg, dataset_cfg, limited_k=None):
	""" Runs one experiment with settings from passed arguments and configs

	Parameters
	----------
	args : argparse.Namespace
		arguments parsed from argparse
	cfg : dict
		experiment configs parsed from the input yaml
	dataset_cfg : dict
		dataset configs parsed from the input yaml
	limited_k : int
		Only for semi-sup: number of training data instances per class available for training 

	Returns
	-------
	Dictionary with metrics and their values
	"""
    experiment_id = generate_experiment_id()
    if args.supervised:
        approach = 'supervised'
    else:
        approach = 'ssl'
    if args.cross_subject_cv:
        approach += '_cscv'
    if args.semi_sup:
        approach += '_semi_sup'

    loggers_list, loggers_dict = init_loggers(args, cfg, experiment_id, fine_tune_only=False, approach=approach)
    ### ssl full pre-training + fine_tuning
    if not (args.supervised or args.fine_tuning):
        encoder, cfg = ssl_pre_training(args, cfg, dataset_cfg, experiment_id, loggers_list, loggers_dict)
        result_metrics = fine_tuning(args, cfg, dataset_cfg, encoder, loggers_list, loggers_dict, experiment_id, limited_k=limited_k)
    ### fine-tuning or supervised training
    else:
        model_cfg = cfg['model'][args.model]
        model_cfg['kwargs'] = {**dataset_cfg, **model_cfg['kwargs']}
        if args.fine_tuning:
            pre_trained_model = init_ssl_pretrained(model_cfg, args.fine_tuning_ckpt_path, cfg['model']['ssl']['kwargs']['projection_hidden'], dataset_cfg[args.dataset]['n_classes'])
            encoder = getattr(pre_trained_model, 'encoder')
        elif args.supervised:
            encoder = init_encoder(model_cfg)
        ft = not args.supervised
        result_metrics = fine_tuning(args, cfg, dataset_cfg, encoder, loggers_list, loggers_dict, experiment_id, ft=ft, limited_k=limited_k)
    return result_metrics


def validate_args(args):
    if args.fine_tuning and not (args.fine_tuning_ckpt_path or args.fine_tuning_ckpt_paths):
        print("Need to provide --fine_tuning_ckpt_path if running with --fine_tuning!")
        exit(1)


def main():
	# parse cli arguments and configs
    args = parse_arguments()
    validate_args(args)
    cfg = load_yaml_to_dict(args.experiment_config_path)
    dataset_cfg = load_yaml_to_dict(args.dataset_config_path)

	# cross-subject cross-validation
    if args.cross_subject_cv:
        for i in range(1, args.num_folds + 1):
            fold_path = os.path.join(dataset_cfg[args.dataset]['cross-subject'], 'fold{}'.format(i))
            dataset_cfg[args.dataset]['train'] = os.path.join(fold_path, 'train')
            dataset_cfg[args.dataset]['val'] = os.path.join(fold_path, 'val')
            dataset_cfg[args.dataset]['test'] = os.path.join(fold_path, 'test')
            if args.fine_tuning:
                args.fine_tuning_ckpt_path = args.fine_tuning_ckpt_paths[i - 1]
            print(dataset_cfg)
            run_one_experiment(args, cfg, dataset_cfg)
	
	# semi-supervised learning scenarios
    if args.semi_sup:
        results = {}
        for k in [1, 2, 5, 10, 25, 50, 100]:
            results[str(k)] = {}
            for i in range(args.semi_sup_runs):
                results[str(k)]['trial_{}'.format(i)] = {}
                metrics = run_one_experiment(args, cfg, dataset_cfg, limited_k=k)
                results[str(k)]['trial_{}'.format(i)][args.framework] = metrics['test_f1-score']
                dict_to_json(results, args.semi_sup_results_path + '_' + args.framework + '.json')         
    
	# single pre-training and fine-tuning experiment (feature representation learning -- when fine-tuning is done on the whole train set)
	else:
        run_one_experiment(args, cfg, dataset_cfg)


if __name__ == '__main__':
    main()
