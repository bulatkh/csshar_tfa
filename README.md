# Contrastive SSL for sensor-based Human Activity Recognition

This repository covers the following papers on SSL for sensor-based Human Activity Recognition:
1. B. Khaertdinov, E. Ghaleb, and S. Asteriadis, “Contrastive Self-supervised Learning for Sensor-based Human Activity Recognition“, in the 2021 International Joint Conference on Biometrics (IJCB 2021), Shenzhen, China, August 4-7, 2021 - [[link]](https://www.researchgate.net/publication/353481823_Contrastive_Self-Supervised_Learning_for_Sensor-based_Human_Activity_Recognition)

	![SIMCLR](/paper_img/simclr.png)

2. Partially (unimodal case) - B. Khaertdinov, E. Ghaleb, and S. Asteriadis, “Contrastive Self-supervised Learning with Temporal Feature alignment for Sensor-based Human Activity Recognition“, - Accepted to IJCB 2022.

Bibtex:
```
@INPROCEEDINGS{Khaertdinov_csshar_2021,
  author={Khaertdinov, Bulat and Ghaleb, Esam and Asteriadis, Stylianos},
  booktitle={2021 IEEE International Joint Conference on Biometrics (IJCB)}, 
  title={Contrastive Self-supervised Learning for Sensor-based Human Activity Recognition}, 
  year={2021},
  pages={1-8},
  doi={10.1109/IJCB52358.2021.9484410}}
  ```

```
@INPROCEEDINGS{Khaertdinov_tfa_2022,
  author={Khaertdinov, Bulat and Asteriadis, Stylianos},
  title={Contrastive Self-supervised Learning with Temporal Feature alignment for Sensor-based Human Activity Recognition}
}
```

# Requirements and environment

We used **PyTorch** and **PyTorch Lightning** for our experiments. Additionally, we used **Weights&Biases** for logging. Below are the versions of the main packages and libraries:
* Python: `3.7.4`
* CUDA: `9.1`
* torch: `1.7.1`
* pytorch_lightning: `1.5.9`
* wandb: `0.12.6`

We also exported our conda environment into ``requirements.txt``. You can create and activate it by running the following command:
```
$ conda create --name csshar --file requirements.txt
$ conda activate csshar
```

For more stable training, we use layer-wise adaptive rate control and wrap optimizers into LARC from **apex**. Installation guidelines can be found via the following link: https://github.com/NVIDIA/apex#quick-start

# Datasets and pre-processing
In this project, we use UCI-HAR, USC-HAD and MobiAct datasets. Specifically, raw accelerometer and gyroscope signals are downsampled to 30Hz and segmented into 50% overlapping time-windows of 1 second length. Then, the training, validation and test splits are created based on subjects. Finally, signals are normalized to have zero mean and unit variance per channel based on training data.

Links to the datasets:
1. UCI-HAR: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
2. USC-HAD: https://sipi.usc.edu/had/
3. MobiAct: https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/

Download the datasets and unzip to `./data` folder in order to use the following scripts without changing them.

The pre-processing scripts can be launched as follows:
## 1. Sampling
```
$ python sample_datasets.py --datasets usc_had --paths data/usc_had/USC-HAD/ --destination sampled_data/usc_had/ --downsample_freq 30

$ python sample_datasets.py --datasets uci_smartphones --paths data/uci_smartphones/RawData/ --destination sampled_data/uci_har/ --downsample_freq 30

$ python sample_datasets.py --datasets mobi_act --paths ./data/mobi_act/MobiAct_Dataset_v2.0/Annotated\ Data/ --destination sampled_data/mobi_act --downsample_freq 30 
```

## 2. Create train-val-test splits or cross-validation folds:
```
$ python split_dataset.py --dataset usc_had --path ./sampled_data/usc_had/usc_had/ --val_users subject11 subject12 --test_users subject13 subject14

$ python split_dataset.py --dataset uci_har --path ./sampled_data/uci_har/uci_smartphones/ --random_subject_split 

$ python split_dataset.py --dataset mobi_act --path ./sampled_data/mobi_act/mobi_act/ --random_subject_split 
```

Train, validation and test splits are created randomly based on subjects for UCI-HAR and MobiAct datasets. Due to subject variability, the final results may vary. Below are the subjects splits we got:
```
mobi_act: {
		'train': ['subject1', 'subject10', 'subject11', 'subject12', 'subject18', 'subject19', 'subject2', 'subject20', 'subject26', 'subject27', 'subject28', 'subject29', 'subject32', 'subject33', 'subject34', 'subject35', 'subject38', 'subject39', 'subject4', 'subject40', 'subject44', 'subject45', 'subject46', 'subject47', 'subject48', 'subject49', 'subject51', 'subject55', 'subject57', 'subject58', 'subject59', 'subject62', 'subject64', 'subject65', 'subject66', 'subject67', 'subject7', 'subject8']
		'val': ['subject16', 'subject21', 'subject22', 'subject23', 'subject24', 'subject3', 'subject37', 'subject52', 'subject54', 'subject9']
		'test': ['subject25', 'subject36', 'subject41', 'subject42', 'subject43', 'subject5', 'subject50', 'subject53', 'subject56', 'subject6', 'subject60', 'subject61', 'subject63']
}

uci_har: {
		'train': ['subject01', 'subject05', 'subject06', 'subject07', 'subject08', 'subject09', 'subject12', 'subject13', 'subject15', 'subject16', 'subject17', 'subject18', 'subject19', 'subject20', 'subject22', 'subject23', 'subject25', 'subject27', 'subject29']
		'val': ['subject04', 'subject10', 'subject21', 'subject24', 'subject28']
		'test': ['subject02', 'subject03', 'subject11', 'subject14', 'subject26', 'subject30']
}

```

Additionally, there is an option to create folds for cross-validation and run experiments in cross-subject cross-validation settings. For this two arguments have to be added. Example for MobiAct:
```
$ python sample_datasets.py --datasets mobi_act --paths data/mobi_act/MobiAct_Dataset_v2.0/Annotated/ --destination sampled_data/cs_mobi_act/ --downsample_freq 30 --cross_subject_splits --num_folds 5
```

## 3. Normalization
This step might take quite some time, especially for the MobiAct dataset.
```
$ python normalization.py --train_path ./sampled_data/usc_had/usc_had/train/ --normalization_paths ./sampled_data/usc_had/usc_had/train/ ./sampled_data/usc_had/usc_had/val ./sampled_data/usc_had/usc_had/test

$ python normalization.py --train_path ./sampled_data/uci_har/uci_smartphones/train/ --normalization_paths ./sampled_data/uci_har/uci_smartphones/train/ ./sampled_data/uci_har/uci_smartphones/val ./sampled_data/uci_har/uci_smartphones/test

$ python normalization.py --train_path ./sampled_data/mobi_act/mobi_act/train/ --normalization_paths ./sampled_data/mobi_act/mobi_act/train/ ./sampled_data/mobi_act/mobi_act/val/ ./sampled_data/mobi_act/mobi_act/test/
```

## Running Experiments:
In this section, we introduce how to run experiments on the MobiAct dataset (the same commands can be used for UCI-HAR and USC-HAD -- you only have to alter some args) for the default folder structure of data obtained in the previous step. In our papers we use CNN Encoder with Transformer-like self-attention layers defined in `models/transformer.py`. You can also use our training scripts with your own encoder derived from `torch.nn.Module` by adjusting an experiment configuration file (e.g. `./configs/models/ssl/simclr/simclr_mobiact.yaml`).

<<<<<<< Updated upstream
=======
For Temporal Feature Alignment, we use SoftDTW implementation from https://github.com/Maghoumi/pytorch-softdtw-cuda. Git clone this repository to `./libraries` or comment out imports related to SoftDTW if you do not wish to use it.

We used [LARC](https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py) optimizer from `apex.parallel`. As Pytorch optimizers require to have `closure` argument in `step` function (`closure` is used by some algorithms that re-evaluate the loss function multiple times -- it is not the case for our implementation), we added this argument to `step` function from `LARC.py` in our site-packages in order to make it compatible. You can also do this if you face the following error: `TypeError: step() got an unexpected keyword argument 'closure'`. Alternatively, you can try to change optimizer in `models/simclr.py` to `FusedLAMB` from `apex` or `LARS` from `lightning_bolts`.

>>>>>>> Stashed changes
1. Pre-training and fine-tuning on the whole training set using SimCLR (CSSHAR). The MLP is evaluated on the validation set after each epoch, and the best model is used for testing. To use, temporal feature alignment (TFA), change argument value to `--framework dtw`.
You can also add `--linear_eval` flag to use a linear evaluation protocol. The results and logged values (loss curves, etc.) could be visualized in the wandb client (or using tensorboard logs).

	```
	$ python ssl_training.py --experiment_config_path ./configs/models/ssl/simclr/simclr_mobiact.yaml --dataset mobi_act --augmentations_path configs/augmentations/jit_scal_rot.yaml --framework simclr --model transformer --num-workers 16
	```

2. Fine-tuning only. Needs a saved model for running.
	```
	$ python ssl_training.py --experiment_config_path ./configs/models/ssl/simclr/simclr_mobiact.yaml --dataset mobi_act --augmentations_path configs/augmentations/jit_scal_rot.yaml --framework simclr --model transformer --num-workers 16 --fine_tuning --fine_tuning_ckpt_path <path-to-pretrained-model>
	```


3. Semi-supervised learning. Pre-trained encoder is frozen and only the output MLP is trained. The F1-scores are aggregated into the json file in `/results`. More performance metrics and values could be checked in the wandb client.
	 
	```
	$ python ssl_training.py --experiment_config_path ./configs/models/ssl/simclr/simclr_mobiact.yaml --dataset mobi_act --augmentations_path configs/augmentations/jit_scal_rot.yaml --framework dtw --model transformer --num-workers 16  --semi_sup --semi_sup_results_path ./results/mobiact_semi_sup --fine_tuning --fine_tuning_ckpt_path <path-to-pretrained-model>
	```

