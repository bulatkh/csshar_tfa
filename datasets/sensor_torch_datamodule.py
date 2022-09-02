from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from datasets.sensor_torch_dataset import SensorTorchDataset


class SensorDataModule(LightningDataModule):
    def __init__(self,
            train_path,
            val_path,
            test_path,
            batch_size,
            train_transforms = {},
            test_transforms = {},
            ssl = False,
            cae=False,
            n_views = 2,
            num_workers = 1,
            limited_k=None,
            store_in_ram=True,
            devices=None,
            noise_devices=None,
            noise_devices_test=None,
            randomly_masked_channels_test=0,
            get_subjects=False):
        super().__init__()
        # paths
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        # batch and transforms
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        # ssl related
        self.ssl = ssl
        self.cae = cae
        self.n_views = n_views
        self.num_workers = num_workers
        self.limited_k = limited_k
        # xai related
        self.noise_devices = noise_devices
        self.noise_devices_test = noise_devices_test
        self.randomly_masked_channels_test = randomly_masked_channels_test

        self.get_subjects = get_subjects
        self._get_use_devices(devices, noise_devices, noise_devices_test)

        self.store_in_ram = store_in_ram

        self._init_dataloaders()
        self.save_hyperparameters("batch_size", "ssl", "cae", "n_views", "limited_k", "noise_devices", "noise_devices_test", "randomly_masked_channels_test")

    def _get_use_devices(self, devices, noise_devices, noise_devices_test):
        use_devices = set(devices)
        if noise_devices is not None:
            for noise_d in noise_devices:
                use_devices.remove(noise_d)
        self.use_devices = list(use_devices)
        if noise_devices_test is None:
            self.use_devices_test = self.use_devices
        else:
            use_devices_test = set(devices)
            for noise_d in noise_devices_test:
                use_devices_test.remove(noise_d)
            self.use_devices_test = list(use_devices_test)

    def _init_dataloaders(self):
        train_dataset = self._create_train_dataset()
        test_dataset = self._create_test_dataset()
        val_dataset = self._create_val_dataset()

        drop_last_ssl = bool(self.ssl)
        self.train = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=drop_last_ssl, num_workers=self.num_workers, pin_memory=True)
        self.test = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers, pin_memory=True)
        self.val = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=drop_last_ssl, num_workers=self.num_workers, pin_memory=True)

    def _create_train_dataset(self):
        return SensorTorchDataset(
            data_path=self.train_path,
            ssl=self.ssl,
            transforms= self.train_transforms,
            limited=True if self.limited_k is not None else False,
            limited_k=self.limited_k,
            instance_data=False,
            cae=self.cae,
            store_in_ram=self.store_in_ram,
            use_devices=self.use_devices,
            get_subjects=self.get_subjects
        )
        
    def _create_val_dataset(self):
        if self.ssl:
            val_transforms = self.train_transforms
        else:
            val_transforms = self.test_transforms
        return SensorTorchDataset(
            data_path=self.val_path,
            ssl = self.ssl,
            transforms=val_transforms,
            cae = self.cae,
            store_in_ram=self.store_in_ram,
            use_devices=self.use_devices,
            get_subjects=self.get_subjects
        )

    def _create_test_dataset(self):
        return SensorTorchDataset(
            data_path=self.test_path,
            store_in_ram=self.store_in_ram,
            use_devices=self.use_devices_test,
            get_subjects=self.get_subjects,
            randomly_masked_channels=self.randomly_masked_channels_test
        )

    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.val

    def test_dataloader(self):
        return self.test


    
