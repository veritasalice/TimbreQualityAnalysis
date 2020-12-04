from glob import glob

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
# from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
# from audio_encoder.audio_processing import random_crop, random_mask, random_multiply
from encoder import Claq, MLPClassifier
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"


class clarinetDataset(torch.utils.data.Dataset):
    """ clarinetTimbre2018 Dataset extraction """
    def __init__(self, csv_file, feature_path, augment):
        labels = []
        if augment is False:
            filenames = []
            with open(csv_file, 'r') as f:
                content = f.readlines()
    #             print(content)
                for x in content:
                    row = x.strip('\n').split(',')
                    # print(row[0].replace("wav", "npy"))
                    filenames.append(os.path.join(feature_path, row[0].replace("wav", "npy")))
                    labels.append(row[1])

        else:
            filenames = sorted(list(glob(str(feature_path / "*.npy"))))
            for i in range(len(filenames)):
                if filenames[i][24] == '3':
                    labels.append('1')
                elif filenames[i][24] == '1':
                    labels.append('0')
                else:
                    print(filenames[i][24])
        self.datalist = filenames
        self.feature_path = feature_path
        self.labels = labels
        # print("Total data: {}".format(len(self.datalist)))
        # print("Total label: {}".format(len(self.labels)))

    def __len__(self):
        """ set the len(object) funciton """
        return len(self.datalist)

    def __getitem__(self, idx):
        """
        Function to extract the spectrogram samples and labels from the audio dataset.
        """
        data_path = self.datalist[idx]
        x = np.load(data_path)
        label = np.asarray(int(self.labels[idx]))

        x = torch.from_numpy(x).type(torch.FloatTensor)
        label = torch.from_numpy(label).type(torch.LongTensor)
        return x, label


def get_data_loaders(batch_size, use_cuda, feature_path, train_dir, eval_dir):

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    # data_transform = transforms.Compose([ToTensor()])

    clarinet_train = clarinetDataset(
        csv_file=train_dir, feature_path=feature_path, augment=True)
    clarinet_test = clarinetDataset(
        csv_file=eval_dir, feature_path=feature_path, augment=False)

# see data form
    print('len(clarinet_train): ', len(clarinet_train))
    print('len(clarinet_test): ', len(clarinet_test))

    train_loader = DataLoader(clarinet_train, batch_size=batch_size, shuffle=True, **kwargs)
#     print(train_loader)
    for ba in train_loader:
        x, y = ba
        break
    print('x.shape ', x.shape)
    print('y.shape ', y.shape)

    val_loader = DataLoader(clarinet_test, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader


class DecayLearningRate(pl.Callback):
    def __init__(self):
        self.old_lrs = []

    def on_train_start(self, trainer, pl_module):
        # track the initial learning rates
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            group = []
            for param_group in optimizer.param_groups:
                group.append(param_group["lr"])
            self.old_lrs.append(group)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            old_lr_group = self.old_lrs[opt_idx]
            new_lr_group = []
            for p_idx, param_group in enumerate(optimizer.param_groups):
                old_lr = old_lr_group[p_idx]
                new_lr = old_lr * 0.99
                new_lr_group.append(new_lr)
                param_group["lr"] = new_lr
            self.old_lrs[opt_idx] = new_lr_group


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_path", default="../clarinetData/audio/")
    parser.add_argument("--feature_path", default="../clarinetData/feature/")
    parser.add_argument("--train_dir", default="../clarinetData/evaluation_setup/train_data.csv")
    parser.add_argument("--eval_dir", default="../clarinetData/evaluation_setup/test_data.csv")
    parser.add_argument("--use_cuda", default=True)
    parser.add_argument("--gpus", default=4)
    parser.add_argument("--batch_size", default=20)
    parser.add_argument("--epochs", default=50)
    args = parser.parse_args()

    feature_path = Path(args.feature_path)
    # filenames = sorted(list(glob(str(feature_path / "*.npy"))))
    # print(filenames[15000])
    # print(filenames[15000][24])

    device = torch.device("cuda" if args.use_cuda else "cpu")
    # Load the data loaders
    train_loader, val_loader = get_data_loaders(args.batch_size, args.use_cuda,
                                                feature_path, args.train_dir, args.eval_dir)

    # model = Claq()
    model = MLPClassifier()

    logger = TensorBoardLogger(
        save_dir=".",
        name="lightning_logs",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_acc", mode="max", filepath="models/", prefix="encoder"
    )

    # change to distributed data parallel (gpus > 1)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[DecayLearningRate()],
        gpus=args.gpus,
        distributed_backend="ddp"
    )

    trainer.fit(model, train_loader, val_loader)

    # trainer.test(test_dataloaders=test_loader)
