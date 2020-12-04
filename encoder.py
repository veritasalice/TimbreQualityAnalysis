import pytorch_lightning as pl
import torch
from efficientnet_pytorch import EfficientNet
from torch.nn import functional as F


class Encoder(torch.nn.Module):
    def __init__(self, drop_connect_rate=0.1):
        super(Encoder, self).__init__()

        self.cnn1 = torch.nn.Conv2d(1, 3, kernel_size=3)
        self.efficientnet = EfficientNet.from_name(
            "efficientnet-b0", include_top=False, drop_connect_rate=drop_connect_rate
        )

    def forward(self, x):
        # print(x.shape) 20,80,80
        x = x.unsqueeze(1)

        x = self.cnn1(x)
        x = self.efficientnet(x)

        y = x.squeeze(3).squeeze(2)

        return y


class Claq(pl.LightningModule):
    def __init__(self, p=0.1):
        super().__init__()
        self.save_hyperparameters()

        self.p = p

        self.do = torch.nn.Dropout(p=self.p)

        self.encoder = Encoder(drop_connect_rate=p)

        self.g = torch.nn.Linear(1280, 512)
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=512)
        self.linear = torch.nn.Linear(512, 2, bias=False)

    def forward(self, data):
        x, y = data

        x = self.do(self.encoder(x))
        x = self.do(self.g(x))
        x = self.do(torch.tanh(self.layer_norm(x)))
        x = self.linear(x)

        return x, y

    def training_step(self, data, batch_idx):
        x, y = self(data)

        loss = F.cross_entropy(x, y)
        _, predicted = torch.max(x, 1)
        acc = (predicted == y).double().mean()

        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    def validation_step(self, data, batch_idx):
        x, y = self(data)

        loss = F.cross_entropy(x, y)
        _, predicted = torch.max(x, 1)
        acc = (predicted == y).double().mean()

        self.log("valid_loss", loss)
        self.log("valid_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


class MLPClassifier(pl.LightningModule):

    def __init__(self):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(80 * 80, 2048)
        self.layer_2 = torch.nn.Linear(2048, 256)
        self.layer_3 = torch.nn.Linear(256, 2)

    def forward(self, x):
        batch_size, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # layer 1 (b, 1*28*28) -> (b, 128)
        x = self.layer_1(x)
        x = torch.relu(x)

        # layer 2 (b, 128) -> (b, 256)
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3 (b, 256) -> (b, 10)
        x = self.layer_3(x)

        # probability distribution over labels
        x = torch.log_softmax(x, dim=1)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        _, predicted = torch.max(logits, 1)
        acc = (predicted == y).double().mean()
        self.log('train_loss', loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        _, predicted = torch.max(logits, 1)
        acc = (predicted == y).double().mean()
        self.log('val_loss', loss)
        self.log("valid_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# class Cola(pl.LightningModule):
#     def __init__(self, p=0.1):
#         super().__init__()
#         self.save_hyperparameters()

#         self.p = p

#         self.do = torch.nn.Dropout(p=self.p)

#         self.encoder = Encoder(drop_connect_rate=p)

#         self.g = torch.nn.Linear(1280, 512)
#         self.layer_norm = torch.nn.LayerNorm(normalized_shape=512)
#         self.linear = torch.nn.Linear(512, 512, bias=False)

#     def forward(self, x):
#         x1, x2 = x

#         x1 = self.do(self.encoder(x1))
#         x1 = self.do(self.g(x1))
#         x1 = self.do(torch.tanh(self.layer_norm(x1)))

#         x2 = self.do(self.encoder(x2))  ##
#         x2 = self.do(self.g(x2))
#         x2 = self.do(torch.tanh(self.layer_norm(x2)))

#         x1 = self.linear(x1)

#         return x1, x2

#     def training_step(self, x, batch_idx):
#         x1, x2 = self(x)

#         y = torch.arange(x1.size(0), device=x1.device)

#         y_hat = torch.mm(x1, x2.t())

#         loss = F.cross_entropy(y_hat, y)

#         _, predicted = torch.max(y_hat, 1)
#         acc = (predicted == y).double().mean()

#         self.log("train_loss", loss)
#         self.log("train_acc", acc)

#         return loss

#     def validation_step(self, x, batch_idx):
#         x1, x2 = self(x)

#         y = torch.arange(x1.size(0), device=x1.device)

#         y_hat = torch.mm(x1, x2.t())

#         loss = F.cross_entropy(y_hat, y)

#         _, predicted = torch.max(y_hat, 1)
#         acc = (predicted == y).double().mean()

#         self.log("valid_loss", loss)
#         self.log("valid_acc", acc)

#     def test_step(self, x, batch_idx):
#         x1, x2 = self(x)

#         y = torch.arange(x1.size(0), device=x1.device)

#         y_hat = torch.mm(x1, x2.t())

#         loss = F.cross_entropy(y_hat, y)

#         _, predicted = torch.max(y_hat, 1)
#         acc = (predicted == y).double().mean()

#         self.log("test_loss", loss)
#         self.log("test_acc", acc)

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=1e-4)


class AudioClassifier(pl.LightningModule):
    def __init__(self, classes=8, p=0.1):
        super().__init__()
        self.save_hyperparameters()

        self.p = p

        self.do = torch.nn.Dropout(p=self.p)

        self.encoder = Encoder(drop_connect_rate=self.p)

        self.g = torch.nn.Linear(1280, 512)
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=512)

        self.fc1 = torch.nn.Linear(512, 256)
        self.fy = torch.nn.Linear(256, classes)

    def forward(self, x):
        x = self.do(self.encoder(x))

        x = self.do(self.g(x))
        x = self.do(torch.tanh(self.layer_norm(x)))

        x = F.relu(self.do(self.fc1(x)))
        y_hat = self.fy(x)

        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("valid_loss", loss)
        self.log("valid_acc", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
