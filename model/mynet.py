import torch
#import torchmetrics
#import pytorch_lightning as pl
from torch import nn


class mynet(nn.Module):
    def __init__(self, in_channel):
        super(mynet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=16, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),


            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
        )

        self.fc1 = torch.nn.Sequential(torch.nn.Linear(2048, 512), torch.nn.ReLU(),torch.nn.BatchNorm1d(512))
        self.fc2 = torch.nn.Sequential(torch.nn.Linear(512, 128),torch.nn.ReLU(),torch.nn.BatchNorm1d(128))
        self.fc3 = torch.nn.Sequential(torch.nn.Linear(128, 32),torch.nn.ReLU(),torch.nn.BatchNorm1d(32))
        self.fc4 = torch.nn.Sequential(torch.nn.Linear(32, 5))


    def forward(self, x):
        y = self.net(x)
        y = y.view(y.size(0),-1)

        y = self.fc1(y)
        y = self.fc2(y)
        y = self.fc3(y)
        y = self.fc4(y)
        return y
'''
class mynet_light(pl.LightningModule):
    def __init__(self
                 #,pretrain_path='pretrain.ckpt'
                 ):
        super(mynet_light, self).__init__()
        self.learning_rate = 1e-4
        self.net = mynet(3)
        #self.net.load_state_dict(torch.load(pretrain_path)['state_dict'],strict=False)
        self.lossfun = nn.CrossEntropyLoss()
        self.trainfun = torchmetrics.Accuracy()
        self.valfun = torchmetrics.Accuracy()
        self.testfun = torchmetrics.Accuracy()

        # self.example_input_array = torch.ones(1,3,512,1024)
    def forward(self,x):
        out = self.net(x)
        return out
    def training_step(self,batch, batch_idx):
        img,gt = batch
        pred = self.net(img)
        train_loss = self.lossfun(pred,gt)
        self.log("train_loss", train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.trainfun.update(pred.argmax(1), gt)
        return train_loss
    def training_epoch_end(self,_):
        trainacc = self.trainfun.compute()
        self.log("train_acc", trainacc,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.trainfun.reset()

    def validation_step(self,batch, batch_idx):
        img,gt = batch
        valpre = self(img)
        val_loss = self.lossfun(valpre,gt)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.valfun.update(valpre.argmax(1),gt)
    def validation_epoch_end(self,_):
        # Make the Progress Bar leave there
        valacc = self.valfun.compute()
        self.log("val_acc", valacc,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.valfun.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
'''