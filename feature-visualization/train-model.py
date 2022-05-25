import os

import pandas as pd
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model.mynet import mynet_light
import pytorch_lightning as pl
import torch

torch.backends.cudnn.deterministic = True
# torch.set_default_tensor_type(torch.FloatTensor)
seed_everything(6, workers=True)

if __name__ == '__main__':
    img_path = r"../data/4Vi/data2"
    data_transform = {
        "train": transforms.Compose([transforms.Resize([101, 101]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize([101, 101]),
                                   # transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    train_dataset = datasets.ImageFolder(root=os.path.join(img_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    batch_size = 128
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=4)

    validate_dataset = datasets.ImageFolder(root=os.path.join(img_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=4)


    model = mynet_light()
    # model.load_state_dict(torch.load('/root/pyfile/noname/tb_logs/bisenetv2_camvid_log/version_159/checkpoints/epoch=491-step=91019.ckpt')['state_dict'])
    # model.net.load_state_dict(torch.load('weights/model_final_v2_city.pth'),strict=False)
    checkpoint_callback = ModelCheckpoint(monitor="val_acc",
                                          mode='max',
                                          every_n_val_epochs=True,
                                          save_top_k=3,
                                          #period=2
                                          )
    early_stop_callback = EarlyStopping(monitor="val_acc",
                                        min_delta=0.00, patience=100,
                                        verbose=True, mode="max")

    logger = TensorBoardLogger("tb_logs", name="10_log")

    trainer = pl.Trainer(max_epochs=200,
                         gpus=[0],
                         precision=32,
                         log_every_n_steps=1,
                         auto_lr_find=True,
                         #resume_from_checkpoint="/root/pyfile/noname/tb_logs/bisenetv2_kuang_log/version_3/checkpoints/epoch=95-step=6335.ckpt",
                         # accelerator='ddp',
                         # profiler="simple",
                         weights_summary="full",
                         default_root_dir="./checkpoints",
                         logger=logger,
                         callbacks=[checkpoint_callback,early_stop_callback]
                         )

    trainer.fit(model, train_dataloaders=train_loader,val_dataloaders=validate_loader)


