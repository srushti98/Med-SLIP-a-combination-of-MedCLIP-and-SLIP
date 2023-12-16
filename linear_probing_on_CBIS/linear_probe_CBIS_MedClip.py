import os
import wandb
import torchmetrics
import numpy as np
# import pytorch_lightning as pl
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as tsfm
# from pytorch_lightning import Trainer, seed_everything
import json
import warnings
import argparse
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel
from torchmetrics import AUROC

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='Learning Probing on CBIS!')
parser.add_argument('--seed', default=77, type=int, help='seed value')
parser.add_argument('--batchsize', default=16, type=int, help='Linear probing Batch size')
parser.add_argument('--pretrain_epoch', default=25, type=int, help='Linear probing epochs')
parser.add_argument('--path', default='/scratch/sxp8182/MLHC/ML_env/MLHC/linear_logs', type=str)
parser.add_argument('--modeltype', default='vit', type=str, help='Give either \'vit\' or \'resnet\'')
parser.add_argument('--ipath',
                    default='/scratch/sxp8182/MLHC/ML_env/MLHC/50_eps_60000_resnet_backup_25_checkpt/image_encoder.pth',
                    type=str, help='Image encoder path')
parser.add_argument('--epochs', default=15, type=int, help='Epochs')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# seed_everything(args.seed)
"""
Define train & valid image transformation
"""
DATASET_IMAGE_MEAN = (0.485, 0.456, 0.406)
DATASET_IMAGE_STD = (0.229, 0.224, 0.225)

train_transform = tsfm.Compose([tsfm.Resize((384, 384)),
                                tsfm.RandomVerticalFlip(p=0.3),
                                tsfm.RandomHorizontalFlip(p=0.3),
                                tsfm.ToTensor(),
                                tsfm.Normalize(DATASET_IMAGE_MEAN, DATASET_IMAGE_STD), ])

val_transform = tsfm.Compose([tsfm.Resize((384, 384)),
                              tsfm.ToTensor(),
                              tsfm.Normalize(DATASET_IMAGE_MEAN, DATASET_IMAGE_STD), ])
"""
Define dataset class
"""


class Dataset(torch.utils.data.Dataset):
    def __init__(self, val_transform, val_path, mode):
        print("==>path:", val_path)
        print("==>mode:", mode)
        annotations = pd.read_csv(val_path)
        self.samples = [(annotations.loc[i, 'image_path'], annotations.loc[i, 'pathology']) for i in
                        range(len(annotations))]
        self.mode = mode
        self.transform = val_transform

    def __getitem__(self, i):
        image_id, target = self.samples[i]
        if self.mode == 'test':
            # image_id=image_id+'.jpg'
            path = os.path.join('/scratch/sxp8182/MLHC/ML_env/MLHC/', image_id)
        elif self.mode == 'val':
            path = os.path.join('/scratch/sxp8182/MLHC/ML_env/MLHC/', image_id)
        else:
            path = os.path.join('/scratch/sxp8182/MLHC/ML_env/MLHC/', image_id)
        # print("=>path for image", i, "is::", path)
        img = pil_loader(path)
        image = self.transform(img)
        target = diagnosis_map[target]
        # print("=>target check for image",path,"is::",target)
        return image, target

    def __len__(self):
        return len(self.samples)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# Send_input_model_to_be_finetuned
class ProjectionHead(nn.Module):
    def __init__(
            self,
            backbone,
            embedding_dim,
            projection_dim,
            classes,
            dropout=0.2
    ):
        super().__init__()
        self.backbone = backbone
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.fc = nn.Linear(projection_dim, classes)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        projected = self.projection(x)
        x = self.fc(projected)
        return x


def train_epoch(model, train_loader, optimizer, criterion, lr_scheduler, metric, accuracy_metric, precision_metric,
                recall_metric, f1_score_metric):
    loss_meter = AvgMeter()
    correct = 0
    total = 0
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    for image, label in tqdm_object:
        image, label = image.to(device), label.to(device)
        prediction = model(image)
        loss = criterion(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        loss_meter.update(loss.item())

        metric.update(prediction, label)  # Existing
        accuracy_metric.update(prediction, label)  # Existing
        precision_metric.update(prediction, label)  # Existing
        f1_score_metric.update(prediction, label)  # Added
        recall_metric.update(prediction, label)  # Added

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    auroc = metric.compute()  # Existing
    acc = accuracy_metric.compute()  # Existing
    prec = precision_metric.compute()  # Existing
    f1_score = f1_score_metric.compute()  # Added
    recall = recall_metric.compute()  # Added

    metric.reset()  # Existing
    accuracy_metric.reset()  # Existing
    precision_metric.reset()  # Existing
    f1_score_metric.reset()  # Added
    recall_metric.reset()  # Added

    wandb.log({'train_AUROC': auroc.item(), 'train_acc': acc.item(), 'train_precision': prec.item(),
               'train_f1_score': f1_score.item(), 'train_recall': recall.item(), 'train_loss': loss_meter.avg})

    return loss_meter, {'AUROC': auroc.item(), 'train acc': acc.item(), 'train pre': prec.item(),
                        'train f1 score': f1_score.item(), 'train recall': recall.item(), 'loss': loss_meter.avg}


def valid_epoch(model, valid_loader, criterion, metric, accuracy_metric, precision_metric, recall_metric,
                f1_score_metric, mode='val'):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))

    confusion_matrix_metric = torchmetrics.ConfusionMatrix(num_classes=len(diagnosis_map), task='multiclass').to(device)

    for image, label in tqdm_object:

        image, label = image.to(device), label.to(device)
        prediction = model(image)
        loss = criterion(prediction, label)

        metric.update(prediction, label)  # Existing
        accuracy_metric.update(prediction, label)  # Existing
        precision_metric.update(prediction, label)  # Existing
        f1_score_metric.update(prediction, label)  # Added
        recall_metric.update(prediction, label)  # Added

        if mode == 'test':
            confusion_matrix_metric.update(prediction.argmax(dim=1), label)

        loss_meter.update(loss.item())
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer), Recall=val_metric)

    val_metric_result = metric.compute()  # Existing
    acc = accuracy_metric.compute()  # Existing
    prec = precision_metric.compute()  # Existing
    f1_score = f1_score_metric.compute()  # Added
    recall = recall_metric.compute()  # Added

    # Reset all metrics
    metric.reset()  # Existing
    accuracy_metric.reset()  # Existing
    precision_metric.reset()  # Existing
    f1_score_metric.reset()  # Added
    recall_metric.reset()  # Added

    ##if mode is test then add it file
    if mode == 'test':
        path_for_saving_logs = os.path.join(args.path, 'linear_probe_logs.txt')
        confusion_matrix = confusion_matrix_metric.compute()
        with open(path_for_saving_logs, 'a') as f:
            f.write('Confusion Matrix:\n')
            f.write(str(confusion_matrix.cpu().numpy()) + '\n')
        confusion_matrix_metric.reset()

    wandb.log({'val_auroc': val_metric_result.item(), 'val_acc': acc.item(), 'val_precision': prec.item(),
               'val_loss': loss_meter.avg, 'val_f1_score': f1_score.item(), 'val_recall': recall.item()})
    print(loss_meter, 'Validation AUROC', val_metric_result.item(), 'val acc', acc.item(), 'val pre', prec.item(),
          'val f1 score', f1_score.item(), 'val recall', recall.item(), 'loss', loss_meter.avg)

    return loss_meter, {'Validation AUROC': val_metric_result.item(), 'val acc': acc.item(), 'val pre': prec.item(),
                        'val f1 score': f1_score.item(), 'val recall': recall.item(), 'loss': loss_meter.avg}


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


if __name__ == '__main__':

    print("=>The file is from from this directory::", os.getcwd())
    print("=> current running file:ft_vision_CBIS.py")
    print("=> pretrained model path::", args.path)
    print("=> batch size::", args.batchsize)
    print("=> seed::", args.seed)

    train_dataset = Dataset(train_transform,
                            os.path.join('/scratch/sxp8182/MLHC/ML_env/MLHC/', 'CBIS-DDSM/csv/train_calcification.csv'),
                            'train')
    val_dataset = Dataset(val_transform,
                          os.path.join('/scratch/sxp8182/MLHC/ML_env/MLHC/', 'CBIS-DDSM/csv/test_calcification.csv'),
                          'val')
    test_dataset = Dataset(val_transform,
                           os.path.join('/scratch/sxp8182/MLHC/ML_env/MLHC/', 'CBIS-DDSM/csv/test_calcification.csv'),
                           'test')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchsize,
        num_workers=4, pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batchsize, shuffle=False,
        num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batchsize, shuffle=False,
        num_workers=4, pin_memory=True)
    diagnosis_map = {"MALIGNANT": 0, "BENIGN": 1, "BENIGN_WITHOUT_CALLBACK": 2}

    wandb_id = os.path.split(args.path)[-1]
    wandb.init(project='medclip_linear_prob', id=wandb_id, config=args, resume='allow')

    ####model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    ####model = MedCLIPModel(vision_cls=MedCLIPVisionModel)

    if args.modeltype == 'vit':
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)

        print("=> We have loaded medclip vit vision model")
    else:
        model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
        print("=> We have loaded mdeclip resnet vision model")

    model.from_pretrained()
    model = model.vision_model


    # for name, param in model.named_parameters():
    #    print(name, param)

    # freezing parameters
    for param in model.parameters():
        param.requires_grad = False

    print("=> checking whether all paramters are set to zero or not")
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    # optim_params = [{"params": p_wd, "weight_decay": args.wd},
    #                 {"params": p_non_wd, "weight_decay": 0}]
    print("p_wd:", len(p_wd))
    print("p_non_wd:", len(p_non_wd))

    model = ProjectionHead(model, 512, 1024, len(diagnosis_map))
    lr = 1e-3 * 16 / 256
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                        T_0=5, eta_min=0, last_epoch=-1)
    train_metric = AUROC(task="multiclass", num_classes=len(diagnosis_map), average="macro", thresholds=None)
    val_metric = AUROC(task="multiclass", num_classes=len(diagnosis_map), average="macro", thresholds=None)
    test_metric = AUROC(task="multiclass", num_classes=len(diagnosis_map), average="macro", thresholds=None)

    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=len(diagnosis_map), average='macro').to(
        device)

    precision_metric = torchmetrics.Precision(task="multiclass", num_classes=len(diagnosis_map), average='macro').to(
        device)

    recall_metric = torchmetrics.Recall(task="multiclass", num_classes=len(diagnosis_map), average='macro').to(device)

    f1_score_metric = torchmetrics.F1Score(task="multiclass", num_classes=len(diagnosis_map), average='macro').to(
        device)

    print("len(diagnosis_map):", len(diagnosis_map))
    best_loss = float('inf')
    model.to(device)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        # print("=> checking whether all paramters are set to zero or not")
        # p_wd, p_non_wd = [], []
        # count = 0
        # for n, p in model.named_parameters():
        #   if not p.requires_grad:
        #      if count < 2:
        #         print("These weights are frozen, printint them")
        #        print(n,":",p)
        #       count =count +1
        #  continue  # frozen weights
        # if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
        #   p_non_wd.append(p)
        # else:
        #   p_wd.append(p)
        # print("len of p_wd:",len(p_wd))
        # print("len of p_non_wd:", len(p_non_wd))
        train_loss, train_report = train_epoch(model, train_loader, optimizer, criterion, lr_scheduler, train_metric,
                                               accuracy_metric, precision_metric, recall_metric, f1_score_metric)
        model.eval()
        with torch.no_grad():
            valid_loss, validation_report = valid_epoch(model, valid_loader, criterion, val_metric, accuracy_metric,
                                                        precision_metric, recall_metric, f1_score_metric)
            print("args.path:", args.path)
            if not os.path.exists(args.path):
                os.makedirs(args.path)
            path_for_saving_logs = os.path.join(args.path, 'linear_probe_logs.txt')
            print("path_for_saving_logs:", path_for_saving_logs)
            with open(path_for_saving_logs, 'a') as f:
                print("writing data at:", path_for_saving_logs)
                f.write(json.dumps(train_report) + '\n')
                f.write(json.dumps(validation_report) + '\n')
            print("=>done with writing::")
            if valid_loss.avg < best_loss:
                best_loss = valid_loss.avg
                # save_path = args.path[:-3] + 'unimodal_linear' + '--ft' + '.pt'
                print("The best loss so far:", best_loss)
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=len(diagnosis_map), average='macro').to(
        device)

    precision_metric = torchmetrics.Precision(task="multiclass", num_classes=len(diagnosis_map), average='macro').to(
        device)

    recall_metric = torchmetrics.Recall(task="multiclass", num_classes=len(diagnosis_map), average='macro').to(device)

    f1_score_metric = torchmetrics.F1Score(task="multiclass", num_classes=len(diagnosis_map), average='macro').to(
        device)
    torch.save(model.state_dict(), 'MLHC_CML.pth')
    model.load_state_dict(torch.load('MLHC_CML.pth'))
    model.eval()
    with torch.no_grad():
        test_loss, test_report = valid_epoch(model, test_loader, criterion, test_metric, accuracy_metric,
                                             precision_metric, recall_metric, f1_score_metric, mode='test')
        print("The final test report is:", test_report)
        with open(path_for_saving_logs, 'a') as f:
            f.write(json.dumps(test_report) + '\n')
