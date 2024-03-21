from albumentations import (
    Compose, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, 
    ShiftScaleRotate, Transpose
    )
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from albumentations.pytorch import ToTensorV2
from sklearn import model_selection, metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datetime import datetime
from torch.optim import Adam
from PIL import Image
import seaborn as sns
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import torch
import timm
import cv2
import os
import gc


def load_path():
    # Use absolute path to define the file path
    current_script_path = os.path.abspath(__file__)
    amls_dir_path = os.path.dirname(os.path.dirname(current_script_path))
    return amls_dir_path

def seed_everything(seed):
    # set the random seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_new_csv():
    # get the init path
    amls_dir_path=load_path()
    # load original csv file
    # '..\AMLSII_23-24_SN23037969\train.csv'
    original_csv_path = os.path.join(amls_dir_path, 'train.csv')
    original_df = pd.read_csv(original_csv_path)
    # dataset folder path
    # '..\AMLSII_23-24_SN23037969\Datasets'
    data_folder = os.path.join(amls_dir_path, 'Datasets')

    new_data = []
    # Get the 'image_id' and 'label' from train.csv
    for filename in os.listdir(data_folder):
        if filename in original_df['image_id'].values:
            label = original_df[original_df['image_id'] == filename]['label'].iloc[0]
            new_data.append({'image_id': filename, 'label': label})
    # make a new dataframe
    new_df = pd.DataFrame(new_data)
    # create a csv to match the datasets
    # '..\AMLSII_23-24_SN23037969\train_1.csv'
    new_csv_path = os.path.join(amls_dir_path, 'train_1.csv')
    new_df.to_csv(new_csv_path, index=False)
    return

# set a class to store all features
class CFG:
    debug=False
    apex=False
    print_freq=50  # Frequency of printing logs
    num_workers=0 
    model_name='resnext50_32x4d'  # model name
    size=256  # image size
    scheduler='CosineAnnealingWarmRestarts' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    epochs=10  # Number of training cycles
    lr=1e-5  # learning rate
    batch_size=16  # Size of each batch of data
    weight_decay=1e-6  # Weight decay for regularization
    seed=1111  # random seed
    target_size=5  # classes number
    factor=0.2 # ReduceLROnPlateau
    patience=4 # ReduceLROnPlateau
    eps=1e-6 # ReduceLROnPlateau
    T_max=10 # CosineAnnealingLR
    T_0=10 # CosineAnnealingWarmRestarts
    min_lr=1e-6 # min learning rate


def preprocess_ResNext1():
    # load data and set seed
    DATA_PATH = load_path()
    seed_everything(CFG.seed)
    create_new_csv()
    # read data
    df = pd.read_csv(os.path.join(DATA_PATH, "train_1.csv"))
    TRAIN_PATH = os.path.join(DATA_PATH, "Datasets")
    # create a class to change the type of iamge
    class CassavaDataset(torch.utils.data.Dataset):
        def __init__(self, df, data_path=DATA_PATH, mode="train", transforms=None):
            super().__init__()
            self.df_data = df.values
            self.data_path = data_path
            self.transforms = transforms
            self.mode = mode
            self.data_dir = "Datasets"

        def __len__(self):
            return len(self.df_data)

        def __getitem__(self, index):
            img_name, label = self.df_data[index]
            img_path = os.path.join(self.data_path, self.data_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)
            if self.transforms is not None:
                image = self.transforms(image=img)['image']
            label = torch.tensor(label).long()
            return image, label
    # split the dataset to train, valid, test
    train_df, tem_df = model_selection.train_test_split(
        df, test_size=0.3, random_state=CFG.seed, stratify=df.label.values
    )
    valid_df, test_df = model_selection.train_test_split(
        tem_df, test_size=0.5, random_state=CFG.seed, stratify=tem_df.label.values
    )
    # create image augmentations
    transforms_train = Compose(
        [
            # Resize(CFG.size, CFG.size),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomResizedCrop(CFG.size, CFG.size),
            ShiftScaleRotate(p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            ToTensorV2(),
        ]
    )
    transforms_valid = Compose(
        [
            Resize(CFG.size, CFG.size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            ToTensorV2(),
        ]
    )
    # use transforms to get train and valid data
    train_dataset = CassavaDataset(train_df, transforms=transforms_train)
    valid_dataset = CassavaDataset(valid_df, transforms=transforms_valid)
    # Use validation transforms for testing
    test_dataset = CassavaDataset(test_df, transforms=transforms_valid)

    return train_dataset,valid_dataset,test_dataset


def model_ResNext1():
    # create a class to define the model
    class CustomResNext(nn.Module):
        def __init__(self, n_classes):
            # load a pre-trained ResNeXt model and create a "fc" layer
            super(CustomResNext, self).__init__()
            self.model = timm.create_model("resnext50_32x4d", pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)

        def forward(self, x):
            # make the forward method for model
            x = self.model(x)
            return x
    
        def train_one_epoch(self, train_loader, criterion, optimizer, device):
            # set init values
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            ###################
            # train the model #
            ###################
            self.model.train()
            for i, (data, target) in enumerate(train_loader):
                # move to CPU
                data, target = data.to(device), target.to(device)
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.forward(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # Calculate Accuracy
                accuracy = (output.argmax(dim=1) == target).float().mean()
                # update training loss and accuracy
                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()
                # set optimizer
                optimizer.step()
            return epoch_loss / len(train_loader), epoch_accuracy / len(train_loader)
    
        def validate_one_epoch(self, valid_loader, criterion, device):
            # set init values
            valid_loss = 0.0
            valid_accuracy = 0.0
            ######################
            # validate the model #
            ######################
            self.model.eval()
            for data, target in valid_loader:
                # move to CPU
                data, target = data.to(device), target.to(device)
                with torch.no_grad():
                    # forward pass: compute predicted outputs by passing inputs to the model
                    output = self.model(data)
                    # calculate the batch loss
                    loss = criterion(output, target)
                    # Calculate Accuracy
                    accuracy = (output.argmax(dim=1) == target).float().mean()
                    # update average validation loss and accuracy
                    valid_loss += loss.item()
                    valid_accuracy += accuracy.item()
            return valid_loss / len(valid_loader), valid_accuracy / len(valid_loader)
    
        def evaluate(self, test_loader, criterion, device):
            # set init values
            test_loss = 0.0
            test_accuracy = 0.0
            all_preds = []
            all_targets = []
            ######################
            # evaluate the model #
            ######################
            self.model.eval()
            for data, target in test_loader:
                # move to CPU
                data, target = data.to(device), target.to(device)
                with torch.no_grad():
                    # turn off gradients for evaluation
                    output = self.model(data)
                    # calculate the batch loss
                    loss = criterion(output, target)
                    # calculate accuracy
                    accuracy = (output.argmax(dim=1) == target).float().mean()
                    # update average test loss and accuracy
                    test_loss += loss.item()
                    test_accuracy += accuracy.item()
                    # get prediction and true values
                    preds = output.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
            # calculate average losses
            test_loss /= len(test_loader)
            test_accuracy /= len(test_loader)
            return test_loss, test_accuracy, np.array(all_preds), np.array(all_targets)
    # make ResNeXt model 
    model = CustomResNext(n_classes=CFG.target_size)

    return model


def train_ResNext1(model,train_dataset,valid_dataset):
    # load and set the init data
    init_path = load_path()
    acc_train = 0
    acc_valid = 0
    ls_train = np.Inf
    ls_valid = np.Inf
    # define a fit function
    def fit(
        model, epochs, device, criterion, optimizer, scheduler, train_loader, valid_loader=None
    ):
        # set init values
        valid_loss_min = np.Inf
        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []
        # set the logic about epoch
        for epoch in range(1, epochs + 1):
            # clear useless data
            gc.collect()

            print(f"{'='*50}")
            print(f"EPOCH {epoch} - TRAINING...")
            train_loss, train_acc = model.train_one_epoch(
                train_loader, criterion, optimizer, device
            )
            print(
                f"\n\t[TRAIN] EPOCH {epoch} - LOSS: {train_loss}, ACCURACY: {train_acc}\n"
            )
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            gc.collect()

            if valid_loader is not None:
                gc.collect()
            
                print(f"EPOCH {epoch} - VALIDATING...")
                valid_loss, valid_acc = model.validate_one_epoch(
                    valid_loader, criterion, device
                )   
                print(f"\t[VALID] LOSS: {valid_loss}, ACCURACY: {valid_acc}\n")
                valid_losses.append(valid_loss)
                valid_accs.append(valid_acc)
                gc.collect()
                # show if validation loss has decreased
                if valid_loss <= valid_loss_min and epoch != 1:
                    print("Validation loss decreased ({:.6f} --> {:.6f}).".format(valid_loss_min, valid_loss))
        
                valid_loss_min = valid_loss
            # set scheduler
            scheduler.step()

        return {
            "train_loss": train_losses,
            "valid_losses": valid_losses,
            "train_acc": train_accs,
            "valid_acc": valid_accs,
        }
    # get train and valid loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )
    # define the different scheduler
    def get_scheduler(optimizer):
        if CFG.scheduler=='ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.eps)
        elif CFG.scheduler=='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
        elif CFG.scheduler=='CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
        return scheduler
    # set parameters
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    model.to(device)
    lr = CFG.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=CFG.weight_decay)
    scheduler = get_scheduler(optimizer)
    # record time
    print(f"INITIALIZING TRAINING ")
    start_time = datetime.now()
    print(f"Start Time: {start_time}")
    # train model
    logs = fit(
        model=model,
        epochs=CFG.epochs,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )

    print(f"Execution time: {datetime.now() - start_time}")

    # plot the accuracy and loss values of training and validation with each epoch
    fig, ax = plt.subplots(2,1)
    ax[0].plot(logs['train_loss'], color='b', label="Training Loss")
    ax[0].plot(logs['valid_losses'], color='r', label="Validation Loss")
    legend = ax[0].legend(loc='best', shadow=True)
    ax[1].plot(logs['train_acc'], color='b', label="Training Accuracy")
    ax[1].plot(logs['valid_acc'], color='r',label="Validation Accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    # save diagram
    plot = os.path.join(init_path, 'plot', 'ResNext_1 accuracy and loss diagram')
    plt.savefig(plot)
    plt.close()
    # get max accuracy and min loss
    for items in logs['train_acc']:
        if items > acc_train:
            acc_train = items

    for items in logs['valid_acc']:
        if items > acc_valid:
            acc_valid = items

    for items in logs['train_loss']:
        if items < ls_train:
            ls_train = items
    
    for items in logs['valid_losses']:
        if items < ls_valid:
            ls_valid = items

    return acc_train, acc_valid, ls_train, ls_valid



def evaluate_ResNext1(model, test_dataset):
    # load data and set parameter
    init_path = load_path()
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    # make test loader
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    # evaluate model
    test_loss, test_acc, all_preds, all_targets = model.evaluate(test_loader, criterion, device)
    # make confusion matrix
    conf_mat = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, fmt='d', ax=ax, cmap="Blues")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    # save diagram
    plot = os.path.join(init_path, 'plot', 'ResNext_1 confusion matrix diagram')
    plt.savefig(plot)
    plt.close()

    accuracy = accuracy_score(all_targets, all_preds)
    num_classes = CFG.target_size

    # make performance metrics for each classes
    # define the values in normal, micro and macro state
    precision = precision_score(all_targets, all_preds, average=None)
    precision_micro = precision_score(all_targets, all_preds, average='micro')
    precision_macro = precision_score(all_targets, all_preds, average='macro')

    recall = recall_score(all_targets, all_preds, average=None)
    recall_micro = recall_score(all_targets, all_preds, average='micro')
    recall_macro = recall_score(all_targets, all_preds, average='macro')

    f1 = f1_score(all_targets, all_preds, average=None)
    f1_micro = f1_score(all_targets, all_preds, average='micro')
    f1_macro= f1_score(all_targets, all_preds, average='macro')

    precision = np.append(precision, [precision_micro, precision_macro])
    recall = np.append(recall, [recall_micro, recall_macro])
    f1 = np.append(f1, [f1_micro, f1_macro])
    classes = list(range(num_classes)) + ['Micro', 'Macro']

    x = np.arange(len(classes))
    width = 0.2

    fig1, ax1 = plt.subplots(figsize=(14, 6))
    rects1 = ax1.bar(x - width, precision, width, label='Precision')
    rects2 = ax1.bar(x, recall, width, label='Recall')
    rects3 = ax1.bar(x + width, f1, width, label='F1')
    # make performance diagram
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics for 5 Classes (Cassava Leaf Disease Classification)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='upper right')
    ax1.grid(True)
    # save diagram
    plot1 = os.path.join(init_path, 'plot', 'ResNext_1 Performance Metrics diagram')
    plt.savefig(plot1)
    plt.close()

    return test_acc, test_loss
