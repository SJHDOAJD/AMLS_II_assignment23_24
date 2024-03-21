from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn import model_selection, metrics
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from datetime import datetime
import seaborn as sns
import torch.nn as nn
from PIL import Image
import pandas as pd
import numpy as np
import random
import torch
import timm
import cv2
import gc
import os

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

def preprocess_ViT():
    # load data and set the seed
    create_new_csv()
    DATA_PATH = load_path()
    seed_everything(1000)
    # get the data through data path
    TRAIN_PATH = os.path.join(DATA_PATH, 'Datasets')
    df = pd.read_csv(os.path.join(DATA_PATH, "train_1.csv"))
    # set image size
    IMG_SIZE = 224
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

            if self.transforms is not None:
                image = self.transforms(img)

            return image, label

    # create image augmentations
    transforms_train = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    transforms_valid = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # split the dataset to create train, valid, and test sets
    train_df, tem_df = model_selection.train_test_split(
        df, test_size=0.3, random_state=42, stratify=df.label.values
    )

    valid_df, test_df = model_selection.train_test_split(
        tem_df, test_size=0.5, random_state=42, stratify=tem_df.label.values
    )

    # transforms the sets
    train_dataset = CassavaDataset(train_df, transforms=transforms_train)
    valid_dataset = CassavaDataset(valid_df, transforms=transforms_valid)
    test_dataset = CassavaDataset(test_df, transforms=transforms_valid)

    return train_dataset, valid_dataset, test_dataset


def model_ViT():
    # create a class to define the model
    class ViTBase16(nn.Module):
        def __init__(self, n_classes):
            # load a pre-trained ViT model and create a "head" layer
            super(ViTBase16, self).__init__()
            self.model = timm.create_model("vit_base_patch16_224", pretrained=True)
            self.model.head = nn.Linear(self.model.head.in_features, n_classes)

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
                # start CPU
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
                # start CPU
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
                # start CPU
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
                    preds = output.argmax(dim=1)
                    # get prediction and true values
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
            # calculate average losses
            test_loss /= len(test_loader)
            test_accuracy /= len(test_loader)
            return test_loss, test_accuracy, np.array(all_preds), np.array(all_targets)
    # make ViT model    
    model = ViTBase16(n_classes=5)

    return model


def train_ViT(model, train_dataset, valid_dataset):
    # load path
    init_path = load_path()
    # give init values and settings
    BATCH_SIZE = 16
    LR = 1e-05
    N_EPOCHS = 10
    acc_train = 0
    acc_valid = 0
    ls_train = np.Inf
    ls_valid = np.Inf

    # define a fit function
    def fit(
        model, epochs, device, criterion, optimizer, train_loader, valid_loader=None
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
                    print(
                        "Validation loss decreased ({:.4f} --> {:.4f}).".format(
                            valid_loss_min, valid_loss
                        )
                    )
                valid_loss_min = valid_loss

        return {
            "train_loss": train_losses,
            "valid_losses": valid_losses,
            "train_acc": train_accs,
            "valid_acc": valid_accs,
        }
    
    # get loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )
    # set parameters
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    model.to(device)
    lr = LR
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # record start time
    print(f"INITIALIZING TRAINING ")
    start_time = datetime.now()
    print(f"Start Time: {start_time}")
    # train model
    logs = fit(
        model=model,
        epochs=N_EPOCHS,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
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
    plot = os.path.join(init_path, 'plot', 'ViT accuracy and loss diagram')
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



def evaluate_ViT(model, test_dataset):
    # get data path
    init_path = load_path()
    # set parameter
    BATCH_SIZE = 16
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    # get loader
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
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
    plot = os.path.join(init_path, 'plot', 'ViT confusion matrix diagram')
    plt.savefig(plot)
    plt.close()

    accuracy = accuracy_score(all_targets, all_preds)
    num_classes = 5

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
    plot1 = os.path.join(init_path, 'plot', 'ViT Performance Metrics diagram')
    plt.savefig(plot1)
    plt.close()

    return test_acc, test_loss



