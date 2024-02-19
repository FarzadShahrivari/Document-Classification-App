# --------------------------------------------------------------------------------------
# Importing necessary libraries inlcluding Numpy, Pandas, Pytorch, Matplotlib
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchmetrics import Accuracy
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Setting seeds for reproducibility
torch.manual_seed(737)
torch.set_float32_matmul_precision('medium')

# Making a custom dataset class
class MyDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.class_folders = sorted(os.listdir(dataset_path))

        self.data = []
        self.labels = []

        self.load_data()

    # Load the dataset from the device location
    def load_data(self):
        for class_idx, class_folder in enumerate(self.class_folders):
            class_path = os.path.join(self.dataset_path, class_folder)
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)

                img = Image.open(image_path)
                img = ImageOps.exif_transpose(img)
                if self.transform:
                    img = self.transform(img)
                self.data.append(img)
                self.labels.append(class_idx)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.data)

# Applying data preprocessing including image resize and normalization
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.485, 0.485), (0.229, 0.229, 0.229)),
])

# Load data using Pytorch DataLoader
dataset = MyDataset(dataset_path="./Dataset", transform=data_transform)

total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

BATCH_SIZE = 25
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)  
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)  

# Visualising a mini-batch that the training dataloader gives us
images, labels = next(iter(train_loader))

fig, ax = plt.subplots(1, 5, figsize=(20, 5))
for i in range(5):
    image = images[i].permute(1, 2, 0)
    ax[i].imshow(image)
    caption = f'Ground Truth: {labels[i]}'
    ax[i].set_title(caption)

# Designing a deep CNN using three convolutional blocks as well as three linear layers
class myCNN(LightningModule):
    def __init__(self, learning_rate=1e-3, optimizer='SGD'):
        super().__init__()
        # The following can be add to fine-tune an overkill! model (RESNET50) in order to get a significantly high accuracy (99%)
        """
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        """
        # Building my own neural network
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(0.05),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4)
            )
        self.lin = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(7*7*256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 10)
        )        

        # Adjusting necessary hyper-parameters, loss function, and metric systems
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy(task='multiclass', num_classes=10)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=10)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=10)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.lin(x)
        return x
        # This part can be added for the overkill model
        """
        return self.model(x)
        """
    
    # Training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        pred = y_hat.argmax(1)
        self.train_accuracy.update(pred, y)

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', self.train_accuracy, prog_bar=False, on_step=False, on_epoch=True)

        return loss

    # Validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        pred = y_hat.argmax(1)
        self.val_accuracy.update(pred, y)

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_accuracy, prog_bar=False, on_step=False, on_epoch=True)

    # Test step
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        pred = y_hat.argmax(1)
        self.test_accuracy.update(pred, y)

        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_accuracy, prog_bar=False, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        pred = y_hat.argmax(1)
        return pred, y, x

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)

        return optimizer

    def train_dataloader(self):
       return train_loader

    def val_dataloader(self):
        return val_loader

    def test_dataloader(self):
        return test_loader

# Training mymodel using SGD optimiser with lr=0.007 and a momentum of 0.9
myModel = myCNN(learning_rate=7e-3, optimizer='SGD')

# Saving the best possible model among all epochs
model_callback = ModelCheckpoint(
        monitor="val_acc",              
        dirpath="PassportClassification/checkpoints/",   
        save_top_k=1,                   
        mode="max",                     
        every_n_epochs=1                
    )

Model = Trainer(
    accelerator="cuda",
    max_epochs=30,
    callbacks=[TQDMProgressBar(refresh_rate=20), model_callback, EarlyStopping('val_loss', patience  = 10, mode = 'min')],
    logger=CSVLogger(save_dir="PassportClassification/logs/"),                          
)

# Observing model accuracy and model loss function on a test data 
Model.fit(myModel)
Model.test()

# Plotting the results
metrics = pd.read_csv(Model.logger.log_dir + "/metrics.csv")
metrics.set_index("epoch", inplace=True)
metrics = metrics.groupby(level=0).sum().drop("step", axis=1)

plt.figure()
plt.plot(metrics["train_loss"][:-1], label="Train Loss")
plt.plot(metrics["val_loss"][:-1], label="Validation Loss")
plt.title("Training Loss of myCNN")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.figure()
plt.plot(metrics["train_acc"][:-1], label="Train Accuracy")
plt.plot(metrics["val_acc"][:-1], label="Validation Accuracy")
plt.title("Validation Accuracy of myCNN")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

result = Model.predict(myModel, test_loader)
mypred, mylabel, myinput = result[0]

myinput = myinput.cpu().numpy()

# Visualizing predictions
fig, ax = plt.subplots(2, 5, figsize=(20, 10))
for i in range(2):
    for j in range(5):
        image = myinput[i*5+j].transpose(1, 2, 0)
        caption = f'Ground Truth: {mylabel[i*5+j]} \n Prediction: {mypred[i*5+j]}'
        ax[i, j].imshow((image*0.229)+0.485)
        ax[i, j].set_title(caption)

# Plotting the evaluation metircs including the confusion matrix + precision, recall, and f1 score for each specefic class
myModel.eval()

all_predictions, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = myModel(inputs)
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

confusion_matrix_sklearn = confusion_matrix(all_labels, all_predictions)
precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None)
overall_precision, overall_recall, overall_f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')

plt.figure(figsize=(10, 8))
ax = sns.heatmap(confusion_matrix_sklearn, annot=True, fmt='g', cmap='Blues')
ax.set_xlabel('Predicted Classes')
ax.set_ylabel('True Classes')

for i in range(len(precision)):
    ax.text(len(precision)+3, i+0.5, f'Precision: {precision[i]:.2f}\nRecall: {recall[i]:.2f}\nF1-score: {f1_score[i]:.2f}', ha='center', va='center', fontsize=10, color='red')

ax.text(-0.5, len(precision)+0.8, f'Overall Precision: {overall_precision:.2f}\nOverall Recall: {overall_recall:.2f}\nOverall F1-score: {overall_f1_score:.2f}', ha='left', va='center', fontsize=10, color='red')

plt.title('Confusion Matrix with Precision, Recall, and F1-score')
plt.show()
