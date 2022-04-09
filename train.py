import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary
from datetime import datetime

from datasets import HousingDataset
from model import Model
from utils import Select, CustomScale

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Running on device: {device}")

mean=np.array([1377, 1354, 1381, 2356])
std=np.array([540, 398, 327, 515])

upperbound = mean + 3 * std
SCALE = upperbound[:, np.newaxis, np.newaxis]

norm_mean = mean / upperbound
norm_std = std / upperbound
    
transformations = [
    transforms.CenterCrop(size=(32, 32)), 
    CustomScale(scale=1/SCALE, clamp=(0, 1.0)),
    transforms.Normalize(mean=norm_mean, std=norm_std),
    Select(dim=-3, indices=[0,1,2]),
]
transform = transforms.Compose(transformations)

reverse_transform = transforms.Normalize(mean=-norm_mean[:3]/norm_std[:3], std=1/norm_std[:3]/SCALE)

train_set = HousingDataset("/atlas/u/erikrozi/housing_event_pred/data/train_seasonal_eff.csv", transform=transform)
print("Train examples:", len(train_set))
val_set = HousingDataset("/atlas/u/erikrozi/housing_event_pred/data/val_seasonal_eff.csv", transform=transform)
print("Val examples:", len(val_set))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=16)
val_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=False, num_workers=16)

if "resnet" in sys.argv:
    encoder = torchvision.models.resnet18(pretrained=True).to(device=device)
    encoder.layer4 = torch.nn.Identity()
    encoder.avgpool = torch.nn.Identity()
    encoder.fc = torch.nn.Identity()
else:
    encoder = torch.nn.Sequential(
        torch.nn.Conv2d(3,16,1,padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(16),
        torch.nn.Conv2d(16,32,1,padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.BatchNorm2d(32),
        torch.nn.Conv2d(32,64,1,padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(64),
        torch.nn.Conv2d(64,128,1,padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.BatchNorm2d(128),
    ).to(device=device)
print(summary(encoder, input_size=(4, 3, 32, 32)))

model = Model(encoder).to(device=device).train()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Beginning training...")

now = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
writer = SummaryWriter(log_dir=f'runs/exp_{now}')
num_epochs = 500
idx = 0
loss_history = [] # Train losses over batches
train_loss_history = [] # Train losses over epochs
val_loss_history = []
train_acc_history = []
val_acc_history = []
fig_dir = f'figures/exp_{now}/'
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)
for epoch in range(num_epochs):
    epoch_losses = []
    epoch_accuracies = []
    
    model.train()
    for i, batch in enumerate(train_loader):
        idx += 1
        img_start = batch["image_start"].to(device=device).float()
        img_end = batch["image_end"].to(device=device).float()
        img_sample = batch["image_sample"].to(device=device).float()
        label = batch["label"].float().to(device=device).unsqueeze(1)
        
        optimizer.zero_grad()
        pred = model(img_start, img_end, img_sample)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        
        loss = loss.item()
        writer.add_scalar('Loss/batch/train', loss, idx)
        loss_history.append(loss)
        epoch_losses.append(loss)
        epoch_accuracies.append(np.mean(1 * ((pred > 0.5) == label).detach().cpu().numpy()))
        if idx % 100 == 0:
            print(f"Epoch {epoch}, batch {idx}, loss = {loss}")

    epoch_loss = sum(epoch_losses) / len(epoch_losses)
    train_loss_history.append(epoch_loss)
    writer.add_scalar('Loss/epoch/train', epoch_loss, epoch)
    
    epoch_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)
    train_acc_history.append(epoch_accuracy)
    writer.add_scalar('Accuracy/epoch/train', epoch_accuracy, epoch)
    
    print(f"Epoch {epoch} loss = {epoch_loss} accuracy = {epoch_accuracy}")
    
    # Validation
    val_predictions = []
    val_labels = []
    model.eval()
    for i, batch in enumerate(val_loader):
        with torch.no_grad():
            img_start = batch["image_start"].to(device=device).float()
            img_end = batch["image_end"].to(device=device).float()
            img_sample = batch["image_sample"].to(device=device).float()
            label = batch["label"].float().to(device=device)

            pred = model(img_start, img_end, img_sample).flatten()
            val_predictions.append(pred.detach().cpu().numpy())
            val_labels.append(label.flatten().cpu().numpy())
    val_predictions = np.concatenate(val_predictions)
    val_labels = np.concatenate(val_labels)
    val_loss = criterion(torch.FloatTensor(val_predictions), torch.FloatTensor(val_labels))
    val_loss_history.append(val_loss)
    writer.add_scalar('Loss/epoch/val', val_loss, epoch)
    
    val_accuracy = np.sum((val_predictions > 0.5) == val_labels.astype(int)) / len(val_labels)
    val_acc_history.append(val_accuracy)
    writer.add_scalar('Accuracy/epoch/val', val_loss, epoch)
    
    print(f"Epoch {epoch} val loss = {val_loss} accuracy = {val_accuracy}")

    
    plt.figure()
    plt.plot(np.arange(1, len(loss_history) + 1), loss_history)
    plt.savefig(fig_dir + 'train_loss_batch.png')

    plt.figure()
    plt.plot(np.arange(1, len(train_loss_history) + 1), train_loss_history)
    plt.plot(np.arange(1, len(val_loss_history) + 1), val_loss_history)
    plt.savefig(fig_dir + 'loss.png')

    plt.figure()
    plt.plot(np.arange(1, len(train_acc_history) + 1), train_acc_history)
    plt.plot(np.arange(1, len(val_acc_history) + 1), val_acc_history)
    plt.savefig(fig_dir + 'accuracy.png')

    model_dir = f'checkpoints/exp_{now}/'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir + f'ckpt_{epoch}.pt')
    
print("Training completed!")
 
print("Figures saved at", fig_dir)
plt.figure()
plt.plot(np.arange(1, len(loss_history) + 1), loss_history)
plt.savefig(fig_dir + 'train_loss_batch.png')

plt.figure()
plt.plot(np.arange(1, len(train_loss_history) + 1), train_loss_history)
plt.plot(np.arange(1, len(val_loss_history) + 1), val_loss_history)
plt.savefig(fig_dir + 'loss.png')

plt.figure()
plt.plot(np.arange(1, len(train_acc_history) + 1), train_acc_history)
plt.plot(np.arange(1, len(val_acc_history) + 1), val_acc_history)
plt.savefig(fig_dir + 'accuracy.png')

def predict(model, dloader, loops=1):
    model.eval()
    predictions = []
    labels = []
    for _ in range(loops):
        for i, batch in enumerate(dloader):
            with torch.no_grad():
                img_start = batch["image_start"].to(device=device).float()
                img_end = batch["image_end"].to(device=device).float()
                img_sample = batch["image_sample"].to(device=device).float()
                label = batch["label"].float().to(device=device)

                pred = model(img_start, img_end, img_sample).flatten()
                predictions.append(pred.detach().cpu().numpy())
                labels.append(label.flatten().cpu().numpy())
    return np.concatenate(predictions), np.concatenate(labels).astype(int)

predictions, labels = predict(model, val_loader, loops=10)
predictions = (predictions > 0.5)
accurate = np.sum(labels == predictions)
total = len(labels)
print(f"Final val accuracy {accurate} / {total} = {accurate/total}")

from sklearn.metrics import classification_report
print(classification_report(labels, predictions))

