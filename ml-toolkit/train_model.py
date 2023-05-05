import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
from tqdm import tqdm
import yaml


from utils.data import prepare_dataloaders
from models import MinimalCNN, sleep_resnet18, LSTMModel




def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device).long()

        optimizer.zero_grad()

        outputs = model(inputs)
        # print(f'types: {outputs.dtype} | {labels.dtype}')
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix(Running_Accuracy=f'{(correct / total):.4f}', refresh=True)

    return running_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Validation")
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device).long()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix(Running_Accuracy=f'{(correct / total):.4f}', refresh=True)

    return running_loss / len(dataloader), correct / total

def main():
    with open("conf/config_train.yaml", "r") as file:
        config = yaml.safe_load(file)

    train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = MinimalCNN(num_classes=8).to(device)
    # model = sleep_resnet18().to(device)
    model = LSTMModel(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])

    epochs = config["training"]["epochs"]

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        train_loss, train_accuracy = train(model, train_dataloader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_dataloader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), f"{config['training']['save_dir']}/model.pth")

if __name__ == "__main__":
    main()
