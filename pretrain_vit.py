import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import tome

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------- 数据增强 ----------
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ---------- CIFAR10 数据集 ----------
    train_set = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )
    test_set = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    # 在 train_set, test_set 定义之后，加这两行，把数据集缩小
    from torch.utils.data import Subset

    train_indices = list(range(1000))  # 只用前 1000 张
    test_indices = list(range(500))  # 只用前 500 张
    train_set_small = Subset(train_set, train_indices)
    test_set_small = Subset(test_set, test_indices)

    train_loader = DataLoader(
        train_set_small,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_set_small,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print("DEBUG Train size:", len(train_set_small), "Test size:", len(test_set_small))

    # ---------- 创建 ViT 模型 ----------
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=True,
        num_classes=10      # CIFAR10 10 类
    )
    model = model.to(device)
    print("Model loaded (vit_tiny_patch16_224 pretrained on ImageNet).")

    # ---------- 训练设置 ----------
    epochs = 1  # 先只跑 1 轮，确认不卡
    lr = 5e-4
    weight_decay = 0.05

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            if batch_idx % 50 == 0:
                print(f"[DEBUG] Epoch {epoch}, batch {batch_idx}/{len(train_loader)}, loss={loss.item():.4f}",
                      flush=True)

        train_loss = running_loss / len(train_loader.dataset)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch [{epoch}/{epochs}] Train Loss: {train_loss:.4f}  Test Acc: {test_acc:.4f}", flush=True)

    baseline_acc = evaluate(model, test_loader, device)
    print("Baseline accuracy (no merge):", baseline_acc)

    # ---------- ToMe ----------
    # 先给 timm 的 ViT 打补丁
    tome.patch.timm(model)

    r_list = [0, 4, 8, 12, 16, 24, 32]

    for r in r_list:
        model.r = r      # 设置 merge 强度
        acc = evaluate(model, test_loader, device)
        print(f"r={r},  Test Acc={acc:.4f}")


if __name__ == "__main__":
    main()
