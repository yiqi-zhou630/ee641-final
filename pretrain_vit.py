import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import tome
import time
from matplotlib import pyplot as plt


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


def train_one_rp(r, p, train_loader, test_loader, device):
    """
    为某个 r 单独训练一个模型，并返回训练时间和测试精度
    """
    print(f"\n==== train start r={r} ====")

    # 1. 初始化模型
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=True,
        num_classes=10
    ).to(device)

    # 2. 打 ToMe 补丁（只做一次）
    tome.patch.timm(model)
    model.r = r
    model.p = p

    # 3. 设置训练超参
    epochs = 1                 # 你可以改成 5
    lr = 5e-4
    wd = 0.05

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # 4. 计时开始
    start_time = time.time()

    # 5. 训练
    for epoch in range(epochs):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    # 6. 计时结束
    end_time = time.time()
    train_time = end_time - start_time

    # 7. 测试
    acc = evaluate(model, test_loader, device)

    print(f"r={r} p={p} use time: {train_time:.2f}s  accuracy: {acc:.4f}")

    return train_time, acc

def visualize_results(r_list, p_list, results):
    time_matrix = []
    acc_matrix = []
    for r in r_list:
        row_t = []
        row_a = []
        for p in p_list:
            t, acc = results[(r, p)]
            row_t.append(t)
            row_a.append(acc)
        time_matrix.append(row_t)
        acc_matrix.append(row_a)

    # ---------- 1. 训练时间热力图 ----------
    plt.figure(figsize=(6, 4))
    plt.imshow(time_matrix, aspect='auto')
    plt.colorbar(label="Train Time (s)")
    plt.xticks(range(len(p_list)), [str(p) for p in p_list])
    plt.yticks(range(len(r_list)), [str(r) for r in r_list])
    plt.xlabel("p")
    plt.ylabel("r")
    plt.title("train time")
    plt.savefig("train_time_heatmap.png")
    plt.show()

    # ---------- 2. 准确率热力图 ----------
    plt.figure(figsize=(6, 4))
    plt.imshow(acc_matrix, aspect='auto')
    plt.colorbar(label="Test Accuracy")
    plt.xticks(range(len(p_list)), [str(p) for p in p_list])
    plt.yticks(range(len(r_list)), [str(r) for r in r_list])
    plt.xlabel("p")
    plt.ylabel("r")
    plt.title("Accuracy")
    plt.savefig("accuracy_heatmap.png")
    plt.show()

    plt.figure(figsize=(6, 4))
    for p in p_list:
        times = [results[(r, p)][0] for r in r_list]
        plt.plot(r_list, times, marker='o', label=f"p={p}")
    plt.xlabel("r")
    plt.ylabel("Train Time")
    plt.title("train time - p")
    plt.legend()
    plt.savefig("train_time_vs_r.png")
    plt.show()

    plt.figure(figsize=(6, 4))
    for p in p_list:
        accs = [results[(r, p)][1] for r in r_list]
        plt.plot(r_list, accs, marker='o', label=f"p={p}")
    plt.xlabel("r")
    plt.ylabel("Test Accuracy")
    plt.title("test accuracy - r")
    plt.legend()
    plt.savefig("accuracy_vs_r.png")
    plt.show()


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

    r_list = [4, 8, 12, 16, 24, 32]
    p_list = [1.0, 0.5, 0.25]

    results = {}

    for r in r_list:
        for p in p_list:
            t, acc = train_one_rp(r, p, train_loader, test_loader, device)
            results[(r, p)] = (t, acc)

    visualize_results(r_list, p_list, results)

if __name__ == "__main__":
    main()
