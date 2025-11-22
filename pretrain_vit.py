import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import tome
import time
import os
import json
from datetime import datetime

import warnings
import sys
import io

# compute FLOPs
from fvcore.nn import FlopCountAnalysis, flop_count_table
FVCORE_AVAILABLE = True


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


def train_one_rp(r, p, train_loader, test_loader, device, epochs=30, log_interval=5):
    """
    为某个 (r, p) 配置训练一个模型，并返回训练时间、测试精度和计算量
    
    Args:
        r: 每层要合并的 token 数量
        p: 参与相似度计算的 token 比例
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        device: 设备 (cpu/cuda)
        epochs: 训练轮数
        log_interval: 每隔多少个 epoch 输出一次状态
    
    Returns:
        train_time: 总训练时间(秒)
        acc: 最终测试精度
        flops: 计算量 (GFLOPs), 如果无法计算则为 None
    """
    print(f"\n{'='*60}")
    print(f"Training Config: r={r}, p={p}, epochs={epochs}")
    print(f"{'='*60}")

    # 1. 初始化模型
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=True,
        num_classes=10
    ).to(device)

    # 2. 打 ToMe 补丁
    tome.patch.timm(model)
    model.r = r
    model.p = p

    # 计算 FLOPs (只计算一次)
    flops_giga = None
    if FVCORE_AVAILABLE:
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            # 屏蔽 FLOPs 计算时的所有输出（包括 unsupported operator 信息）
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            
            try:
                flops = FlopCountAnalysis(model, dummy_input)
                total_flops = flops.total()
                flops_giga = total_flops / 1e9
            finally:
                # 恢复标准输出
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        except Exception as e:
            pass  # 静默失败

    # 3. 设置训练超参
    lr = 5e-4
    wd = 0.05
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    # 学习率调度器（余弦退火）- 适合长时间训练
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4. 训练
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 统计训练指标
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        # 更新学习率
        scheduler.step()
        
        # 每隔 log_interval 个 epoch 输出状态
        if (epoch + 1) % log_interval == 0 or epoch == 0 or epoch == epochs - 1:
            avg_loss = running_loss / len(train_loader)
            train_acc = correct / total
            elapsed_time = time.time() - start_time
            
            test_acc = evaluate(model, test_loader, device)
            
            print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Test Acc: {test_acc:.4f} | "
                  f"Time: {elapsed_time:.1f}s")

    # 5. 最终评估
    end_time = time.time()
    train_time = end_time - start_time
    final_acc = evaluate(model, test_loader, device)

    print(f"{'-'*60}")
    print(f"Finished: r={r}, p={p} | "
          f"Time: {train_time:.2f}s | "
          f"Accuracy: {final_acc:.4f} | " 
          f"FLOPs: {flops_giga:.4f} GFLOPs")

    return train_time, final_acc, flops_giga


def save_results_to_json(r_list, p_list, results):
    """
    保存实验结果到 JSON 文件
    results: {(r, p): (train_time, accuracy, flops)}
    """
    os.makedirs("results", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_data = {
        "timestamp": timestamp,
        "r_list": r_list,
        "p_list": p_list,
        "results": {}
    }
    
    for (r, p), (t, acc, flops) in results.items():
        results_data["results"][f"r{r}_p{p}"] = {
            "train_time": t,
            "accuracy": acc,
            "flops_gflops": flops
        }
    
    json_path = f"results/experiment_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)
    
    print(f"\nResults saved to: {json_path}")
    return json_path



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
    train_indices = list(range(5000))  # 只用前 5000 张
    test_indices = list(range(1000))  # 只用前 1000 张
    train_data = Subset(train_set, train_indices)
    test_data = Subset(test_set, test_indices)

    train_loader = DataLoader(
        train_data,
        batch_size=128,  # Kaggle GPU 可以用更大的 batch size
        shuffle=True,
        num_workers=2,  # Kaggle 上设置为 2
        pin_memory=True  # GPU 训练时启用
    )
    test_loader = DataLoader(
        test_data,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print("Train size:", len(train_data), "Test size:", len(test_data))

    # 本地 CPU
    # r_list = [64]
    # p_list = [0.8, 0.6]
    # epochs = 1
    # log_interval = 1
    
    # GPU
    r_list = [16, 64]
    p_list = [1.0, 0.8, 0.6, 0.4, 0.2]
    epochs = 5  # 完整训练
    log_interval = 2  # 每 2 个 epoch 输出一次

    results = {}

    for r in r_list:
        for p in p_list:
            t, acc, flops = train_one_rp(r, p, train_loader, test_loader, device, 
                                         epochs=epochs, log_interval=log_interval)
            results[(r, p)] = (t, acc, flops)

    # saving results to JSON
    json_path = save_results_to_json(r_list, p_list, results)
    print(f"\n Training complete! Use 'python evaluate.py {json_path}' to visualize results.")

if __name__ == "__main__":
    main()