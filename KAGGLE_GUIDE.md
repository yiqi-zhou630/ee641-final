# Kaggle 使用指南

## 🎯 方法 1：从 GitHub 克隆（最推荐）

### 超简单 3 步：

**Step 1: 在 Kaggle 创建新 Notebook**
- 进入 [Kaggle Notebooks](https://www.kaggle.com/code)
- 点击 "New Notebook"
- Settings → Accelerator → 选择 **GPU T4 x2**

**Step 2: 在第一个 Cell 运行以下代码**

```python
# 安装依赖
!pip install timm==0.4.12 fvcore iopath -q

# 克隆 GitHub 仓库
!git clone https://github.com/yiqi-zhou630/ee641-final.git
%cd ee641-final

# 验证环境
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**Step 3: 在第二个 Cell 运行训练**

```python
# 运行训练脚本
!python pretrain_vit.py
```

完成！结果会保存在 `results/` 文件夹，可以直接下载 JSON 文件。

---

## 📦 方法 2：使用提供的 setup 脚本（更简单）

直接复制粘贴 `kaggle_setup.py` 的内容到 Kaggle notebook，一键运行！

---

## 📦 方法 3：手动上传文件

在 Kaggle Notebook 的第一个 cell 中运行：

```python
# 安装必要的包
!pip install timm==0.4.12 fvcore iopath -q

# 验证安装
import torch
import torchvision
import timm
import fvcore
print(f"PyTorch: {torch.__version__}")
print(f"timm: {timm.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## 📁 方法 2：上传整个项目

### 步骤 1：准备文件
上传这些文件到 Kaggle Notebook：
- `pretrain_vit.py`（主训练脚本）
- `evaluate.py`（可视化脚本）
- `tome/` 文件夹（整个文件夹及其子文件）
- `requirements.txt`（可选）

### 步骤 2：在 Kaggle 中安装依赖
```python
# Cell 1: 安装依赖
!pip install -r requirements.txt -q

# 或者直接安装
!pip install timm==0.4.12 fvcore iopath -q
```

### 步骤 3：运行训练
```python
# Cell 2: 导入并运行
import sys
sys.path.append('/kaggle/working')  # 确保能找到 tome 模块

# 运行训练脚本
!python pretrain_vit.py
```

### 步骤 4：下载结果
训练完成后，下载 `results/experiment_*.json` 文件到本地，然后用 `evaluate.py` 可视化。

## 🚀 方法 3：直接在 Notebook 中运行代码（最简单）

### Cell 1: 安装依赖
```python
!pip install timm==0.4.12 fvcore iopath -q
```

### Cell 2: 上传 tome 文件夹
点击 Kaggle 右侧的 "Add Data" → "Upload" → 上传 `tome/` 整个文件夹

或者手动创建 tome 模块（复制粘贴代码）

### Cell 3: 复制 pretrain_vit.py 的代码
直接把 `pretrain_vit.py` 的全部代码复制到一个 cell 中运行

### Cell 4: 运行训练
```python
# 代码会自动运行 main() 函数
```

### Cell 5: 下载结果
```python
# 显示生成的结果文件
!ls results/

# 下载到本地（点击文件即可下载）
from IPython.display import FileLink
import os

result_files = [f for f in os.listdir('results/') if f.endswith('.json')]
if result_files:
    latest = sorted(result_files)[-1]
    print(f"下载这个文件: results/{latest}")
    display(FileLink(f'results/{latest}'))
```

## ⚙️ Kaggle 环境说明

### 预装的包（无需安装）：
- ✅ `torch` (PyTorch)
- ✅ `torchvision`
- ✅ `numpy`
- ✅ `matplotlib`
- ✅ `scipy`
- ✅ `pillow`

### 需要手动安装的包：
- ❌ `timm` (需要安装特定版本 0.4.12)
- ❌ `fvcore` (用于 FLOPs 计算)
- ❌ `iopath` (fvcore 的依赖)

### GPU 设置：
- 进入 Notebook Settings (右侧)
- Accelerator 选择：**GPU T4 x2** 或 **GPU P100**
- 每周有免费的 GPU 使用时长（约 30 小时）

## 📊 实验配置建议

### 快速测试（约 10 分钟）：
```python
r_list = [4, 16]
p_list = [1.0, 0.6]
epochs = 5
```

### 中等规模（约 1-2 小时）：
```python
r_list = [4, 16, 32]
p_list = [1.0, 0.8, 0.6]
epochs = 15
```

### 完整实验（约 3-5 小时）：
```python
r_list = [4, 8, 16, 32, 64]
p_list = [1.0, 0.8, 0.6, 0.4, 0.2]
epochs = 30
```

## 💡 常见问题

### Q: 提示找不到 tome 模块
A: 确保 `tome/` 文件夹在工作目录，或添加：
```python
import sys
sys.path.append('/kaggle/working')
```

### Q: CUDA out of memory
A: 减小 batch_size：
```python
batch_size = 64  # 从 128 改为 64
```

### Q: 训练时间太长
A: 减少实验配置或 epochs：
```python
r_list = [4, 16]  # 只测试 2 个值
epochs = 10  # 减少 epoch 数
```

### Q: 如何保存中间结果
A: 在 `train_one_rp` 函数中添加模型保存：
```python
# 保存最佳模型
torch.save(model.state_dict(), f'model_r{r}_p{p}.pth')
```

## 📝 完整的 Kaggle Notebook 模板

### 从 GitHub 克隆版本（推荐）：

```python
# ========== Cell 1: Setup ==========
!pip install timm==0.4.12 fvcore iopath -q
!git clone https://github.com/yiqi-zhou630/ee641-final.git
%cd ee641-final

import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ========== Cell 2: Run Training ==========
!python pretrain_vit.py

# ========== Cell 3: Check Results ==========
!ls -lh results/

# ========== Cell 4: Download Results ==========
from IPython.display import FileLink
import os

result_files = [f for f in os.listdir('results/') if f.endswith('.json')]
if result_files:
    latest = sorted(result_files)[-1]
    print(f"📥 Download this file: results/{latest}")
    display(FileLink(f'results/{latest}'))
```

### 手动上传版本：

```python
# ========== Cell 1: 安装依赖 ==========
!pip install timm==0.4.12 fvcore iopath -q

# ========== Cell 2: 验证环境 ==========
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# ========== Cell 3: 上传 tome 文件夹后，导入模块 ==========
import sys
sys.path.append('/kaggle/working')

# ========== Cell 4: 粘贴 pretrain_vit.py 的全部代码 ==========
# [粘贴代码]

# ========== Cell 5: 运行训练 ==========
if __name__ == "__main__":
    main()

# ========== Cell 6: 查看结果 ==========
!ls results/
```

## 🎯 推荐工作流程

1. **本地测试**（1 epoch, 小数据集）→ 确保代码能运行
2. **Kaggle 快速验证**（5 epochs, 2-3 个配置）→ 验证 GPU 训练正常
3. **Kaggle 完整实验**（30 epochs, 完整配置）→ 获取最终结果
4. **本地可视化**（`evaluate.py`）→ 生成论文图表
