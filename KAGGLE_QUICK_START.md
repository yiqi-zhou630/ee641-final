# ToMe (Token Merging) for Vision Transformers - Kaggle Quick Start

## 🚀 在 Kaggle 中运行 (3 步搞定)

### Step 1: 创建 Kaggle Notebook
1. 访问 https://www.kaggle.com/code
2. 点击 "New Notebook"
3. Settings (右上角) → Accelerator → 选择 **GPU T4 x2**

### Step 2: 安装并克隆仓库
在第一个 Cell 中运行：

```python
# 安装依赖包
!pip install timm==0.4.12 fvcore iopath -q

# 克隆 GitHub 仓库
!git clone https://github.com/yiqi-zhou630/ee641-final.git
%cd ee641-final

# 验证 GPU 可用
import torch
print(f"✓ CUDA Available: {torch.cuda.is_available()}")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
```

### Step 3: 运行训练
在第二个 Cell 中运行：

```python
!python pretrain_vit.py
```

训练完成后，下载 `results/experiment_*.json` 文件到本地。

---

## 📊 本地可视化结果

下载 JSON 文件后，在本地运行：

```bash
python evaluate.py results/experiment_YYYYMMDD_HHMMSS.json
```

会生成 4 张图表：
- 训练时间热力图
- 准确率热力图
- 训练时间 vs r 曲线
- 准确率 vs r 曲线

---

## ⚙️ 修改实验配置

如需调整实验参数，在 Kaggle 中修改 `pretrain_vit.py` 的这些行：

```python
# 快速测试 (约 10 分钟)
r_list = [4, 16]
p_list = [1.0, 0.6]
epochs = 5

# 完整实验 (约 3-5 小时，当前默认配置)
r_list = [4, 8, 16, 32, 64]
p_list = [1.0, 0.8, 0.6, 0.4, 0.2]
epochs = 30
```

---

## 📖 详细说明

查看 [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md) 了解：
- 完整的配置选项
- 常见问题解答
- 其他部署方法

---

## 🔗 相关链接

- **论文**: [Token Merging: Your ViT But Faster (ICLR 2023)](https://arxiv.org/abs/2210.09461)
- **原始仓库**: https://github.com/facebookresearch/ToMe
- **项目文档**: 查看 README.md

---

需要帮助？请查看 [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md) 或提交 Issue。
