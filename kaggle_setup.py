"""
Kaggle Setup Script
在 Kaggle Notebook 中运行此脚本，自动克隆 GitHub 仓库并安装依赖
"""

# ========== Cell 1: 克隆仓库并安装依赖 ==========
print("📦 Installing dependencies...")
!pip install timm==0.4.12 fvcore iopath -q

print("\n📥 Cloning repository from GitHub...")
!git clone https://github.com/yiqi-zhou630/ee641-final.git
%cd ee641-final

print("\n✅ Setup complete!")
print("\n" + "="*60)

# 验证环境
import torch
import timm
print(f"PyTorch version: {torch.__version__}")
print(f"timm version: {timm.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("="*60)

# ========== Cell 2: 运行训练 ==========
print("\n🚀 Starting training...")
!python pretrain_vit.py

# ========== Cell 3: 查看结果文件 ==========
print("\n📊 Results generated:")
!ls -lh results/

# ========== Cell 4: (可选) 在 Kaggle 中直接可视化 ==========
# 如果想在 Kaggle 中直接看图，运行下面的代码
"""
import os
result_files = [f for f in os.listdir('results/') if f.endswith('.json')]
if result_files:
    latest = sorted(result_files)[-1]
    print(f"Visualizing: results/{latest}")
    !python evaluate.py results/{latest}
    
    # 显示生成的图片
    from IPython.display import Image, display
    import glob
    
    png_files = glob.glob(f'results/*{latest.replace(".json", "")}*.png')
    for png in png_files:
        print(f"\n{png}")
        display(Image(png))
"""
