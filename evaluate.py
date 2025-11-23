import json
import os
import sys
from matplotlib import pyplot as plt


def load_results(json_path):
    """
    Load experiment results from a JSON file
    """
    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def visualize_results(data, json_path):
    """
    Generate visualizations based on JSON data
    将图表保存到与 JSON 文件相同的文件夹中
    """
    # 获取 JSON 文件所在的目录作为输出目录
    output_dir = os.path.dirname(json_path)
    
    timestamp = data["timestamp"]
    r_list = data["r_list"]
    p_list = data["p_list"]
    results_dict = data["results"]
    
    # 重建 results 字典
    results = {}
    for key, value in results_dict.items():
        # key format: "r4_p0.8"
        parts = key.split('_')
        r = int(parts[0][1:])  # 去掉 'r' 前缀
        p = float(parts[1][1:])  # 去掉 'p' 前缀
        # 兼容旧格式（没有 flops）和新格式（有 flops）
        flops = value.get("flops_gflops", None)
        results[(r, p)] = (value["train_time"], value["accuracy"], flops)
    
    # 构建矩阵
    time_matrix = []
    acc_matrix = []
    for r in r_list:
        row_t = []
        row_a = []
        for p in p_list:
            t, acc, flops = results[(r, p)]
            row_t.append(t)
            row_a.append(acc)
        time_matrix.append(row_t)
        acc_matrix.append(row_a)

    # output_dir 已经在函数开头从 json_path 获取，无需再创建

    # ---------- 1. 训练时间热力图 ----------
    plt.figure(figsize=(8, 6))
    plt.imshow(time_matrix, aspect='auto', cmap='YlOrRd')
    plt.colorbar(label="Train Time (s)")
    plt.xticks(range(len(p_list)), [str(p) for p in p_list])
    plt.yticks(range(len(r_list)), [str(r) for r in r_list])
    plt.xlabel("p (Proportion of tokens for matching)")
    plt.ylabel("r (Number of tokens to merge)")
    plt.title("Training Time Heatmap")
    
    # 在格子上显示数值
    for i in range(len(r_list)):
        for j in range(len(p_list)):
            plt.text(j, i, f'{time_matrix[i][j]:.1f}s',
                    ha="center", va="center", color="black", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/train_time_heatmap.png", dpi=150)
    plt.close()
    print(f"✓ Saved: train_time_heatmap.png")

    # ---------- 2. 准确率热力图 ----------
    plt.figure(figsize=(8, 6))
    plt.imshow(acc_matrix, aspect='auto', cmap='RdYlGn')
    plt.colorbar(label="Test Accuracy")
    plt.xticks(range(len(p_list)), [str(p) for p in p_list])
    plt.yticks(range(len(r_list)), [str(r) for r in r_list])
    plt.xlabel("p (Proportion of tokens for matching)")
    plt.ylabel("r (Number of tokens to merge)")
    plt.title("Test Accuracy Heatmap")
    
    # 在格子上显示数值
    for i in range(len(r_list)):
        for j in range(len(p_list)):
            plt.text(j, i, f'{acc_matrix[i][j]:.4f}',
                    ha="center", va="center", color="black", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_heatmap.png", dpi=150)
    plt.close()
    print(f"✓ Saved: accuracy_heatmap.png")

    # ---------- 3. 训练时间 vs r (不同 p) ----------
    plt.figure(figsize=(10, 6))
    for p in p_list:
        times = [results[(r, p)][0] for r in r_list]  # [0] 是 train_time
        plt.plot(r_list, times, marker='o', label=f"p={p}", linewidth=2)
    plt.xlabel("r (Number of tokens to merge)")
    plt.ylabel("Training Time (s)")
    plt.title("Training Time vs r for Different p Values")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/train_time_vs_r.png", dpi=150)
    plt.close()
    print(f"✓ Saved: train_time_vs_r.png")

    # ---------- 4. 准确率 vs r (不同 p) ----------
    plt.figure(figsize=(10, 6))
    for p in p_list:
        accs = [results[(r, p)][1] for r in r_list]  # [1] 是 accuracy
        plt.plot(r_list, accs, marker='o', label=f"p={p}", linewidth=2)
    plt.xlabel("r (Number of tokens to merge)")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs r for Different p Values")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_vs_r.png", dpi=150)
    plt.close()
    print(f"✓ Saved: accuracy_vs_r.png")
    
    # ---------- 5. 训练曲线（如果有 history 数据）----------
    has_history = any('history' in value for value in results_dict.values())
    if has_history:
        print("\n Generating training curves...")
        
        # 方案1: 固定 r，不同 p 的曲线画在一起
        # 按 r 值分组
        r_groups = {}
        for key, value in results_dict.items():
            if 'history' not in value or not value['history']:
                continue
            
            # 解析 r 和 p
            parts = key.split('_')
            r = int(parts[0][1:])
            p = float(parts[1][1:])
            
            if r not in r_groups:
                r_groups[r] = []
            r_groups[r].append((p, value['history']))
        
        # 为每个 r 值生成一张图
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for r in sorted(r_groups.keys()):
            configs = sorted(r_groups[r], key=lambda x: x[0], reverse=True)  # 按 p 降序
            
            # 创建 2x1 子图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle(f'Training Curves for r={r} (Different p Values)', 
                        fontsize=14, fontweight='bold')
            
            # 子图1: Loss 曲线
            for idx, (p, history) in enumerate(configs):
                epochs = history['epochs']
                color = colors[idx % len(colors)]
                ax1.plot(epochs, history['train_loss'], 
                        color=color, linestyle='-', linewidth=2, 
                        marker='o', markersize=4, label=f'p={p}')
            
            ax1.set_xlabel('Epoch', fontsize=11)
            ax1.set_ylabel('Training Loss', fontsize=11)
            ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
            ax1.legend(loc='best', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # 子图2: Test Accuracy 曲线
            for idx, (p, history) in enumerate(configs):
                epochs = history['epochs']
                color = colors[idx % len(colors)]
                ax2.plot(epochs, history['test_acc'], 
                        color=color, linestyle='-', linewidth=2, 
                        marker='s', markersize=4, label=f'p={p}')
            
            ax2.set_xlabel('Epoch', fontsize=11)
            ax2.set_ylabel('Test Accuracy', fontsize=11)
            ax2.set_title('Test Accuracy', fontsize=12, fontweight='bold')
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/training_curves_r{r}.png", dpi=150)
            plt.close()
            print(f"✓ Saved: training_curves_r{r}.png (grouped by p)")
        
        print(f"✓ Saved {len(r_groups)} combined training curve plots")

    print(f"\n✓ All plots saved to {output_dir}/ folder")


def print_summary(data):
    """
    打印实验结果摘要
    """
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Timestamp: {data['timestamp']}")
    
    # 打印实验配置信息（如果有）
    if 'experiment_config' in data:
        config = data['experiment_config']
        print("\n--- Experiment Configuration ---")
        print(f"Model:        {config.get('model', 'N/A')}")
        print(f"Dataset:      {config.get('dataset', 'N/A')}")
        print(f"Train Size:   {config.get('train_size', 'N/A')}")
        print(f"Test Size:    {config.get('test_size', 'N/A')}")
        print(f"Batch Size:   {config.get('batch_size', 'N/A')}")
        print(f"Epochs:       {config.get('epochs', 'N/A')}")
        print(f"Learning Rate: {config.get('learning_rate', 'N/A')}")
        print(f"Optimizer:    {config.get('optimizer', 'N/A')}")
        print(f"Device:       {config.get('device', 'N/A')}")
        print(f"Image Size:   {config.get('image_size', 'N/A')}")
    
    print(f"\n--- ToMe Parameters ---")
    print(f"r values tested: {data['r_list']}")
    print(f"p values tested: {data['p_list']}")
    
    print("\n--- Results ---")
    print("-"*80)
    print(f"{'Config':<15} {'Train Time (s)':<18} {'Accuracy':<12} {'FLOPs (G)':<12}")
    print("-"*80)
    
    results_dict = data["results"]
    for key in sorted(results_dict.keys()):
        value = results_dict[key]
        flops_str = f"{value['flops_gflops']:.4f}" if 'flops_gflops' in value and value['flops_gflops'] is not None else "N/A"
        print(f"{key:<15} {value['train_time']:<18.2f} {value['accuracy']:<12.4f} {flops_str:<12}")
    print("="*80 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <path_to_json_file>")
        print("Example: python evaluate.py results/experiment_20251122_120000/results.json")
        
        # 尝试找最新的实验文件夹
        if os.path.exists("results"):
            # 查找所有实验文件夹
            exp_dirs = [d for d in os.listdir("results") 
                       if os.path.isdir(os.path.join("results", d)) and d.startswith("experiment_")]
            if exp_dirs:
                latest_dir = max([os.path.join("results", d) for d in exp_dirs], 
                               key=os.path.getmtime)
                latest_json = os.path.join(latest_dir, "results.json")
                if os.path.exists(latest_json):
                    print(f"\nFound latest result: {latest_json}")
                    print(f"   Run: python evaluate.py {latest_json}")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    # 加载数据
    print(f"Loading results from: {json_path}")
    data = load_results(json_path)
    
    # 打印摘要
    print_summary(data)
    
    # 生成可视化
    print("\nGenerating visualizations...")
    visualize_results(data, json_path)
    
    print("\n✓ Evaluation complete!")
    print(f"  All files saved to: {os.path.dirname(json_path)}/")


if __name__ == "__main__":
    main()
