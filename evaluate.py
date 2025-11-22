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


def visualize_results(data, output_dir="results"):
    """
    Generate visualizations based on JSON data
    """
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

    os.makedirs(output_dir, exist_ok=True)

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
    plt.savefig(f"{output_dir}/train_time_heatmap_{timestamp}.png", dpi=150)
    plt.close()
    print(f"  Saved: train_time_heatmap_{timestamp}.png")

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
    plt.savefig(f"{output_dir}/accuracy_heatmap_{timestamp}.png", dpi=150)
    plt.close()
    print(f"  Saved: accuracy_heatmap_{timestamp}.png")

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
    plt.savefig(f"{output_dir}/train_time_vs_r_{timestamp}.png", dpi=150)
    plt.close()
    print(f"  Saved: train_time_vs_r_{timestamp}.png")

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
    plt.savefig(f"{output_dir}/accuracy_vs_r_{timestamp}.png", dpi=150)
    plt.close()
    print(f"  Saved: accuracy_vs_r_{timestamp}.png")

    print(f"\n  All plots saved to {output_dir}/ folder")


def print_summary(data):
    """
    打印实验结果摘要
    """
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Timestamp: {data['timestamp']}")
    print(f"r values tested: {data['r_list']}")
    print(f"p values tested: {data['p_list']}")
    print("\nResults:")
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
        print("Example: python evaluate.py results/experiment_20250121_143052.json")
        
        # 尝试找最新的 JSON 文件
        if os.path.exists("results"):
            json_files = [f for f in os.listdir("results") if f.endswith('.json')]
            if json_files:
                latest_json = max([os.path.join("results", f) for f in json_files], 
                                key=os.path.getmtime)
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
    print("Generating visualizations...")
    visualize_results(data)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
