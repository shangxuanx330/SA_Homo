import numpy as np
import os

def categorize_samples_by_scale_diff(scale_diff_list, scale_ranges):
    """
    根据每个样本的scale_diff值和指定范围分为小、中、大三组
    
    Args:
        scale_diff_list: 每个样本的scale_diff值列表
        scale_ranges: 三个组的范围，格式为[[min1,max1], [min2,max2], [min3,max3]]
        
    Returns:
        list: 每个样本对应的组别（'small', 'medium', 'large'）
    """
    if len(scale_ranges) != 3:
        raise ValueError("scale_ranges必须包含3个范围，对应small、medium、large三组")
    
    small_range = scale_ranges[0]  # [min, max] for small group
    medium_range = scale_ranges[1]  # [min, max] for medium group  
    large_range = scale_ranges[2]  # [min, max] for large group
    
    sample_groups = []
    
    for scale_diff in scale_diff_list:
        # 判断属于哪个组（使用左闭右开区间，除了最后一个组）
        if small_range[0] <= scale_diff < small_range[1]:
            sample_groups.append('small')
        elif medium_range[0] <= scale_diff < medium_range[1]:
            sample_groups.append('medium')
        elif large_range[0] <= scale_diff <= large_range[1]:  # 最后一个组使用闭区间
            sample_groups.append('large')
        else:
            # 如果不在任何范围内，根据最接近的范围分配
            distances = [
                min(abs(scale_diff - small_range[0]), abs(scale_diff - small_range[1])),
                min(abs(scale_diff - medium_range[0]), abs(scale_diff - medium_range[1])),
                min(abs(scale_diff - large_range[0]), abs(scale_diff - large_range[1]))
            ]
            closest_group_idx = distances.index(min(distances))
            group_names = ['small', 'medium', 'large']
            sample_groups.append(group_names[closest_group_idx])
    
    return sample_groups

def analyze_and_save_results(mace_list, dataset_name_list, img_path_list, scale_diff_list=None, 
                           scale_diff_ranges=None, save_dir=None, dataset="unknown", 
                           checkpoint="unknown", logger=None):
    """
    统计分析MACE结果并保存
    
    Args:
        mace_list: 所有样本的MACE值列表
        dataset_name_list: 对应的数据集名称列表
        img_path_list: 对应的图片路径列表
        scale_diff_list: 对应的scale_diff值列表（必需）
        scale_diff_ranges: 三个组的scale_diff范围，格式为[[min1,max1], [min2,max2], [min3,max3]]（必需）
        save_dir: 保存目录
        dataset: 数据集名称
        logger: 日志记录器
    Returns:
        dict: 包含统计结果的字典
    """
    
    # 如果没有传入logger，使用print
    def log_info(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    # 验证输入
    expected_length = len(mace_list)
    if len(dataset_name_list) != expected_length or len(img_path_list) != expected_length:
        raise ValueError("mace_list, dataset_name_list, img_path_list 长度必须一致")
    
    if scale_diff_list is None or scale_diff_ranges is None:
        raise ValueError("scale_diff_list和scale_diff_ranges参数是必需的")
    
    if len(scale_diff_list) != expected_length:
        raise ValueError("scale_diff_list长度必须与其他列表一致")
    
    if len(mace_list) == 0:
        log_info("警告: 没有测试样本")
        return {}
    
    # 转换为numpy数组便于计算
    mace_array = np.array(mace_list)
    
    # 全局统计
    total_samples = len(mace_list)
    overall_avg_mace = np.mean(mace_array)
    
    # 对所有损失值进行排序以计算分位数统计
    sorted_mace = np.sort(mace_array)
    
    # 按比例划分
    split_30 = int(0.3 * total_samples)
    split_60 = int(0.6 * total_samples)
    
    # 计算各区间平均值
    avg_0_30 = sorted_mace[:split_30].mean() if split_30 > 0 else 0.0
    avg_30_60 = sorted_mace[split_30:split_60].mean() if split_60 > split_30 else 0.0
    avg_60_100 = sorted_mace[split_60:].mean() if len(sorted_mace) > split_60 else 0.0
    
    # 按数据集分类统计
    dataset_metrics = {}
    for i, dataset_name in enumerate(dataset_name_list):
        if dataset_name not in dataset_metrics:
            dataset_metrics[dataset_name] = {
                'mace_list': [],
                'scale_diff_list': [],
                'total_samples': 0
            }
        dataset_metrics[dataset_name]['mace_list'].append(mace_list[i])
        dataset_metrics[dataset_name]['scale_diff_list'].append(scale_diff_list[i])
        dataset_metrics[dataset_name]['total_samples'] += 1
    
    # 计算每个数据集的平均指标
    dataset_avg_metrics = {}
    for dataset_name, metrics in dataset_metrics.items():
        if len(metrics['mace_list']) > 0:
            avg_mace = np.mean(metrics['mace_list'])
            avg_scale_diff = np.mean(metrics['scale_diff_list'])
            dataset_avg_metrics[dataset_name] = {
                'avg_mace': avg_mace,
                'avg_scale_diff': avg_scale_diff,
                'total_samples': metrics['total_samples']
            }
    
    # 根据scale_diff对样本进行分组
    sample_groups = categorize_samples_by_scale_diff(scale_diff_list, scale_diff_ranges)
    log_info(f"使用scale_diff值对样本进行分组，范围设置为: {scale_diff_ranges}")
    
    # 计算三大组的统计信息
    group_avg_metrics = {}
    detailed_mace_records = {
        'all': [],
        'by_dataset': {},
        'by_group': {'small': [], 'medium': [], 'large': []}
    }
    
    # 构建详细记录
    for i, (mace_val, dataset_name, img_path, scale_diff, group) in enumerate(
        zip(mace_list, dataset_name_list, img_path_list, scale_diff_list, sample_groups)):
        
        record = {
            'sample_index': i + 1,
            'dataset_name': dataset_name,
            'img_path': img_path,
            'mace': float(mace_val),
            'scale_diff': float(scale_diff),
            'group': group
        }
        
        # 添加到总体记录
        detailed_mace_records['all'].append(record)
        
        # 添加到数据集分类记录
        if dataset_name not in detailed_mace_records['by_dataset']:
            detailed_mace_records['by_dataset'][dataset_name] = []
        detailed_mace_records['by_dataset'][dataset_name].append(record)
        
        # 添加到组记录
        detailed_mace_records['by_group'][group].append(record)
    
    # 为三大组计算统计信息
    for group_name in ['small', 'medium', 'large']:
        group_records = detailed_mace_records['by_group'][group_name]
        
        if len(group_records) > 0:
            # 收集该组所有样本的MACE值
            group_mace_list = [record['mace'] for record in group_records]
            group_total_samples = len(group_mace_list)
            
            avg_mace = np.mean(group_mace_list)
            
            # 计算该组的分位数统计
            sorted_group_mace = np.sort(group_mace_list)
            group_samples = len(sorted_group_mace)
            
            group_split_30 = int(0.3 * group_samples)
            group_split_60 = int(0.6 * group_samples)
            
            group_avg_0_30 = sorted_group_mace[:group_split_30].mean() if group_split_30 > 0 else 0.0
            group_avg_30_60 = sorted_group_mace[group_split_30:group_split_60].mean() if group_split_60 > group_split_30 else 0.0
            group_avg_60_100 = sorted_group_mace[group_split_60:].mean() if group_samples > group_split_60 else 0.0
            
            # 计算该组的scale_diff范围
            group_scale_diff_list = [record['scale_diff'] for record in group_records]
            min_scale_diff = min(group_scale_diff_list)
            max_scale_diff = max(group_scale_diff_list)
            avg_scale_diff = np.mean(group_scale_diff_list)
            
            group_avg_metrics[group_name] = {
                'avg_mace': avg_mace,
                'total_samples': group_total_samples,
                'avg_0_30': group_avg_0_30,
                'avg_30_60': group_avg_30_60, 
                'avg_60_100': group_avg_60_100,
                'min_scale_diff': min_scale_diff,
                'max_scale_diff': max_scale_diff,
                'avg_scale_diff': avg_scale_diff,
                'scale_diff_range': scale_diff_ranges[['small', 'medium', 'large'].index(group_name)]
            }
    
    # 打印结果
    log_info("="*80)
    log_info("全局测试结果:")
    log_info("="*80)
    log_info(f"总样本数: {total_samples}")
    log_info(f"总体平均 MACE: {overall_avg_mace:.4f}")
    log_info(f"0-30% 平均 MACE: {avg_0_30:.4f}")
    log_info(f"30-60% 平均 MACE: {avg_30_60:.4f}")
    log_info(f"60-100% 平均 MACE: {avg_60_100:.4f}")
    
    # 打印按数据集分类的结果
    log_info("="*80)
    log_info("按具体数据集分类的测试结果:")
    log_info("="*80)
    for dataset_name, metrics in sorted(dataset_avg_metrics.items()):
        log_info(f"数据集 {dataset_name}:")
        log_info(f"  样本数: {metrics['total_samples']}")
        log_info(f"  平均 MACE: {metrics['avg_mace']:.4f}")
        log_info(f"  平均 Scale Diff: {metrics['avg_scale_diff']:.4f}")
        log_info("-" * 60)
    
    # 打印按三大组分类的结果
    log_info("="*80)
    log_info("按三大组分类的测试结果（基于Scale Diff值对样本分组）:")
    log_info("="*80)
    
    for group_name, metrics in group_avg_metrics.items():
        log_info(f"{group_name.upper()} Group ({group_name}组):")
        log_info(f"  Scale Diff范围设置: {metrics['scale_diff_range']}")
        log_info(f"  实际Scale Diff范围: [{metrics['min_scale_diff']:.4f}, {metrics['max_scale_diff']:.4f}]")
        log_info(f"  样本数: {metrics['total_samples']}")
        log_info(f"  平均 Scale Diff: {metrics['avg_scale_diff']:.4f}")
        log_info(f"  平均 MACE: {metrics['avg_mace']:.4f}")
        log_info(f"  0-30% 平均 MACE: {metrics['avg_0_30']:.4f}")
        log_info(f"  30-60% 平均 MACE: {metrics['avg_30_60']:.4f}")
        log_info(f"  60-100% 平均 MACE: {metrics['avg_60_100']:.4f}")
        log_info("-" * 80)
    
    # 显示分组详情
    log_info("="*80)
    log_info("样本分组详情（基于Scale Diff值分组）:")
    log_info("="*80)
    
    for group_name in ['small', 'medium', 'large']:
        group_count = sum(1 for g in sample_groups if g == group_name)
        log_info(f"{group_name}组: {group_count}个样本")
    
    # 保存结果到文件
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 保存总体MACE结果
        overall_results_path = os.path.join(save_dir, 'overall_results.txt')
        with open(overall_results_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Overall MACE Test Results\n")
            f.write("="*80 + "\n")
            f.write(f"Dataset: {dataset}\n")
            f.write(f"Batch Size: 1 (fixed)\n")
            f.write("="*80 + "\n\n")
            
            f.write("Global Test Results:\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Samples: {total_samples}\n")
            f.write(f"Overall Average MACE: {overall_avg_mace:.4f}\n")
            f.write(f"0-30% Average MACE: {avg_0_30:.4f}\n")
            f.write(f"30-60% Average MACE: {avg_30_60:.4f}\n")
            f.write(f"60-100% Average MACE: {avg_60_100:.4f}\n")
            f.write("\n" + "="*80 + "\n")
        
        log_info(f"总体结果已保存到: {overall_results_path}")
        
        # 2. 保存三大组的结果（分别保存到不同文件）
        group_descriptions = {
            'small': 'SMALL Group',
            'medium': 'MEDIUM Group',
            'large': 'LARGE Group'
        }
        
        for group_name, metrics in group_avg_metrics.items():
            group_results_path = os.path.join(save_dir, f'{group_name}_group_results.txt')
            with open(group_results_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write(f"{group_descriptions.get(group_name, group_name)} Test Results\n")
                f.write("="*80 + "\n")
                f.write(f"Dataset: {dataset}\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"{group_descriptions.get(group_name, group_name)} Statistical Results:\n")
                f.write("-"*40 + "\n")
                f.write(f"Scale Diff Range Setting: {metrics['scale_diff_range']}\n")
                f.write(f"Actual Scale Diff Range: [{metrics['min_scale_diff']:.4f}, {metrics['max_scale_diff']:.4f}]\n")
                f.write(f"Sample Count: {metrics['total_samples']}\n")
                f.write(f"Average Scale Diff: {metrics['avg_scale_diff']:.4f}\n")
                f.write(f"Average MACE: {metrics['avg_mace']:.4f}\n")
                f.write(f"0-30% Average MACE: {metrics['avg_0_30']:.4f}\n")
                f.write(f"30-60% Average MACE: {metrics['avg_30_60']:.4f}\n")
                f.write(f"60-100% Average MACE: {metrics['avg_60_100']:.4f}\n")
                f.write("\n" + "="*80 + "\n")
            
            log_info(f"{group_name}组结果已保存到: {group_results_path}")
        
    return {
        'global_metrics': {
            'overall_avg_mace': overall_avg_mace,
            'avg_0_30': avg_0_30,
            'avg_30_60': avg_30_60,
            'avg_60_100': avg_60_100,
            'total_samples': total_samples
        },
        'dataset_metrics': dataset_avg_metrics,
        'group_metrics': group_avg_metrics,
        'sample_groups': sample_groups,
        'detailed_records': detailed_mace_records
    }
