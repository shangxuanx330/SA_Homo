import importlib
from torch.optim.lr_scheduler import _LRScheduler

def get_obj_from_str(string, reload=False):
    #与 split() 方法类似，但是从右边开始分割， 只分割一次
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class WarmupStepLR(_LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.7, warmup_epochs=5, warmup_lr_init=1e-6, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_init = warmup_lr_init
        super(WarmupStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.warmup_lr_init + (base_lr - self.warmup_lr_init) * (self.last_epoch / self.warmup_epochs)
                    for base_lr in self.base_lrs]
        else:
            return [base_lr * self.gamma ** ((self.last_epoch - self.warmup_epochs) // self.step_size)
                    for base_lr in self.base_lrs]
        
def print_config(config, logger,indent=0):
    """
    递归打印配置字典，支持嵌套字典
    每个顶级键占一行，子字典的键值对缩进显示
    """
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(' ' * indent + f"{key}:")
            print_config(value,logger, indent + 2)
        else:
            logger.info(' ' * indent + f"{key}: {value}")


def write_epoch_results(file_path, epoch_data, header=False):
    """
    将每个epoch的结果写入文件，自动对齐列名和数据
    :param file_path: 输出文件路径
    :param epoch_data: 字典，包含所有需要记录的数据
    :param header: 是否需要写入表头
    """
    # 判断文件是否为空
    file_empty = True
    try:
        with open(file_path, 'r') as f:
            if f.read(1):
                file_empty = False  # 文件非空
    except FileNotFoundError:
        file_empty = True  # 文件不存在时视为文件为空

    # 如果是第一次写入（文件为空），写入表头和列宽信息
    if header or file_empty:
        headers = list(epoch_data.keys())

        # 先将所有数据转换为字符串，以计算最大宽度
        str_values = []
        for key, value in epoch_data.items():
            if key == 'epoch':
                str_values.append(f"{value:03d}")
            elif isinstance(value, (int, float)):
                if value == 100000:
                    str_values.append('NA')
                elif abs(value) < 0.0001:
                    str_values.append(f"{value:.7e}")
                else:
                    str_values.append(f"{value:.7f}")
            else:
                str_values.append(str(value))

        # 计算每列的最大宽度（列名和数据的最大宽度）
        column_widths = {}
        for header, value in zip(headers, str_values):
            column_widths[header] = max(len(header), len(value))

        # 写入列宽信息和表头
        with open(file_path, 'w') as f:
            # 写入列宽信息作为注释
            widths_str = ','.join(f"{k}:{v}" for k, v in column_widths.items())
            f.write(f"#WIDTHS={widths_str}\n")
            # 写入对齐的表头
            header_line = ','.join(h.ljust(column_widths[h]) for h in headers)
            f.write(header_line + '\n')
    else:
        # 文件已存在，从文件中读取列宽信息
        column_widths = {}
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith('#WIDTHS='):
                widths_str = first_line[8:]  # 去掉 '#WIDTHS=' 前缀
                for item in widths_str.split(','):
                    key, width = item.split(':')
                    column_widths[key] = int(width)
            else:
                # 如果没有列宽信息，重新计算
                # 这里可以根据需要设置默认宽度或重新计算
                raise ValueError("文件格式错误：缺少列宽信息")

    # 写入数据行
    with open(file_path, 'a') as f:
        values = []
        for key, value in epoch_data.items():
            if key == 'epoch':
                values.append(f"{value:03d}".ljust(column_widths[key]))
            elif isinstance(value, (int, float)):
                if value == 100000:
                    values.append('NA'.ljust(column_widths[key]))
                elif abs(value) < 0.0001:
                    values.append(f"{value:.7e}".ljust(column_widths[key]))
                else:
                    values.append(f"{value:.7f}".ljust(column_widths[key]))
            else:
                values.append(str(value).ljust(column_widths[key]))

        f.write(','.join(values) + '\n')

def read_epoch_results(file_path):
    """
    读取训练记录文件，将每列数据保存到对应的列表中
    :param file_path: 训练记录文件路径
    :return: 字典，键为列名，值为该列的所有数据组成的列表
    """
    results = {}
    
    with open(file_path, 'r') as f:
        # 跳过列宽信息行
        f.readline()
        # 读取表头
        headers = [h.strip() for h in f.readline().strip().split(',')]
        for header in headers:
            results[header] = []
            
        # 读取数据行
        for line in f:
            values = [v.strip() for v in line.strip().split(',')]
            for header, value in zip(headers, values):
                if value == 'NA':
                    results[header].append(None)
                elif header == 'epoch':
                    results[header].append(int(value))
                else:
                    try:
                        results[header].append(float(value))
                    except ValueError:
                        results[header].append(value)
                        
    return results