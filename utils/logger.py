import logging
import os
import sys

def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 如果是分布式训练且 rank > 0，则不记录日志
    if distributed_rank > 0:
        return logger

    # StreamHandler 处理控制台输出
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 如果 save_dir 存在，则创建 FileHandler 并记录到文件
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def save_metrics_periodically(metrics_dict, dir_text, epoch, period, logger=None):
    """
    在每个 epoch 是指定周期的倍数时，将指定的指标保存到文本文件中。

    Args:
        metrics_dict (dict): 包含要保存的指标的字典，键为指标名称，值为指标值。
        dir_text (str): 要保存日志文件的目标目录路径。
        epoch (int): 当前的 epoch 数。
        period (int): 保存周期，例如 50 表示每 50 个 epoch 保存一次。
        logger (logging.Logger, optional): 用于记录日志的 logger 对象。
                                         如果为 None,则不记录日志。
                                         默认为 None。
    """
    # --- 检查保存条件 ---
    # 通常我们希望在 epoch 50, 100, 150... 时保存
    # 同时要避免在 epoch 0 时保存 (如果你的 epoch 从 0 开始)
    if epoch > 0 and epoch % period == 0:
        try:
            # --- 确保目录存在 ---
            os.makedirs(dir_text, exist_ok=True)

            # --- 定义文件名 ---
            # 你可以根据需要修改文件名，例如包含模型名称等
            file_name = "training_metrics_log.txt"
            file_path = os.path.join(dir_text, file_name)

            # --- 准备要写入的内容 ---
            # 添加一个分隔符使日志更易读
            log_entry = f"--- Epoch: {epoch} ---\n"
            
            # 遍历字典中的所有指标
            for metric_name, metric_value in metrics_dict.items():
                log_entry += f"{metric_name}: {metric_value}\n"
            
            log_entry += f"{'-'*30}\n"  # 分隔线

            # --- 以追加模式 ('a') 打开文件并写入 ---
            # 使用 'utf-8' 编码以支持更广泛的字符
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)

            if logger:
                logger.info(f"指标已于 Epoch {epoch} 保存到: {file_path}")

        except OSError as e:
            if logger:
                logger.error(f"无法创建目录或写入文件 {file_path}: {e}")
        except Exception as e:
            if logger:
                logger.error(f"在 Epoch {epoch} 保存指标时发生意外错误: {e}")



