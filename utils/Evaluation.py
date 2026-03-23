
def calculate_intervals_statistics_logger_version(values, logger):
    """统计各个区间的数值，并计算 error 的排序及指定百分比区间的平均值

    Args:
        values (float list): 一个浮点数列表，包含需要统计的值
        logger: 日志记录器

    Returns:
        cumulative_results, dict
        specific_results, dict
        error_percentage_statistics, dict
    """
    # 定义累积区间的上界
    cumulative_limits = [2, 4, 6, 10, 20, 40, 100]
    cumulative_limits.append(float('inf'))

    # 定义具体区间
    specific_intervals = [(float('-inf'), 2), (2, 4), (4, 6), (6, 10), (10, 20), (20, 40), (40, 100)]
    specific_intervals.append((100, float('inf')))

    # 初始化结果字典
    cumulative_results = {}
    specific_results = {}

    # 计算累积区间的统计信息
    for upper_limit in cumulative_limits:
        interval_values = [v for v in values if v < upper_limit]
        if interval_values:
            count = len(interval_values)
            average = sum(interval_values) / count
            percentage = count / len(values) * 100
            cumulative_results[f"<={upper_limit}"] = {
                "count": count,
                "average": average,
                "percentage": percentage
            }

    # 计算具体区间的统计信息
    for lower_limit, upper_limit in specific_intervals:
        eplison = 1e-6
        interval_values = [v for v in values if lower_limit < v <= upper_limit+eplison]
        if interval_values:
            count = len(interval_values)
            average = sum(interval_values) / count
            specific_results[f"{lower_limit}-{upper_limit}"] = {
                "count": count,
                "average": average
            }

    # 新增：添加 error 的排序和百分比区间的平均值计算
    sorted_errors = sorted(values)  # 将 error 从大到小排序
    total_count = len(sorted_errors)  # 总元素个数

    # 定义百分比区间：0-30%、30-60%、60-100%
    percentage_intervals = [(0, 30), (30, 60), (60, 100)]
    error_percentage_statistics = {}

    for lower_percent, upper_percent in percentage_intervals:
        lower_index = int((lower_percent / 100) * total_count)
        upper_index = int((upper_percent / 100) * total_count)
        interval_errors = sorted_errors[lower_index:upper_index]  # 提取对应区间的 error

        if interval_errors:
            average_error = sum(interval_errors) / len(interval_errors)
            error_percentage_statistics[f"{lower_percent}-{upper_percent}%"] = {
                "average": average_error,
                "count": len(interval_errors)
            }
        else:
            error_percentage_statistics[f"{lower_percent}-{upper_percent}%"] = {
                "average": 0,
                "count": 0
            }

    # 格式化输出函数
    def format_table_row(*columns):
        return "| " + " | ".join(f"{col:>12}" for col in columns) + " |"

    def format_table_separator(column_count):
        return "+" + "+".join("-" * 14 for _ in range(column_count)) + "+"

    # 使用logger输出累积区间统计信息
    logger.info("Cumulative Intervals:")
    logger.info(format_table_separator(4))
    logger.info(format_table_row("Interval", "Count", "Average", "Percentage"))
    logger.info(format_table_separator(4))

    for interval, stats in cumulative_results.items():
        logger.info(format_table_row(
            interval,
            stats['count'],
            f"{stats['average']:.2f}",
            f"{stats['percentage']:.2f}%"
        ))

    logger.info(format_table_separator(4))

    # 输出具体区间统计信息
    logger.info("\nSpecific Intervals:")
    logger.info(format_table_separator(3))
    logger.info(format_table_row("Interval", "Count", "Average"))
    logger.info(format_table_separator(3))

    for interval, stats in specific_results.items():
        logger.info(format_table_row(
            interval,
            stats['count'],
            f"{stats['average']:.2f}"
        ))

    logger.info(format_table_separator(3))

    # 输出百分比区间统计信息
    logger.info("\nError Percentage Intervals:")
    logger.info(format_table_separator(3))
    logger.info(format_table_row("Interval", "Count", "Average"))
    logger.info(format_table_separator(3))

    for interval, stats in error_percentage_statistics.items():
        logger.info(format_table_row(
            interval,
            stats['count'],
            f"{stats['average']:.2f}"
        ))

    logger.info(format_table_separator(3))

    return cumulative_results, specific_results, error_percentage_statistics
