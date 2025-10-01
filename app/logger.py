import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

# 定义日志格式
LOG_FORMAT = "%(asctime)s - [%(name)s] %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 初始化日志
def get_logger(tag: str, log_file: str = "/log/app.log", max_bytes: int = 5 * 1024 * 1024, backup_count: int = 3):
    """
    创建一个日志记录器。

    :param tag: 日志的标签名称，用于标记日志来源。
    :param log_file: 日志文件路径。
    :param max_bytes: 单个日志文件的最大大小（字节）。
    :param backup_count: 保留的日志文件备份数量。
    :return: 配置好的日志记录器。
    """
    # 创建日志记录器
    logger = logging.getLogger(tag)
    logger.setLevel(logging.DEBUG)

    # 防止重复添加处理器
    if not logger.handlers:
        # 创建文件处理器（带日志轮转）
        file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setLevel(logging.DEBUG)

        # 设置日志格式
        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        file_handler.setFormatter(formatter)

        # 添加处理器
        logger.addHandler(file_handler)

    return logger
