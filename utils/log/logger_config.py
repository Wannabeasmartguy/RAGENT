# logger_config.py
import sys
from loguru import logger

def setup_logger():
    folder_ = "./log/"
    prefix_ = "ragent-copilot-"
    rotation_ = "10 MB"
    retention_ = "30 days"
    encoding_ = "utf-8"
    backtrace_ = True
    diagnose_ = True

    format_ = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> ' \
              '| <magenta>{process}</magenta>:<yellow>{thread}</yellow> ' \
              '| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<yellow>{line}</yellow> - <level>{message}</level>'

    try:
        logger.remove(0)  # 移除默认的日志记录器
    except ValueError:
        pass

    # debug
    logger.add(folder_ + prefix_ + "debug.log", level="DEBUG", backtrace=backtrace_, diagnose=diagnose_,
               format=format_, colorize=False,
               rotation=rotation_, retention=retention_, encoding=encoding_,
               filter=lambda record: record["level"].no >= logger.level("DEBUG").no)

    # info
    logger.add(folder_ + prefix_ + "info.log", level="INFO", backtrace=backtrace_, diagnose=diagnose_,
               format=format_, colorize=False,
               rotation=rotation_, retention=retention_, encoding=encoding_,
               filter=lambda record: record["level"].no >= logger.level("INFO").no)

    # warning
    logger.add(folder_ + prefix_ + "warning.log", level="WARNING", backtrace=backtrace_, diagnose=diagnose_,
               format=format_, colorize=False,
               rotation=rotation_, retention=retention_, encoding=encoding_,
               filter=lambda record: record["level"].no >= logger.level("WARNING").no)

    # error
    logger.add(folder_ + prefix_ + "error.log", level="ERROR", backtrace=backtrace_, diagnose=diagnose_,
               format=format_, colorize=False,
               rotation=rotation_, retention=retention_, encoding=encoding_,
               filter=lambda record: record["level"].no >= logger.level("ERROR").no)

    # critical
    logger.add(folder_ + prefix_ + "critical.log", level="CRITICAL", backtrace=backtrace_, diagnose=diagnose_,
               format=format_, colorize=False,
               rotation=rotation_, retention=retention_, encoding=encoding_,
               filter=lambda record: record["level"].no >= logger.level("CRITICAL").no)

    # logger.add(sys.stderr, level="CRITICAL", backtrace=backtrace_, diagnose=diagnose_,
    #            format=format_, colorize=True,
    #            filter=lambda record: record["level"].no >= logger.level("CRITICAL").no)
    
    # logger.add(sys.stdout, level="INFO", backtrace=backtrace_, diagnose=diagnose_,
    #            format=format_, colorize=True,
    #            filter=lambda record: record["level"].no == logger.level("INFO").no)

def log_dict_changes(original_dict, new_dict):
    # 找出变化的键值对
    changed_keys = set(original_dict.keys()) & set(new_dict.keys())
    changed_keys = [key for key in changed_keys if original_dict[key] != new_dict[key]]
    
    # 找出新增和删除的键
    added_keys = set(new_dict.keys()) - set(original_dict.keys())
    removed_keys = set(original_dict.keys()) - set(new_dict.keys())
    
    # 记录变化
    if changed_keys or added_keys or removed_keys:
        logger.debug("Dictionary changes:")
        for key in changed_keys:
            logger.debug(f"Key '{key}' changed from '{original_dict[key]}' to '{new_dict[key]}'")
        for key in added_keys:
            logger.debug(f"Key '{key}' added with value '{new_dict[key]}'")
        for key in removed_keys:
            logger.debug(f"Key '{key}' removed (old value was '{original_dict[key]}')")

setup_logger()