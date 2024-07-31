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

setup_logger()