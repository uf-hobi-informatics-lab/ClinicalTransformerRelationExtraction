import logging


class TransformerLogger:
    LOG_LVLs = {
        'i': logging.INFO,
        'd': logging.DEBUG,
        'e': logging.ERROR,
        'w': logging.WARN
    }

    def __init__(self, logger_file=None, logger_level='d'):
        self.lf = logger_file
        self.lvl = logger_level

    def set_log_info(self, logger_file, logger_level):
        self.lf = logger_file
        self.lvl = logger_level

    def _create_logger(self, logger_name=""):
        logger = logging.getLogger(logger_name)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
        logger.setLevel(self.LOG_LVLs[self.lvl])
        if self.lf:
            fh = logging.FileHandler(self.lf)
            fh.setFormatter(formatter)
            fh.setLevel(self.LOG_LVLs[self.lvl])
            logger.addHandler(fh)
        else:
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            ch.setLevel(self.LOG_LVLs[self.lvl])
            logger.addHandler(ch)

        return logger

    def get_logger(self):
        return self._create_logger("Transformer_Relation_Extraction")
