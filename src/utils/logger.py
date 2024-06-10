import logging
import coloredlogs
import sys

class Logger:
    def __init__(self, name, logfile=None):
        self.logger = self.setup_logger(name, logfile)
        sys.excepthook = self.handle_exception

    def setup_logger(self, name, logfile=None):
        logger_instance = logging.getLogger(name)
        logger_instance.setLevel(logging.DEBUG)  # Set the logger level to DEBUG or any desired level

        if logfile:
            i_handler = logging.FileHandler(logfile)
            i_handler.setLevel(logging.INFO)
            logger_instance.addHandler(i_handler)
        
        coloredlogs.install(
            level='DEBUG', logger=logger_instance,
            fmt='%(asctime)s %(name)s %(levelname)s %(message)s'
        )

        return logger_instance

    def handle_exception(self, exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        self.logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
