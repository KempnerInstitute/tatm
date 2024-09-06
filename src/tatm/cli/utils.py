import logging


def configure_cli_logging(level: int):
    cli_logger = logging.getLogger("tatm")
    cli_logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    cli_logger.addHandler(handler)
    return cli_logger
