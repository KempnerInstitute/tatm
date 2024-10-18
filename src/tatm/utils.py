from enum import Enum
import logging


def configure_logging(log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

class TatmOptionEnum(str, Enum):
    """base class for creating a set of Enum options for the Tatm package.
       
    Adds has_value method to Enum class to check if a value is in the Enum.
    
    """

    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)
