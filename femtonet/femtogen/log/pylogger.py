import logging
import functools

def create_logger():
    """
    Creates a logging object and returns it
    """
    _logger = logging.getLogger(__name__)

    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG)

    f_handler = logging.FileHandler('error.log')
    f_handler.setLevel(logging.ERROR)

    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(fmt)
    f_handler.setFormatter(fmt)

    _logger.addHandler(c_handler)
    _logger.addHandler(f_handler)

    return _logger


def logger(function):
    """
    A decorator that wraps the passed in function and logs
    exceptions should one occur
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        _logger = create_logger()
        try:
            _logger.debug("{0} - {1} - {2}".format(function.__name__, args, kwargs))
            result = function(*args, **kwargs)
            _logger.debug(result)

            return result

        except:
            error = "There was an exception."
            error += function.__name__
            _logger.exception(error)

            raise

    return wrapper
