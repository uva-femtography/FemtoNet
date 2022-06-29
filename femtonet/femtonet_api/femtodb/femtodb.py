import redis3 as redis
import configparser

from os import environ
from colorama import Fore


class FemtoDB:
    def __init__(self, host='localhost', port=6379):
        self.host = host
        self.port = port
        self.secret_key = None
        self.connect()

    def connect(self):
        """
           Configure Redis database connection using local environment variable. This should be the default for live
           running. If no environment variables are found, try using configuration file.
        """
        try:
            assert environ.get('SECRET_KEY') is not None
            assert environ.get('REDIS_HOST') is not None
            assert environ.get('REDIS_PORT') is not None

            self.secret_key = environ['SECRET_KEY']
            self.host = environ['REDIS_HOST']
            self.port = environ['REDIS_PORT']
            self.db = redis.Redis(host=self.host, port=self.port)

        except AssertionError:
            print(Fore.BLUE + 'Database info not found in environment. Checking for configuration file.' + Fore.WHITE)

            self.read_config()
            self.db = redis.Redis(host=self.host, port=self.port)

    def read_config(self):
        """
            Read configuration files and get database connection information. Configuration is determined by
            one of a few sources:

            1) Check local environment variables for configuration information.
            2) Check for local configuration files.
            3) Default to localhost and default port.

            4) Fail to connect.
        """

        config = configparser.ConfigParser()
        if len(config.read('config/redis.config')) != 0:
            connection = config['connection']
            self.host = connection.get('host', fallback='localhost')
            self.port = int(connection.get('port', fallback='6379'))
        else:
            print(Fore.BLUE + 'Local configuration file not found. Using defaults.' + Fore.WHITE)

    def get_model_list(self):
        """
            Helper function used to return a list of models available for analysis.
        """

        models = []
        for key in self.db.keys():
            models.append(key.decode())

        return models

    def get_parameter_limits(self, model, parameter, dtype=float):
        """
            Get model parameter limits form database.
        """
        try:
            if self.db.exists(model) == 0:
                raise Exception('ModelException')

        except Exception as ex:
            print(Fore.RED + '{}: Model does not exists in database.'.format(ex) + Fore.WHITE)
            return

        return tuple(map(dtype, self.db.hget(model, parameter).decode().split(':')))

    def get_parameter(self, model, parameter, dtype=str):
        """
            Get model parameter from database.
        """
        try:
            if self.db.exists(model) == 0:
                raise Exception('ModelException')

        except Exception as ex:
            print(Fore.RED + '{}: Model does not exists in database.'.format(ex) + Fore.WHITE)
            return

        return dtype(self.db.hget(model, parameter).decode())

    def set_parameter(self, model, parameter, value, dtype=str):
        """
            Get model parameter from database.
        """
        try:
            if self.db.exists(model) == 0:
                raise Exception('ModelException')

        except Exception as ex:
            print(Fore.RED + '{}: Model does not exists in database.'.format(ex) + Fore.WHITE)
            return

        self.db.hset(model, parameter, dtype(value))

    def add_model(self, name=None, origin=None, xbj_limits=None, t_limits=None, q2_limits=None, model_dict=None):
        """
            Add new models to Redis database.
        """
        try:
            if self.db.exists(name):
                raise Exception('ModelException')
        except Exception as ex:
            print(Fore.RED + '{}: Model exists in database already.'.format(ex) + Fore.WHITE)

        if model_dict is not None:
            model = model_dict
        else:
            model = {'origin': origin,
                     'xbj_limits': xbj_limits,
                     't_limits': t_limits,
                     'q2_limits': q2_limits}

        for key, value in model.items():
            try:
                self.db.hset(name, key, value)
            except ValueError as ex:
                print(Fore.RED + 'Database error: {}', format(ex.args) + Fore.WHITE)
