from termcolor import colored


def CRITICAL():
    return colored("(CRITICAL)", "red")


def WARNING():
    return colored("(WARNING)", "yellow")


class Error(Exception):
    def __init__(self, message, severity=CRITICAL):
        super().__init__(f"{colored('dvcs_xsx:', 'blue')} {severity()} {message}")
