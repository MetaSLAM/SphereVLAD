""" Init file for utils"""


def red(sthg):
    ''' red color '''
    return "\033[01;31m{0}\033[00m".format(str(sthg))


def green(sthg):
    ''' green color '''
    return "\033[01;32m{0}\033[00m".format(str(sthg))


def lblue(sthg):
    ''' green color '''
    return "\033[01;36m{0}\033[00m".format(str(sthg))


def pink(sthg):
    ''' pink color '''
    return "\033[01;35m{0}\033[00m".format(str(sthg))


def blue(sthg):
    ''' blue color '''
    return "\033[01;34m{0}\033[00m".format(str(sthg))


def yellow(sthg):
    ''' yellow color '''
    return "\033[01;93m{0}\033[00m".format(str(sthg))


def log_print(sthg, color='', logger=None):
    ''' Print colored stream, color options:r, g, b, lb, y, p. '''
    if color == "r":
        print(red(sthg))
    elif color == "g":
        print(green(sthg))
    elif color == "b":
        print(blue(sthg))
    elif color == "lb":
        print(lblue(sthg))
    elif color == "y":
        print(yellow(sthg))
    elif color == "p":
        print(pink(sthg))
    else:
        print(sthg)
    if logger != None:
        logger.info(sthg)
