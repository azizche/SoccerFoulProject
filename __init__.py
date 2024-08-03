import logging



LOGGING_NAME='MVFoul'
def set_logger(name='LOGGING NAME',verbose=True):
    level = logging.INFO if verbose else logging.WARNING

    #Setting up stream handler
    streamHandler= logging.StreamHandler()
    formatter= logging.Formatter('%(message)s')
    streamHandler.setFormatter(formatter)
    streamHandler.setLevel(level)

    #Setting up logger
    logger=logging.getLogger(name)
    logger.addHandler(streamHandler)
    logger.setLevel(level)
    return logger
LOGGER=set_logger(LOGGING_NAME,verbose=True)