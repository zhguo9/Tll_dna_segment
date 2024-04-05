import logging

level    = logging.INFO
format   = '  %(message)s'
handlers = [logging.FileHandler('filename.log'), logging.StreamHandler()]

logging.basicConfig(level = level, format = format, handlers = handlers)
logging.info('Hey, this is working!')