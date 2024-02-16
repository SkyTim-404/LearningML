'''
Utils function for reading yaml files
'''

import yaml

class Dict2Object():
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if type(value) is dict:
                setattr(self, key, Dict2Object(value))
            else:
                setattr(self, key, value)

class YamlParser(object):
    def __init__(self, filename):
        with open(filename, 'r') as file:
            config = yaml.safe_load(file)
        for key, value in config.items():
            if type(value) is dict:
                setattr(self, key, Dict2Object(value))
            else:
                setattr(self, key, value)