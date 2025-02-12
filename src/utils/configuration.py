import configparser
from configparser import ConfigParser
from src.utils import metaclasses
import ast
from src.utils.metaclasses import Singleton


class Configuration(metaclass=Singleton):
    def __init__(self, inifilename):
        self.board = dict()
        self.load(inifilename)

    def get(self, key):
        return self.board[key]

    def reset(self, inifilename):
        self.board = dict()
        self.load(inifilename)

    def put(self, key, value):
        self.board[key] = value

    def clean(self):
        self.board = dict()

    @staticmethod
    def tolist(value):
        retval = value.split(',')
        retval = list(map(lambda x: x.strip(),retval))
        return retval

    def loadSection(self, reader, s):
        temp = dict()
        options = []
        try:
            options = reader.options(s)
        except configparser.NoSectionError as nse:
            temp[s] = ['*']
        for o in options:
            try:
                value = reader[s][o]
                temp[s] = self.tolist(value)
            except:
                print("exception on %s!" % o)
        return temp

    def loadOption(self, reader, option):
        temp = dict()
        options = reader.options(option)
        for o in options:
            value = reader[option][o]
            temp[o] = self.tolist(value)
        retval = dict()
        retval[option] = temp
        return retval

    def load(self, inifile):
        reader = ConfigParser()
        reader.read(inifile)
        try:
            # Main
            temp = reader['main'].get('imagefolder',None)
            self.put('imagefolder', temp)
            temp = reader['main'].get('croppedfolder', None)
            self.put('croppedfolder', temp)
            temp = reader['main'].get('splittedfolder', None)
            self.put('splittedfolder', temp)
            temp = reader['main'].get('maskfolder', None)
            self.put('maskfolder', temp)
            temp = reader['main'].get('imagetype', None)
            self.put('imagetype', temp)
            temp = reader['main'].get('sam_model', None)
            self.put('sam_model', temp)
            temp = reader['main'].get('areaofinterest', None)
            temp = ast.literal_eval(temp)
            self.put('areaofinterest', temp)
            temp = reader['main'].get('channels', 'plain')
            temp = self.tolist(temp)
            self.put('channels', temp)
        except Exception as s:
            print(s)
