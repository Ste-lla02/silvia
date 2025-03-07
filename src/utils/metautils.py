from src.utils.configuration import Configuration
import shutil
import os

class FileCleaner():
    def __init__(self):
        self.folder_names = ['maskfolder', 'croppedfolder', 'splittedfolder']

    def clean(self):
        configuration = Configuration()
        for folder_name in self.folder_names:
            folder = configuration.get(folder_name)
            shutil.rmtree(folder)
            os.makedirs(folder)


