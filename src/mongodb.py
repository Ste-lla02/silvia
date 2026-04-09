import os
import zipfile
from pathlib import Path
import requests
from src.models import FileResponse
from utils.configuration import Configuration
import tempfile

class MongoInterface:
    def __init__(self, iip, pport):
        self.ip = iip
        self.port = pport

    def write(self, name, binaryfilename):
        mongourl = 'http://' + self.ip + ':' + str(self.port) + '/matforpat?configuration_name=' + name
        retval = FileResponse(binaryfilename, False)
        try:
            handler = open(binaryfilename, 'rb')
            files = {"file": (handler.name, handler, "multipart/form-data")}
            resp = requests.post(url=mongourl, files=files)
            retval.result = resp.json()['success']
        except Exception:
            pass
        return retval

    def read(self, name: str, base_folder: str, src_folder: str, dumps_folder: str) -> str:
        mongourl = 'http://' + self.ip + ':' + str(self.port) + '/matforpat?configuration_name=' + name
        resp = requests.get(url=mongourl)
        tmpdirname = tempfile.mkdtemp(dir=base_folder)
        file_path = os.path.join(tmpdirname, name + '.zip')
        with open(file_path, 'wb') as handler:
            handler.write(resp.content)
        with zipfile.ZipFile(file_path, 'r') as zf:
            zf.extractall(tmpdirname + '/' + src_folder)
        Path(tmpdirname + '/' + dumps_folder).mkdir(parents=True, exist_ok=True)
        return tmpdirname
