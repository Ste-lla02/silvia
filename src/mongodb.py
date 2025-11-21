import requests
from src.models import FileResponse

class MongoWriter:
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