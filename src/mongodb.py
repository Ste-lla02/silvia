import requests
from src.models import FileResponse
from utils.configuration import Configuration


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

    def read(self, name):
        conf = Configuration()
        mongoport = conf.get('mongo_port')
        mongoip = conf.get('mongo_address')
        mongourl = ('http://' + mongoip + ':' + str(mongoport) + '/matforpat?identifier=' + str(t_req.modelid) +
                    '&version=' + str(t_req.modelversion))
        resp = requests.get(url=mongourl)
        tmp = tempfile.NamedTemporaryFile()
        handler = open(tmp.name, 'wb')
        handler.write(resp.content)
        handler.flush()
        reader = open(tmp.name, 'rb')
        fileContent = reader.read()
        trainedmodel = TrainedModel()
        trainedmodel.load(fileContent)
