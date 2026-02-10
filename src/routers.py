import glob
from src.utils.configuration import folderextraction
#from src.core import grab
from src.mongodb import MongoInterface
from src.models import *
from fastapi import APIRouter
from src.utils.utils import messagemaker

router = APIRouter()

@router.post("/execute")
async def create_item(item: Request):
    response = Response(list())
    for analysis in item.analyses:
        name = analysis.experimentname
        files = analysis.files
        content = analysis.configuration


        folder = folderextraction(content)

        pass
    #     writer = MongoWriter(router.mongodb_address, router.mongodb_port)
    #     image_retval, data_retval = None, None #grab(content) todo: qui non c'è un grab quindi bisogna modificare
    #     message = messagemaker(image_retval, data_retval)
    #     single_response = SingleResponse(name, image_retval and data_retval, message, list())
    #     for file in glob.glob(folder + "/*"):
    #         fileresponse = writer.write(name, file)
    #         single_response.files.append(fileresponse)
    #     response.reponses.append(single_response)
    return response.dict()