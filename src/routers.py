import glob
import os
import shutil

from core.core_functions import Processor
from mongodb import MongoInterface
from src.models import *
from fastapi import APIRouter

from utils.configuration import Configuration

router = APIRouter()

@router.post("/execute")
async def create_item(item: Request):
    response = Response(list())
    for analysis in item.analyses:
        name = analysis.experimentname
        content = analysis.configuration
        configuration = Configuration(content)
        mongo_ip = configuration.get('mongo_ip')
        mongo_port = configuration.get('mongo_port')
        mongo_client = MongoInterface(mongo_ip, mongo_port)
        data_directory = configuration.get('basefolder')
        if os.path.exists(data_directory) and os.path.isdir(data_directory):
            for item in os.listdir(data_directory):
                shutil.rmtree(os.path.join(data_directory, item), ignore_errors=True)
        temp_dir_name = mongo_client.read(name, configuration.get('basefolder'), 'src')
        processor = Processor(configuration, temp_dir_name)
        processor.full_process()
        pass
        #     writer = MongoWriter(router.mongodb_address, router.mongodb_port)
    return response.dict()