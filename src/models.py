from typing import List
from pydantic import BaseModel
from dataclasses import dataclass, asdict, field

class Experiment(BaseModel):
    name: str
    content: str


class Experiments(BaseModel):
    configurations: List[Experiment]


@dataclass
class FileResponse:
    filename: str
    result: bool

    def dict(self):
        _dict = self.__dict__.copy()
        return _dict


@dataclass
class SingleResponse:
    configurationname: str
    response: bool
    errormessage: str
    files: List[FileResponse]

    def dict(self):
        _dict = self.__dict__.copy()
        return _dict


@dataclass
class Response:
    reponses: List[SingleResponse]

    def dict(self):
        _dict = self.__dict__.copy()
        return _dict