from pydantic import BaseModel, field_validator
from typing import List
from dataclasses import dataclass

class Analysis(BaseModel):
    experimentname: str
    files: List[str]
    configuration: str

    @field_validator('configuration', mode='before')
    @classmethod
    def unescape_config(cls, v: str) -> str:
        if isinstance(v, str):
            v = v.encode('utf-8').decode('unicode_escape')
        return v

class Request(BaseModel):
    analyses: List[Analysis]

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