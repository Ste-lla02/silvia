import sys
import uvicorn
from fastapi import FastAPI
from src.routers import router

def main(applicationport, mongodb_ipaddress, mongodb_port):
    app = FastAPI()
    app.include_router(router)
    router.mongodb_address = mongodb_ipaddress
    router.mongodb_port = mongodb_port
    uvicorn.run(app, port=applicationport)


if __name__ == "__main__":
    port = int(sys.argv[1])
    mongoip = sys.argv[2]
    mongoport = int(sys.argv[3])
    main(port, mongoip, mongoport)