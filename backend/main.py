from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
import aiofiles
import os, json
import utils
import debug, model, preprocess

from pydantic import BaseModel


app = FastAPI()


from fastapi import File, UploadFile


@app.post("/predict")
def predict(file: UploadFile = File(...), debug: bool = False):
    result = {}
    try:
        contents = file.file.read()
        with open(file.filename, "wb+") as f:
            f.write(contents)
        result = utils.predict(file.filename, debug)
    except Exception as e:
        return {"message": f"There was an error uploading the file: {e}"}
    finally:
        file.file.close()

    return {
        "payload":result.get("prediction", "error"),
        "filenames":result.get("debug_imgs", "")
    }


@app.post("/predict")
async def get_user():

    return json.dumps({"features": []}, indent=4, ensure_ascii=False)
