import json
import os
from itertools import zip_longest

import numpy as np
import pydantic
import redis
import tqdm
import typer
import uvicorn
from fastapi import FastAPI, HTTPException

database = redis.Redis()

API_KEY = "ehiqODe6gEDbyWeJeNQgrbQrx7j7wiBc9AYrHVOwN6B0fXL0Qi5HcTat9ODbQWJE8IurApUyWD1jnLwsolvKDM4VZwcxtPAPfUkA24AaSBtItSuDhgi1GNcmAdsXzs5vHodsPqqoqQsJ20IJXpJArZdQyWqcUMKcZOJurhIcyilv89ChpQB9DwQxM5EJFwUDdPWAcZykNRBtDRSiGrYo2QivftVh6Swxp3QKBMptCzIYKpzR4pozM2W5kasLbuj3"


class APIKey(pydantic.BaseModel):
    data: str


class APIUrl(APIKey):
    url: str


def check(data: APIKey):
    if data.data != API_KEY:
        raise HTTPException(status_code=500)


async def get_url(data: APIKey) -> str:
    check(data)
    try:
        item = database.lpop("items")
        database.rpush("items", 1, item)  # rotate values so none are lost
        return item
    except:
        raise HTTPException(status_code=400)


async def post_url(data: APIUrl):
    check(data)
    try:
        database.lrem("items", 1, data.url)
    except:
        return


def main(url_dir: str = "./urls", max_seconds: int = 3600 * 4, min_seconds: int = 300, context: int = 64,
         fps: int = 16, seed: int = 0, clear: bool = False, chunk_size: int = 2 ** 16):
    durations = []
    ids = []
    for path in os.listdir(url_dir):
        with open(f'{url_dir}/{path}', 'rb') as f:
            vals = json.load(f)
            durations.extend([x for i in vals["duration"] for x in i])
            ids.extend([x for i in vals["id"] for x in i])

    durations = np.array(durations)
    ids = np.array(ids)
    where = np.where(np.logical_and(durations < max_seconds, durations > max(context / fps, min_seconds)))
    ids = ids[where]
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    if clear:
        database.delete("items")
        ids = iter(ids)
        for itm in tqdm.tqdm(list(zip_longest(*[ids] * chunk_size))):
            database.rpush("items", *(i for i in itm if i is not None))

    api = FastAPI()
    api.get("/url")(get_url)
    api.post("/url")(post_url)

    uvicorn.run(api, host='0.0.0.0', port=443, log_level='info', workers=1,
                ssl_certfile='/etc/letsencrypt/live/limitless.sh/fullchain.pem',
                ssl_keyfile='/etc/letsencrypt/live/limitless.sh/privkey.pem')


if __name__ == "__main__":
    typer.run(main)
