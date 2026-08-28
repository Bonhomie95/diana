"""Entrypoint: serve Diana over HTTP (8080) and HTTPS (8443) from one process."""
import asyncio
import logging

import uvicorn

from . import certs, config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")


async def main():
    certfile, keyfile = certs.ensure()
    http_cfg = uvicorn.Config("diana.main:app", host="0.0.0.0",
                              port=config.PORT, log_level="info")
    https_cfg = uvicorn.Config("diana.main:app", host="0.0.0.0",
                               port=config.TLS_PORT, log_level="info",
                               ssl_certfile=certfile, ssl_keyfile=keyfile)
    await asyncio.gather(uvicorn.Server(http_cfg).serve(),
                         uvicorn.Server(https_cfg).serve())


if __name__ == "__main__":
    asyncio.run(main())
