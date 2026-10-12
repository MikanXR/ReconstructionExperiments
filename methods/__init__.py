"""Method registry.

Each method module exposes:

    NAME: str                      directory name under out/<capture>/
    LICENSE: str                   weights license, copied into the report
    predict(capture, frame_ids, cache, cfg) -> None
                                   writes depth and weight for every frame id
                                   into the DepthCache (cache.begin / write / finish)

Modules are imported lazily so a method's heavy dependencies load only when it
is run.
"""

from __future__ import annotations

import importlib

METHODS = {
    "lidar_tsdf": "methods.lidar_tsdf",
    "promptda": "methods.promptda",
    "da3": "methods.da3",
    "mapanything": "methods.mapanything",
    "milo": "methods.milo",
}


def load(name: str):
    if name not in METHODS:
        raise KeyError(f"unknown method {name!r}; known: {', '.join(METHODS)}")
    return importlib.import_module(METHODS[name])
