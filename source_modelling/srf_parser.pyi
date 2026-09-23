from typing import Any

import numpy as np
import numpy.typing as npt

class PySrfPlane:
    elon: float
    elat: float
    nstk: int
    ndip: int
    len: float
    wid: float
    stk: float
    dip: float
    dtop: float
    shyp: float
    dhyp: float
    def __init__(
        self,
        elon: float,
        elat: float,
        nstk: int,
        ndip: int,
        len: float,
        wid: float,
        stk: float,
        dip: float,
        dtop: float,
        shyp: float,
        dhyp: float,
    ) -> None: ...

class PyCsrMatrix:
    row_ptr: npt.NDArray[np.uintp]
    indices: npt.NDArray[np.uintp]
    data: npt.NDArray[np.float32]
    def __init__(
        self,
        row_ptr: npt.NDArray[np.uintp],
        indices: npt.NDArray[np.uintp],
        data: npt.NDArray[np.float32],
    ) -> None: ...

class PySrfMetadata:
    lon: npt.NDArray[np.float32]
    lat: npt.NDArray[np.float32]
    dep: npt.NDArray[np.float32]
    stk: npt.NDArray[np.float32]
    dip: npt.NDArray[np.float32]
    area: npt.NDArray[np.float32]
    tinit: npt.NDArray[np.float32]
    dt: npt.NDArray[np.float32]
    rake: npt.NDArray[np.float32]
    slip1: npt.NDArray[np.float32]
    rise: npt.NDArray[np.float32]
    vs: npt.NDArray[np.float32] | None
    density: npt.NDArray[np.float32] | None
    def __init__(
        self,
        lon: npt.NDArray[np.float32],
        lat: npt.NDArray[np.float32],
        dep: npt.NDArray[np.float32],
        stk: npt.NDArray[np.float32],
        dip: npt.NDArray[np.float32],
        area: npt.NDArray[np.float32],
        tinit: npt.NDArray[np.float32],
        dt: npt.NDArray[np.float32],
        rake: npt.NDArray[np.float32],
        slip1: npt.NDArray[np.float32],
        rise: npt.NDArray[np.float32],
        vs: npt.NDArray[np.float32] | None = None,
        density: npt.NDArray[np.float32] | None = None,
    ) -> None: ...

class PySrfFile:
    planes: list[PySrfPlane]
    metadata: PySrfMetadata
    slipt1: PyCsrMatrix
    def __init__(
        self,
        planes: list[PySrfPlane],
        metadata: PySrfMetadata,
        slipt1: PyCsrMatrix,
    ) -> None: ...

def parse_srf(buffer: Any) -> PySrfFile: ...
def write_srf(py_srf_file: PySrfFile, file_path: str) -> None: ...
