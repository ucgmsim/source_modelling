use numpy::PyArray1;
use pyo3::prelude::*;

use crate::pytypes::{PyCsrMatrix, PySrfFile, PySrfMetadata, PySrfPlane};

#[derive(Debug, Copy, Clone)]
pub struct SrfPlane {
    pub elon: f32,
    pub elat: f32,
    pub nstk: usize,
    pub ndip: usize,
    pub len: f32,
    pub wid: f32,
    pub stk: f32,
    pub dip: f32,
    pub dtop: f32,
    pub shyp: f32,
    pub dhyp: f32,
}

impl SrfPlane {
    pub fn points(&self) -> usize {
        self.nstk * self.ndip
    }
}

impl<'py> IntoPyObject<'py> for SrfPlane {
    type Target = PySrfPlane;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Py::new(
            py,
            PySrfPlane {
                elon: self.elon,
                elat: self.elat,
                nstk: self.nstk,
                ndip: self.ndip,
                len: self.len,
                wid: self.wid,
                stk: self.stk,
                dip: self.dip,
                dtop: self.dtop,
                shyp: self.shyp,
                dhyp: self.dhyp,
            },
        )?
        .into_bound(py))
    }
}

#[derive(Debug)]
pub struct CsrMatrix {
    pub row_ptr: Vec<usize>,
    pub data: Vec<f32>,
}

impl<'py> IntoPyObject<'py> for CsrMatrix {
    type Target = PyCsrMatrix;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Py::new(
            py,
            PyCsrMatrix {
                row_ptr: PyArray1::from_vec(py, self.row_ptr).unbind(),
                data: PyArray1::from_vec(py, self.data).unbind(),
            },
        )?
        .into_bound(py))
    }
}

impl CsrMatrix {
    pub fn new(row_capacity: usize, data_capacity: usize) -> Self {
        CsrMatrix {
            row_ptr: Vec::with_capacity(row_capacity),
            data: Vec::with_capacity(data_capacity),
        }
    }

    pub fn push_row(&mut self) {
        self.row_ptr.push(self.data.len());
    }

    pub fn push(&mut self, v: f32) {
        self.data.push(v);
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Point {
    pub lon: f32,
    pub lat: f32,
    pub dep: f32,
    pub stk: f32,
    pub dip: f32,
    pub area: f32,
    pub tinit: f32,
    pub dt: f32,
    pub rake: f32,
    pub slip1: f32,
    pub rise: f32,
}

#[derive(Debug)]
pub struct SrfMetadata {
    pub lon: Vec<f32>,
    pub lat: Vec<f32>,
    pub dep: Vec<f32>,
    pub stk: Vec<f32>,
    pub dip: Vec<f32>,
    pub area: Vec<f32>,
    pub tinit: Vec<f32>,
    pub dt: Vec<f32>,
    pub rake: Vec<f32>,
    pub slip1: Vec<f32>,
    pub rise: Vec<f32>,
}

impl SrfMetadata {
    pub fn with_capacity(n: usize) -> Self {
        SrfMetadata {
            lon: Vec::with_capacity(n),
            lat: Vec::with_capacity(n),
            dep: Vec::with_capacity(n),
            stk: Vec::with_capacity(n),
            dip: Vec::with_capacity(n),
            area: Vec::with_capacity(n),
            tinit: Vec::with_capacity(n),
            dt: Vec::with_capacity(n),
            rake: Vec::with_capacity(n),
            slip1: Vec::with_capacity(n),
            rise: Vec::with_capacity(n),
        }
    }

    pub fn push(&mut self, point: &Point) {
        self.lon.push(point.lon);
        self.lat.push(point.lat);
        self.dep.push(point.dep);
        self.stk.push(point.stk);
        self.dip.push(point.dip);
        self.area.push(point.area);
        self.tinit.push(point.tinit);
        self.dt.push(point.dt);
        self.rake.push(point.rake);
        self.slip1.push(point.slip1);
        self.rise.push(point.rise);
    }
}

impl<'py> IntoPyObject<'py> for SrfMetadata {
    type Target = PySrfMetadata;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Py::new(
            py,
            PySrfMetadata {
                lon: PyArray1::from_vec(py, self.lon).unbind(),
                lat: PyArray1::from_vec(py, self.lat).unbind(),
                dep: PyArray1::from_vec(py, self.dep).unbind(),
                stk: PyArray1::from_vec(py, self.stk).unbind(),
                dip: PyArray1::from_vec(py, self.dip).unbind(),
                area: PyArray1::from_vec(py, self.area).unbind(),
                tinit: PyArray1::from_vec(py, self.tinit).unbind(),
                dt: PyArray1::from_vec(py, self.dt).unbind(),
                rake: PyArray1::from_vec(py, self.rake).unbind(),
                slip1: PyArray1::from_vec(py, self.slip1).unbind(),
                rise: PyArray1::from_vec(py, self.rise).unbind(),
                vs: None,
                density: None,
            },
        )?
        .into_bound(py))
    }
}

#[derive(Debug, Copy, Clone)]
pub struct PointV2 {
    pub base: Point,
    pub vs: f32,
    pub density: f32,
}

#[derive(Debug)]
pub struct SrfMetadataV2 {
    pub base: SrfMetadata,
    pub vs: Vec<f32>,
    pub density: Vec<f32>,
}

impl SrfMetadataV2 {
    pub fn with_capacity(n: usize) -> Self {
        SrfMetadataV2 {
            base: SrfMetadata::with_capacity(n),
            vs: Vec::with_capacity(n),
            density: Vec::with_capacity(n),
        }
    }

    pub fn push(&mut self, point: &PointV2) {
        self.base.push(&point.base);
        self.vs.push(point.vs);
        self.density.push(point.density);
    }
}

impl<'py> IntoPyObject<'py> for SrfMetadataV2 {
    type Target = PySrfMetadata;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let base = self.base.into_pyobject(py)?;
        {
            let mut base_ref = base.borrow_mut();
            base_ref.vs = Some(PyArray1::from_vec(py, self.vs).unbind());
            base_ref.density = Some(PyArray1::from_vec(py, self.density).unbind());
        }
        Ok(base)
    }
}

#[derive(Debug)]
pub enum SrfMetadataVersioned {
    V1(SrfMetadata),
    V2(SrfMetadataV2),
}

impl SrfMetadataVersioned {
    pub fn base(&self) -> &SrfMetadata {
        match self {
            Self::V1(metadata) => metadata,
            Self::V2(metadata) => &metadata.base,
        }
    }
}

impl<'py> IntoPyObject<'py> for SrfMetadataVersioned {
    type Target = PySrfMetadata;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Self::V1(metadata) => metadata.into_pyobject(py),
            Self::V2(metadata) => metadata.into_pyobject(py),
        }
    }
}

#[derive(Debug)]
pub struct SrfFile {
    pub planes: Vec<SrfPlane>,
    pub metadata: SrfMetadataVersioned,
    pub slipt1: CsrMatrix,
}

impl<'py> IntoPyObject<'py> for SrfFile {
    type Target = PySrfFile;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let mut planes: Vec<Py<PySrfPlane>> = Vec::with_capacity(self.planes.len());
        for plane in self.planes {
            planes.push(plane.into_pyobject(py)?.unbind());
        }
        let metadata = self.metadata.into_pyobject(py)?.unbind();
        let slipt1 = self.slipt1.into_pyobject(py)?.unbind();
        Ok(Py::new(
            py,
            PySrfFile {
                planes,
                metadata,
                slipt1,
            },
        )?
        .into_bound(py))
    }
}
