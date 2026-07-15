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

impl From<&PySrfPlane> for SrfPlane {
    fn from(plane: &PySrfPlane) -> Self {
        SrfPlane {
            elon: plane.elon,
            elat: plane.elat,
            nstk: plane.nstk,
            ndip: plane.ndip,
            len: plane.len,
            wid: plane.wid,
            stk: plane.stk,
            dip: plane.dip,
            dtop: plane.dtop,
            shyp: plane.shyp,
            dhyp: plane.dhyp,
        }
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

/// CSR matrix over any storage: `Vec`s when parsing (the parser appends), or
/// borrowed slices when writing data that another allocator (e.g. numpy) owns.
#[derive(Debug)]
pub struct CsrMatrix<R = Vec<usize>, D = Vec<f32>> {
    pub row_ptr: R,
    pub data: D,
}

pub type CsrMatrixView<'a> = CsrMatrix<&'a [usize], &'a [f32]>;

impl<'py> IntoPyObject<'py> for CsrMatrix {
    type Target = PyCsrMatrix;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(mut self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        // from_vec keeps the Vec as the numpy array's backing store, so any
        // excess capacity from the parser's size guess would stay reserved
        // for the array's whole lifetime unless released here.
        self.row_ptr.shrink_to_fit();
        self.data.shrink_to_fit();
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

impl<R: AsRef<[usize]>, D: AsRef<[f32]>> CsrMatrix<R, D> {
    pub fn rows(&self) -> CsrRowIter<'_> {
        CsrRowIter {
            row_ptr: self.row_ptr.as_ref(),
            data: self.data.as_ref(),
            index: 0,
        }
    }
}

pub struct CsrRowIter<'a> {
    row_ptr: &'a [usize],
    data: &'a [f32],
    index: usize,
}

impl<'a> Iterator for CsrRowIter<'a> {
    type Item = &'a [f32];

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.index;
        if i >= self.row_ptr.len() {
            return None;
        }
        self.index += 1;
        let start = self.row_ptr[i];
        let end = self
            .row_ptr
            .get(i + 1)
            .copied()
            .unwrap_or(self.data.len());
        Some(&self.data[start..end])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.row_ptr.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for CsrRowIter<'_> {}

impl<'a, R: AsRef<[usize]>, D: AsRef<[f32]>> IntoIterator for &'a CsrMatrix<R, D> {
    type Item = &'a [f32];
    type IntoIter = CsrRowIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.rows()
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

/// Per-point metadata in struct-of-arrays layout. Generic over storage:
/// `Vec<f32>` (the default) when the parser builds it, `&[f32]` when viewing
/// numpy-owned arrays for writing.
#[derive(Debug)]
pub struct SrfMetadata<S = Vec<f32>> {
    pub lon: S,
    pub lat: S,
    pub dep: S,
    pub stk: S,
    pub dip: S,
    pub area: S,
    pub tinit: S,
    pub dt: S,
    pub rake: S,
    pub slip1: S,
    pub rise: S,
}

pub type SrfMetadataView<'a> = SrfMetadata<&'a [f32]>;

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

impl<S: AsRef<[f32]>> SrfMetadata<S> {
    pub fn iter(&self) -> PointIter<'_> {
        PointIter {
            lon: self.lon.as_ref(),
            lat: self.lat.as_ref(),
            dep: self.dep.as_ref(),
            stk: self.stk.as_ref(),
            dip: self.dip.as_ref(),
            area: self.area.as_ref(),
            tinit: self.tinit.as_ref(),
            dt: self.dt.as_ref(),
            rake: self.rake.as_ref(),
            slip1: self.slip1.as_ref(),
            rise: self.rise.as_ref(),
            index: 0,
        }
    }
}

pub struct PointIter<'a> {
    lon: &'a [f32],
    lat: &'a [f32],
    dep: &'a [f32],
    stk: &'a [f32],
    dip: &'a [f32],
    area: &'a [f32],
    tinit: &'a [f32],
    dt: &'a [f32],
    rake: &'a [f32],
    slip1: &'a [f32],
    rise: &'a [f32],
    index: usize,
}

impl Iterator for PointIter<'_> {
    type Item = Point;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.index;
        if i >= self.lon.len() {
            return None;
        }
        self.index += 1;
        Some(Point {
            lon: self.lon[i],
            lat: self.lat[i],
            dep: self.dep[i],
            stk: self.stk[i],
            dip: self.dip[i],
            area: self.area[i],
            tinit: self.tinit[i],
            dt: self.dt[i],
            rake: self.rake[i],
            slip1: self.slip1[i],
            rise: self.rise[i],
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.lon.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for PointIter<'_> {}

impl<'a, S: AsRef<[f32]>> IntoIterator for &'a SrfMetadata<S> {
    type Item = Point;
    type IntoIter = PointIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
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
pub struct SrfMetadataV2<S = Vec<f32>> {
    pub base: SrfMetadata<S>,
    pub vs: S,
    pub density: S,
}

pub type SrfMetadataV2View<'a> = SrfMetadataV2<&'a [f32]>;

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

impl<S: AsRef<[f32]>> SrfMetadataV2<S> {
    pub fn iter(&self) -> PointV2Iter<'_> {
        PointV2Iter {
            points: self.base.iter(),
            vs: self.vs.as_ref(),
            density: self.density.as_ref(),
        }
    }
}

pub struct PointV2Iter<'a> {
    points: PointIter<'a>,
    vs: &'a [f32],
    density: &'a [f32],
}

impl Iterator for PointV2Iter<'_> {
    type Item = PointV2;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.points.index;
        let base = self.points.next()?;
        Some(PointV2 {
            base,
            vs: self.vs[i],
            density: self.density[i],
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.points.size_hint()
    }
}

impl ExactSizeIterator for PointV2Iter<'_> {}

impl<'a, S: AsRef<[f32]>> IntoIterator for &'a SrfMetadataV2<S> {
    type Item = PointV2;
    type IntoIter = PointV2Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
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
pub enum SrfMetadataVersioned<S = Vec<f32>> {
    V1(SrfMetadata<S>),
    V2(SrfMetadataV2<S>),
}

impl<S> SrfMetadataVersioned<S> {
    pub fn base(&self) -> &SrfMetadata<S> {
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
pub struct SrfFile<S = Vec<f32>, R = Vec<usize>> {
    pub planes: Vec<SrfPlane>,
    pub metadata: SrfMetadataVersioned<S>,
    pub slipt1: CsrMatrix<R, S>,
}

pub type SrfFileView<'a> = SrfFile<&'a [f32], &'a [usize]>;

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
