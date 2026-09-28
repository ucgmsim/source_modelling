use numpy::PyArray1;
use pyo3::prelude::*;
use thiserror::Error;

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
///
/// The layout is exactly scipy's `csr_array` with int32 index arrays, so the
/// matrix crosses the Python boundary in both directions without copies.
/// `row_ptr` follows the scipy `indptr` convention: an n-row matrix has n+1
/// entries, `row_ptr[0] == 0`, `row_ptr[n] == data.len()`, and row i occupies
/// `indices[row_ptr[i]..row_ptr[i + 1]]` and `data[row_ptr[i]..row_ptr[i + 1]]`.
/// This holds after every `add_row`, so the matrix can be handed to scipy or
/// iterated at any point without a fixup pass.
///
/// Row i, column j is the slip rate of point i at timestep j on the global
/// timeline. Columns the matrix leaves out are zero, as in scipy.
#[derive(Debug)]
pub struct CsrMatrix<R = Vec<i32>, D = Vec<f32>> {
    pub indices: R,
    pub row_ptr: R,
    pub data: D,
}

pub type CsrMatrixView<'a> = CsrMatrix<&'a [i32], &'a [f32]>;

#[derive(Debug, Error, PartialEq, Eq)]
#[error(
    "slipt1 needs more than i32::MAX values or timesteps, which int32 CSR indices cannot address"
)]
pub struct CsrIndexOverflow;

impl<'py> IntoPyObject<'py> for CsrMatrix {
    type Target = PyCsrMatrix;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(mut self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        self.finalise();
        Ok(Py::new(
            py,
            PyCsrMatrix {
                row_ptr: PyArray1::from_vec(py, self.row_ptr).unbind(),
                indices: PyArray1::from_vec(py, self.indices).unbind(),
                data: PyArray1::from_vec(py, self.data).unbind(),
            },
        )?
        .into_bound(py))
    }
}

impl CsrMatrix {
    pub fn new(row_capacity: usize, data_capacity: usize) -> Self {
        CsrMatrix {
            row_ptr: {
                let mut row_ptr = Vec::with_capacity(row_capacity + 1);
                row_ptr.push(0);
                row_ptr
            },
            indices: Vec::with_capacity(data_capacity),
            data: Vec::with_capacity(data_capacity),
        }
    }

    /// Appends a row whose values sit at consecutive columns from `starting`.
    pub fn add_row<I>(&mut self, starting: usize, values: I) -> Result<(), CsrIndexOverflow>
    where
        I: Iterator<Item = f32>,
    {
        let starting = i32::try_from(starting).map_err(|_| CsrIndexOverflow)?;
        for (i, v) in values.enumerate() {
            let column = i32::try_from(i)
                .ok()
                .and_then(|i| starting.checked_add(i))
                .ok_or(CsrIndexOverflow)?;
            self.indices.push(column);
            self.data.push(v);
        }
        self.row_ptr
            .push(i32::try_from(self.data.len()).map_err(|_| CsrIndexOverflow)?);
        Ok(())
    }

    pub fn finalise(&mut self) {
        self.row_ptr.shrink_to_fit();
        self.data.shrink_to_fit();
        self.indices.shrink_to_fit();
    }
}

/// Column of a point's first slipt1 sample on the global timeline. The parser
/// and writer must agree on this, or written rows shift in time on re-read.
pub fn starting_column(tinit: f32, dt: f32) -> usize {
    // The choice between round and floor is relatively arbitrary. We choose floor here.
    (tinit / dt).floor() as usize
}

impl<R: AsRef<[i32]>, D: AsRef<[f32]>> CsrMatrix<R, D> {
    pub fn rows(&self) -> CsrRowIter<'_> {
        CsrRowIter {
            row_ptr: self.row_ptr.as_ref(),
            indices: self.indices.as_ref(),
            data: self.data.as_ref(),
            index: 0,
        }
    }
}

/// One row of a `CsrMatrix`: the stored columns and their values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CsrRow<'a> {
    pub indices: &'a [i32],
    pub data: &'a [f32],
}

/// Why a CSR row cannot be written as an SRF slip-rate function.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SlipRowError {
    #[error(
        "column indices are not sorted and unique (call sum_duplicates() on the csr_array first)"
    )]
    NonCanonical,
    #[error(
        "it has a value at timestep {column}, before its first timestep floor(tinit / dt) = {start}"
    )]
    BeforeStart { column: i64, start: usize },
}

impl<'a> CsrRow<'a> {
    /// The row as an SRF stores it: one sample per timestep from `start` (see
    /// `starting_column`) to the last stored column. An SRF only records where
    /// a row starts, so columns the CSR leaves out must come back as explicit
    /// zeros or the samples after them shift earlier in time on re-read.
    pub fn samples(&self, start: usize) -> Result<Samples<'a>, SlipRowError> {
        if self.indices.windows(2).any(|w| w[0] >= w[1]) {
            return Err(SlipRowError::NonCanonical);
        }
        let (Some(&first), Some(&last)) = (self.indices.first(), self.indices.last()) else {
            return Ok(Samples {
                indices: &[],
                data: &[],
                column: start,
                end: start,
            });
        };
        if i64::from(first) < start as i64 {
            return Err(SlipRowError::BeforeStart {
                column: first.into(),
                start,
            });
        }
        Ok(Samples {
            indices: self.indices,
            data: self.data,
            column: start,
            end: last as usize + 1,
        })
    }
}

/// Dense samples of a `CsrRow`, yielding 0.0 for columns it does not store.
pub struct Samples<'a> {
    indices: &'a [i32],
    data: &'a [f32],
    column: usize,
    end: usize,
}

impl Iterator for Samples<'_> {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.column == self.end {
            return None;
        }
        let value = match self.indices.first() {
            Some(&stored) if stored as usize == self.column => {
                let value = self.data[0];
                self.indices = &self.indices[1..];
                self.data = &self.data[1..];
                value
            }
            _ => 0.0,
        };
        self.column += 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.column;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for Samples<'_> {}

pub struct CsrRowIter<'a> {
    row_ptr: &'a [i32],
    indices: &'a [i32],
    data: &'a [f32],
    index: usize,
}

impl<'a> Iterator for CsrRowIter<'a> {
    type Item = CsrRow<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.index;
        // n rows are described by n+1 row_ptr entries, so the last valid row
        // index is row_ptr.len() - 2.
        if i + 1 >= self.row_ptr.len() {
            return None;
        }
        self.index += 1;
        let span = self.row_ptr[i] as usize..self.row_ptr[i + 1] as usize;
        Some(CsrRow {
            indices: &self.indices[span.clone()],
            data: &self.data[span],
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.row_ptr.len().saturating_sub(self.index + 1);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for CsrRowIter<'_> {}

impl<'a, R: AsRef<[i32]>, D: AsRef<[f32]>> IntoIterator for &'a CsrMatrix<R, D> {
    type Item = CsrRow<'a>;
    type IntoIter = CsrRowIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.rows()
    }
}

#[cfg(test)]
mod csr_tests {
    use super::*;

    fn build(rows: &[&[f32]]) -> CsrMatrix {
        let mut matrix = CsrMatrix::new(rows.len(), rows.iter().map(|row| row.len()).sum());
        for row in rows {
            matrix.add_row(0, row.iter().copied()).unwrap();
        }
        matrix
    }

    fn row_values(matrix: &CsrMatrix) -> Vec<Vec<f32>> {
        matrix.rows().map(|row| row.data.to_vec()).collect()
    }

    // row_ptr must be a valid scipy indptr after every add_row, not just once
    // finalise has run.
    fn assert_indptr_invariant(matrix: &CsrMatrix, rows: usize) {
        assert_eq!(matrix.row_ptr.len(), rows + 1);
        assert_eq!(matrix.row_ptr[0], 0);
        assert_eq!(*matrix.row_ptr.last().unwrap() as usize, matrix.data.len());
        assert!(matrix.row_ptr.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn empty_matrix_has_zero_rows() {
        let matrix = build(&[]);
        assert_indptr_invariant(&matrix, 0);
        assert_eq!(matrix.rows().count(), 0);
    }

    #[test]
    fn invariant_holds_after_every_add_row() {
        let mut matrix = CsrMatrix::new(3, 6);
        for (i, row) in [&[1.0f32, 2.0][..], &[][..], &[3.0][..]].iter().enumerate() {
            matrix.add_row(0, row.iter().copied()).unwrap();
            assert_indptr_invariant(&matrix, i + 1);
        }
    }

    // Each row's column indices start at the point's own offset on the global
    // timeline (floor(tinit / dt)), so rows can sit at different columns.
    #[test]
    fn starting_offset_shifts_column_indices() {
        let mut matrix = CsrMatrix::new(2, 4);
        matrix.add_row(0, [1.0f32, 2.0].into_iter()).unwrap();
        matrix.add_row(1, [3.0f32, 4.0].into_iter()).unwrap();
        assert_eq!(matrix.indices, vec![0, 1, 1, 2]);
        assert_eq!(matrix.row_ptr, vec![0, 2, 4]);
        // Widest column touched is 2, so the matrix spans 3 columns.
        assert_eq!(matrix.indices.iter().max().copied(), Some(2));
    }

    #[test]
    fn column_beyond_i32_is_rejected() {
        let mut matrix = CsrMatrix::new(1, 1);
        let starting = i32::MAX as usize + 1;
        assert_eq!(
            matrix.add_row(starting, [1.0f32].into_iter()),
            Err(CsrIndexOverflow)
        );
    }

    #[test]
    fn rows_yields_exactly_the_added_rows() {
        let matrix = build(&[&[1.0, 2.0], &[], &[3.0]]);
        assert_eq!(row_values(&matrix), vec![vec![1.0, 2.0], vec![], vec![3.0]]);
        let indices: Vec<&[i32]> = matrix.rows().map(|row| row.indices).collect();
        assert_eq!(indices, vec![&[0, 1][..], &[][..], &[0][..]]);
    }

    // The bug this guards: a trailing empty phantom row, previously produced by
    // iterating a finalised row_ptr and masked downstream by zip().
    #[test]
    fn finalise_does_not_change_the_rows() {
        let mut matrix = build(&[&[1.0, 2.0], &[3.0]]);
        let before = row_values(&matrix);
        let row_ptr_before = matrix.row_ptr.clone();
        matrix.finalise();
        assert_eq!(before, row_values(&matrix));
        assert_eq!(matrix.row_ptr, row_ptr_before);
    }

    #[test]
    fn exact_size_hint_matches_rows_produced() {
        let matrix = build(&[&[1.0, 2.0], &[], &[3.0]]);
        let mut iter = matrix.rows();
        for expected in (0..=3).rev() {
            assert_eq!(iter.len(), expected);
            assert_eq!(iter.size_hint(), (expected, Some(expected)));
            if iter.next().is_none() {
                break;
            }
        }
    }

    fn samples(indices: &[i32], data: &[f32], start: usize) -> Result<Vec<f32>, SlipRowError> {
        CsrRow { indices, data }
            .samples(start)
            .map(|samples| samples.collect())
    }

    // The bug this guards: rows without explicit zeros (e.g. scipy's
    // csr_array(dense)) were written packed together, shifting them earlier
    // in time on re-read.
    #[test]
    fn samples_fill_leading_and_interior_zeros() {
        assert_eq!(
            samples(&[6, 8], &[1.0, 2.0], 5),
            Ok(vec![0.0, 1.0, 0.0, 2.0])
        );
    }

    #[test]
    fn samples_of_dense_row_are_its_values() {
        assert_eq!(
            samples(&[5, 6, 7], &[1.0, 2.0, 3.0], 5),
            Ok(vec![1.0, 2.0, 3.0])
        );
    }

    #[test]
    fn samples_of_empty_row_are_empty() {
        let row = CsrRow {
            indices: &[],
            data: &[],
        };
        assert_eq!(row.samples(5).unwrap().len(), 0);
    }

    #[test]
    fn samples_length_is_exact() {
        let row = CsrRow {
            indices: &[6, 8],
            data: &[1.0, 2.0],
        };
        assert_eq!(row.samples(5).unwrap().len(), 4);
    }

    #[test]
    fn samples_before_start_are_rejected() {
        assert_eq!(
            samples(&[4, 5], &[1.0, 2.0], 5),
            Err(SlipRowError::BeforeStart {
                column: 4,
                start: 5
            })
        );
        assert_eq!(
            samples(&[-1], &[1.0], 0),
            Err(SlipRowError::BeforeStart {
                column: -1,
                start: 0
            })
        );
    }

    #[test]
    fn samples_of_non_canonical_row_are_rejected() {
        assert_eq!(
            samples(&[7, 5], &[1.0, 2.0], 5),
            Err(SlipRowError::NonCanonical)
        );
        assert_eq!(
            samples(&[5, 5], &[1.0, 2.0], 5),
            Err(SlipRowError::NonCanonical)
        );
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
pub struct SrfFile<S = Vec<f32>, R = Vec<i32>> {
    pub planes: Vec<SrfPlane>,
    pub metadata: SrfMetadataVersioned<S>,
    pub slipt1: CsrMatrix<R, S>,
}

pub type SrfFileView<'a> = SrfFile<&'a [f32], &'a [i32]>;

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
