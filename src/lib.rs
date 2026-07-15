pub mod pytypes;
mod scanner;
mod srf_parser;
mod srf_writer;
mod types;

use lexical_core::BUFFER_SIZE;
use memmap::MmapOptions;
use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::error;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Error, Write};

use crate::pytypes::{PyCsrMatrix, PySrfFile, PySrfMetadata, PySrfPlane};
use crate::types::{
    CsrMatrixView, SrfFileView, SrfMetadataV2View, SrfMetadataVersioned, SrfMetadataView, SrfPlane,
};

fn marshall_os_error<T>(e: Error) -> PyResult<T> {
    Err(PyErr::new::<PyOSError, _>(e.to_string()))
}

fn marshall_value_error<T, U: error::Error>(e: U) -> PyResult<T> {
    Err(PyErr::new::<PyValueError, _>(e.to_string()))
}

#[pyfunction]
pub fn parse_srf<'py>(py: Python<'py>, file_path: &str) -> PyResult<Py<PySrfFile>> {
    let file = File::open(file_path).or_else(marshall_os_error)?;
    let mmap = unsafe { MmapOptions::new().map(&file) }.or_else(marshall_os_error)?;
    let mut scanner = scanner::Scanner::new(&mmap);
    let srf_file = srf_parser::read_srf_struct(&mut scanner).or_else(marshall_value_error)?;
    Ok(srf_file.into_pyobject(py)?.unbind())
}

#[pyfunction]
pub fn write_srf(py: Python<'_>, py_srf_file: Py<PySrfFile>, file_path: &str) -> PyResult<()> {
    let srf = py_srf_file.borrow(py);
    let metadata = srf.metadata.borrow(py);
    let slipt1 = srf.slipt1.borrow(py);

    let planes: Vec<SrfPlane> = srf
        .planes
        .iter()
        .map(|plane| SrfPlane::from(&*plane.borrow(py)))
        .collect();

    // Readonly borrows of the numpy buffers; the slices below view them
    // directly, so these guards must outlive the write call.
    let lon = metadata.lon.bind(py).readonly();
    let lat = metadata.lat.bind(py).readonly();
    let dep = metadata.dep.bind(py).readonly();
    let stk = metadata.stk.bind(py).readonly();
    let dip = metadata.dip.bind(py).readonly();
    let area = metadata.area.bind(py).readonly();
    let tinit = metadata.tinit.bind(py).readonly();
    let dt = metadata.dt.bind(py).readonly();
    let rake = metadata.rake.bind(py).readonly();
    let slip1 = metadata.slip1.bind(py).readonly();
    let rise = metadata.rise.bind(py).readonly();
    let vs = metadata.vs.as_ref().map(|arr| arr.bind(py).readonly());
    let density = metadata.density.as_ref().map(|arr| arr.bind(py).readonly());
    let row_ptr = slipt1.row_ptr.bind(py).readonly();
    let data = slipt1.data.bind(py).readonly();

    let base: SrfMetadataView = SrfMetadataView {
        lon: lon.as_slice()?,
        lat: lat.as_slice()?,
        dep: dep.as_slice()?,
        stk: stk.as_slice()?,
        dip: dip.as_slice()?,
        area: area.as_slice()?,
        tinit: tinit.as_slice()?,
        dt: dt.as_slice()?,
        rake: rake.as_slice()?,
        slip1: slip1.as_slice()?,
        rise: rise.as_slice()?,
    };

    let metadata_view = match (&vs, &density) {
        (Some(vs), Some(density)) => SrfMetadataVersioned::V2(SrfMetadataV2View {
            base,
            vs: vs.as_slice()?,
            density: density.as_slice()?,
        }),
        (None, None) => SrfMetadataVersioned::V1(base),
        _ => {
            return Err(PyErr::new::<PyValueError, _>(
                "vs and density must both be set (SRF v2) or both be None (SRF v1)",
            ))
        }
    };

    let srf_view: SrfFileView = SrfFileView {
        planes,
        metadata: metadata_view,
        slipt1: CsrMatrixView {
            row_ptr: row_ptr.as_slice()?,
            data: data.as_slice()?,
        },
    };

    let file = File::create(file_path).or_else(marshall_os_error)?;
    let mut writer = BufWriter::new(file);
    srf_writer::write_srf(&mut writer, &srf_view).or_else(marshall_os_error)?;
    writer.flush().or_else(marshall_os_error)
}

#[pyfunction]
pub fn write_srf_points(
    _py: Python<'_>,
    file_path: &str,
    points_metadata: PyReadonlyArray2<f32>,
    row_ptr: PyReadonlyArray1<usize>,
    data: PyReadonlyArray1<f32>,
) -> PyResult<()> {
    let file = OpenOptions::new()
        .append(true)
        .open(file_path)
        .or_else(marshall_os_error)?;
    let mut buffered_writer = BufWriter::new(file);
    let metadata_array = points_metadata.as_array();
    let row_array = row_ptr.as_slice()?;
    let data_array = data.as_slice()?;
    let mut buffer = [0u8; BUFFER_SIZE];
    let summary_length = 8;

    for (i, row) in metadata_array.outer_iter().enumerate() {
        for v in row.iter().take(summary_length) {
            let slice = lexical_core::write(*v, &mut buffer);
            buffered_writer
                .write_all(slice)
                .or_else(marshall_os_error)?;
            buffered_writer.write_all(b" ").or_else(marshall_os_error)?;
        }
        buffered_writer
            .write_all(b"\n")
            .or_else(marshall_os_error)?;
        for v in row.iter().skip(summary_length) {
            let slice = lexical_core::write(*v, &mut buffer);
            buffered_writer
                .write_all(slice)
                .or_else(marshall_os_error)?;
            buffered_writer.write_all(b" ").or_else(marshall_os_error)?;
        }

        let row_idx = row_array[i];
        let next_row_idx = row_array.get(i + 1).copied().unwrap_or(row_idx);
        let nt = next_row_idx - row_idx;
        let slice = lexical_core::write(nt, &mut buffer);
        buffered_writer
            .write_all(slice)
            .or_else(marshall_os_error)?;
        buffered_writer
            .write_all(b" 0.0 0 0.0 0")
            .or_else(marshall_os_error)?;
        if nt > 0 {
            buffered_writer
                .write_all(b"\n")
                .or_else(marshall_os_error)?;
            for v in &data_array[row_idx..next_row_idx] {
                let slice = lexical_core::write(*v, &mut buffer);
                buffered_writer
                    .write_all(slice)
                    .or_else(marshall_os_error)?;
                buffered_writer.write_all(b" ").or_else(marshall_os_error)?;
            }
        }
        buffered_writer
            .write_all(b"\n")
            .or_else(marshall_os_error)?;
    }

    buffered_writer.flush().or_else(marshall_os_error)?;
    Ok(())
}

#[pymodule]
fn srf_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySrfPlane>()?;
    m.add_class::<PyCsrMatrix>()?;
    m.add_class::<PySrfMetadata>()?;
    m.add_class::<PySrfFile>()?;
    m.add_function(wrap_pyfunction!(write_srf_points, m)?)?;
    m.add_function(wrap_pyfunction!(write_srf, m)?)?;
    m.add_function(wrap_pyfunction!(parse_srf, m)?)?;
    Ok(())
}
