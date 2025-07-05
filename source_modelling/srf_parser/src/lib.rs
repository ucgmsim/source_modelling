use memmap::MmapOptions;
use numpy::PyArray1;
use numpy::PyArray2;
use pyo3::exceptions::PyOSError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use std::fs::File;
use std::io::Error;

#[derive(Default)]
struct SparseMatrix {
    row_ptr: Vec<usize>,
    col_ptr: Vec<usize>,
    data: Vec<f32>,
}

impl SparseMatrix {
    fn into_csr_matrix(&self, py: Python<'_>) -> PyResult<PyObject> {
        if self.data.len() == 0 {
            return Ok(py.None());
        }
        let data = PyArray1::from_vec(py, self.data.clone());
        let indices = PyArray1::from_vec(py, self.col_ptr.clone());
        let indptr = PyArray1::from_vec(py, self.row_ptr.clone());

        let csr = py
            .import("scipy.sparse")?
            .getattr("csr_array")?
            .call1(((&data, &indices, &indptr),))?;
        Ok(csr.to_owned().into())
    }
}

#[derive(Default)]
struct Metadata {
    lat: Vec<f32>,
    lon: Vec<f32>,
    dep: Vec<f32>,
    stk: Vec<f32>,
    dip: Vec<f32>,
    area: Vec<f32>,
    tinit: Vec<f32>,
    dt: Vec<f32>,
    rake: Vec<f32>,
    slip1: Vec<f32>,
    rise_time: Vec<f32>,
}

fn marshall_os_error<T>(e: Error) -> PyResult<T> {
    Err(PyErr::new::<PyOSError, _>(e.to_string()))
}

fn marshall_value_error<T>(e: lexical_core::Error) -> PyResult<T> {
    Err(PyErr::new::<PyValueError, _>(e.to_string()))
}

fn space_index(data: &[u8]) -> Result<usize, lexical_core::Error> {
    let nonwhitespace = data
        .iter()
        .enumerate()
        .find(|&(_, &x)| x != 32 && x != 10)
        .map(|(idx, _)| idx);
    match nonwhitespace {
        Some(x) => Ok(x),
        _ => Err(lexical_core::Error::InvalidDigit(0)),
    }
}

fn parse_value<T: lexical_core::FromLexical>(
    data: &[u8],
    index: &mut usize,
) -> Result<T, lexical_core::Error> {
    *index += space_index(&data[*index..])?;
    let (val, read) = lexical_core::parse_partial(&data[*index..])?;
    *index += read;
    Ok(val)
}

fn read_srf_points(
    data: &[u8],
    point_count: usize,
) -> Result<(Metadata, SparseMatrix), lexical_core::Error> {
    let mut index: usize = 0;
    let mut metadata = Metadata::default();
    let mut slipt1 = SparseMatrix::default();

    for _ in 0..point_count {
        let lat = parse_value::<f32>(data, &mut index)?;
        metadata.lat.push(lat);

        let lon = parse_value::<f32>(data, &mut index)?;
        metadata.lon.push(lon);

        let dep = parse_value::<f32>(data, &mut index)?;
        metadata.dep.push(dep);

        let stk = parse_value::<f32>(data, &mut index)?;
        metadata.stk.push(stk);

        let dip = parse_value::<f32>(data, &mut index)?;
        metadata.dip.push(dip);

        let area = parse_value::<f32>(data, &mut index)?;
        metadata.area.push(area);

        let tinit = parse_value::<f32>(data, &mut index)?;
        metadata.tinit.push(tinit);

        let dt = parse_value::<f32>(data, &mut index)?;
        metadata.dt.push(dt);

        let rake = parse_value::<f32>(data, &mut index)?;
        metadata.rake.push(rake);

        let slip1 = parse_value::<f32>(data, &mut index)?;
        metadata.slip1.push(slip1);

        let nt = parse_value::<usize>(data, &mut index)?;

        let _nt2 = parse_value::<f32>(data, &mut index)?;
        let _slip2 = parse_value::<i64>(data, &mut index)?;
        let _slip3 = parse_value::<f32>(data, &mut index)?;
        let _nt3 = parse_value::<i64>(data, &mut index)?;

        metadata.rise_time.push((nt as f32) * dt);

        let start_column_index: usize = (tinit / dt).floor() as usize;
        slipt1.row_ptr.push(slipt1.data.len());
        for i in start_column_index..start_column_index + nt {
            let slip = parse_value::<f32>(data, &mut index)?;
            slipt1.col_ptr.push(i);
            slipt1.data.push(slip);
        }
    }
    slipt1.row_ptr.push(slipt1.data.len());
    Ok((metadata, slipt1))
}

#[pyfunction]
fn parse_srf(
    py: Python<'_>,
    file_path: &str,
    offset: usize,
    num_points: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let file = File::open(file_path).or_else(marshall_os_error)?;
    let mmap = unsafe { MmapOptions::new().map(&file) }.or_else(marshall_os_error)?;
    let (metadata, sparse_matrix) =
        read_srf_points(&mmap[offset..], num_points).or_else(marshall_value_error)?;

    let metadata_array = PyArray2::from_vec2(
        py,
        &[
            metadata.lat,
            metadata.lon,
            metadata.dep,
            metadata.stk,
            metadata.dip,
            metadata.area,
            metadata.tinit,
            metadata.dt,
            metadata.rake,
            metadata.slip1,
            metadata.rise_time,
        ],
    )?;
    let csr = sparse_matrix.into_csr_matrix(py)?;
    Ok((metadata_array.to_owned().into(), csr))
}

#[pymodule]
fn srf_parser(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_srf, m)?)?;
    Ok(())
}
