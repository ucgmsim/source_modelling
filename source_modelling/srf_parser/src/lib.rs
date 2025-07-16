use lexical_core::BUFFER_SIZE;
use memmap::MmapOptions;
use numpy::PyArray1;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyOSError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3::wrap_pyfunction;

use std::error;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::io::Error;
use std::io::Write;

mod scanner;

struct SrfPlane {
    elon: f32,
    elat: f32,
    nstk: usize,
    ndip: usize,
    len: f32,
    wid: f32,
    stk: f32,
    dip: f32,
    dtop: f32,
    shyp: f32,
    dhyp: f32,
}

struct SrfFile {
    version: String,
    planes: Vec<SrfPlane>,
    metadata: Vec<f32>,
    row_ptr: Vec<usize>,
    slipt1: Vec<f32>,
}

fn read_srf_header(
    scanner: &mut scanner::Scanner,
    plane_count: usize,
) -> Result<Vec<SrfPlane>, scanner::ScannerError> {
    let mut plane_vec = Vec::new();
    for _ in 0..plane_count {
        let elon = scanner.next()?;
        let elat = scanner.next()?;
        let nstk = scanner.next()?;
        let ndip = scanner.next()?;
        let len = scanner.next()?;
        let wid = scanner.next()?;
        let stk = scanner.next()?;
        let dip = scanner.next()?;
        let dtop = scanner.next()?;
        let shyp = scanner.next()?;
        let dhyp = scanner.next()?;
        plane_vec.push(SrfPlane {
            elon,
            elat,
            nstk,
            ndip,
            len,
            wid,
            stk,
            dip,
            dtop,
            shyp,
            dhyp,
        })
    }
    return Ok(plane_vec);
}

fn estimate_slipt1_array_size(
    scanner: &mut scanner::Scanner,
    point_count: usize,
) -> Result<usize, scanner::ScannerError> {
    // Sample size is the minimum of 500 and point count
    let sample_size = 500.min(point_count);
    let mut total_slip_samples: usize = 0;
    let index = scanner.index;
    for _ in 0..sample_size {
        // The first 10 floats are unused
        for _ in 0..10 {
            let _: f32 = scanner.next()?;
        }
        // Extract nt
        let nt: usize = scanner.next()?;
        total_slip_samples += nt;
        // Skip rest of unused values and the actual slip data
        for _ in 0..(nt + 4) {
            let _: f32 = scanner.next()?;
        }
    }
    let avg_slip = (total_slip_samples as f64) / (sample_size as f64);
    scanner.index = index;
    Ok((point_count as f64 * avg_slip).ceil() as usize)
}

fn read_srf_struct(scanner: &mut scanner::Scanner) -> Result<SrfFile, scanner::ScannerError> {
    let version = String::from_utf8_lossy(scanner.line()?).into_owned();
    scanner.skip_token(b"PLANE")?;
    let plane_count: usize = scanner.next()?;
    let header = read_srf_header(scanner, plane_count)?;
    scanner.skip_token(b"POINTS")?;

    let point_count = scanner.next()?;
    let mut metadata = Vec::with_capacity(point_count * 11);
    let mut row_ptr = Vec::with_capacity(point_count);
    let slipt1_capacity = estimate_slipt1_array_size(scanner, point_count)?;
    let mut slipt1 = Vec::with_capacity(slipt1_capacity);

    for _ in 0..point_count {
        metadata.push(scanner.next()?); // lon
        metadata.push(scanner.next()?); // lat
        metadata.push(scanner.next()?); // dep
        metadata.push(scanner.next()?); // stk
        metadata.push(scanner.next()?); // dip
        metadata.push(scanner.next()?); // area
        metadata.push(scanner.next()?); // tinit
        let dt = scanner.next()?;
        metadata.push(dt);
        metadata.push(scanner.next()?); // rake
        metadata.push(scanner.next()?); // slip1
        let nt = scanner.next::<usize>()?;
        let _slip2 = scanner.next::<f32>()?;
        let _nt2 = scanner.next::<usize>()?;
        let _slip3 = scanner.next::<f32>()?;
        let _nt3 = scanner.next::<usize>()?;
        metadata.push((nt as f32) * dt);

        row_ptr.push(slipt1.len());
        for _ in 0..nt {
            slipt1.push(scanner.next()?);
        }
    }
    Ok(SrfFile {
        version: version,
        planes: header,
        metadata: metadata,
        row_ptr: row_ptr,
        slipt1: slipt1,
    })
}

#[pyfunction]
fn parse_srf(
    py: Python<'_>,
    file_path: &str,
) -> PyResult<(Py<PyString>, Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let file = File::open(file_path).or_else(marshall_os_error)?;
    let mmap = unsafe { MmapOptions::new().map(&file) }.or_else(marshall_os_error)?;
    let mut scanner = scanner::Scanner::new(&mmap);
    let srf_file = read_srf_struct(&mut scanner).or_else(marshall_value_error)?;

    let metadata = PyArray1::from_vec(py, srf_file.metadata);
    let row_ptr = PyArray1::from_vec(py, srf_file.row_ptr);
    let slipt1 = PyArray1::from_vec(py, srf_file.slipt1);
    let mut header: Vec<f32> = Vec::new();
    for plane in &srf_file.planes {
        header.extend_from_slice(&[
            plane.elon,
            plane.elat,
            plane.nstk as f32,
            plane.ndip as f32,
            plane.len,
            plane.wid,
            plane.stk,
            plane.dip,
            plane.dtop,
            plane.shyp,
            plane.dhyp,
        ]);
    }
    let header_array = PyArray1::from_vec(py, header);

    Ok((
        PyString::new(py, &srf_file.version).to_owned().into(),
        header_array.to_owned().into(),
        metadata.to_owned().into(),
        row_ptr.to_owned().into(),
        slipt1.to_owned().into(),
    ))
}

fn marshall_os_error<T>(e: Error) -> PyResult<T> {
    Err(PyErr::new::<PyOSError, _>(e.to_string()))
}

fn marshall_value_error<T, U: error::Error>(e: U) -> PyResult<T> {
    Err(PyErr::new::<PyValueError, _>(e.to_string()))
}

#[pyfunction]
fn write_srf_points(
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

    for (i, row) in metadata_array.outer_iter().enumerate() {
        // Write all but last element
        for v in row.iter().take(row.len() - 1) {
            let slice = lexical_core::write(*v, &mut buffer);
            buffered_writer
                .write_all(slice)
                .or_else(marshall_os_error)?;
            buffered_writer.write_all(b" ").or_else(marshall_os_error)?;
        }

        let row_idx = row_array[i] as usize;
        let next_row_idx = row_array.get(i + 1).map(|&x| x as usize).unwrap_or(row_idx);
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
fn srf_parser(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(write_srf_points, m)?)?;
    m.add_function(wrap_pyfunction!(parse_srf, m)?)?;
    Ok(())
}
