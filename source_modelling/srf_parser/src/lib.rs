use lexical_core::Error::*;
use lexical_core::BUFFER_SIZE;
use memmap::MmapOptions;
use numpy::PyArray1;
use numpy::PyArray2;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyOSError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use std::error;
use std::fmt;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::io::Error;
use std::io::Write;

fn read_srf_points(
    scanner: &mut Scanner,
    point_count: usize,
) -> Result<(Vec<f32>, Vec<usize>, Vec<f32>), ScannerError> {
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

    Ok((metadata, row_ptr, slipt1))
}

pub struct SrfPlane {
    elon: f32,
    elat: f32,
    nstk: usize,
    ndip: usize,
    len: f32,
    wid: f32,
    dip: f32,
    dtop: f32,
    shyp: f32,
    dhyp: f32,
}

fn read_srf_header(
    scanner: &mut Scanner,
    plane_count: usize,
) -> Result<Vec<SrfPlane>, ScannerError> {
    let mut plane_vec = Vec::new();
    for _ in 0..plane_count {
        let elon = scanner.next()?;
        let elat = scanner.next()?;
        let nstk = scanner.next()?;
        let ndip = scanner.next()?;
        let len = scanner.next()?;
        let wid = scanner.next()?;
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
            dip,
            dtop,
            shyp,
            dhyp,
        })
    }
    return Ok(plane_vec);
}

#[pyclass(subclass)]
pub struct SrfFileBackend {
    header: Vec<SrfPlane>,

    // Corresponds to the 'points' DataFrame in the Python SrfFile.
    // Stores the metadata for each point (lon, lat, dep, stk, dip, area, tinit, dt, rake, slip1, total_time).
    metadata: Vec<f32>,

    // Corresponds to the 'row_ptr' numpy array in the Python SrfFile.
    // Stores the starting index in the 'data' array for each point's slip data.
    row_ptr: Vec<usize>,

    // Corresponds to the 'slipt1_array' numpy array in the Python SrfFile.
    // Stores the actual slip data for all points concatenated.
    data: Vec<f32>,
}

#[pymethods]
impl SrfFileBackend {
    /// Reads an SRF file from the given path and returns an SrfFileBackend instance.
    /// This method leverages the `parse_srf` function to handle the file parsing.
    #[staticmethod]
    fn from_file(py: Python<'_>, file_path: &str) -> PyResult<Self> {
        // parse_srf now returns raw Vecs directly
        let file = File::open(file_path).or_else(marshall_os_error)?;
        let mmap = unsafe { MmapOptions::new().map(&file) }.or_else(marshall_os_error)?;
        let mut scanner = Scanner::new(&mmap);
        let version = scanner.line().or_else(marshall_value_error)?;
        scanner.skip_token(b"PLANE").or_else(marshall_value_error)?;
        let plane_count: usize = scanner.next().or_else(marshall_value_error)?;
        let header = read_srf_header(&mut scanner, plane_count).or_else(marshall_value_error)?;
        scanner
            .skip_token(b"POINTS")
            .or_else(marshall_value_error)?;
        let num_points = scanner.next().or_else(marshall_value_error)?;
        let (metadata, row_ptr, data) =
            read_srf_points(&mut scanner, num_points).or_else(marshall_value_error)?;

        Ok(SrfFileBackend {
            header,
            metadata,
            row_ptr,
            data,
        })
    }

    #[getter]
    fn header(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut header: Vec<f32> = Vec::new();
        for plane in &self.header {
            header.extend_from_slice(&[
                plane.elon,
                plane.elat,
                plane.nstk as f32,
                plane.ndip as f32,
                plane.len,
                plane.wid,
                plane.dip,
                plane.dtop,
                plane.shyp,
                plane.dhyp,
            ]);
        }
        let py_array = PyArray1::from_vec(py, header);
        Ok(py_array.to_owned().into())
    }

    /// Returns a copy of the metadata as a 2D NumPy array.
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let num_points = self.row_ptr.len();
        // Create a new PyArray2 from the metadata Vec
        let py_array = PyArray1::from_vec(py, self.metadata.clone());
        Ok(py_array.to_owned().into())
    }

    /// Returns a copy of the row_ptr as a 1D NumPy array.
    #[getter]
    fn row_ptr(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Create a new PyArray1 from the row_ptr Vec
        let py_array = PyArray1::from_vec(py, self.row_ptr.clone());
        Ok(py_array.to_owned().into())
    }

    /// Returns a copy of the data (slipt1_array) as a 1D NumPy array.
    #[getter]
    fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Create a new PyArray1 from the data Vec
        let py_array = PyArray1::from_vec(py, self.data.clone());
        Ok(py_array.to_owned().into())
    }
}

#[derive(Debug)]
struct ScannerError {
    context: String,
    error: Box<dyn error::Error>,
}

impl ScannerError {
    fn new(data: &[u8], error: impl error::Error + 'static) -> Self {
        Self {
            context: String::from_utf8_lossy(data).into_owned(),
            error: Box::new(error),
        }
    }
}

impl fmt::Display for ScannerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, context: {}", self.error, self.context)
    }
}

impl error::Error for ScannerError {}

#[derive(Debug)]
enum ScannerErrorReason {
    InvalidToken(String, String),
    NoNewlineFound,
}

impl fmt::Display for ScannerErrorReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidToken(expected, found) => {
                write!(f, "Invalid token, expected: {}, found: {}", expected, found)
            }
            Self::NoNewlineFound => write!(f, "Could not find newline"),
        }
    }
}

impl error::Error for ScannerErrorReason {}

struct Scanner<'a> {
    data: &'a [u8],
    index: usize,
}

impl<'a> Scanner<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, index: 0 }
    }

    fn next<T: lexical_core::FromLexical>(&mut self) -> Result<T, ScannerError> {
        self.skip_spaces()?;
        let (val, read) = lexical_core::parse_partial(&self.data[self.index..])
            .map_err(|err| match err {
                Overflow(offset) => Overflow(self.index + offset),
                Underflow(offset) => Underflow(self.index + offset),
                InvalidDigit(offset) => InvalidDigit(self.index + offset),
                Empty(offset) => Empty(self.index + offset),
                EmptyMantissa(offset) => EmptyMantissa(self.index + offset),
                EmptyExponent(offset) => EmptyExponent(self.index + offset),
                EmptyInteger(offset) => EmptyInteger(self.index + offset),
                EmptyFraction(offset) => EmptyFraction(self.index + offset),
                InvalidPositiveMantissaSign(offset) => {
                    InvalidPositiveMantissaSign(self.index + offset)
                }
                MissingMantissaSign(offset) => MissingMantissaSign(self.index + offset),
                InvalidExponent(offset) => InvalidExponent(self.index + offset),
                InvalidPositiveExponentSign(offset) => {
                    InvalidPositiveExponentSign(self.index + offset)
                }
                MissingExponentSign(offset) => MissingExponentSign(self.index + offset),
                ExponentWithoutFraction(offset) => ExponentWithoutFraction(self.index + offset),
                InvalidLeadingZeros(offset) => InvalidLeadingZeros(self.index + offset),
                MissingExponent(offset) => MissingExponent(self.index + offset),
                MissingSign(offset) => MissingSign(self.index + offset),
                InvalidPositiveSign(offset) => InvalidPositiveSign(self.index + offset),
                InvalidNegativeSign(offset) => InvalidNegativeSign(self.index + offset),
                e => e,
            })
            .map_err(|err| ScannerError::new(self.context(), err))?;
        self.index += read;
        Ok(val)
    }

    fn skip_spaces(&mut self) -> Result<(), ScannerError> {
        let nonwhitespace = &self.data[self.index..]
            .iter()
            .enumerate()
            .find(|&(_, &x)| !x.is_ascii_whitespace())
            .map(|(idx, _)| idx);
        match nonwhitespace {
            Some(x) => {
                self.index += x;
                Ok(())
            }
            _ => Err(ScannerError::new(self.context(), InvalidDigit(self.index))),
        }
    }

    fn line(&mut self) -> Result<&[u8], ScannerError> {
        let newline_index = &self.data[self.index..]
            .iter()
            .enumerate()
            .find(|&(_, &x)| x == b'\n')
            .map(|(idx, _)| idx);
        match newline_index {
            Some(x) => {
                let res = Ok(&self.data[self.index..self.index + x]);
                self.index += x + 1; // plus 1 to skip the newline itself.
                res
            }
            _ => Err(ScannerError::new(
                self.context(),
                ScannerErrorReason::NoNewlineFound,
            )),
        }
    }

    fn skip_token(&mut self, token: &[u8]) -> Result<(), ScannerError> {
        self.skip_spaces()?;
        let next = &self.data[self.index..self.index + token.len()];
        if next == token {
            self.index += token.len();
            Ok(())
        } else {
            Err(ScannerError::new(
                self.context(),
                ScannerErrorReason::InvalidToken(
                    String::from_utf8_lossy(token).into_owned(),
                    String::from_utf8_lossy(next).into_owned(),
                ),
            ))
        }
    }

    fn context(&self) -> &[u8] {
        return &self.data[self.index..(self.index + 20).min(self.data.len())];
    }

    fn reset(&mut self) {
        self.index = 0;
    }
}

fn marshall_os_error<T>(e: Error) -> PyResult<T> {
    Err(PyErr::new::<PyOSError, _>(e.to_string()))
}

fn marshall_value_error<T, U: error::Error>(e: U) -> PyResult<T> {
    Err(PyErr::new::<PyValueError, _>(e.to_string()))
}

fn estimate_slipt1_array_size(
    scanner: &mut Scanner,
    point_count: usize,
) -> Result<usize, ScannerError> {
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
    m.add_class::<SrfFileBackend>()?;
    m.add_function(wrap_pyfunction!(write_srf_points, m)?)?;
    Ok(())
}
