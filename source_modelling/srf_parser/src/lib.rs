use lexical_core::BUFFER_SIZE;
use memmap::MmapOptions;
use numpy::PyArray1;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyOSError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use std::fs::File;
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::io::Error;
use std::io::Write;

#[derive(Default)]
struct SparseMatrix {
    row_ptr: Vec<i64>,
    col_ptr: Vec<i64>,
    data: Vec<f32>,
}

impl SparseMatrix {
    fn into_csr_matrix(self, py: Python<'_>) -> PyResult<PyObject> {
        if self.data.is_empty() {
            return Ok(py.None());
        }
        let data = PyArray1::from_vec(py, self.data);
        let indices = PyArray1::from_vec(py, self.col_ptr);
        let indptr = PyArray1::from_vec(py, self.row_ptr);

        let csr = py
            .import("scipy.sparse")?
            .getattr("csr_array")?
            .call1(((&data, &indices, &indptr),))?;
        Ok(csr.to_owned().into())
    }
}


struct Scanner<'a> {
    data: &'a [u8],
    index: usize,
}

impl<'a> Scanner<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, index: 0 }
    }

    fn next<T: lexical_core::FromLexical>(
        &mut self,
    ) -> Result<T, lexical_core::Error> {
        self.skip_spaces()?;
        let (val, read) = lexical_core::parse_partial(&self.data[self.index..]).map_err(|err| match err {
            lexical_core::Error::Overflow(offset) => lexical_core::Error::Overflow(self.index + offset),
            lexical_core::Error::Underflow(offset) => lexical_core::Error::Underflow(self.index + offset),
            lexical_core::Error::InvalidDigit(offset) => lexical_core::Error::InvalidDigit(self.index + offset),
            lexical_core::Error::Empty(offset) => lexical_core::Error::Empty(self.index + offset),
            lexical_core::Error::EmptyMantissa(offset) => {
                    let context = &self.data[self.index + offset..self.data.len().min(self.index + offset + 20)];
                    eprintln!(
                        "lexical error at index {}: EmptyMantissa\ncontext: {:?}",
                        self.index + offset,
                        String::from_utf8_lossy(context)
                    );
                lexical_core::Error::EmptyMantissa(self.index + offset)},
            lexical_core::Error::EmptyExponent(offset) => lexical_core::Error::EmptyExponent(self.index + offset),
            lexical_core::Error::EmptyInteger(offset) => lexical_core::Error::EmptyInteger(self.index + offset),
            lexical_core::Error::EmptyFraction(offset) => lexical_core::Error::EmptyFraction(self.index + offset),
            lexical_core::Error::InvalidPositiveMantissaSign(offset) => lexical_core::Error::InvalidPositiveMantissaSign(self.index + offset),
            lexical_core::Error::MissingMantissaSign(offset) => lexical_core::Error::MissingMantissaSign(self.index + offset),
            lexical_core::Error::InvalidExponent(offset) => lexical_core::Error::InvalidExponent(self.index + offset),
            lexical_core::Error::InvalidPositiveExponentSign(offset) => lexical_core::Error::InvalidPositiveExponentSign(self.index + offset),
            lexical_core::Error::MissingExponentSign(offset) => lexical_core::Error::MissingExponentSign(self.index + offset),
            lexical_core::Error::ExponentWithoutFraction(offset) => lexical_core::Error::ExponentWithoutFraction(self.index + offset),
            lexical_core::Error::InvalidLeadingZeros(offset) => lexical_core::Error::InvalidLeadingZeros(self.index + offset),
            lexical_core::Error::MissingExponent(offset) => lexical_core::Error::MissingExponent(self.index + offset),
            lexical_core::Error::MissingSign(offset) => lexical_core::Error::MissingSign(self.index + offset),
            lexical_core::Error::InvalidPositiveSign(offset) => lexical_core::Error::InvalidPositiveSign(self.index + offset),
            lexical_core::Error::InvalidNegativeSign(offset) => lexical_core::Error::InvalidNegativeSign(self.index + offset),
            e => e
        })?;
        self.index += read;
        Ok(val)
    }

    fn skip_spaces(&mut self) -> Result<(), lexical_core::Error> {
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
            _ => Err(lexical_core::Error::InvalidDigit(self.index)),
        }
    }

    fn reset(&mut self) {
        self.index = 0;
    }
}


fn marshall_os_error<T>(e: Error) -> PyResult<T> {
    Err(PyErr::new::<PyOSError, _>(e.to_string()))
}

fn marshall_value_error<T>(e: lexical_core::Error) -> PyResult<T> {
    Err(PyErr::new::<PyValueError, _>(e.to_string()))
}



fn estimate_slipt1_array_size(
    scanner: &mut Scanner,
    point_count: usize,
) -> Result<usize, lexical_core::Error> {
    // Sample size is the minimum of 500 and point count
    let sample_size = 500.min(point_count);
    let mut total_slip_samples: usize = 0;

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
    scanner.reset();
    Ok((point_count as f64 * avg_slip).ceil() as usize)
}

fn read_srf_points(
    data: &[u8],
    point_count: usize,
) -> Result<(Vec<f32>, Vec<usize>, Vec<f32>), lexical_core::Error> {
    let mut scanner = Scanner::new(data);
    let mut metadata = Vec::with_capacity(point_count * 11);
    let mut row_ptr = Vec::with_capacity(point_count);
    let slipt1_capacity = estimate_slipt1_array_size(&mut scanner, point_count)?;
    let mut slipt1 = Vec::with_capacity(slipt1_capacity);

    for i in 0..point_count {
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

#[pyfunction]
fn parse_srf(
    py: Python<'_>,
    file_path: &str,
    offset: usize,
    num_points: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    use numpy::PyArray1;

    let file = File::open(file_path).or_else(marshall_os_error)?;
    let mmap = unsafe { MmapOptions::new().map(&file) }.or_else(marshall_os_error)?;

    let (metadata, row_ptr, slipt1) =
        read_srf_points(&mmap[offset..], num_points).or_else(marshall_value_error)?;

    let metadata_array = PyArray1::from_vec(py, metadata);
    let row_ptr_array = PyArray1::from_vec(py, row_ptr);
    let slipt1_array = PyArray1::from_vec(py, slipt1);

    Ok((
        metadata_array.to_owned().into(),
        row_ptr_array.to_owned().into(),
        slipt1_array.to_owned().into(),
    ))
}
#[pyfunction]
fn write_srf_points(
    _py: Python<'_>,
    file_path: &str,
    points_metadata: PyReadonlyArray2<f32>,
    row_ptr: PyReadonlyArray1<i64>,
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
    m.add_function(wrap_pyfunction!(parse_srf, m)?)?;
    m.add_function(wrap_pyfunction!(write_srf_points, m)?)?;
    Ok(())
}
