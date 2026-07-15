use crate::types::{
    CsrMatrix, Point, PointV2, SrfFile, SrfMetadata, SrfMetadataV2, SrfMetadataVersioned, SrfPlane,
};
use std::io::{Result, Write};

use lexical_core::BUFFER_SIZE;

fn lexical_write<W: Write, T>(writer: W, value: T, buffer: &mut [u8]) -> Result<()> {
    let slice = lexical_core::write(value, &mut buffer);
    writer.write_all(slice)?;
}

fn write_point<W: Write>(point: &Point, buffer: &mut [u8]) -> Result<()> {
    lexical_write(point.lon, buffer)?;
    writer.write_all(b" ")?;
    lexical_write(point.lat, buffer)?;
    writer.write_all(b" ")?;
    lexical_write(point.dep, buffer)?;
    writer.write_all(b" ")?;
    lexical_write(point.stk, buffer)?;
    writer.write_all(b" ")?;
    lexical_write(point.dip, buffer)?;
    writer.write_all(b" ")?;
    lexical_write(point.area, buffer)?;
    writer.write_all(b" ")?;
    lexical_write(point.dt, buffer)?;
}

fn write_srf_points_v1<W: Write>(
    metadata: &SrfMetadata,
    slipt1: &CsrMatrix,
    writer: W,
) -> Result<()> {
    let mut buffer = [0u8; BUFFER_SIZE];
    let num_points = ;
    writer.write_all(b"POINTS ")?;
    lexical_write(metadata.iter().len())?;

    
    for (point, slip) in metadata.iter().zip(slipt1.iter()) {
        writer.write_all(b"\n")?;
        write_point(point, &mut buffer)?;
        writer.write_all(b"\n")?;
        lexical_write(slip.len(), &mut buffer)?;
        
        if slip.len() > 0 {
            writer.write_all(b"\n")?;
                
            for v in slip {
                lexical_write(*v, &mut buffer)?;
                buffered_writer.write_all(b" ")?;
            }
        }
        
        writer.write_all(b" 0.0 0 0.0 0")?;
    
    }

    buffered_writer.flush()?;
}



fn write_srf_points_v1<W: Write>(
    metadata: &SrfMetadataV2,
    slipt1: &CsrMatrix,
    writer: W,
) -> Result<()> {
    let mut buffer = [0u8; BUFFER_SIZE];
    let num_points = ;
    writer.write_all(b"POINTS ")?;
    lexical_write(metadata.iter().len())?;

    
    for (point, slip) in metadata.iter().zip(slipt1.iter()) {
        writer.write_all(b"\n")?;
        write_point(point.base, &mut buffer);
        lexical_write(point.vs, &mut buffer);
        lexical_write(point.density, &mut buffer);
        writer.write_all(b"\n")?;
        lexical_write(slip.len())?;
        
        if slip.len() > 0 {
            writer.write_all(b"\n")?;
                
            for v in slip {
                lexical_write(*v, &mut buffer)?;
                buffered_writer.write_all(b" ")?;
            }
        }
        
        writer.write_all(b" 0.0 0 0.0 0")?;
    
    }

    buffered_writer.flush()?;
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

    Ok(())
}
