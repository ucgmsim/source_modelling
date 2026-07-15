use crate::types::{
    CsrMatrix, Point, SrfFile, SrfMetadata, SrfMetadataV2, SrfMetadataVersioned, SrfPlane,
};
use std::io::{Result, Write};

use lexical_core::{ToLexical, BUFFER_SIZE};

fn lexical_write<W: Write, T: ToLexical>(
    writer: &mut W,
    value: T,
    buffer: &mut [u8],
) -> Result<()> {
    let slice = lexical_core::write(value, buffer);
    writer.write_all(slice)?;
    Ok(())
}

const POINTS: &[u8] = b"POINTS ";
const EMPTY_SLIP_TAIL: &[u8] = b" 0.0 0 0.0 0";

fn write_point<W: Write>(writer: &mut W, point: &Point, buffer: &mut [u8]) -> Result<()> {
    lexical_write(writer, point.lon, buffer)?;
    writer.write_all(b" ")?;
    lexical_write(writer, point.lat, buffer)?;
    writer.write_all(b" ")?;
    lexical_write(writer, point.dep, buffer)?;
    writer.write_all(b" ")?;
    lexical_write(writer, point.stk, buffer)?;
    writer.write_all(b" ")?;
    lexical_write(writer, point.dip, buffer)?;
    writer.write_all(b" ")?;
    lexical_write(writer, point.area, buffer)?;
    writer.write_all(b" ")?;
    lexical_write(writer, point.dt, buffer)?;
    Ok(())
}

fn write_slip_history<W: Write>(writer: &mut W, slip: &[f32], buffer: &mut [u8]) -> Result<()> {
    lexical_write(writer, slip.len(), buffer)?;

    if slip.len() > 0 {
        writer.write_all(b"\n")?;

        for v in slip {
            lexical_write(writer, *v, buffer)?;
            writer.write_all(b" ")?;
        }
    }

    writer.write_all(EMPTY_SLIP_TAIL)?;
    Ok(())
}

fn write_srf_points_v1<W: Write>(
    writer: &mut W,
    metadata: &SrfMetadata,
    slipt1: &CsrMatrix,
) -> Result<()> {
    let mut buffer = [0u8; BUFFER_SIZE];
    writer.write_all(POINTS)?;
    lexical_write(writer, metadata.iter().len(), &mut buffer)?;

    for (point, slip) in metadata.iter().zip(slipt1.rows()) {
        writer.write_all(b"\n")?;
        write_point(writer, &point, &mut buffer)?;
        writer.write_all(b"\n")?;
        lexical_write(writer, slip.len(), &mut buffer)?;
        write_slip_history(writer, slip, &mut buffer)?;
    }

    Ok(())
}

fn write_srf_points_v2<W: Write>(
    writer: &mut W,
    planes: &[SrfPlane],
    metadata: &SrfMetadataV2,
    slipt1: &CsrMatrix,
) -> Result<()> {
    let mut buffer = [0u8; BUFFER_SIZE];
    let mut point_iter = metadata.iter().zip(slipt1.rows());
    for plane in planes {
        writer.write_all(POINTS)?;
        let plane_point_count = plane.points();
        lexical_write(writer, plane_point_count, &mut buffer)?;
        for (point, slip) in point_iter.by_ref().take(plane_point_count) {
            writer.write_all(b"\n")?;
            write_point(writer, &point.base, &mut buffer)?;
            writer.write_all(b" ")?;
            lexical_write(writer, point.vs, &mut buffer)?;
            writer.write_all(b" ")?;
            lexical_write(writer, point.density, &mut buffer)?;
            writer.write_all(b"\n")?;
            write_slip_history(writer, slip, &mut buffer)?;
        }
    }

    Ok(())
}

fn write_plane_header<W: Write>(writer: &mut W, planes: &[SrfPlane]) -> Result<()> {
    writeln!(writer, "PLANES {}", planes.len())?;
    for plane in planes {
        writeln!(
            writer,
            "{} {} {} {} {} {} {} {} {} {} {}",
            plane.elon,
            plane.elat,
            plane.nstk,
            plane.ndip,
            plane.len,
            plane.wid,
            plane.stk,
            plane.dip,
            plane.dtop,
            plane.shyp,
            plane.dhyp
        )?;
    }
    Ok(())
}

const VERSION_1: &[u8] = b"VERSION 1.0\n";
const VERSION_2: &[u8] = b"VERSION 2.0\n";

fn write_version<W: Write>(writer: &mut W, metadata: &SrfMetadataVersioned) -> Result<()> {
    writer.write_all(match metadata {
        SrfMetadataVersioned::V1(_) => VERSION_1,
        SrfMetadataVersioned::V2(_) => VERSION_2,
    })
}

pub fn write_srf<W: Write>(writer: &mut W, srf_file: &SrfFile) -> Result<()> {
    write_version(writer, &srf_file.metadata)?;
    write_plane_header(writer, &srf_file.planes)?;
    match &srf_file.metadata {
        SrfMetadataVersioned::V1(metadata) => {
            write_srf_points_v1(writer, metadata, &srf_file.slipt1)
        }
        SrfMetadataVersioned::V2(metadata) => {
            write_srf_points_v2(writer, &srf_file.planes, metadata, &srf_file.slipt1)
        }
    }
}
