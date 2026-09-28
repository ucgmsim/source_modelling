use crate::types::{
    CsrMatrix, Point, SrfFile, SrfMetadata, SrfMetadataV2, SrfMetadataVersioned, SrfPlane,
    starting_column,
};
use std::io::{self, Write};
use std::iter;
use thiserror::Error;

use lexical_core::{BUFFER_SIZE, ToLexical};

#[derive(Debug, Error)]
pub enum WriteError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(
        "slipt1 row {point} must have sorted, unique column indices no earlier than its first timestep floor(tinit / dt) = {start}"
    )]
    Slip { point: usize, start: usize },
}

type Result<T> = std::result::Result<T, WriteError>;

// The samples an SRF stores for a point: one per timestep from floor(tinit /
// dt) to the row's last stored column. An SRF records only where a row starts,
// so columns slipt1 leaves out must be written as explicit zeros or the
// samples after them shift earlier in time on re-read.
fn point_samples<'a>(
    point: usize,
    (indices, data): (&'a [i32], &'a [f32]),
    tinit: f32,
    dt: f32,
) -> Result<impl ExactSizeIterator<Item = f32> + 'a> {
    let start = starting_column(tinit, dt);
    let columns_increase_from_start = iter::once(start as i64 - 1)
        .chain(indices.iter().map(|&column| i64::from(column)))
        .is_sorted_by(|a, b| a < b);
    if !columns_increase_from_start {
        return Err(WriteError::Slip { point, start });
    }
    let end = indices.last().map_or(start, |&last| last as usize + 1);
    let mut stored = indices.iter().zip(data).peekable();
    Ok((start..end).map(move |column| {
        stored
            .next_if(|&(&stored_column, _)| stored_column as usize == column)
            .map_or(0.0, |(_, &value)| value)
    }))
}

/// Expands slipt1 into the per-point sample runs SRF-style formats store: the
/// sample count of each point and all of their samples concatenated.
pub fn dense_slip_rows<R: AsRef<[i32]>, D: AsRef<[f32]>>(
    slipt1: &CsrMatrix<R, D>,
    tinit: &[f32],
    dt: &[f32],
) -> Result<(Vec<i32>, Vec<f32>)> {
    let mut counts = Vec::with_capacity(slipt1.rows().len());
    let mut data = Vec::with_capacity(slipt1.data.as_ref().len());
    for (point, ((row, &tinit), &dt)) in slipt1.rows().zip(tinit).zip(dt).enumerate() {
        let samples = point_samples(point, row, tinit, dt)?;
        // Columns are int32 and the Python side keeps them below i32::MAX, so
        // a row's sample count fits too.
        counts.push(samples.len() as i32);
        data.extend(samples);
    }
    Ok((counts, data))
}

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
    lexical_write(writer, point.tinit, buffer)?;
    writer.write_all(b" ")?;
    lexical_write(writer, point.dt, buffer)?;
    Ok(())
}

// Writes "rake slip1 nt1 0.0 0 0.0 0" followed by the nt1 slipt1 values on
// the next line, matching the record layout read_slip_row expects. slip2/nt2
// and slip3/nt3 are always written empty.
fn write_slip_row<W: Write>(
    writer: &mut W,
    point: &Point,
    samples: impl ExactSizeIterator<Item = f32>,
    buffer: &mut [u8],
) -> Result<()> {
    lexical_write(writer, point.rake, buffer)?;
    writer.write_all(b" ")?;
    lexical_write(writer, point.slip1, buffer)?;
    writer.write_all(b" ")?;
    lexical_write(writer, samples.len(), buffer)?;
    writer.write_all(EMPTY_SLIP_TAIL)?;

    for (i, v) in samples.enumerate() {
        writer.write_all(if i == 0 { b"\n" } else { b" " })?;
        lexical_write(writer, v, buffer)?;
    }
    Ok(())
}

fn write_srf_points_v1<W: Write, S: AsRef<[f32]>, R: AsRef<[i32]>, D: AsRef<[f32]>>(
    writer: &mut W,
    metadata: &SrfMetadata<S>,
    slipt1: &CsrMatrix<R, D>,
) -> Result<()> {
    let mut buffer = [0u8; BUFFER_SIZE];
    writer.write_all(POINTS)?;
    lexical_write(writer, metadata.iter().len(), &mut buffer)?;
    writer.write_all(b"\n")?;

    for (point_index, (point, row)) in metadata.iter().zip(slipt1.rows()).enumerate() {
        let samples = point_samples(point_index, row, point.tinit, point.dt)?;
        write_point(writer, &point, &mut buffer)?;
        writer.write_all(b"\n")?;
        write_slip_row(writer, &point, samples, &mut buffer)?;
        writer.write_all(b"\n")?;
    }

    Ok(())
}

fn write_srf_points_v2<W: Write, S: AsRef<[f32]>, R: AsRef<[i32]>, D: AsRef<[f32]>>(
    writer: &mut W,
    planes: &[SrfPlane],
    metadata: &SrfMetadataV2<S>,
    slipt1: &CsrMatrix<R, D>,
) -> Result<()> {
    let mut buffer = [0u8; BUFFER_SIZE];
    let mut point_iter = metadata.iter().zip(slipt1.rows()).enumerate();
    for plane in planes {
        writer.write_all(POINTS)?;
        let plane_point_count = plane.points();
        lexical_write(writer, plane_point_count, &mut buffer)?;
        writer.write_all(b"\n")?;
        for (point_index, (point, row)) in point_iter.by_ref().take(plane_point_count) {
            let samples = point_samples(point_index, row, point.base.tinit, point.base.dt)?;
            write_point(writer, &point.base, &mut buffer)?;
            writer.write_all(b" ")?;
            lexical_write(writer, point.vs, &mut buffer)?;
            writer.write_all(b" ")?;
            lexical_write(writer, point.density, &mut buffer)?;
            writer.write_all(b"\n")?;
            write_slip_row(writer, &point.base, samples, &mut buffer)?;
            writer.write_all(b"\n")?;
        }
    }

    Ok(())
}

fn write_plane_header<W: Write>(writer: &mut W, planes: &[SrfPlane]) -> Result<()> {
    writeln!(writer, "PLANE {}", planes.len())?;
    for plane in planes {
        writeln!(
            writer,
            "{} {} {} {} {} {}",
            plane.elon, plane.elat, plane.nstk, plane.ndip, plane.len, plane.wid,
        )?;
        writeln!(
            writer,
            "{} {} {} {} {}",
            plane.stk, plane.dip, plane.dtop, plane.shyp, plane.dhyp
        )?;
    }
    Ok(())
}

const VERSION_1: &[u8] = b"1.0\n";
const VERSION_2: &[u8] = b"2.0\n";

fn write_version<W: Write, S>(writer: &mut W, metadata: &SrfMetadataVersioned<S>) -> Result<()> {
    writer.write_all(match metadata {
        SrfMetadataVersioned::V1(_) => VERSION_1,
        SrfMetadataVersioned::V2(_) => VERSION_2,
    })?;
    Ok(())
}

pub fn write_srf<W: Write, S: AsRef<[f32]>, R: AsRef<[i32]>>(
    writer: &mut W,
    srf_file: &SrfFile<S, R>,
) -> Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scanner::Scanner;
    use crate::srf_parser::read_srf_struct;
    use crate::types::{CsrMatrixView, SrfFileView, SrfMetadataVersioned};

    const SRF_V1: &[u8] = b"1.0\n\
PLANE 1\n\
0.0 0.0 2 1 4.0 2.0\n\
90.0 45.0 0.0 0.0 1.0\n\
POINTS 2\n\
0.1 -43.0 5.0 90.0 45.0 1.0e10 0.5 0.1\n\
30.0 1.5 3 0.0 0 0.0 0\n\
0.1 0.2 0.3\n\
0.2 -43.1 5.5 90.0 45.0 1.0e10 0.6 0.1\n\
45.0 2.0 2 0.0 0 0.0 0\n\
0.4 0.5\n";

    const SRF_V2_TWO_PLANES: &[u8] = b"2.0\n\
PLANE 2\n\
0.0 0.0 1 1 4.0 2.0\n\
90.0 45.0 0.0 0.0 1.0\n\
0.5 0.5 1 1 4.0 2.0\n\
90.0 45.0 0.0 0.0 1.0\n\
POINTS 1\n\
0.1 -43.0 5.0 90.0 45.0 1.0e10 0.5 0.1 3.5 2.7\n\
30.0 1.5 2 0.0 0 0.0 0\n\
0.1 0.2\n\
POINTS 1\n\
0.2 -43.1 5.5 90.0 45.0 1.0e10 0.6 0.1 3.6 2.8\n\
45.0 2.0 1 0.0 0 0.0 0\n\
0.4\n";

    fn parse(data: &[u8]) -> SrfFile {
        let mut scanner = Scanner::new(data);
        read_srf_struct(&mut scanner).unwrap()
    }

    fn write_to_vec<S: AsRef<[f32]>, R: AsRef<[i32]>>(srf: &SrfFile<S, R>) -> Vec<u8> {
        let mut out = Vec::new();
        write_srf(&mut out, srf).unwrap();
        out
    }

    #[test]
    fn v1_roundtrips_through_parser() {
        let srf = parse(SRF_V1);
        let reparsed = parse(&write_to_vec(&srf));

        assert_eq!(reparsed.planes.len(), 1);
        let (metadata, reparsed_metadata) = match (&srf.metadata, &reparsed.metadata) {
            (SrfMetadataVersioned::V1(a), SrfMetadataVersioned::V1(b)) => (a, b),
            _ => panic!("expected V1 metadata"),
        };
        assert_eq!(metadata.lon, reparsed_metadata.lon);
        assert_eq!(metadata.lat, reparsed_metadata.lat);
        assert_eq!(metadata.dep, reparsed_metadata.dep);
        assert_eq!(metadata.area, reparsed_metadata.area);
        assert_eq!(metadata.tinit, reparsed_metadata.tinit);
        assert_eq!(metadata.rake, reparsed_metadata.rake);
        assert_eq!(metadata.slip1, reparsed_metadata.slip1);
        assert_eq!(metadata.rise, reparsed_metadata.rise);
        assert_eq!(srf.slipt1.row_ptr, reparsed.slipt1.row_ptr);
        assert_eq!(srf.slipt1.data, reparsed.slipt1.data);
    }

    #[test]
    fn v2_multi_plane_roundtrips_through_parser() {
        let srf = parse(SRF_V2_TWO_PLANES);
        let reparsed = parse(&write_to_vec(&srf));

        assert_eq!(reparsed.planes.len(), 2);
        let (metadata, reparsed_metadata) = match (&srf.metadata, &reparsed.metadata) {
            (SrfMetadataVersioned::V2(a), SrfMetadataVersioned::V2(b)) => (a, b),
            _ => panic!("expected V2 metadata"),
        };
        assert_eq!(metadata.base.lon, reparsed_metadata.base.lon);
        assert_eq!(metadata.base.rake, reparsed_metadata.base.rake);
        assert_eq!(metadata.vs, reparsed_metadata.vs);
        assert_eq!(metadata.density, reparsed_metadata.density);
        assert_eq!(srf.slipt1.row_ptr, reparsed.slipt1.row_ptr);
        assert_eq!(srf.slipt1.data, reparsed.slipt1.data);
    }

    #[test]
    fn borrowed_view_writes_same_bytes_as_owned() {
        let srf = parse(SRF_V1);
        let metadata = match &srf.metadata {
            SrfMetadataVersioned::V1(metadata) => metadata,
            SrfMetadataVersioned::V2(_) => panic!("expected V1 metadata"),
        };

        let view: SrfFileView = SrfFile {
            planes: srf.planes.clone(),
            metadata: SrfMetadataVersioned::V1(SrfMetadata {
                lon: &metadata.lon,
                lat: &metadata.lat,
                dep: &metadata.dep,
                stk: &metadata.stk,
                dip: &metadata.dip,
                area: &metadata.area,
                tinit: &metadata.tinit,
                dt: &metadata.dt,
                rake: &metadata.rake,
                slip1: &metadata.slip1,
                rise: &metadata.rise,
            }),
            slipt1: CsrMatrixView {
                row_ptr: &srf.slipt1.row_ptr,
                indices: &srf.slipt1.indices,
                data: &srf.slipt1.data,
            },
        };

        assert_eq!(write_to_vec(&view), write_to_vec(&srf));
    }

    #[test]
    fn empty_slip_row_roundtrips() {
        let data = b"1.0\n\
PLANE 1\n\
0.0 0.0 1 1 4.0 2.0\n\
90.0 45.0 0.0 0.0 1.0\n\
POINTS 1\n\
0.1 -43.0 5.0 90.0 45.0 1.0e10 0.5 0.1\n\
30.0 1.5 0 0.0 0 0.0 0\n";
        let srf = parse(data);
        let reparsed = parse(&write_to_vec(&srf));
        assert_eq!(srf.slipt1.row_ptr, reparsed.slipt1.row_ptr);
        assert!(reparsed.slipt1.data.is_empty());
    }

    // A single V1 point with tinit = 0.5, dt = 0.1, so its first timestep is
    // column 5.
    fn single_point_srf(indices: Vec<i32>, data: Vec<f32>) -> SrfFile {
        let mut srf = parse(
            b"1.0\n\
PLANE 1\n\
0.0 0.0 1 1 4.0 2.0\n\
90.0 45.0 0.0 0.0 1.0\n\
POINTS 1\n\
0.1 -43.0 5.0 90.0 45.0 1.0e10 0.5 0.1\n\
30.0 1.5 0 0.0 0 0.0 0\n",
        );
        srf.slipt1 = CsrMatrix {
            row_ptr: vec![0, indices.len() as i32],
            indices,
            data,
        };
        srf
    }

    // The bug this guards: the writer used to drop column indices, so rows
    // without explicit zeros (e.g. scipy's csr_array(dense)) shifted earlier
    // in time on re-read.
    #[test]
    fn implicit_zeros_are_written_explicitly() {
        // Dense row from column 5: [0, 1, 0, 2]. The leading zero at column 5
        // and the interior zero at column 7 are both implicit.
        let srf = single_point_srf(vec![6, 8], vec![1.0, 2.0]);
        let reparsed = parse(&write_to_vec(&srf));
        assert_eq!(reparsed.slipt1.indices, vec![5, 6, 7, 8]);
        assert_eq!(reparsed.slipt1.data, vec![0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn value_before_tinit_is_rejected() {
        let srf = single_point_srf(vec![4, 5], vec![1.0, 2.0]);
        let err = write_srf(&mut Vec::new(), &srf).unwrap_err();
        assert!(matches!(err, WriteError::Slip { point: 0, start: 5 }));
    }

    #[test]
    fn non_canonical_row_is_rejected() {
        let srf = single_point_srf(vec![7, 5], vec![1.0, 2.0]);
        let err = write_srf(&mut Vec::new(), &srf).unwrap_err();
        assert!(matches!(err, WriteError::Slip { point: 0, start: 5 }));
    }

    #[test]
    fn dense_slip_rows_match_written_samples() {
        let slipt1 = CsrMatrix {
            row_ptr: vec![0, 2, 2, 3],
            indices: vec![6, 8, 1],
            data: vec![1.0, 2.0, 3.0],
        };
        let (counts, data) = dense_slip_rows(&slipt1, &[0.5, 0.0, 0.0], &[0.1, 0.1, 0.1]).unwrap();
        assert_eq!(counts, vec![4, 0, 2]);
        assert_eq!(data, vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0]);
    }
}
