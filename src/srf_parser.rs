use thiserror::Error;

use crate::scanner;
use crate::types::{
    CsrMatrix, Point, PointV2, SrfFile, SrfMetadata, SrfMetadataV2, SrfMetadataVersioned, SrfPlane,
};

#[derive(Debug)]
pub enum Version {
    V1,
    V2,
}

impl Version {
    fn parse(line: &[u8]) -> Result<Self, SrfParseError> {
        match line.trim_ascii() {
            b"1.0" => Ok(Version::V1),
            b"2.0" => Ok(Version::V2),
            other => Err(SrfParseError::UnknownVersion(
                String::from_utf8_lossy(other).into_owned(),
            )),
        }
    }
}

#[derive(Debug, Error)]
pub enum SrfParseError {
    #[error(transparent)]
    Scanner(#[from] scanner::ScannerError),
    #[error("unknown SRF version: {0}")]
    UnknownVersion(String),
    #[error("PLANE headers expect {expected} total points but POINTS declares {declared}")]
    PointCountMismatch { declared: usize, expected: usize },
    #[error("plane {plane} expects {expected} points but its POINTS block declares {declared}")]
    PlanePointCountMismatch {
        plane: usize,
        declared: usize,
        expected: usize,
    },
}

fn read_srf_header(
    scanner: &mut scanner::Scanner,
    plane_count: usize,
) -> Result<Vec<SrfPlane>, SrfParseError> {
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
    Ok(plane_vec)
}

// Tiny PointHeader used to construct actual points later.
struct PointHeader {
    lon: f32,
    lat: f32,
    dep: f32,
    stk: f32,
    dip: f32,
    area: f32,
    tinit: f32,
    dt: f32,
}

// Common point header values for both V1 and V2.
fn read_point_header(scanner: &mut scanner::Scanner) -> Result<PointHeader, SrfParseError> {
    Ok(PointHeader {
        lon: scanner.next()?,
        lat: scanner.next()?,
        dep: scanner.next()?,
        stk: scanner.next()?,
        dip: scanner.next()?,
        area: scanner.next()?,
        tinit: scanner.next()?,
        dt: scanner.next()?,
    })
}

fn read_slip_row(
    scanner: &mut scanner::Scanner,
    slipt1: &mut CsrMatrix,
    nt: usize,
) -> Result<(), SrfParseError> {
    let _slip2 = scanner.next::<f32>()?;
    let _nt2 = scanner.next::<usize>()?;
    let _slip3 = scanner.next::<f32>()?;
    let _nt3 = scanner.next::<usize>()?;

    slipt1.push_row();
    for _ in 0..nt {
        slipt1.push(scanner.next()?);
    }
    Ok(())
}

fn read_srf_points_v1(
    planes: &[SrfPlane],
    scanner: &mut scanner::Scanner,
) -> Result<(SrfMetadata, CsrMatrix), SrfParseError> {
    scanner.skip_token(b"POINTS")?;
    let plane_point_count = planes.iter().map(|plane| plane.points()).sum();
    let point_count = scanner.next()?;
    if point_count != plane_point_count {
        return Err(SrfParseError::PointCountMismatch {
            declared: point_count,
            expected: plane_point_count,
        });
    }
    let slipt1_capacity = scanner.remaining() / 8;
    let mut metadata = SrfMetadata::with_capacity(point_count);
    let mut slipt1 = CsrMatrix::new(point_count, slipt1_capacity);

    for _ in 0..point_count {
        let header = read_point_header(scanner)?;
        // technically the read_srf routines in EMOD3D don't need to have a
        // newline here but it allows us to distinguish between a mislabelled
        // version 1.0 SRF and a version 2.0 SRF because EOL conveniently
        // follows the point header (where-as SRF V2.0 has two vs/density still
        // to go):
        //
        // - read_srf (in the srf_subs.c versions): reads the floats with scanf() that is newline tolerant.
        // - write_srf (same files): always writes a newline after the header (header + vs + density in SRF 2.0).
        scanner.expect_end_of_line()?;
        let rake = scanner.next()?;
        let slip1 = scanner.next()?;
        let nt = scanner.next::<usize>()?;
        let rise = (nt as f32) * header.dt;

        let point = Point {
            lon: header.lon,
            lat: header.lat,
            dep: header.dep,
            stk: header.stk,
            dip: header.dip,
            area: header.area,
            tinit: header.tinit,
            dt: header.dt,
            rake,
            slip1,
            rise,
        };
        metadata.push(&point);
        read_slip_row(scanner, &mut slipt1, nt)?;
    }
    Ok((metadata, slipt1))
}

fn read_srf_points_v2(
    planes: &[SrfPlane],
    scanner: &mut scanner::Scanner,
) -> Result<(SrfMetadataV2, CsrMatrix), SrfParseError> {
    let point_count = planes.iter().map(|plane| plane.points()).sum();
    let slipt1_capacity = scanner.remaining() / 8;

    let mut metadata = SrfMetadataV2::with_capacity(point_count);
    let mut slipt1 = CsrMatrix::new(point_count, slipt1_capacity);

    for (i, plane) in planes.iter().enumerate() {
        // In version 2.0 (and version 2.0 only), it is possible to construct SRFs
        // with multiple POINT instantiations. Technically V1.0 SRFs could be
        // constructed with multiple POINTS instantiations but we are deliberately
        // parsing a stricter subset of the format.

        scanner.skip_token(b"POINTS")?;
        let plane_point_count = scanner.next()?;
        if plane.points() != plane_point_count {
            return Err(SrfParseError::PlanePointCountMismatch {
                plane: i,
                declared: plane_point_count,
                expected: plane.points(),
            });
        }

        for _ in 0..plane_point_count {
            let header = read_point_header(scanner)?;
            let vs = scanner.next()?;
            let density = scanner.next()?;
            // Like the V1 parser, expecting EOL here helps detect mangled SRFs
            scanner.expect_end_of_line()?;
            let rake = scanner.next()?;
            let slip1 = scanner.next()?;

            let nt = scanner.next::<usize>()?;
            let rise = (nt as f32) * header.dt;

            let point = Point {
                lon: header.lon,
                lat: header.lat,
                dep: header.dep,
                stk: header.stk,
                dip: header.dip,
                area: header.area,
                tinit: header.tinit,
                dt: header.dt,
                rake,
                slip1,
                rise,
            };

            metadata.push(&PointV2 {
                base: point,
                vs,
                density,
            });
            read_slip_row(scanner, &mut slipt1, nt)?;
        }
    }
    Ok((metadata, slipt1))
}

pub fn read_srf_struct(scanner: &mut scanner::Scanner) -> Result<SrfFile, SrfParseError> {
    let version = Version::parse(scanner.line()?)?;

    scanner.skip_token(b"PLANE")?;
    let plane_count: usize = scanner.next()?;
    let planes = read_srf_header(scanner, plane_count)?;

    let (metadata, slipt1) = match version {
        Version::V1 => {
            let (metadata, slipt1) = read_srf_points_v1(&planes, scanner)?;
            (SrfMetadataVersioned::V1(metadata), slipt1)
        }
        Version::V2 => {
            let (metadata, slipt1) = read_srf_points_v2(&planes, scanner)?;
            (SrfMetadataVersioned::V2(metadata), slipt1)
        }
    };

    Ok(SrfFile {
        planes,
        metadata,
        slipt1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRF_V1: &[u8] = b"1.0\n\
PLANE 1\n\
0.0 0.0 2 1 4.0 2.0 90.0 45.0 0.0 0.0 1.0\n\
POINTS 2\n\
0.1 -43.0 5.0 90.0 45.0 1.0e10 0.5 0.1\n\
30.0 1.5 3 0.0 0 0.0 0\n\
0.1 0.2 0.3\n\
0.2 -43.1 5.5 90.0 45.0 1.0e10 0.6 0.1\n\
45.0 2.0 2 0.0 0 0.0 0\n\
0.4 0.5\n";

    const SRF_V2: &[u8] = b"2.0\n\
PLANE 1\n\
0.0 0.0 2 1 4.0 2.0 90.0 45.0 0.0 0.0 1.0\n\
POINTS 2\n\
0.1 -43.0 5.0 90.0 45.0 1.0e10 0.5 0.1 3.5 2.7\n\
30.0 1.5 3 0.0 0 0.0 0\n\
0.1 0.2 0.3\n\
0.2 -43.1 5.5 90.0 45.0 1.0e10 0.6 0.1 3.6 2.8\n\
45.0 2.0 2 0.0 0 0.0 0\n\
0.4 0.5\n";

    #[test]
    fn parses_v1() {
        let mut scanner = scanner::Scanner::new(SRF_V1);
        let srf = read_srf_struct(&mut scanner).unwrap();
        assert_eq!(srf.planes.len(), 1);
        let metadata = match &srf.metadata {
            SrfMetadataVersioned::V1(metadata) => metadata,
            SrfMetadataVersioned::V2(_) => panic!("expected V1 metadata"),
        };
        assert_eq!(metadata.lon, vec![0.1, 0.2]);
        assert_eq!(metadata.rake, vec![30.0, 45.0]);
        assert_eq!(metadata.rise, vec![3.0 * 0.1f32, 2.0 * 0.1f32]);
        assert_eq!(srf.slipt1.row_ptr, vec![0, 3]);
        assert_eq!(srf.slipt1.data, vec![0.1, 0.2, 0.3, 0.4, 0.5]);
    }

    #[test]
    fn parses_v2() {
        let mut scanner = scanner::Scanner::new(SRF_V2);
        let srf = read_srf_struct(&mut scanner).unwrap();
        let metadata = match &srf.metadata {
            SrfMetadataVersioned::V2(metadata) => metadata,
            SrfMetadataVersioned::V1(_) => panic!("expected V2 metadata"),
        };
        assert_eq!(metadata.base.lon, vec![0.1, 0.2]);
        assert_eq!(metadata.base.rake, vec![30.0, 45.0]);
        assert_eq!(metadata.vs, vec![3.5, 3.6]);
        assert_eq!(metadata.density, vec![2.7, 2.8]);
        assert_eq!(srf.slipt1.data, vec![0.1, 0.2, 0.3, 0.4, 0.5]);
    }

    // Two 1x1 planes, each with its own POINTS block.
    const SRF_V2_TWO_PLANES: &[u8] = b"2.0\n\
PLANE 2\n\
0.0 0.0 1 1 4.0 2.0 90.0 45.0 0.0 0.0 1.0\n\
0.5 0.5 1 1 4.0 2.0 90.0 45.0 0.0 0.0 1.0\n\
POINTS 1\n\
0.1 -43.0 5.0 90.0 45.0 1.0e10 0.5 0.1 3.5 2.7\n\
30.0 1.5 2 0.0 0 0.0 0\n\
0.1 0.2\n\
POINTS 1\n\
0.2 -43.1 5.5 90.0 45.0 1.0e10 0.6 0.1 3.6 2.8\n\
45.0 2.0 1 0.0 0 0.0 0\n\
0.4\n";

    #[test]
    fn rejects_unknown_version() {
        let mut scanner = scanner::Scanner::new(b"3.0\nPLANE 1\n");
        let err = read_srf_struct(&mut scanner).unwrap_err();
        assert!(matches!(err, SrfParseError::UnknownVersion(version) if version == "3.0"));
    }

    #[test]
    fn parses_v2_with_multiple_planes() {
        let mut scanner = scanner::Scanner::new(SRF_V2_TWO_PLANES);
        let srf = read_srf_struct(&mut scanner).unwrap();
        assert_eq!(srf.planes.len(), 2);
        let metadata = match &srf.metadata {
            SrfMetadataVersioned::V2(metadata) => metadata,
            SrfMetadataVersioned::V1(_) => panic!("expected V2 metadata"),
        };
        assert_eq!(metadata.base.lon, vec![0.1, 0.2]);
        assert_eq!(metadata.vs, vec![3.5, 3.6]);
        assert_eq!(metadata.density, vec![2.7, 2.8]);
        assert_eq!(srf.slipt1.row_ptr, vec![0, 2]);
        assert_eq!(srf.slipt1.data, vec![0.1, 0.2, 0.4]);
    }

    #[test]
    fn rejects_v1_point_count_mismatch() {
        // Plane declares 2x1 points but POINTS declares 3.
        let data = replace_once(SRF_V1, b"POINTS 2", b"POINTS 3");
        let mut scanner = scanner::Scanner::new(&data);
        let err = read_srf_struct(&mut scanner).unwrap_err();
        assert!(matches!(
            err,
            SrfParseError::PointCountMismatch {
                declared: 3,
                expected: 2,
            }
        ));
    }

    #[test]
    fn rejects_v2_plane_point_count_mismatch() {
        // Second plane is 1x1 but its POINTS block declares 2; the first
        // plane must parse cleanly before the mismatch is hit.
        let data = replace_once(SRF_V2_TWO_PLANES, b"POINTS 1\n0.2", b"POINTS 2\n0.2");
        let mut scanner = scanner::Scanner::new(&data);
        let err = read_srf_struct(&mut scanner).unwrap_err();
        assert!(matches!(
            err,
            SrfParseError::PlanePointCountMismatch {
                plane: 1,
                declared: 2,
                expected: 1,
            }
        ));
    }

    #[test]
    fn truncated_file_never_panics() {
        // Truncating mid-token can still parse ("0.5" cut to "0." is a valid
        // float), so not every prefix errors — but none may panic, and a cut
        // mid-record must error.
        for len in 0..SRF_V1.len() {
            let mut scanner = scanner::Scanner::new(&SRF_V1[..len]);
            let _ = read_srf_struct(&mut scanner);
        }
        let mut scanner = scanner::Scanner::new(&SRF_V1[..SRF_V1.len() / 2]);
        assert!(read_srf_struct(&mut scanner).is_err());
    }

    #[test]
    fn rejects_v1_data_mislabeled_as_v2() {
        // read_srf_points_v2 reads vs/density right after the point header,
        // so on v1 data it swallows the v1 line's rake/slip1 tokens and the
        // expect_end_of_line() check right after fails to land on a newline.
        let data = replace_once(SRF_V1, b"1.0\n", b"2.0\n");
        let mut scanner = scanner::Scanner::new(&data);
        let err = read_srf_struct(&mut scanner).unwrap_err();
        assert!(matches!(
            err,
            SrfParseError::Scanner(scanner::ScannerError::NoNewlineFound { .. })
        ));
    }

    #[test]
    fn rejects_v2_data_mislabeled_as_v1() {
        // read_srf_points_v1 checks for end-of-line right after the point
        // header, but v2 data still has vs/density left on that same line.
        let data = replace_once(SRF_V2, b"2.0\n", b"1.0\n");
        let mut scanner = scanner::Scanner::new(&data);
        let err = read_srf_struct(&mut scanner).unwrap_err();
        assert!(matches!(
            err,
            SrfParseError::Scanner(scanner::ScannerError::NoNewlineFound { .. })
        ));
    }

    #[test]
    fn version_line_tolerates_surrounding_whitespace() {
        let data = replace_once(SRF_V1, b"1.0\n", b"1.0 \r\n");
        let mut scanner = scanner::Scanner::new(&data);
        assert!(read_srf_struct(&mut scanner).is_ok());
    }

    #[test]
    fn mismatch_messages_read_correctly() {
        let err = SrfParseError::PointCountMismatch {
            declared: 3,
            expected: 2,
        };
        assert_eq!(
            err.to_string(),
            "PLANE headers expect 2 total points but POINTS declares 3"
        );
        let err = SrfParseError::PlanePointCountMismatch {
            plane: 1,
            declared: 2,
            expected: 1,
        };
        assert_eq!(
            err.to_string(),
            "plane 1 expects 1 points but its POINTS block declares 2"
        );
    }

    fn replace_once(data: &[u8], from: &[u8], to: &[u8]) -> Vec<u8> {
        let start = data
            .windows(from.len())
            .position(|window| window == from)
            .expect("pattern not found in fixture");
        let mut result = data[..start].to_vec();
        result.extend_from_slice(to);
        result.extend_from_slice(&data[start + from.len()..]);
        result
    }
}
