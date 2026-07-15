use crate::scanner;
use crate::types::{CsrMatrix, Point, SrfFile, SrfMetadata, SrfMetadataVersioned, SrfPlane};

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
    Ok(plane_vec)
}

fn estimate_slipt1_array_size(
    scanner: &mut scanner::Scanner,
    point_count: usize,
) -> Result<usize, scanner::ScannerError> {
    let sample_size = 500.min(point_count);
    let mut total_slip_samples: usize = 0;
    let index = scanner.index;
    for _ in 0..sample_size {
        for _ in 0..10 {
            let _: f32 = scanner.next()?;
        }
        let nt: usize = scanner.next()?;
        total_slip_samples += nt;
        for _ in 0..(nt + 4) {
            let _: f32 = scanner.next()?;
        }
    }
    let avg_slip = (total_slip_samples as f64) / (sample_size as f64);
    scanner.index = index;
    Ok((point_count as f64 * avg_slip).ceil() as usize)
}

pub fn read_srf_struct(
    scanner: &mut scanner::Scanner,
) -> Result<SrfFile, scanner::ScannerError> {
    let version = String::from_utf8_lossy(scanner.line()?).into_owned();
    scanner.skip_token(b"PLANE")?;
    let plane_count: usize = scanner.next()?;
    let header = read_srf_header(scanner, plane_count)?;
    scanner.skip_token(b"POINTS")?;

    let point_count = scanner.next()?;
    let mut metadata = SrfMetadata::with_capacity(point_count);

    let slipt1_capacity = estimate_slipt1_array_size(scanner, point_count)?;
    let mut slipt1 = CsrMatrix::new(point_count, slipt1_capacity);

    for _ in 0..point_count {
        let lon = scanner.next()?;
        let lat = scanner.next()?;
        let dep = scanner.next()?;
        let stk = scanner.next()?;
        let dip = scanner.next()?;
        let area = scanner.next()?;
        let tinit = scanner.next()?;
        let dt = scanner.next()?;
        let rake = scanner.next()?;
        let slip1 = scanner.next()?;
        let nt = scanner.next::<usize>()?;
        let rise = (nt as f32) * dt;

        metadata.push(Point {
            lon,
            lat,
            dep,
            stk,
            dip,
            area,
            tinit,
            dt,
            rake,
            slip1,
            rise,
        });

        let _slip2 = scanner.next::<f32>()?;
        let _nt2 = scanner.next::<usize>()?;
        let _slip3 = scanner.next::<f32>()?;
        let _nt3 = scanner.next::<usize>()?;

        slipt1.push_row();
        for _ in 0..nt {
            slipt1.push(scanner.next()?);
        }
    }
    Ok(SrfFile {
        version,
        planes: header,
        metadata: SrfMetadataVersioned::V1(metadata),
        slipt1,
    })
}
