// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2026 Adrian <adrian.eddy at gmail>

#[derive(Debug, Clone, PartialEq)]
pub struct Lut3D {
    pub size: u32,
    pub domain_min: f32,
    pub domain_max: f32,
    /// `size³ × 3` values, RGB-major, red varying fastest (`.cube` order).
    pub data: Vec<f32>,
    pub checksum: u32,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum LutError {
    #[error("1D LUTs are not supported, only 3D .cube files")]
    OneDimensionalNotSupported,
    #[error("LUT_3D_SIZE not found in file")]
    MissingSize,
    #[error("LUT size {0} is out of range, must be between 2 and 64")]
    SizeOutOfRange(u32),
    #[error("LUT is incomplete: expected {expected} entries but found {found}")]
    EntryCountMismatch { expected: usize, found: usize },
    #[error("Invalid number on line {0}")]
    InvalidNumber(usize),
    #[error("Value is not a finite number on line {0}")]
    NonFiniteValue(usize),
    #[error("Per-channel LUT domains are not supported, DOMAIN_MIN/DOMAIN_MAX must be equal for R, G and B")]
    NonUniformDomain,
}

pub fn parse_cube(text: &str) -> Result<Lut3D, LutError> {
    let mut size: Option<u32> = None;
    let mut domain_min = 0.0f32;
    let mut domain_max = 1.0f32;
    let mut data: Vec<f32> = Vec::new();

    fn parse_triplet(parts: &[&str], line_no: usize) -> Result<[f32; 3], LutError> {
        let mut out = [0.0f32; 3];
        for (i, p) in parts.iter().take(3).enumerate() {
            let v: f32 = p.parse().map_err(|_| LutError::InvalidNumber(line_no))?;
            if !v.is_finite() { return Err(LutError::NonFiniteValue(line_no)); }
            out[i] = v;
        }
        Ok(out)
    }

    for (i, raw_line) in text.lines().enumerate() {
        let line_no = i + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }

        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts[0].to_ascii_uppercase().as_str() {
            "TITLE" => continue,
            "LUT_1D_SIZE" => return Err(LutError::OneDimensionalNotSupported),
            "LUT_3D_SIZE" => {
                let n: u32 = parts.get(1).ok_or(LutError::InvalidNumber(line_no))?
                                  .parse().map_err(|_| LutError::InvalidNumber(line_no))?;
                if !(2..=64).contains(&n) { return Err(LutError::SizeOutOfRange(n)); }
                size = Some(n);
                data.reserve((n as usize).pow(3) * 3);
            }
            "DOMAIN_MIN" | "DOMAIN_MAX" => {
                if parts.len() < 4 { return Err(LutError::InvalidNumber(line_no)); }
                let v = parse_triplet(&parts[1..], line_no)?;
                if (v[0] - v[1]).abs() > f32::EPSILON || (v[1] - v[2]).abs() > f32::EPSILON {
                    return Err(LutError::NonUniformDomain);
                }
                if parts[0].eq_ignore_ascii_case("DOMAIN_MIN") { domain_min = v[0]; } else { domain_max = v[0]; }
            }
            _ => {
                if parts.len() < 3 { return Err(LutError::InvalidNumber(line_no)); }
                let v = parse_triplet(&parts, line_no)?;
                data.extend_from_slice(&v);
            }
        }
    }

    let size = size.ok_or(LutError::MissingSize)?;
    let expected = (size as usize).pow(3);
    let found = data.len() / 3;
    if found != expected { return Err(LutError::EntryCountMismatch { expected, found }); }

    let checksum = crc32fast::hash(bytemuck::cast_slice(&data));
    Ok(Lut3D { size, domain_min, domain_max, data, checksum })
}

/// Manual trilinear interpolation. Deliberately not a hardware sampler — this
/// exact arithmetic is mirrored in every GPU backend so results match.
pub fn sample_trilinear(lut: &Lut3D, rgb: [f32; 3]) -> [f32; 3] {
    let n = lut.size as i32;
    let scale = 1.0 / (lut.domain_max - lut.domain_min).max(1e-9);

    let mut base = [0i32; 3];
    let mut frac = [0f32; 3];
    for c in 0..3 {
        let normalized = ((rgb[c] - lut.domain_min) * scale).clamp(0.0, 1.0);
        let pos = normalized * (n - 1) as f32;
        let i0 = (pos.floor() as i32).clamp(0, n - 1);
        base[c] = i0;
        frac[c] = pos - i0 as f32;
    }

    let fetch = |r: i32, g: i32, b: i32| -> [f32; 3] {
        let r = r.clamp(0, n - 1); let g = g.clamp(0, n - 1); let b = b.clamp(0, n - 1);
        let idx = ((b * n * n + g * n + r) * 3) as usize;
        [lut.data[idx], lut.data[idx + 1], lut.data[idx + 2]]
    };

    let mut out = [0f32; 3];
    for c in 0..3 {
        let c000 = fetch(base[0],     base[1],     base[2])[c];
        let c100 = fetch(base[0] + 1, base[1],     base[2])[c];
        let c010 = fetch(base[0],     base[1] + 1, base[2])[c];
        let c110 = fetch(base[0] + 1, base[1] + 1, base[2])[c];
        let c001 = fetch(base[0],     base[1],     base[2] + 1)[c];
        let c101 = fetch(base[0] + 1, base[1],     base[2] + 1)[c];
        let c011 = fetch(base[0],     base[1] + 1, base[2] + 1)[c];
        let c111 = fetch(base[0] + 1, base[1] + 1, base[2] + 1)[c];

        let c00 = c000 + (c100 - c000) * frac[0];
        let c10 = c010 + (c110 - c010) * frac[0];
        let c01 = c001 + (c101 - c001) * frac[0];
        let c11 = c011 + (c111 - c011) * frac[0];
        let c0 = c00 + (c10 - c00) * frac[1];
        let c1 = c01 + (c11 - c01) * frac[1];
        out[c] = c0 + (c1 - c0) * frac[2];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Smallest valid LUT: 2³ = 8 entries, identity.
    const IDENTITY_2: &str = "\
TITLE \"identity\"
LUT_3D_SIZE 2
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0
1.0 1.0 1.0
";

    #[test]
    fn parses_minimal_identity() {
        let lut = parse_cube(IDENTITY_2).unwrap();
        assert_eq!(lut.size, 2);
        assert_eq!(lut.data.len(), 8 * 3);
        assert_eq!(lut.domain_min, 0.0);
        assert_eq!(lut.domain_max, 1.0);
    }

    #[test]
    fn skips_comments_and_blank_lines() {
        let text = format!("# a comment\n\n{}\n# trailing\n", IDENTITY_2);
        assert_eq!(parse_cube(&text).unwrap().size, 2);
    }

    #[test]
    fn handles_crlf_line_endings() {
        let text = IDENTITY_2.replace('\n', "\r\n");
        assert_eq!(parse_cube(&text).unwrap().size, 2);
    }

    #[test]
    fn handles_extra_whitespace() {
        let text = IDENTITY_2.replace("0.0 0.0 0.0", "  0.0\t0.0   0.0  ");
        assert_eq!(parse_cube(&text).unwrap().size, 2);
    }

    #[test]
    fn reads_uniform_domain() {
        let text = IDENTITY_2.replace("LUT_3D_SIZE 2", "DOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 4.0 4.0 4.0\nLUT_3D_SIZE 2");
        let lut = parse_cube(&text).unwrap();
        assert_eq!(lut.domain_max, 4.0);
    }

    #[test]
    fn rejects_non_uniform_domain() {
        let text = IDENTITY_2.replace("LUT_3D_SIZE 2", "DOMAIN_MAX 1.0 2.0 1.0\nLUT_3D_SIZE 2");
        assert_eq!(parse_cube(&text), Err(LutError::NonUniformDomain));
    }

    #[test]
    fn rejects_1d_luts() {
        let text = "LUT_1D_SIZE 2\n0.0 0.0 0.0\n1.0 1.0 1.0\n";
        assert_eq!(parse_cube(text), Err(LutError::OneDimensionalNotSupported));
    }

    #[test]
    fn rejects_missing_size() {
        assert_eq!(parse_cube("0.0 0.0 0.0\n"), Err(LutError::MissingSize));
    }

    #[test]
    fn rejects_truncated_file() {
        let text = IDENTITY_2.replace("1.0 1.0 1.0\n", "");
        assert_eq!(parse_cube(&text), Err(LutError::EntryCountMismatch { expected: 8, found: 7 }));
    }

    #[test]
    fn rejects_size_out_of_range() {
        assert_eq!(parse_cube("LUT_3D_SIZE 1\n"), Err(LutError::SizeOutOfRange(1)));
        assert_eq!(parse_cube("LUT_3D_SIZE 65\n"), Err(LutError::SizeOutOfRange(65)));
    }

    #[test]
    fn rejects_non_finite_values() {
        let text = IDENTITY_2.replace("0.0 0.0 0.0", "nan 0.0 0.0");
        assert!(matches!(parse_cube(&text), Err(LutError::NonFiniteValue(_))));
    }

    #[test]
    fn rejects_bad_number_with_line_number() {
        let text = IDENTITY_2.replace("0.0 1.0 0.0", "0.0 banana 0.0");
        assert!(matches!(parse_cube(&text), Err(LutError::InvalidNumber(_))));
    }

    #[test]
    fn identity_lut_samples_to_input() {
        let lut = parse_cube(IDENTITY_2).unwrap();
        for probe in [[0.0f32, 0.0, 0.0], [1.0, 1.0, 1.0], [0.25, 0.5, 0.75], [0.9, 0.1, 0.6]] {
            let out = sample_trilinear(&lut, probe);
            for i in 0..3 {
                assert!((out[i] - probe[i]).abs() < 1e-6, "channel {i}: {} != {}", out[i], probe[i]);
            }
        }
    }

    #[test]
    fn sampling_clamps_out_of_range_input() {
        let lut = parse_cube(IDENTITY_2).unwrap();
        assert_eq!(sample_trilinear(&lut, [-1.0, 2.0, 0.5])[0], 0.0);
        assert_eq!(sample_trilinear(&lut, [-1.0, 2.0, 0.5])[1], 1.0);
    }
}
