// Copyright © 2026 Gyroflow

//! Video scopes - luma waveform, RGB histogram and vectorscope - computed from an RGBA8 frame.
//!
//! Everything works on 8-bit R'G'B' code values exactly as displayed: the preview is RGBA8 and
//! full range, so 0 is 0% and 255 is 100%. Luma and chroma use Rec.709 coefficients on those
//! non-linear values, which is what a grading scope shows.
//!
//! No Qt here. `compute` returns a premultiplied RGBA8 image of the trace; the UI scales it
//! into its box and draws the graticule on top, using `vectorscope_position` for the targets so
//! the graticule and the plotted trace cannot disagree.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScopeKind {
    Waveform = 0,
    Histogram = 1,
    Vectorscope = 2,
}

impl ScopeKind {
    pub fn from_index(index: i32) -> Option<Self> {
        match index {
            0 => Some(Self::Waveform),
            1 => Some(Self::Histogram),
            2 => Some(Self::Vectorscope),
            _ => None,
        }
    }
}

/// Trace resolution of each scope. Fixed, independent of the source, so the cost of `compute`
/// depends only on the sample count and the UI can scale the image freely.
pub const WAVEFORM_SIZE: (usize, usize) = (512, 256);
pub const HISTOGRAM_SIZE: (usize, usize) = (256, 128);
pub const VECTORSCOPE_SIZE: (usize, usize) = (256, 256);

/// How many pixels `Samples::from_frame` keeps by default: ~1/9 of a 1080p frame, enough
/// that the traces look continuous, and a copy small enough (~1 MB) to take on the render thread.
pub const DEFAULT_MAX_SAMPLES: usize = 256 * 1024;

/// Scales chroma so the most saturated colours (100% green and magenta, radius ~0.596) still
/// fit inside the unit-diameter plot.
pub const VECTORSCOPE_GAIN: f32 = 0.8;

/// The skin-tone ("I") line, degrees counter-clockwise from the +Cb axis.
pub const SKIN_TONE_ANGLE_DEG: f64 = 123.0;

/// A borrowed RGBA8 frame. `stride` is in bytes.
pub struct Frame<'a> {
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub data: &'a [u8],
}

/// A packed, subsampled copy of a frame: every `step`-th pixel in both directions, so the
/// aspect ratio is kept and waveform columns still map onto frame columns.
#[derive(Clone, Debug)]
pub struct Samples {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

impl Samples {
    /// `None` when the frame is empty or `data` is too short for `width`, `height` and `stride`.
    pub fn from_frame(frame: &Frame, max_pixels: usize) -> Option<Self> {
        let (w, h, stride) = (frame.width, frame.height, frame.stride);
        if w == 0 || h == 0 || stride < w * 4 || frame.data.len() < (h - 1) * stride + w * 4 {
            return None;
        }
        let step = (((w * h) as f64 / max_pixels.max(1) as f64).sqrt().ceil() as usize).max(1);
        let (sw, sh) = (w.div_ceil(step), h.div_ceil(step));

        let mut rgba = Vec::with_capacity(sw * sh * 4);
        for y in (0..h).step_by(step) {
            let row = &frame.data[y * stride..y * stride + w * 4];
            for px in row.chunks_exact(4).step_by(step) {
                rgba.extend_from_slice(px);
            }
        }
        Some(Self { width: sw, height: sh, rgba })
    }

    fn pixels(&self) -> impl Iterator<Item = (usize, usize, [u8; 3])> + '_ {
        let w = self.width;
        self.rgba.chunks_exact(4).enumerate().map(move |(i, p)| (i % w, i / w, [p[0], p[1], p[2]]))
    }
}

/// Premultiplied RGBA8, `width * height * 4` bytes, top row first.
#[derive(Clone, Debug)]
pub struct ScopeImage {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

/// Rec.709 luma of 8-bit R'G'B', in code values (0..=255).
pub fn luma(rgb: [u8; 3]) -> f32 {
    0.2126 * rgb[0] as f32 + 0.7152 * rgb[1] as f32 + 0.0722 * rgb[2] as f32
}

/// Rec.709 (Cb, Cr) of 8-bit R'G'B', each in -0.5..=0.5.
pub fn chroma(rgb: [u8; 3]) -> (f32, f32) {
    let (r, b) = (rgb[0] as f32 / 255.0, rgb[2] as f32 / 255.0);
    let y = luma(rgb) / 255.0;
    ((b - y) / 1.8556, (r - y) / 1.5748)
}

/// Where a colour lands on the vectorscope, as fractions of the plot (0..1, y down):
/// +Cb to the right, +Cr up, neutral grey in the centre.
pub fn vectorscope_position(rgb: [u8; 3]) -> (f32, f32) {
    let (cb, cr) = chroma(rgb);
    (0.5 + cb * VECTORSCOPE_GAIN, 0.5 - cr * VECTORSCOPE_GAIN)
}

/// The six 75% colour-bar targets: label and plot position.
pub fn vectorscope_targets() -> [(&'static str, (f32, f32)); 6] {
    const V: u8 = 191; // 75% of 255
    [
        ("R",  vectorscope_position([V, 0, 0])),
        ("Mg", vectorscope_position([V, 0, V])),
        ("B",  vectorscope_position([0, 0, V])),
        ("Cy", vectorscope_position([0, V, V])),
        ("G",  vectorscope_position([0, V, 0])),
        ("Yl", vectorscope_position([V, V, 0])),
    ]
}

pub fn compute(kind: ScopeKind, samples: &Samples) -> ScopeImage {
    match kind {
        ScopeKind::Waveform => {
            // No wider than the samples: with more columns than sample columns, some columns
            // get no hits at all and the trace turns into speckle (e.g. a vertical clip).
            let (w, h) = (WAVEFORM_SIZE.0.min(samples.width.max(1)), WAVEFORM_SIZE.1);
            render_density(&waveform_counts(samples, w, h), w, h, [190, 255, 190])
        }
        ScopeKind::Histogram => render_histogram(&histogram(samples), HISTOGRAM_SIZE.1),
        ScopeKind::Vectorscope => {
            let (w, h) = VECTORSCOPE_SIZE;
            render_density(&vectorscope_counts(samples, w, h), w, h, [255, 255, 255])
        }
    }
}

/// Hits per cell of a `w` x `h` grid: column from the pixel's x, row from its luma
/// (100% at the top row, 0% at the bottom).
pub fn waveform_counts(samples: &Samples, w: usize, h: usize) -> Vec<u32> {
    let mut counts = vec![0u32; w * h];
    if samples.width == 0 { return counts; }
    for (x, _, rgb) in samples.pixels() {
        let col = (x * w / samples.width).min(w - 1);
        let row = (((255.0 - luma(rgb)) / 255.0) * (h - 1) as f32).round() as usize;
        counts[row.min(h - 1) * w + col] += 1;
    }
    counts
}

/// Per-channel counts of each code value, `[r, g, b]`.
pub fn histogram(samples: &Samples) -> [[u32; 256]; 3] {
    let mut bins = [[0u32; 256]; 3];
    for (_, _, rgb) in samples.pixels() {
        for c in 0..3 {
            bins[c][rgb[c] as usize] += 1;
        }
    }
    bins
}

pub fn vectorscope_counts(samples: &Samples, w: usize, h: usize) -> Vec<u32> {
    let mut counts = vec![0u32; w * h];
    for (_, _, rgb) in samples.pixels() {
        let (x, y) = vectorscope_position(rgb);
        let col = (x * (w - 1) as f32).round().clamp(0.0, (w - 1) as f32) as usize;
        let row = (y * (h - 1) as f32).round().clamp(0.0, (h - 1) as f32) as usize;
        counts[row * w + col] += 1;
    }
    counts
}

/// Log-scaled density: a cell hit once is faint but visible, the busiest cell is full `tint`.
pub fn render_density(counts: &[u32], w: usize, h: usize, tint: [u8; 3]) -> ScopeImage {
    let max = counts.iter().copied().max().unwrap_or(0).max(1) as f32;
    let norm = (1.0 + max).ln();
    let mut rgba = vec![0u8; w * h * 4];
    for (px, &c) in rgba.chunks_exact_mut(4).zip(counts) {
        if c == 0 { continue; }
        // Floor of 0.25 so a single hit is not lost against the video behind the scope.
        let v = (0.25 + 0.75 * (1.0 + c as f32).ln() / norm).min(1.0);
        px[0] = (tint[0] as f32 * v) as u8;
        px[1] = (tint[1] as f32 * v) as u8;
        px[2] = (tint[2] as f32 * v) as u8;
        px[3] = (255.0 * v) as u8;
    }
    ScopeImage { width: w, height: h, rgba }
}

/// Overlapping R, G and B bars, added together (so where all three overlap it is white).
///
/// Scaled to the tallest bin *excluding* 0 and 255: clipped shadows or highlights pile into one
/// bin and would otherwise squash every other bar flat. Those two bins are drawn clamped.
pub fn render_histogram(bins: &[[u32; 256]; 3], h: usize) -> ScopeImage {
    let w = 256;
    let inner_max = bins.iter().flat_map(|b| b[1..255].iter().copied()).max().unwrap_or(0);
    let max = if inner_max > 0 { inner_max } else { bins.iter().flat_map(|b| b.iter().copied()).max().unwrap_or(0) }.max(1) as f32;

    let mut rgba = vec![0u8; w * h * 4];
    const ALPHA: u8 = 210;
    for x in 0..w {
        let heights = [0, 1, 2].map(|c| ((bins[c][x] as f32 / max).min(1.0) * h as f32).round() as usize);
        for (from_bottom, y) in (0..h).rev().enumerate() {
            let on = heights.map(|hc| from_bottom < hc);
            if !on.iter().any(|&b| b) { continue; }
            let px = &mut rgba[(y * w + x) * 4..(y * w + x) * 4 + 4];
            for c in 0..3 {
                px[c] = if on[c] { ALPHA } else { 0 }; // premultiplied: 255 * ALPHA / 255
            }
            px[3] = ALPHA;
        }
    }
    ScopeImage { width: w, height: h, rgba }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: usize, h: usize, rgb: [u8; 3]) -> Vec<u8> {
        (0..w * h).flat_map(|_| [rgb[0], rgb[1], rgb[2], 255]).collect()
    }
    fn samples_of(w: usize, h: usize, data: &[u8]) -> Samples {
        Samples::from_frame(&Frame { width: w, height: h, stride: w * 4, data }, usize::MAX).unwrap()
    }
    /// Cell with the most hits, as (col, row).
    fn peak(counts: &[u32], w: usize) -> (usize, usize) {
        let i = counts.iter().enumerate().max_by_key(|&(_, &c)| c).unwrap().0;
        (i % w, i / w)
    }

    #[test]
    fn sampling_honours_stride_and_budget() {
        // 4x2 frame with 8 bytes of row padding; the padding is poison and must never be read.
        let (w, h, stride) = (4, 2, 24);
        let mut data = vec![0xEEu8; stride * h];
        for y in 0..h { for x in 0..w { data[y * stride + x * 4..y * stride + x * 4 + 4].copy_from_slice(&[x as u8, y as u8, 0, 255]); } }
        let s = Samples::from_frame(&Frame { width: w, height: h, stride, data: &data }, usize::MAX).unwrap();
        assert_eq!((s.width, s.height), (4, 2));
        assert!(s.rgba.chunks_exact(4).all(|p| p[0] != 0xEE));

        let big = solid(1920, 1080, [1, 2, 3]);
        let s = Samples::from_frame(&Frame { width: 1920, height: 1080, stride: 1920 * 4, data: &big }, DEFAULT_MAX_SAMPLES).unwrap();
        assert!(s.width * s.height <= DEFAULT_MAX_SAMPLES, "{}x{}", s.width, s.height);
        assert_eq!(s.rgba.len(), s.width * s.height * 4);
    }

    #[test]
    fn sampling_rejects_short_or_empty_buffers() {
        let data = solid(4, 4, [0, 0, 0]);
        assert!(Samples::from_frame(&Frame { width: 4, height: 5, stride: 16, data: &data }, 100).is_none());
        assert!(Samples::from_frame(&Frame { width: 4, height: 4, stride: 8,  data: &data }, 100).is_none());
        assert!(Samples::from_frame(&Frame { width: 0, height: 4, stride: 16, data: &data }, 100).is_none());
    }

    #[test]
    fn waveform_maps_x_to_columns_and_luma_to_rows() {
        // Left half black, right half white.
        let (w, h) = (8, 4);
        let data: Vec<u8> = (0..w * h).flat_map(|i| { let v = if i % w < w / 2 { 0 } else { 255 }; [v, v, v, 255] }).collect();
        let (gw, gh) = WAVEFORM_SIZE;
        let counts = waveform_counts(&samples_of(w, h, &data), gw, gh);
        for col in 0..gw {
            let rows: Vec<usize> = (0..gh).filter(|r| counts[r * gw + col] > 0).collect();
            if rows.is_empty() { continue; }
            let expected = if col < gw / 2 { gh - 1 } else { 0 };
            assert_eq!(rows, vec![expected], "column {col}");
        }
        assert_eq!(counts.iter().sum::<u32>() as usize, w * h);
    }

    #[test]
    fn waveform_has_no_empty_columns_when_samples_are_narrow() {
        let data = solid(300, 40, [128, 128, 128]);
        let img = compute(ScopeKind::Waveform, &samples_of(300, 40, &data));
        assert!(img.width <= 300);
        for col in 0..img.width {
            assert!((0..img.height).any(|r| img.rgba[(r * img.width + col) * 4 + 3] > 0), "column {col} is empty");
        }
    }

    #[test]
    fn waveform_puts_mid_grey_mid_height() {
        let data = solid(16, 16, [128, 128, 128]);
        let (gw, gh) = WAVEFORM_SIZE;
        let (_, row) = peak(&waveform_counts(&samples_of(16, 16, &data), gw, gh), gw);
        assert!((row as i32 - (gh as i32 - 1) / 2).abs() <= 1, "row {row}");
    }

    #[test]
    fn histogram_counts_each_channel_separately() {
        let data = solid(5, 3, [10, 20, 30]);
        let bins = histogram(&samples_of(5, 3, &data));
        assert_eq!((bins[0][10], bins[1][20], bins[2][30]), (15, 15, 15));
        assert_eq!(bins.iter().map(|b| b.iter().sum::<u32>()).sum::<u32>(), 45);
    }

    #[test]
    fn histogram_scale_ignores_clipped_spikes() {
        // 90% clipped white, 10% mid grey: the grey bar must still reach the top.
        let (w, h) = (10, 10);
        let data: Vec<u8> = (0..w * h).flat_map(|i| { let v = if i < 10 { 128 } else { 255 }; [v, v, v, 255] }).collect();
        let img = render_histogram(&histogram(&samples_of(w, h, &data)), HISTOGRAM_SIZE.1);
        let top_row_alpha = img.rgba[(128 * 4) + 3];
        assert!(top_row_alpha > 0, "the grey bar was flattened by the clipped bin");
    }

    #[test]
    fn vectorscope_centres_neutrals() {
        for v in [0u8, 64, 128, 255] {
            let (x, y) = vectorscope_position([v, v, v]);
            assert!((x - 0.5).abs() < 1e-3 && (y - 0.5).abs() < 1e-3, "{v}: {x},{y}");
        }
    }

    #[test]
    fn vectorscope_plots_colour_bars_on_their_targets() {
        let (gw, gh) = VECTORSCOPE_SIZE;
        for (name, (tx, ty)) in vectorscope_targets() {
            let rgb = match name { "R" => [191, 0, 0], "Mg" => [191, 0, 191], "B" => [0, 0, 191], "Cy" => [0, 191, 191], "G" => [0, 191, 0], _ => [191, 191, 0] };
            let data = solid(8, 8, rgb);
            let (col, row) = peak(&vectorscope_counts(&samples_of(8, 8, &data), gw, gh), gw);
            assert!((col as f32 - tx * (gw - 1) as f32).abs() <= 1.0 && (row as f32 - ty * (gh - 1) as f32).abs() <= 1.0, "{name}");
        }
    }

    #[test]
    fn vectorscope_orientation_and_fit() {
        // Red is up-left (+Cr, -Cb), blue is right (+Cb): the conventional layout.
        let (rx, ry) = vectorscope_position([255, 0, 0]);
        let (bx, _) = vectorscope_position([0, 0, 255]);
        assert!(ry < 0.5 && rx < 0.5 && bx > 0.5);
        // Every fully saturated primary and secondary stays inside the plot circle.
        for rgb in [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]] {
            let (x, y) = vectorscope_position(rgb);
            assert!(((x - 0.5).powi(2) + (y - 0.5).powi(2)).sqrt() < 0.5, "{rgb:?}");
        }
    }

    #[test]
    fn skin_tone_line_passes_near_typical_skin() {
        let (cb, cr) = chroma([224, 172, 140]);
        let angle = (cr as f64).atan2(cb as f64).to_degrees();
        assert!((angle - SKIN_TONE_ANGLE_DEG).abs() < 8.0, "{angle}");
    }

    #[test]
    fn compute_returns_images_of_the_declared_sizes() {
        let data = solid(32, 18, [200, 100, 50]);
        let s = samples_of(32, 18, &data);
        for (kind, (w, h)) in [(ScopeKind::Waveform, (32, WAVEFORM_SIZE.1)), (ScopeKind::Histogram, HISTOGRAM_SIZE), (ScopeKind::Vectorscope, VECTORSCOPE_SIZE)] {
            let img = compute(kind, &s);
            assert_eq!((img.width, img.height, img.rgba.len()), (w, h, w * h * 4), "{kind:?}");
            assert!(img.rgba.chunks_exact(4).any(|p| p[3] > 0), "{kind:?} drew nothing");
            // Premultiplied: no channel may exceed alpha.
            assert!(img.rgba.chunks_exact(4).all(|p| p[0] <= p[3] && p[1] <= p[3] && p[2] <= p[3]), "{kind:?}");
        }
    }
}
