// Copyright © 2026 Gyroflow

//! The eight stocks of the handover, verbatim (Photo360 `film.js` STOCKS), and the film formats.

#[derive(Clone, Copy, Debug)]
pub struct Layer { pub gamma: f64, pub log_e0: f64, pub u_min: f64, pub u_max: f64 }
#[derive(Clone, Copy, Debug)]
pub struct GrainSpec { pub radius_median_px: f64, pub radius_log_sigma: f64, pub filter_sigma_px: f64 }
#[derive(Clone, Copy, Debug)]
pub struct Halation { pub gain: f64, pub tint: [f64; 3] }
#[derive(Clone, Copy, Debug)]
pub struct Stock {
    pub id: &'static str,
    pub name: &'static str,
    pub bw: bool,
    /// 3x3 row-major: linear rgb -> layer exposure.
    pub sensitivity: [f64; 9],
    pub layers: [Layer; 3],
    pub grain: GrainSpec,
    /// 3x3 row-major: developed layers -> output rgb.
    pub crosstalk: [f64; 9],
    pub halation: Halation,
}

const fn l(gamma: f64, log_e0: f64, u_min: f64, u_max: f64) -> Layer { Layer { gamma, log_e0, u_min, u_max } }
const fn g(radius_median_px: f64, radius_log_sigma: f64) -> GrainSpec { GrainSpec { radius_median_px, radius_log_sigma, filter_sigma_px: 0.8 } }
const HALO_STD: Halation = Halation { gain: 0.6, tint: [1.0, 0.28, 0.05] };

pub const STOCKS: [Stock; 8] = [
    Stock { id: "portrait160", name: "Portrait 160", bw: false,
        sensitivity: [1.02, -0.01, -0.01, 0.00, 1.00, 0.00, -0.01, -0.02, 0.96],
        layers: [l(0.50, -0.82, 0.040, 0.960), l(0.50, -0.82, 0.040, 0.960), l(0.49, -0.80, 0.045, 0.955)],
        grain: g(0.09, 0.30), crosstalk: [1.04, -0.03, -0.01, -0.02, 1.04, -0.02, -0.01, -0.03, 1.04], halation: HALO_STD },
    Stock { id: "golden200", name: "Golden 200", bw: false,
        sensitivity: [1.06, -0.02, -0.04, -0.01, 1.00, 0.01, -0.03, -0.05, 0.92],
        layers: [l(0.62, -0.80, 0.035, 0.965), l(0.62, -0.79, 0.030, 0.965), l(0.60, -0.76, 0.045, 0.950)],
        grain: g(0.12, 0.35), crosstalk: [1.12, -0.08, -0.04, -0.06, 1.10, -0.04, -0.03, -0.09, 1.12], halation: HALO_STD },
    Stock { id: "vivid100", name: "Vivid 100", bw: false,
        sensitivity: [1.08, -0.05, -0.03, -0.04, 1.08, -0.04, -0.03, -0.06, 1.09],
        layers: [l(0.70, -0.80, 0.030, 0.970), l(0.70, -0.79, 0.028, 0.970), l(0.68, -0.78, 0.035, 0.960)],
        grain: g(0.07, 0.30), crosstalk: [1.16, -0.10, -0.06, -0.07, 1.15, -0.08, -0.04, -0.11, 1.15], halation: HALO_STD },
    Stock { id: "pastel400", name: "Pastel 400", bw: false,
        sensitivity: [0.96, 0.02, 0.02, 0.00, 0.99, 0.05, 0.02, 0.04, 0.98],
        layers: [l(0.41, -0.90, 0.085, 0.952), l(0.41, -0.92, 0.090, 0.952), l(0.40, -0.88, 0.095, 0.948)],
        grain: g(0.15, 0.35), crosstalk: [0.95, 0.03, 0.02, 0.03, 0.95, 0.02, 0.02, 0.03, 0.95], halation: HALO_STD },
    Stock { id: "verdant400", name: "Verdant 400", bw: false,
        sensitivity: [0.97, -0.01, -0.01, -0.01, 1.04, -0.01, -0.01, 0.00, 1.04],
        layers: [l(0.60, -0.76, 0.030, 0.960), l(0.58, -0.85, 0.055, 0.960), l(0.60, -0.80, 0.040, 0.955)],
        grain: g(0.16, 0.35), crosstalk: [1.08, -0.05, -0.03, -0.04, 1.08, -0.04, -0.02, -0.06, 1.08], halation: HALO_STD },
    Stock { id: "tungsten800", name: "Tungsten 800", bw: false,
        sensitivity: [0.80, -0.01, -0.01, -0.02, 1.00, 0.02, 0.00, 0.05, 1.22],
        layers: [l(0.58, -0.82, 0.045, 0.955), l(0.58, -0.80, 0.040, 0.955), l(0.56, -0.78, 0.050, 0.950)],
        grain: g(0.20, 0.40), crosstalk: [1.10, -0.06, -0.04, -0.05, 1.09, -0.04, -0.03, -0.08, 1.11],
        halation: Halation { gain: 1.8, tint: [1.0, 0.15, 0.02] } },
    Stock { id: "classic400bw", name: "Classic 400 B&W", bw: true,
        sensitivity: [0.30, 0.59, 0.11, 0.30, 0.59, 0.11, 0.30, 0.59, 0.11],
        layers: [l(0.78, -0.74, 0.020, 0.980), l(0.78, -0.74, 0.020, 0.980), l(0.78, -0.74, 0.020, 0.980)],
        grain: g(0.22, 0.40), crosstalk: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], halation: HALO_STD },
    Stock { id: "smooth400bw", name: "Smooth 400 B&W", bw: true,
        sensitivity: [0.25, 0.62, 0.13, 0.25, 0.62, 0.13, 0.25, 0.62, 0.13],
        layers: [l(0.56, -0.78, 0.040, 0.960), l(0.56, -0.78, 0.040, 0.960), l(0.56, -0.78, 0.040, 0.960)],
        grain: g(0.13, 0.30), crosstalk: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], halation: HALO_STD },
];

pub fn stock(id: &str) -> Option<&'static Stock> { STOCKS.iter().find(|s| s.id == id) }

/// The film gauge the frame stands for: a smaller gate means coarser grain and wider halation
/// at the same output size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FilmFormat { Super8, Super16, #[default] Mm35, Medium120 }

impl FilmFormat {
    pub const ALL: [FilmFormat; 4] = [Self::Super8, Self::Super16, Self::Mm35, Self::Medium120];
    /// Stable key for the project file, CLI and QML. Never localise.
    pub fn id(self) -> &'static str { match self { Self::Super8 => "s8", Self::Super16 => "s16", Self::Mm35 => "35", Self::Medium120 => "120" } }
    pub fn from_id(id: &str) -> Option<Self> { Self::ALL.into_iter().find(|f| f.id() == id) }
    pub fn label(self) -> &'static str { match self { Self::Super8 => "Super 8", Self::Super16 => "Super 16", Self::Mm35 => "35 mm", Self::Medium120 => "120 (6×6)" } }
    pub fn gate_short_mm(self) -> f64 { match self { Self::Super8 => 4.01, Self::Super16 => 7.41, Self::Mm35 => 24.0, Self::Medium120 => 56.0 } }
}
