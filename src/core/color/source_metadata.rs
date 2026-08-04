// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2026 Adrian <adrian.eddy at gmail>

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SourceExposure {
    pub white_balance_k: Option<f64>,
    pub wb_mode: Option<String>,
    pub iso: Option<f64>,
    pub shutter_s: Option<f64>,
    pub aperture: Option<f64>,
    pub ev: Option<f64>,
}

/// Takes the leading numeric run and ignores the rest, because these values
/// arrive as camera-specific display strings ("5500", "5500K", "f/2.8", "Auto").
pub fn parse_leading_number(s: &str) -> Option<f64> {
    let s = s.trim();
    let start = s.find(|c: char| c.is_ascii_digit() || c == '-' || c == '.')?;
    let rest = &s[start..];
    let end = rest.find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
                  .unwrap_or(rest.len());
    let v: f64 = rest[..end].parse().ok()?;
    v.is_finite().then_some(v)
}

/// Shutter arrives either as a fraction ("1/50") or as seconds ("0.02").
pub fn parse_shutter(s: &str) -> Option<f64> {
    let s = s.trim();
    if let Some((num, den)) = s.split_once('/') {
        let n = parse_leading_number(num)?;
        let d = parse_leading_number(den)?;
        if d.abs() < f64::EPSILON { return None; }
        return Some(n / d);
    }
    parse_leading_number(s).filter(|v| *v > 0.0)
}

impl SourceExposure {
    pub fn from_recording_settings(v: &serde_json::Value) -> Self {
        let get = |k: &str| -> Option<&str> { v.get(k).and_then(|x| x.as_str()) };

        let white_balance_k = get("White balance").and_then(parse_leading_number).filter(|k| (1000.0..=50000.0).contains(k));
        let wb_mode = get("White balance mode").map(|s| s.to_string());
        let iso = get("ISO").and_then(parse_leading_number).filter(|v| *v > 0.0);
        let shutter_s = get("Shutter speed").and_then(parse_shutter);
        let aperture = get("Iris").and_then(parse_leading_number).filter(|v| *v > 0.0);

        // EV = log2(N²/t) - log2(ISO/100), only when every input is present.
        let ev = match (aperture, shutter_s, iso) {
            (Some(n), Some(t), Some(iso)) if t > 0.0 => Some(((n * n) / t).log2() - (iso / 100.0).log2()),
            _ => None,
        };

        Self { white_balance_k, wb_mode, iso, shutter_s, aperture, ev }
    }

    pub fn is_empty(&self) -> bool {
        self.white_balance_k.is_none() && self.wb_mode.is_none() && self.iso.is_none()
            && self.shutter_s.is_none() && self.aperture.is_none() && self.ev.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_leading_number, parse_shutter, SourceExposure};

    #[test]
    fn parses_leading_numbers_forgivingly() {
        assert_eq!(parse_leading_number("5500"),    Some(5500.0));
        assert_eq!(parse_leading_number("5500K"),   Some(5500.0));
        assert_eq!(parse_leading_number("5500 K"),  Some(5500.0));
        assert_eq!(parse_leading_number(" 5500 K"), Some(5500.0));
        assert_eq!(parse_leading_number("f/2.8"),   Some(2.8));
        assert_eq!(parse_leading_number("Auto"),    None);
        assert_eq!(parse_leading_number(""),        None);
        assert_eq!(parse_leading_number("---"),     None);
    }

    #[test]
    fn parses_shutter_as_fraction() {
        assert_eq!(parse_shutter("1/50"), Some(0.02));
        assert_eq!(parse_shutter("0.02"), Some(0.02));
        assert_eq!(parse_shutter("Auto"), None);
    }

    #[test]
    fn extracts_white_balance_and_mode() {
        let json = serde_json::json!({ "White balance": "5500", "White balance mode": "Manual" });
        let s = SourceExposure::from_recording_settings(&json);
        assert_eq!(s.white_balance_k, Some(5500.0));
        assert_eq!(s.wb_mode.as_deref(), Some("Manual"));
    }

    #[test]
    fn auto_white_balance_yields_no_kelvin_but_keeps_mode() {
        let json = serde_json::json!({ "White balance": "Auto", "White balance mode": "Auto" });
        let s = SourceExposure::from_recording_settings(&json);
        assert_eq!(s.white_balance_k, None);
        assert_eq!(s.wb_mode.as_deref(), Some("Auto"));
    }

    #[test]
    fn computes_ev_when_all_inputs_present() {
        // ISO 100, 1/100 s, f/2.8  ->  log2(2.8²/0.01) - log2(1) = log2(784) ≈ 9.61
        let json = serde_json::json!({ "ISO": "100", "Shutter speed": "1/100", "Iris": "f/2.8" });
        let s = SourceExposure::from_recording_settings(&json);
        assert!((s.ev.unwrap() - 9.614).abs() < 0.01, "ev was {:?}", s.ev);
    }

    #[test]
    fn ev_accounts_for_iso() {
        // Doubling ISO lowers EV by exactly 1 stop.
        let base = serde_json::json!({ "ISO": "100", "Shutter speed": "1/100", "Iris": "f/2.8" });
        let high = serde_json::json!({ "ISO": "200", "Shutter speed": "1/100", "Iris": "f/2.8" });
        let a = SourceExposure::from_recording_settings(&base).ev.unwrap();
        let b = SourceExposure::from_recording_settings(&high).ev.unwrap();
        assert!((a - b - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ev_is_none_when_any_input_missing() {
        let json = serde_json::json!({ "ISO": "100", "Shutter speed": "1/100" });
        assert_eq!(SourceExposure::from_recording_settings(&json).ev, None);
    }

    #[test]
    fn empty_settings_yield_all_none() {
        let s = SourceExposure::from_recording_settings(&serde_json::json!({}));
        assert!(s.white_balance_k.is_none() && s.iso.is_none() && s.ev.is_none());
        assert!(s.is_empty());
    }
}
