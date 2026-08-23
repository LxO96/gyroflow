// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2026 Adrian <adrian.eddy at gmail>

//! What colour space the *exported file* should declare itself to be in.
//!
//! This changes only the tags written on the output, never a single pixel. It exists because
//! baking a LUT usually changes what the footage *is* - an S-Log3 clip graded through a
//! log-to-709 LUT is Rec.709 afterwards - while the export otherwise inherits the source's
//! tags verbatim. Sony XAVC leaves the matrix, transfer and primaries unspecified at the
//! container level, so the result was a file carrying `color_range=pc` and nothing else.
//! Players with no declared matrix fall back to their own defaults, and enough of them assume
//! limited range for 10-bit YUV that a correct full-range export gets stretched on playback.
//! That was the whole of the "the export looks more contrasty than the preview" report.
//!
//! Gyroflow cannot *know* what a given `.cube` outputs, so this is the user's declaration.
//! `from_lut_name` only supplies the initial guess.

/// The tags to declare. Deliberately a short list of what LUTs actually target; anything
/// exotic stays `Unspecified`, which reproduces the old behaviour of inheriting the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputColorSpace {
    /// Inherit whatever the source declared. The historical behaviour, and still correct when
    /// no LUT is baked.
    #[default]
    Unspecified,
    Rec709,
    Srgb,
    DciP3D65,
    Rec2020Pq,
    Rec2020Hlg,
}

impl OutputColorSpace {
    pub const ALL: [OutputColorSpace; 6] = [
        Self::Unspecified, Self::Rec709, Self::Srgb, Self::DciP3D65, Self::Rec2020Pq, Self::Rec2020Hlg,
    ];

    /// Stable key for the project file and for QML. Never localise this.
    pub fn id(self) -> &'static str {
        match self {
            Self::Unspecified => "unspecified",
            Self::Rec709      => "rec709",
            Self::Srgb        => "srgb",
            Self::DciP3D65    => "dci_p3_d65",
            Self::Rec2020Pq   => "rec2020_pq",
            Self::Rec2020Hlg  => "rec2020_hlg",
        }
    }

    /// Shown in the dropdown.
    pub fn label(self) -> &'static str {
        match self {
            Self::Unspecified => "Same as source",
            Self::Rec709      => "Rec. 709",
            Self::Srgb        => "sRGB",
            Self::DciP3D65    => "DCI-P3 D65",
            Self::Rec2020Pq   => "Rec. 2020 PQ (HDR10)",
            Self::Rec2020Hlg  => "Rec. 2020 HLG",
        }
    }

    pub fn from_id(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|x| x.id() == s)
    }

    /// A first guess from the LUT's filename, offered as the dropdown's initial value and
    /// always overridable. Creators name LUTs after what they produce far more often than not
    /// ("S-Log3 to Rec709.cube"), so this is right most of the time and harmless when it is
    /// not - the user can see and change it before exporting.
    ///
    /// Order matters: the transfer-function markers are checked before the primaries, because
    /// "BT2020 PQ" contains both and the transfer is the more specific claim.
    pub fn from_lut_name(name: &str) -> Self {
        let n = name.to_ascii_lowercase();

        if contains_token(&n, "hlg")                                 { return Self::Rec2020Hlg; }
        if contains_token(&n, "pq") || contains_token(&n, "2084")    { return Self::Rec2020Pq;  }
        if n.contains("srgb") || n.contains("s-rgb")                 { return Self::Srgb;       }
        if contains_token(&n, "p3")                                  { return Self::DciP3D65;   }
        if contains_token(&n, "709")                                 { return Self::Rec709;     }

        // A bare "2020" is left alone on purpose. It is ambiguous - Rec.2020 says nothing
        // about the transfer - and it is the one token likely to appear in a filename for an
        // unrelated reason, namely a year.
        Self::Unspecified
    }
}

/// Substring search that refuses matches running into a neighbouring digit, so "709" does not
/// fire on "1709" and "2084" does not fire on "12084". Letters are not guarded: LUT names
/// concatenate words freely ("dcip3", "tos709") and requiring a word boundary there would miss
/// most real names. This is a guess with a visible override, not a parser.
fn contains_token(haystack: &str, needle: &str) -> bool {
    let bytes = haystack.as_bytes();
    let mut from = 0;
    while let Some(rel) = haystack[from..].find(needle) {
        let start = from + rel;
        let end = start + needle.len();
        let before_ok = start == 0 || !bytes[start - 1].is_ascii_digit();
        let after_ok  = end >= bytes.len() || !bytes[end].is_ascii_digit();
        if before_ok && after_ok { return true; }
        from = start + 1;
        if from >= haystack.len() { break; }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_round_trip_and_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for s in OutputColorSpace::ALL {
            assert!(seen.insert(s.id()), "duplicate id {}", s.id());
            assert_eq!(OutputColorSpace::from_id(s.id()), Some(s));
        }
        assert_eq!(OutputColorSpace::from_id("nonsense"), None);
    }

    #[test]
    fn default_is_unspecified_so_untagged_exports_behave_as_before() {
        assert_eq!(OutputColorSpace::default(), OutputColorSpace::Unspecified);
    }

    /// The real LUT from the report that started this.
    #[test]
    fn infers_rec709_from_the_sony_lut_name() {
        assert_eq!(OutputColorSpace::from_lut_name("SL3SG3Ctos709.cube"), OutputColorSpace::Rec709);
    }

    #[test]
    fn infers_from_common_lut_names() {
        for (name, expected) in [
            ("S-Log3 to Rec709.cube",        OutputColorSpace::Rec709),
            ("slog3_to_bt709_v2.cube",       OutputColorSpace::Rec709),
            ("VLog_to_sRGB.cube",            OutputColorSpace::Srgb),
            ("LogC to DCIP3 D65.cube",       OutputColorSpace::DciP3D65),
            ("HLG_to_2020.cube",             OutputColorSpace::Rec2020Hlg),
            ("slog3-to-bt2020-pq.cube",      OutputColorSpace::Rec2020Pq),
            ("st2084_grade.cube",            OutputColorSpace::Rec2020Pq),
            ("teal_and_orange.cube",         OutputColorSpace::Unspecified),
            ("",                             OutputColorSpace::Unspecified),
        ] {
            assert_eq!(OutputColorSpace::from_lut_name(name), expected, "for {name:?}");
        }
    }

    /// PQ and HLG both name Rec.2020 primaries, so a name carrying both a transfer marker and
    /// "2020" must resolve by the transfer rather than falling through.
    #[test]
    fn transfer_markers_win_over_bare_primaries() {
        assert_eq!(OutputColorSpace::from_lut_name("rec2020.cube"),     OutputColorSpace::Unspecified);
        assert_eq!(OutputColorSpace::from_lut_name("rec2020_pq.cube"),  OutputColorSpace::Rec2020Pq);
        assert_eq!(OutputColorSpace::from_lut_name("rec2020_hlg.cube"), OutputColorSpace::Rec2020Hlg);
    }

    /// A year in the filename must not be read as Rec.2020, and a longer number must not be
    /// read as Rec.709 - both are plausible in a real LUT library.
    #[test]
    fn digits_do_not_bleed_between_tokens() {
        assert_eq!(OutputColorSpace::from_lut_name("summer_2020_look.cube"), OutputColorSpace::Unspecified);
        assert_eq!(OutputColorSpace::from_lut_name("preset_1709.cube"),      OutputColorSpace::Unspecified);
        assert_eq!(OutputColorSpace::from_lut_name("preset_7091.cube"),      OutputColorSpace::Unspecified);
        // ...but a digit-adjacent letter is fine, which is how most names actually read.
        assert_eq!(OutputColorSpace::from_lut_name("SL3SG3Ctos709"),         OutputColorSpace::Rec709);
    }

    #[test]
    fn serde_uses_the_stable_ids() {
        for s in OutputColorSpace::ALL {
            let j = serde_json::to_value(s).unwrap();
            assert_eq!(j.as_str(), Some(s.id()));
            assert_eq!(serde_json::from_value::<OutputColorSpace>(j).unwrap(), s);
        }
    }
}
