// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2026 Adrian <adrian.eddy at gmail>

//! Whether the exported file's luma occupies the full code range or the narrower video range.
//!
//! Unlike [`super::OutputColorSpace`], which only writes tags, this one *converts pixels*. The
//! two are easy to confuse and must not be: declaring `color_range=tv` on samples that still
//! span the full range would produce exactly the fault this exists to fix, only worse.
//!
//! The problem it solves: baking a log-to-display LUT expands contrast into the full 0..1023
//! code range, and Gyroflow then encodes that verbatim and tags it `color_range=pc`. The tag is
//! accurate, but a great many players ignore the range tag for YUV and assume video range
//! regardless, mapping code 64 to black and 940 to white. On a full-range file that applies an
//! unwanted gain of 1023/876 = 1.168 about a pivot at code 445 - blacks crush, highlights clip,
//! and the export looks "hotter" than the preview it came from.
//!
//! Sources usually hide this. Camera-original log footage lives well inside the video-range box
//! (the Sony XAVC clip this was traced on spans codes 78..883), so decoding it either way looks
//! nearly the same. The grade is what pushes it out to 31..1016 and makes the viewer's wrong
//! assumption visible - which is why the report was "the *export* looks hot", not "the camera
//! files look hot".
//!
//! [`Limited`](OutputRange::Limited) sidesteps the disagreement by encoding what those players
//! already assume, at the cost of ~15% of the code range (a real but usually invisible loss of
//! quantisation headroom, and a real one to avoid if the file is going back into a grading
//! suite rather than to a viewer).

/// What range to encode the output in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputRange {
    /// Encode whatever range the decoded source frame carried. The historical behaviour.
    #[default]
    SameAsSource,
    /// Video/TV range: black at 16/255, white at 235/255 (64 and 940 at 10-bit). What almost
    /// every player assumes for YUV, so it is the safe choice for a file being watched rather
    /// than re-graded.
    Limited,
    /// Full/PC range, 0 to max. Preserves every code value, and relies on the player honouring
    /// `color_range=pc`.
    Full,
}

impl OutputRange {
    pub const ALL: [OutputRange; 3] = [Self::SameAsSource, Self::Limited, Self::Full];

    /// Stable key for the project file and for QML. Never localise this.
    pub fn id(self) -> &'static str {
        match self {
            Self::SameAsSource => "same_as_source",
            Self::Limited      => "limited",
            Self::Full         => "full",
        }
    }

    /// Shown in the dropdown.
    pub fn label(self) -> &'static str {
        match self {
            Self::SameAsSource => "Same as source",
            Self::Limited      => "Limited (video, 16-235)",
            Self::Full         => "Full (0-255)",
        }
    }

    pub fn from_id(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|x| x.id() == s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_round_trip_and_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for r in OutputRange::ALL {
            assert!(seen.insert(r.id()), "duplicate id {}", r.id());
            assert_eq!(OutputRange::from_id(r.id()), Some(r));
        }
        assert_eq!(OutputRange::from_id("nonsense"), None);
    }

    /// The default has to be the do-nothing option: converting range is a pixel change, and
    /// silently applying one to everybody's existing exports would be a regression, not a fix.
    #[test]
    fn default_inherits_the_source() {
        assert_eq!(OutputRange::default(), OutputRange::SameAsSource);
    }

    #[test]
    fn serde_uses_the_stable_ids() {
        for r in OutputRange::ALL {
            let j = serde_json::to_value(r).unwrap();
            assert_eq!(j.as_str(), Some(r.id()));
            assert_eq!(serde_json::from_value::<OutputRange>(j).unwrap(), r);
        }
    }
}
