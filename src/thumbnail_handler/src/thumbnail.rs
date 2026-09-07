// SPDX-License-Identifier: GPL-3.0-or-later

//! The part of the handler that has nothing to do with COM: project JSON in, RGBA pixels out.
//!
//! Kept separate from `lib.rs` precisely so it can be tested. COM registration is machine
//! state and a shell host is not something a unit test should conjure, but this half is pure
//! and carries all the parsing that can realistically be wrong.

use base64::Engine;

/// Everything that can go wrong reading a project's thumbnail.
///
/// A shell handler's only sane response to any of these is "no thumbnail" - Explorer then
/// draws the generic icon, which is what it drew before this feature existed. They are
/// distinguished only so failures can be told apart while debugging.
#[derive(Debug, PartialEq)]
pub enum ThumbError {
    /// The file is not JSON at all. A `.gyroflow` that fails here is corrupt.
    NotJson,
    /// Valid JSON with no `thumbnail` key: a project saved by an older Gyroflow. This is the
    /// expected, uninteresting case, not an error in any real sense.
    NoThumbnail,
    /// The key exists but is not shaped like a thumbnail, or the payload is not valid base64.
    Malformed,
}

/// A decoded thumbnail: JPEG bytes plus the dimensions the project claims for them.
pub struct RawThumbnail {
    pub jpeg: Vec<u8>,
    /// What the project claims the image is. Parsed and asserted by the tests, but NOT used
    /// for decoding: WIC reports the true dimensions from the JPEG itself, and trusting a
    /// number in the file over the image it describes is how you get a buffer mismatch.
    #[allow(dead_code)] pub declared_w: u32,
    #[allow(dead_code)] pub declared_h: u32,
}

/// Pull the JPEG payload out of a `.gyroflow` file's bytes.
///
/// The declared width/height are carried through but deliberately *not* trusted for decoding -
/// WIC reports the true dimensions from the JPEG itself. They exist so a caller can pick a
/// sensible scale before decoding, and so a mismatch is detectable.
pub fn extract(project_bytes: &[u8]) -> Result<RawThumbnail, ThumbError> {
    let value: serde_json::Value = serde_json::from_slice(project_bytes).map_err(|_| ThumbError::NotJson)?;

    let thumb = match value.get("thumbnail") {
        None | Some(serde_json::Value::Null) => return Err(ThumbError::NoThumbnail),
        Some(t) => t,
    };

    let b64 = thumb.get("jpeg_base64").and_then(|x| x.as_str()).ok_or(ThumbError::Malformed)?;
    if b64.is_empty() {
        return Err(ThumbError::Malformed);
    }

    // Tolerate a `data:` URL prefix even though Gyroflow does not write one. The same base64
    // travels through the UI as a data URL, so a hand-edited or third-party file may well
    // carry the prefix, and accepting it costs one line.
    let payload = b64.split_once(",").map(|x| x.1).unwrap_or(b64);

    let jpeg = base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .map_err(|_| ThumbError::Malformed)?;

    // A JPEG always starts FF D8 FF. Checking it here means a truncated or mislabelled payload
    // is rejected as data rather than handed to WIC to fail on.
    if jpeg.len() < 3 || jpeg[0] != 0xFF || jpeg[1] != 0xD8 || jpeg[2] != 0xFF {
        return Err(ThumbError::Malformed);
    }

    Ok(RawThumbnail {
        jpeg,
        declared_w: thumb.get("w").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        declared_h: thumb.get("h").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The smallest valid JPEG header we accept, base64'd.
    fn jpeg_b64() -> String {
        base64::engine::general_purpose::STANDARD.encode([0xFFu8, 0xD8, 0xFF, 0xE0, 0x00, 0x10])
    }

    #[test]
    fn extracts_a_well_formed_thumbnail() {
        let json = format!(r#"{{"title":"Gyroflow data file","thumbnail":{{"w":288,"h":512,"jpeg_base64":"{}"}}}}"#, jpeg_b64());
        let t = extract(json.as_bytes()).expect("should extract");
        assert_eq!((t.declared_w, t.declared_h), (288, 512));
        assert_eq!(&t.jpeg[..3], &[0xFF, 0xD8, 0xFF]);
    }

    /// A project written before this feature. This is the common case for existing files and
    /// must be distinguishable from corruption, not lumped in with it.
    #[test]
    fn a_project_without_a_thumbnail_is_not_an_error_condition() {
        let json = r#"{"title":"Gyroflow data file","version":4,"videofile":"file:///x.mp4"}"#;
        assert_eq!(extract(json.as_bytes()).err(), Some(ThumbError::NoThumbnail));
    }

    #[test]
    fn a_data_url_prefix_is_accepted() {
        let json = format!(r#"{{"thumbnail":{{"w":1,"h":1,"jpeg_base64":"data:image/jpg;base64,{}"}}}}"#, jpeg_b64());
        assert!(extract(json.as_bytes()).is_ok(), "a prefixed payload should still decode");
    }

    /// Explorer hands this DLL whatever is on disk under a `.gyroflow` name. None of it may
    /// panic - a panic in a shell extension is a crash in the host process.
    #[test]
    fn junk_is_rejected_without_panicking() {
        let cases: [(&[u8], ThumbError); 5] = [
            (b"not json at all", ThumbError::NotJson),
            (b"", ThumbError::NotJson),
            (br#"{"thumbnail":42}"#, ThumbError::Malformed),
            (br#"{"thumbnail":{"w":1,"h":1,"jpeg_base64":""}}"#, ThumbError::Malformed),
            (br#"{"thumbnail":{"w":1,"h":1,"jpeg_base64":"!!!not base64!!!"}}"#, ThumbError::Malformed),
        ];
        for (input, expected) in cases {
            assert_eq!(extract(input).err(), Some(expected), "input: {:?}", String::from_utf8_lossy(input));
        }
    }

    /// Valid base64 that is not a JPEG must be rejected here rather than passed to WIC.
    #[test]
    fn valid_base64_that_is_not_a_jpeg_is_rejected() {
        let b64 = base64::engine::general_purpose::STANDARD.encode(b"this decodes fine but is not an image");
        let json = format!(r#"{{"thumbnail":{{"w":1,"h":1,"jpeg_base64":"{b64}"}}}}"#);
        assert_eq!(extract(json.as_bytes()).err(), Some(ThumbError::Malformed));
    }

    /// Missing dimensions are survivable: WIC reports the real ones from the JPEG, so the
    /// declared values are a hint, not a requirement.
    #[test]
    fn missing_dimensions_still_yield_the_image() {
        let json = format!(r#"{{"thumbnail":{{"jpeg_base64":"{}"}}}}"#, jpeg_b64());
        let t = extract(json.as_bytes()).expect("dimensions are optional");
        assert_eq!((t.declared_w, t.declared_h), (0, 0));
    }
}
