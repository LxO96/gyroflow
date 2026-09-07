// SPDX-License-Identifier: GPL-3.0-or-later

//! Windows Explorer thumbnail handler for `.gyroflow` project files.
//!
//! Explorer loads this DLL for every project file it draws, so it stays deliberately small:
//! JSON parse, base64 decode, and WIC (part of Windows) for the JPEG. No ffmpeg, no Qt, no
//! mdk, no `gyroflow-core`. The picture itself is embedded in the project by Gyroflow when
//! the video is loaded — see `rendering::generate_project_thumbnail`.
//!
//! It is registered to run **out of process** via an `AppID` with an empty `DllSurrogate`
//! (see `register.rs`), for two practical reasons: a fault here cannot take Explorer down
//! with it, and Explorer never holds a file lock on the DLL, which would otherwise make
//! updating Gyroflow fail with "file in use" until `explorer.exe` is killed.

#![cfg(windows)]

mod thumbnail;
mod register;
mod helper;

// The wire format is compiled from the helper crate's own source rather than copied, so the
// two can never disagree about layout or bounds.
#[path = "../../thumbnailer/src/protocol.rs"]
mod protocol;

use std::cell::RefCell;
use windows::core::*;
use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Gdi::*;
use windows::Win32::Graphics::Imaging::*;
use windows::Win32::System::Com::*;
use windows::Win32::UI::Shell::*;
use windows::Win32::UI::Shell::PropertiesSystem::*;

/// The handler's CLSID. Stable for the life of the product: it is written into the registry
/// and into installer manifests, so changing it orphans every existing registration.
pub const CLSID_GYROFLOW_THUMBNAIL: GUID = GUID::from_u128(0x6f1d8a42_9c3b_4f27_ae15_2b7c0d4e8a31);

/// A SECOND handler, for video files, registered under its own CLSID.
///
/// Two CLSIDs rather than one because the shell's two initialisation interfaces cannot be
/// mixed in a single object usefully: the shell prefers `IInitializeWithStream` whenever a
/// handler offers it, and a stream yields only a bare filename via `Stat` - no directory, so
/// nothing the decode helper can open. Declining the stream does not make the shell retry
/// through `IInitializeWithFile`; it abandons the handler entirely and falls back to whatever
/// Windows itself can do (measured: the request never reached this DLL at all).
///
/// So the video handler is a separate class that offers ONLY `IInitializeWithFile`, and its
/// registration sets `DisableProcessIsolation` so the shell will load a handler that cannot
/// work from a stream.
pub const CLSID_GYROFLOW_VIDEO_THUMBNAIL: GUID = GUID::from_u128(0x7c2e9b51_4d6a_4e83_9f27_1a58c3b7d094);

#[implement(IInitializeWithStream, IThumbnailProvider)]
/// Serves two quite different jobs behind one CLSID, and the two initialisation interfaces
/// exist to route between them.
///
/// The shell prefers `IInitializeWithStream` and will not load a handler that lacks it, so a
/// project file is read straight from the stream - no path needed, no subprocess. A video
/// cannot work that way: the helper must open the file itself, and `IStream::Stat` yields
/// only a bare filename. So the stream initialiser *declines* video files, which makes the
/// shell fall back to `IInitializeWithFile` and hand over the full path.
struct GyroflowThumbnailProvider {
    /// Project bytes, read from the stream in `Initialize`.
    data: RefCell<Option<Vec<u8>>>,
}

impl Default for GyroflowThumbnailProvider {
    fn default() -> Self { Self { data: RefCell::new(None) } }
}


impl IInitializeWithStream_Impl for GyroflowThumbnailProvider_Impl {
    fn Initialize(&self, stream: Ref<'_, IStream>, _grfmode: u32) -> Result<()> {
        let stream = stream.ok()?;

        // A project file: read it. Bounded, because Explorer hands this DLL whatever is on
        // disk under a `.gyroflow` name.
        const MAX_PROJECT_BYTES: usize = 8 * 1024 * 1024;
        let mut buf = Vec::new();
        let mut chunk = [0u8; 64 * 1024];
        loop {
            let mut read = 0u32;
            unsafe { stream.Read(chunk.as_mut_ptr() as *mut _, chunk.len() as u32, Some(&mut read)) }.ok()?;
            if read == 0 { break; }
            buf.extend_from_slice(&chunk[..read as usize]);
            if buf.len() > MAX_PROJECT_BYTES { return Err(Error::from(E_FAIL)); }
        }
        *self.data.borrow_mut() = Some(buf);
        Ok(())
    }
}

impl IThumbnailProvider_Impl for GyroflowThumbnailProvider_Impl {
    fn GetThumbnail(&self, cx: u32, phbmp: *mut HBITMAP, pdwalpha: *mut WTS_ALPHATYPE) -> Result<()> {
        let data = self.data.borrow();
        let data = data.as_ref().ok_or_else(|| Error::from(E_UNEXPECTED))?;

        // Any parse failure — including the ordinary "saved by an older Gyroflow" case — is
        // reported as "no thumbnail available". Explorer then draws the generic icon, exactly
        // as it did before this handler existed.
        let thumb = thumbnail::extract(data).map_err(|_| Error::from(E_FAIL))?;

        let bitmap = unsafe { decode_to_hbitmap(&thumb.jpeg, cx) }?;

        unsafe {
            *phbmp = bitmap;
            // The JPEG has no alpha channel, and claiming one would make Explorer composite
            // against uninitialised bits.
            *pdwalpha = WTSAT_RGB;
        }
        Ok(())
    }
}

/// Decode JPEG bytes with WIC and produce a 32-bit top-down DIB scaled to fit `max_edge`.
///
/// WIC rather than a Rust image crate: it ships with Windows, so the DLL gains no dependency
/// and no decoder of its own to keep patched.
unsafe fn decode_to_hbitmap(jpeg: &[u8], max_edge: u32) -> Result<HBITMAP> {
    unsafe {
        let factory: IWICImagingFactory = CoCreateInstance(&CLSID_WICImagingFactory, None, CLSCTX_INPROC_SERVER)?;

        let stream = factory.CreateStream()?;
        stream.InitializeFromMemory(jpeg)?;

        let decoder = factory.CreateDecoderFromStream(&stream, std::ptr::null(), WICDecodeMetadataCacheOnLoad)?;
        let frame = decoder.GetFrame(0)?;

        let (mut w, mut h) = (0u32, 0u32);
        frame.GetSize(&mut w, &mut h)?;
        if w == 0 || h == 0 { return Err(Error::from(E_FAIL)); }

        // Fit the long edge and never upscale: Explorer asks for the box it wants to fill, and
        // enlarging a 512px still to fill a 1024px request only produces a blurrier picture.
        let scale = (max_edge as f64 / w.max(h) as f64).min(1.0);
        let tw = ((w as f64 * scale).round() as u32).max(1);
        let th = ((h as f64 * scale).round() as u32).max(1);

        let scaler = factory.CreateBitmapScaler()?;
        scaler.Initialize(&frame, tw, th, WICBitmapInterpolationModeFant)?;

        let converter = factory.CreateFormatConverter()?;
        converter.Initialize(&scaler, &GUID_WICPixelFormat32bppBGR, WICBitmapDitherTypeNone, None, 0.0, WICBitmapPaletteTypeCustom)?;

        // Negative height requests a top-down DIB, matching WIC's row order. With a positive
        // height the picture comes out vertically mirrored.
        let mut info = BITMAPINFO::default();
        info.bmiHeader.biSize = std::mem::size_of::<BITMAPINFOHEADER>() as u32;
        info.bmiHeader.biWidth = tw as i32;
        info.bmiHeader.biHeight = -(th as i32);
        info.bmiHeader.biPlanes = 1;
        info.bmiHeader.biBitCount = 32;
        info.bmiHeader.biCompression = BI_RGB.0;

        let mut bits: *mut std::ffi::c_void = std::ptr::null_mut();
        let bitmap = CreateDIBSection(None, &info, DIB_RGB_COLORS, &mut bits, None, 0)?;
        if bits.is_null() {
            let _ = DeleteObject(bitmap.into());
            return Err(Error::from(E_FAIL));
        }

        let stride = tw * 4;
        let size = (stride * th) as usize;
        let dest = std::slice::from_raw_parts_mut(bits as *mut u8, size);
        if let Err(e) = converter.CopyPixels(std::ptr::null(), stride, dest) {
            // Otherwise the DIB leaks on every failed thumbnail, in a process that may be
            // asked for thousands of them.
            let _ = DeleteObject(bitmap.into());
            return Err(e);
        }

        Ok(bitmap)
    }
}

/// Wrap tightly packed BGRA in a top-down 32-bit DIB.
///
/// The helper already scaled and rotated, so this is a straight copy. Kept separate from the
/// WIC path because the two share nothing but the DIB shape.
unsafe fn bgra_to_hbitmap(bgra: &[u8], w: u32, h: u32) -> Result<HBITMAP> {
    unsafe {
        let expected = (w as usize) * (h as usize) * 4;
        // The helper's header is already bounds-checked, but this is the last point before a
        // raw pointer write - a mismatch here would read past the buffer.
        if w == 0 || h == 0 || bgra.len() != expected { return Err(Error::from(E_FAIL)); }

        let mut info = BITMAPINFO::default();
        info.bmiHeader.biSize = std::mem::size_of::<BITMAPINFOHEADER>() as u32;
        info.bmiHeader.biWidth = w as i32;
        // Negative height = top-down, matching the helper's row order. Positive would show
        // the picture vertically mirrored.
        info.bmiHeader.biHeight = -(h as i32);
        info.bmiHeader.biPlanes = 1;
        info.bmiHeader.biBitCount = 32;
        info.bmiHeader.biCompression = BI_RGB.0;

        let mut bits: *mut std::ffi::c_void = std::ptr::null_mut();
        let bitmap = CreateDIBSection(None, &info, DIB_RGB_COLORS, &mut bits, None, 0)?;
        if bits.is_null() {
            let _ = DeleteObject(bitmap.into());
            return Err(Error::from(E_FAIL));
        }
        std::ptr::copy_nonoverlapping(bgra.as_ptr(), bits as *mut u8, expected);
        Ok(bitmap)
    }
}

// ---------------------------------------------------------------------------------------
// COM plumbing
// ---------------------------------------------------------------------------------------

/// The video half: initialised with a real path, decodes through the helper subprocess.
#[implement(IInitializeWithFile, IThumbnailProvider)]
struct GyroflowVideoThumbnailProvider {
    path: RefCell<Option<String>>,
}

impl Default for GyroflowVideoThumbnailProvider {
    fn default() -> Self { Self { path: RefCell::new(None) } }
}

impl IInitializeWithFile_Impl for GyroflowVideoThumbnailProvider_Impl {
    fn Initialize(&self, path: &PCWSTR, _grfmode: u32) -> Result<()> {
        *self.path.borrow_mut() = Some(unsafe { path.to_string() }.map_err(|_| Error::from(E_INVALIDARG))?);
        Ok(())
    }
}

impl IThumbnailProvider_Impl for GyroflowVideoThumbnailProvider_Impl {
    fn GetThumbnail(&self, cx: u32, phbmp: *mut HBITMAP, pdwalpha: *mut WTS_ALPHATYPE) -> Result<()> {
        let path = self.path.borrow();
        let path = path.as_ref().ok_or_else(|| Error::from(E_UNEXPECTED))?;

        // Only decode what we actually claim. Registration is machine state that can be edited
        // or left behind by an older build, and spawning a decoder for a file type we never
        // advertised (a .braw, say, which needs a vendor SDK) would waste a process to fail.
        if !helper::is_video_extension(path) { return Err(Error::from(E_FAIL)); }

        // Bounded by a timeout inside `decode_frame`, so a corrupt or pathological file cannot
        // wedge the shell. Every failure becomes "no thumbnail" and Explorer draws its icon.
        let frame = helper::decode_frame(path, cx).ok_or_else(|| Error::from(E_FAIL))?;
        let bitmap = unsafe { bgra_to_hbitmap(&frame.bgra, frame.w, frame.h) }?;
        unsafe { *phbmp = bitmap; *pdwalpha = WTSAT_RGB; }
        Ok(())
    }
}

#[implement(IClassFactory)]
struct ClassFactory(bool); // true = video handler

impl IClassFactory_Impl for ClassFactory_Impl {
    fn CreateInstance(&self, outer: Ref<'_, IUnknown>, iid: *const GUID, object: *mut *mut std::ffi::c_void) -> Result<()> {
        if !outer.is_null() { return Err(Error::from(CLASS_E_NOAGGREGATION)); }
        if self.0 {
            let provider: IInitializeWithFile = GyroflowVideoThumbnailProvider::default().into();
            unsafe { provider.query(iid, object).ok() }
        } else {
            let provider: IInitializeWithStream = GyroflowThumbnailProvider::default().into();
            unsafe { provider.query(iid, object).ok() }
        }
    }
    fn LockServer(&self, _lock: BOOL) -> Result<()> { Ok(()) }
}

#[no_mangle]
pub extern "system" fn DllGetClassObject(rclsid: *const GUID, riid: *const GUID, ppv: *mut *mut std::ffi::c_void) -> HRESULT {
    unsafe {
        if rclsid.is_null() || riid.is_null() || ppv.is_null() { return E_INVALIDARG; }
        let is_video = *rclsid == CLSID_GYROFLOW_VIDEO_THUMBNAIL;
        if *rclsid != CLSID_GYROFLOW_THUMBNAIL && !is_video { return CLASS_E_CLASSNOTAVAILABLE; }
        let factory: IClassFactory = ClassFactory(is_video).into();
        factory.query(riid, ppv)
    }
}

/// Always `S_OK`: the surrogate host decides when to unload, and reporting "busy" from a
/// handler this small only keeps a `dllhost.exe` alive for no reason.
#[no_mangle]
pub extern "system" fn DllCanUnloadNow() -> HRESULT { S_OK }

#[no_mangle]
pub extern "system" fn DllRegisterServer() -> HRESULT { register::register(true) }

#[no_mangle]
pub extern "system" fn DllUnregisterServer() -> HRESULT { register::unregister(true) }

/// Per-user variants for the portable/zip build. `HKCU\Software\Classes` needs no admin
/// rights, so an unpacked Gyroflow can register itself without an elevation prompt.
#[no_mangle]
pub extern "system" fn DllRegisterServerUser() -> HRESULT { register::register(false) }

#[no_mangle]
pub extern "system" fn DllUnregisterServerUser() -> HRESULT { register::unregister(false) }

/// Opt-in registration for video extensions, driven by the Gyroflow setting. Separate from
/// the `.gyroflow` registration because claiming `.mp4` means owning every mp4 thumbnail on
/// the machine - see `register::register_video_types`.
#[no_mangle]
pub extern "system" fn DllRegisterVideoTypesUser() -> HRESULT { register::register_video_types(false) }

#[no_mangle]
pub extern "system" fn DllUnregisterVideoTypesUser() -> HRESULT { register::unregister_video_types(false) }

#[no_mangle]
pub extern "system" fn DllRegisterVideoTypes() -> HRESULT { register::register_video_types(true) }

#[no_mangle]
pub extern "system" fn DllUnregisterVideoTypes() -> HRESULT { register::unregister_video_types(true) }
