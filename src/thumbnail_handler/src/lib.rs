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

#[implement(IInitializeWithStream, IThumbnailProvider)]
struct GyroflowThumbnailProvider {
    /// The project file's bytes, captured in `Initialize`. Explorer always calls that before
    /// `GetThumbnail`; if it somehow does not, `GetThumbnail` fails rather than guessing.
    data: RefCell<Option<Vec<u8>>>,
}

impl Default for GyroflowThumbnailProvider {
    fn default() -> Self { Self { data: RefCell::new(None) } }
}

impl IInitializeWithStream_Impl for GyroflowThumbnailProvider_Impl {
    fn Initialize(&self, stream: Ref<'_, IStream>, _grfmode: u32) -> Result<()> {
        let stream = stream.ok()?;

        // Read the whole file. Project files are tens of kilobytes; streaming a JSON document
        // to find one key would be more code and no faster. The cap exists so that a file
        // which is not really a project cannot make the host allocate without bound.
        const MAX_PROJECT_BYTES: usize = 8 * 1024 * 1024;

        let mut buf = Vec::new();
        let mut chunk = [0u8; 64 * 1024];
        loop {
            let mut read = 0u32;
            unsafe { stream.Read(chunk.as_mut_ptr() as *mut _, chunk.len() as u32, Some(&mut read)) }.ok()?;
            if read == 0 { break; }
            buf.extend_from_slice(&chunk[..read as usize]);
            if buf.len() > MAX_PROJECT_BYTES {
                return Err(Error::from(E_FAIL));
            }
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

// ---------------------------------------------------------------------------------------
// COM plumbing
// ---------------------------------------------------------------------------------------

#[implement(IClassFactory)]
struct ClassFactory;

impl IClassFactory_Impl for ClassFactory_Impl {
    fn CreateInstance(&self, outer: Ref<'_, IUnknown>, iid: *const GUID, object: *mut *mut std::ffi::c_void) -> Result<()> {
        if !outer.is_null() { return Err(Error::from(CLASS_E_NOAGGREGATION)); }
        let provider: IInitializeWithStream = GyroflowThumbnailProvider::default().into();
        unsafe { provider.query(iid, object).ok() }
    }
    fn LockServer(&self, _lock: BOOL) -> Result<()> { Ok(()) }
}

#[no_mangle]
pub extern "system" fn DllGetClassObject(rclsid: *const GUID, riid: *const GUID, ppv: *mut *mut std::ffi::c_void) -> HRESULT {
    unsafe {
        if rclsid.is_null() || riid.is_null() || ppv.is_null() { return E_INVALIDARG; }
        if *rclsid != CLSID_GYROFLOW_THUMBNAIL { return CLASS_E_CLASSNOTAVAILABLE; }
        let factory: IClassFactory = ClassFactory.into();
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
