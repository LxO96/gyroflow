// SPDX-License-Identifier: GPL-3.0-or-later

//! Registry wiring for the thumbnail handler.
//!
//! Two scopes, deliberately: `HKLM` for the MSI (machine-wide, needs elevation) and `HKCU`
//! for the portable/zip build, which must never write `HKLM` — an unpacked Gyroflow that
//! demanded elevation to show icons would be worse than showing none. Both write the same
//! shape under `Software\Classes`, which is where `HKCR` resolves from anyway.

use windows::core::*;
use windows::Win32::Foundation::*;
use windows::Win32::System::LibraryLoader::*;
use windows::Win32::System::Registry::*;

use crate::{ CLSID_GYROFLOW_THUMBNAIL, CLSID_GYROFLOW_VIDEO_THUMBNAIL };

/// Shell's `IThumbnailProvider` handler slot. Not ours — this GUID is Windows'.
const THUMBNAIL_HANDLER_SLOT: &str = "{e357fccd-a995-4576-b01f-234630154e96}";

const PROGID: &str = "Gyroflow.Project";
const EXTENSION: &str = ".gyroflow";

fn clsid_string() -> String {
    format!("{{{:?}}}", CLSID_GYROFLOW_THUMBNAIL)
}

fn video_clsid_string() -> String {
    format!("{{{:?}}}", CLSID_GYROFLOW_VIDEO_THUMBNAIL)
}

fn root(machine_wide: bool) -> HKEY {
    if machine_wide { HKEY_LOCAL_MACHINE } else { HKEY_CURRENT_USER }
}

/// Full path of this DLL. Registration stores an absolute path, so moving or renaming the
/// install directory leaves a dangling handler — which is why uninstall must unregister.
pub fn this_dll_path() -> Result<String> {
    unsafe {
        let mut module = Default::default();
        GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            PCSTR(this_dll_path as *const () as *const u8),
            &mut module,
        )?;
        let mut buf = [0u16; 32768];
        let len = GetModuleFileNameW(Some(module.into()), &mut buf);
        if len == 0 { return Err(Error::from(E_FAIL)); }
        Ok(String::from_utf16_lossy(&buf[..len as usize]))
    }
}

fn set_value(root: HKEY, subkey: &str, name: Option<&str>, value: &str) -> Result<()> {
    unsafe {
        let mut key = HKEY::default();
        let sub: Vec<u16> = subkey.encode_utf16().chain(std::iter::once(0)).collect();
        RegCreateKeyExW(root, PCWSTR(sub.as_ptr()), None, PWSTR::null(), REG_OPTION_NON_VOLATILE, KEY_WRITE, None, &mut key, None).ok()?;

        let data: Vec<u16> = value.encode_utf16().chain(std::iter::once(0)).collect();
        let bytes = std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2);
        let name_w: Option<Vec<u16>> = name.map(|n| n.encode_utf16().chain(std::iter::once(0)).collect());
        let name_ptr = name_w.as_ref().map(|n| PCWSTR(n.as_ptr())).unwrap_or(PCWSTR::null());

        let r = RegSetValueExW(key, name_ptr, None, REG_SZ, Some(bytes));
        let _ = RegCloseKey(key);
        r.ok()
    }
}

fn set_dword(root: HKEY, subkey: &str, name: &str, value: u32) -> Result<()> {
    unsafe {
        let mut key = HKEY::default();
        let sub: Vec<u16> = subkey.encode_utf16().chain(std::iter::once(0)).collect();
        RegCreateKeyExW(root, PCWSTR(sub.as_ptr()), None, PWSTR::null(), REG_OPTION_NON_VOLATILE, KEY_WRITE, None, &mut key, None).ok()?;
        let name_w: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
        let r = RegSetValueExW(key, PCWSTR(name_w.as_ptr()), None, REG_DWORD, Some(&value.to_ne_bytes()));
        let _ = RegCloseKey(key);
        r.ok()
    }
}

fn delete_tree(root: HKEY, subkey: &str) {
    unsafe {
        let sub: Vec<u16> = subkey.encode_utf16().chain(std::iter::once(0)).collect();
        // Failure is ignored on purpose: unregistering something that was never registered,
        // or was already removed, is a success from the caller's point of view.
        let _ = RegDeleteTreeW(root, PCWSTR(sub.as_ptr()));
    }
}

pub fn register(machine_wide: bool) -> HRESULT {
    match try_register(machine_wide) {
        Ok(()) => S_OK,
        Err(e) => e.code(),
    }
}

fn try_register(machine_wide: bool) -> Result<()> {
    let hk = root(machine_wide);
    let clsid = clsid_string();
    let dll = this_dll_path()?;
    let classes = "Software\\Classes";

    // The COM server itself.
    let clsid_key = format!("{classes}\\CLSID\\{clsid}");
    set_value(hk, &clsid_key, None, "Gyroflow project thumbnail handler")?;
    set_value(hk, &format!("{clsid_key}\\InprocServer32"), None, &dll)?;
    set_value(hk, &format!("{clsid_key}\\InprocServer32"), Some("ThreadingModel"), "Apartment")?;

    // Run in a surrogate host rather than inside Explorer. An empty `DllSurrogate` means
    // "use the default host" (dllhost.exe). This is what keeps a fault here from taking
    // Explorer with it, and what stops Explorer holding a lock on the DLL during updates.
    set_value(hk, &format!("{clsid_key}"), Some("AppID"), &clsid)?;
    set_value(hk, &format!("{classes}\\AppID\\{clsid}"), Some("DllSurrogate"), "")?;

    // A ProgID, so the file type also gets a name and an icon rather than just a picture.
    set_value(hk, &format!("{classes}\\{PROGID}"), None, "Gyroflow project")?;
    set_value(hk, &format!("{classes}\\{EXTENSION}"), None, PROGID)?;

    // The handler slot, registered in BOTH places on purpose.
    //
    // Microsoft documents the ProgID location, and that is the one that travels with the file
    // type. But the shell resolves thumbnail handlers from the EXTENSION key: with only the
    // ProgID entry present, `IShellItemImageFactory::GetImage` returned
    // WTS_E_FAILEDEXTRACTION (0x8004B200) without ever loading this DLL - confirmed by
    // instrumenting it and watching it never be called. Adding the extension entry made the
    // same call succeed immediately.
    set_value(hk, &format!("{classes}\\{PROGID}\\ShellEx\\{THUMBNAIL_HANDLER_SLOT}"), None, &clsid)?;
    set_value(hk, &format!("{classes}\\{EXTENSION}\\ShellEx\\{THUMBNAIL_HANDLER_SLOT}"), None, &clsid)?;

    Ok(())
}

/// Register the handler for the video extensions listed in `helper::VIDEO_EXTENSIONS`.
///
/// Separate from `register()` because this is opt-in and `.gyroflow` is not. Windows has no
/// thumbnail-handler fallback chaining: claiming `.mp4` means owning EVERY mp4 thumbnail on
/// the machine, including the 8-bit ones Windows already handles well. That is the user's
/// call to make, not ours, which is why it sits behind a setting and ships off.
pub fn register_video_types(machine_wide: bool) -> HRESULT {
    match try_register_video(machine_wide) {
        Ok(()) => S_OK,
        Err(e) => e.code(),
    }
}

fn try_register_video(machine_wide: bool) -> Result<()> {
    let hk = root(machine_wide);
    let clsid = video_clsid_string();
    let dll = this_dll_path()?;
    let classes = "Software\\Classes";

    // The video handler is its own COM server under its own CLSID.
    let key = format!("{classes}\\CLSID\\{clsid}");
    set_value(hk, &key, None, "Gyroflow video thumbnail handler")?;
    set_value(hk, &format!("{key}\\InprocServer32"), None, &dll)?;
    set_value(hk, &format!("{key}\\InprocServer32"), Some("ThreadingModel"), "Apartment")?;

    // This handler initialises from a FILE rather than a stream, because the decode
    // helper must open the file itself and `IStream::Stat` yields only a bare filename.
    // The shell's sandboxed thumbnail host accepts only stream-based handlers and
    // silently declines one that cannot work that way - measured: the request never
    // reached this DLL at all. Disabling process isolation is what makes it load.
    set_dword(hk, &key, "DisableProcessIsolation", 1)?;

    for ext in crate::helper::VIDEO_EXTENSIONS {
        // Only the handler slot. No ProgID and no default value: the file type belongs to
        // whatever the user has associated it with, and we are adding a thumbnail, not
        // taking over the association.
        set_value(hk, &format!("{classes}\\{ext}\\ShellEx\\{THUMBNAIL_HANDLER_SLOT}"), None, &clsid)?;
    }
    Ok(())
}

/// Remove the video registrations, leaving `.gyroflow` alone.
pub fn unregister_video_types(machine_wide: bool) -> HRESULT {
    let hk = root(machine_wide);
    let classes = "Software\\Classes";
    for ext in crate::helper::VIDEO_EXTENSIONS {
        delete_tree(hk, &format!("{classes}\\{ext}\\ShellEx\\{THUMBNAIL_HANDLER_SLOT}"));
    }
    delete_tree(hk, &format!("{classes}\\CLSID\\{}", video_clsid_string()));
    S_OK
}

pub fn unregister(machine_wide: bool) -> HRESULT {
    let hk = root(machine_wide);
    let clsid = clsid_string();
    let classes = "Software\\Classes";

    // Order matters only in that the handler slot should go first: if a later delete fails,
    // Explorer is already no longer pointed at a DLL that may be about to disappear.
    delete_tree(hk, &format!("{classes}\\{PROGID}\\ShellEx\\{THUMBNAIL_HANDLER_SLOT}"));
    delete_tree(hk, &format!("{classes}\\CLSID\\{clsid}"));
    delete_tree(hk, &format!("{classes}\\AppID\\{clsid}"));
    delete_tree(hk, &format!("{classes}\\{PROGID}"));

    // Opt-in video registrations go too: an uninstall must not leave a handler pointing
    // at a DLL that is about to disappear.
    let _ = unregister_video_types(machine_wide);

    // Our handler slot under the extension goes too, but the extension key itself stays:
    // another application may have taken over `.gyroflow` since, and removing that would
    // break their association rather than ours.
    delete_tree(hk, &format!("{classes}\\{EXTENSION}\\ShellEx\\{THUMBNAIL_HANDLER_SLOT}"));

    S_OK
}
