# Windows thumbnails for Gyroflow project files

**Date:** 2026-09-03
**Status:** approved (design), implementation to follow directly — no separate plan document,
at the user's instruction.

## The problem

A `.gyroflow` project shows a generic icon in Windows Explorer, so picking the right project
out of a folder means opening them one at a time. The goal is to see, at a glance, which file
you are about to open.

## What Windows already handles

Measured on the development machine by querying
`HKLM\SOFTWARE\Classes\<ext>\ShellEx\{e357fccd-a995-4576-b01f-234630154e96}`:

| Already thumbnailed | Blank today |
|---|---|
| `.mp4` `.mov` `.mkv` `.webm` `.insv` `.braw` `.dng` | `.gyroflow` `.mxf` `.360` `.r3d` `.nev` `.exr` |

`.insv` and `.braw` are only covered because Insta360 Studio and the BRAW player are installed;
on a clean machine they are blank too.

**Scope decision:** this design covers `.gyroflow` only. The media formats are excluded, for a
reason that is structural rather than a matter of effort: `.braw`, `.r3d` and `.nev` decode
through **mdk** with vendor SDK plugins (`mdk_processor.rs:41-47`), not through ffmpeg. Putting
that stack behind an Explorer thumbnail request means loading a media SDK into a shell
extension for every file browsed. `.mxf`, `.360` and `.exr` could go through ffmpeg, but that
is still a media stack in Explorer's process for a marginal gain. If those are wanted later
they are their own project.

## Approach

Rejected alternatives, briefly:

- **Decode on demand.** The handler links ffmpeg and decodes a frame itself. Covers old project
  files and more formats, but pays a decode per thumbnail, and a project whose source video has
  moved yields nothing — a common state for an editing project.
- **Everything via mdk.** Ruled out above.

**Chosen: embed the thumbnail in the project file, and keep the handler dependency-free.**

Gyroflow already produces exactly the required image. `util::image_data_to_base64`
(`src/util.rs:450`) encodes a decoded frame as a **base64 JPEG at quality 50**, and the render
queue already calls it (`render_queue.rs:1258`). The producer side is therefore a reuse, not new
imaging code.

The consequence to accept: **only projects saved after this change get a thumbnail.** Existing
files stay blank until re-saved. This was chosen over the decode-on-demand fallback because it
keeps the handler free of any media dependency, which is what makes it fast and safe.

## Part 1 — the file format

New top-level key in `.gyroflow`:

```json
"thumbnail": { "w": 512, "h": 288, "jpeg_base64": "…" }
```

- **Capped at 512 px on the long edge.** Explorer normally requests 96–256 px and rarely more
  than 1024. 512 covers real use at roughly 8–20 KB per project.
- **Written on save**, from the same decoded frame the render queue uses.
- **Never written into presets.** A preset is settings meant to apply to any clip; one clip's
  frame would be both wrong and bloat. The export path already distinguishes presets via
  `is_preset`.
- **Failure is silent and non-fatal.** Source offline, unsupported codec, decode error: the key
  is absent and the file saves normally. A project failing to save because a thumbnail failed
  would be far worse than a blank icon.
- **No version bump.** The importer reads with `obj.get(...)` and ignores unknown keys, so older
  Gyroflow builds open these files unchanged. *This must be verified in code, not assumed.*

## Part 2 — the Explorer thumbnail handler

A new standalone crate `src/thumbnail_handler/`, `crate-type = ["cdylib"]`, producing
`gyroflow_thumbnail.dll`. Deliberately separate from the Gyroflow binary: Explorer loads it on
demand, so weight is paid per file browsed.

The repo has no `[workspace]` table (`gyroflow-core` is a path dependency), so this follows the
existing pattern as another separately-built crate rather than introducing a workspace.

**Interfaces**

- `IInitializeWithStream` — Explorer supplies an `IStream`. Preferred over
  `IInitializeWithFile`: it also works for files inside archives and on network shares, and
  needs no path handling of its own.
- `IThumbnailProvider::GetThumbnail(cx)` — parse the JSON, take `thumbnail.jpeg_base64`, decode
  via **WIC** (`IWICImagingFactory`, part of Windows), scale to `cx`, return an `HBITMAP` with
  `WTSAT_RGB`.

Using WIC instead of a Rust JPEG crate keeps dependencies at `windows` (already a dependency at
0.62 with `Win32_System_Com`, `Win32_UI_Shell`, `Win32_System_Registry`) plus `serde_json`.

**Registration**

- `CLSID\{…}\InprocServer32` → the DLL, `ThreadingModel = Apartment`.
- **`AppID` with an empty `DllSurrogate`**, so it runs out-of-process in `dllhost.exe`. Two
  practical reasons: a crash or hang cannot take Explorer down, and Explorer never holds a lock
  on the DLL — otherwise updating Gyroflow fails with "file in use" until `explorer.exe` is
  killed.
- `.gyroflow\ShellEx\{e357fccd-a995-4576-b01f-234630154e96}` → our CLSID.
- **MSI (WiX)** writes to `HKLM`; **MSIX** declares the handler in its manifest; the
  **portable/zip** build registers under `HKCU\Software\Classes`, which needs no admin rights.

A ProgID for `.gyroflow` is registered at the same time, giving the file type a proper name and
app icon in Explorer — a small addition to registry work that is happening anyway.

## Part 3 — in-app

Gyroflow opens files through Qt's `FileDialog` (`src/ui/App.qml:112`), which on Windows is the
**native** `IFileOpenDialog`. Native dialogs render shell thumbnails, so the in-app open dialog
gets project thumbnails **from Part 2 with no further work**.

One in-app change is worth making: the render queue re-decodes the source video to build its
thumbnail on every job add. When a job comes from a project that carries an embedded thumbnail,
it should use that instead — same image, no decode, and one definition of "the thumbnail for
this project".

**Not included:** a recent-files list. Gyroflow has none today, and building one means a
persisted MRU, a QML panel and its own settings — a new feature rather than a thumbnail
feature, and not needed to answer "which file am I about to open".

## Testing

- **Producer side** is ordinary Rust: saving a project yields a well-formed key, valid base64,
  within the size cap, absent for presets, and absent (without failing the save) when no frame
  is available.
- **Handler's pure part** — JSON + base64 → decoded pixels — is testable directly.
- **Registration is machine-global state** and is not something a unit test should mutate. It
  gets a documented manual check, following the existing precedent of `#[ignore]`d GPU tests
  run by hand.

## Risks

- **Explorer's thumbnail cache** will serve a stale blank icon and make a working handler look
  broken. Development needs a documented cache-clear step or this will be misdiagnosed.
- **Registration is global.** A half-registered CLSID or a moved DLL path breaks `.gyroflow`
  thumbnails system-wide. Uninstall must unregister cleanly, and the portable build must never
  write `HKLM`.
- **Older-version compatibility** rests on the importer ignoring unknown keys — to be verified
  as a task, not assumed.
- **The MSIX path cannot be tested here.** It will be written from the manifest and marked
  explicitly unverified rather than implied to be tested.
- **Build wiring:** the new crate must be added to whatever produces release artifacts, or it
  silently ships unregistered.
