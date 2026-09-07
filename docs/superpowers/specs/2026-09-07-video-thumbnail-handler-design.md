# Windows thumbnails for video files Windows cannot preview

**Date:** 2026-09-07
**Status:** design approved, spec awaiting review
**Follows:** `2026-09-03-windows-thumbnails-design.md` (the `.gyroflow` handler this extends)

## The problem

Windows shows no thumbnail for 10-bit 4:2:2 H.264. Measured on the development machine
against `testfiles/C3105.MP4` (H.264 High 4:2:2 Predictive, `yuv422p10le`):

| file | codec | shell thumbnail |
|---|---|---|
| `C3105.MP4` | 10-bit 4:2:2 H.264 | **fails**, `WTS_E_FAILEDEXTRACTION` (0x8004B200) |
| `C3105_off.mp4` | 8-bit H.264 | OK, 144x256 |

Windows *has* a thumbnail handler registered for `.mp4`; it simply cannot decode this
profile, so it fails and Explorer falls back to the file-type icon.

This corrects the earlier spec, which recorded `.mp4` as "already handled". That measurement
checked whether a handler was *registered*, not whether it *worked*.

## Constraint that shapes everything

**Windows has no thumbnail-handler fallback chaining.** A registered handler owns the
extension outright; there is no "try ours, fall back to the system's". Registering for
`.mp4` therefore means owning *every* mp4 thumbnail on the machine, including the 8-bit ones
Windows already handles well.

This is why the feature is opt-in and ships off.

## Approach

Three candidates for giving the handler a decoder, with measured startup costs:

| | cost | verdict |
|---|---|---|
| Link ffmpeg into the DLL | fastest | rejected: 61 MB / 8 DLLs must resolve inside `dllhost.exe`, whose search path is not our install dir; and it discards the current 157 KB dependency-free DLL |
| DLL spawns `gyroflow.exe` | **314 ms** Qt startup before decoding | rejected: roughly doubles cost, runs a GUI binary from a shell process |
| **DLL spawns a small ffmpeg-only helper** | **43 ms** startup, ~350 ms total | **chosen** |

First-frame decode was measured at **~400 ms** including process startup (374-407 ms across
runs on a 193 MB 10-bit file), so performance is not the blocker.

The decisive argument for the helper is not speed: **a separate process can be killed on a
timeout.** A shell extension that hangs on a corrupt file is a classic way to make Explorer
unusable, and neither of the other options bounds that cleanly.

## Components

**`gyroflow-thumbnailer.exe`** — new crate. Links ffmpeg only; no Qt, no OpenCV. Takes an
input path and a max edge, decodes the first frame, applies rotation, and writes **raw BGRA
to stdout** behind a small fixed header (magic, width, height). No JPEG round-trip between
processes and no temp files. Must be built `#![windows_subsystem = "windows"]` — a console
subsystem binary flashes a console window on every thumbnail.

`rendering/mod.rs::rotate_rgba` moves into the helper rather than being duplicated.

**`gyroflow_thumbnail.dll`** — the existing handler, extended. `.gyroflow` keeps today's fast
path (JSON + base64 + WIC, no subprocess). A video extension spawns the helper, reads BGRA
off the pipe, and builds the DIB with the code already there. **The DLL never links ffmpeg**;
it gains exactly one ability, running a helper and reading pixels.

**The opt-in** — a checkbox in `Advanced.qml` persisted through `settings.rs`, calling a new
`DllRegisterVideoTypesUser` / `DllUnregisterVideoTypesUser`. Per-user (`HKCU`), so no
elevation prompt — which is what makes an in-app checkbox viable. `.gyroflow` registration is
independent of the toggle.

## Failure model

**Timeout is the point of the helper.** Spawned with a hard 3 s wall-clock budget — generous
against ~400 ms measured — and killed on overrun. A hung decoder costs one dead subprocess
and a generic icon.

**Every failure degrades to "no thumbnail".** Helper missing, non-zero exit, malformed header,
short read, timeout, unsupported codec: all return the same `E_FAIL` the handler already
returns for a project with no embedded image. There is no path that produces a *wrong*
picture.

**Bounded reads.** Buffer size is computed from the header and checked against a cap before
allocating; the read stops at exactly that many bytes. A helper claiming 40000x40000 is
rejected rather than allocating 6 GB inside a shell process.

**No stdin, stderr swallowed.** The helper never prompts, and diagnostics go to a log rather
than the pipe, so a stray ffmpeg warning cannot corrupt the pixel stream.

**Spawned by absolute path** derived from the DLL's own location (`GetModuleFileNameW`), never
from `PATH`: `dllhost.exe` runs with the user's token but not their environment.

## Scope

**Registered when enabled:** `.mp4`, `.mov`, `.mxf`, `.360`, `.insv`, `.exr`. One ffmpeg path
serves all of them, so the list is a registration decision, not a code one.

**Excluded: `.braw`, `.r3d`, `.nev`.** These decode only through mdk with vendor SDKs
(`mdk_processor.rs:41-47`). Including them would mean shipping a helper that silently fails on
exactly the formats those users care about.

Format support verified against the bundled ffmpeg: EXR decoder present; MXF is a container
whose demuxer is present, with DNxHD and ProRes decoders both available.

## Build and packaging

**No new redistribution.** The helper installs beside `Gyroflow.exe`, which already ships the
8 ffmpeg DLLs (61 MB). Windows resolves a binary's imports from its own directory first, so
the helper finds them with no extra copies, no `PATH` manipulation, and no installer growth.
Licensing is unchanged: the bundled ffmpeg is GPL, Gyroflow is GPL-3.0-or-later.

Build wiring follows the DLL's existing pattern in `_scripts/windows.just`: its own
`cargo build` (it must not link Qt or OpenCV) plus an `xcopy` beside the existing one.

## Testing

- **Helper** is an ordinary binary: run it on real files; assert header, buffer size, and a
  known frame's dimensions and orientation.
- **DLL/helper contract** gets a test running the real helper on a real file.
- **Timeout and missing-helper paths** are testable by pointing the DLL at a stub that sleeps
  or does not exist.
- **Registration** stays a documented manual check, as in the previous spec.

## Risks

- **DLL search path in `dllhost.exe`** is the most likely "works here, fails on a clean
  install" failure. Must be verified from an *installed layout*, not from `target/release`.
  This is a task, not a comment.
- **We own every mp4 once enabled**, so a regression hits files that previously worked.
  Mitigated by opt-in, default off.
- **Console window flash** if the helper is not a windows-subsystem binary. Trivial to fix,
  very visible when wrong, easy to miss in a headless test.
- **Explorer's cache** will make correct work look broken during development, as it did for
  the `.gyroflow` handler. Verification steps need a cache-clear built in.
- **Toggling off** deregisters but leaves already-generated thumbnails in Explorer's cache.
  Harmless — they are correct pictures — but worth knowing before it is reported as a leak.
- **MSI and MSIX installs remain untestable here**, as in the previous spec. Installer changes
  will be XML-checked only and marked as such rather than implied to be tested.

## Definition of done

Helper output correct on a 10-bit mp4, an 8-bit mp4, an `.exr` and an `.mxf`; the shell API
returning a real bitmap for a 10-bit mp4 (the case that prompted this); `.gyroflow` still
served by its no-subprocess fast path; the timeout demonstrably killing a stuck helper; and
toggling off cleanly removing the registration.
