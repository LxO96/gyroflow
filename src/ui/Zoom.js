// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2026 Gyroflow

.pragma library

// Preview zoom maths, kept free of Qt so it can be tested outside the app.
//
// Units: `zoom` is relative to fit (1 = the whole frame fits the view, 2 = twice as big).
// A "pixel ratio" is device pixels per output pixel (1 = 100%, one output pixel per screen
// pixel). The pan is stored as `u`: the image coordinate (0..1) at the centre of the view, so
// it survives the view being resized.

const MAX_PIXEL_RATIO = 4;      // 400%
const STOPS = [0.5, 1, 2, 4];   // pixel ratios the zoom-in/out keys step through

function zoomForPixelRatio(ratio, fitWidth, dpr, outputWidth) {
    return ratio * outputWidth / Math.max(1, fitWidth * dpr);
}
function pixelRatio(zoom, fitWidth, dpr, outputWidth) {
    return zoom * fitWidth * dpr / Math.max(1, outputWidth);
}

// Zooming costs no rendering: the preview texture is always the "Preview resolution" size
// (qml-video-rs sizes it from the decode surface, not the item), and a bigger item only shows
// more of it. So the only limit is how far enlarging is still useful.
function maxZoom(fitWidth, dpr, outputWidth) {
    return Math.max(1, zoomForPixelRatio(MAX_PIXEL_RATIO, fitWidth, dpr, outputWidth));
}
function clampZoom(zoom, max) {
    return Math.min(Math.max(zoom, 1), Math.max(1, max));
}

// The next zoom stop in `direction` (+1 in, -1 out): fit, the STOPS that lie above fit, and
// the cap itself, so zooming in always ends exactly at the limit.
function nextStop(zoom, direction, fitWidth, dpr, outputWidth, max) {
    const stops = [1];
    for (const r of STOPS) {
        const z = zoomForPixelRatio(r, fitWidth, dpr, outputWidth);
        if (z > 1 && z < max) stops.push(z);
    }
    stops.push(max);
    const eps = 1e-3;
    if (direction > 0) {
        for (const z of stops) if (z > zoom * (1 + eps)) return z;
        return max;
    }
    for (let i = stops.length - 1; i >= 0; i--) if (stops[i] < zoom * (1 - eps)) return stops[i];
    return 1;
}

// Offset of the image's leading edge within the view, for one axis. Clamped so the image
// always covers the view: never a gap at either side.
function panOffset(view, zoom, u) {
    const content = view * zoom;
    return Math.min(0, Math.max(view - content, view / 2 - u * content));
}
// The `u` that produces the (clamped) offset `off`. Storing the clamped value is what stops
// a drag past the edge from leaving a dead zone to drag back through.
function uForOffset(view, zoom, off) {
    const content = view * zoom;
    return (view / 2 - Math.min(0, Math.max(view - content, off))) / content;
}

// New `u` after zooming from z0 to z1 with the image point under `cursor` held still.
function zoomAround(view, z0, u0, z1, cursor) {
    const off0 = panOffset(view, z0, u0);
    const point = (cursor - off0) / (view * z0);
    return uForOffset(view, z1, cursor - point * view * z1);
}

// New `u` after dragging by `delta` view pixels.
function dragTo(view, zoom, u, delta) {
    return uForOffset(view, zoom, panOffset(view, zoom, u) + delta);
}
