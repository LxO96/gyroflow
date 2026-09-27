// Tests for src/ui/Zoom.js, a QML .pragma library with no Qt dependency.
// Run: node _scripts/tests/zoom_test.js
const fs = require('fs');
const path = require('path');
const src = fs.readFileSync(path.join(__dirname, '../../src/ui/Zoom.js'), 'utf8').replace(/^\.pragma library.*$/m, '');
const Z = new Function(src + '; return { zoomForPixelRatio, pixelRatio, maxZoom, clampZoom, nextStop, panOffset, zoomAround, dragTo };')();

let failed = 0;
function eq(got, want, msg, tol = 1e-6) {
    if (Math.abs(got - want) > tol) { failed++; console.log(`FAIL ${msg}: got ${got}, want ${want}`); }
    else console.log(`ok   ${msg}`);
}

// A 4K output shown in a 1000 px wide box at DPR 1.25.
const fitW = 1000, dpr = 1.25, outW = 3840;
const z100 = Z.zoomForPixelRatio(1, fitW, dpr, outW);
eq(z100, 3840 / 1250, '100% means one output pixel per device pixel');
eq(Z.pixelRatio(z100, fitW, dpr, outW), 1, 'pixelRatio inverts zoomForPixelRatio');
eq(Z.pixelRatio(1, fitW, dpr, outW), 1250 / 3840, 'fit shows the whole frame');

// Cap: 400%, never below fit.
eq(Z.maxZoom(fitW, dpr, outW), Z.zoomForPixelRatio(4, fitW, dpr, outW), 'capped at 400%');
eq(Z.maxZoom(9000, 1, 1000), 1, 'never below fit (400% of a small output is smaller than the box)');
eq(Z.clampZoom(0.3, 5), 1, 'clamped up to fit');
eq(Z.clampZoom(9, 5), 5, 'clamped down to max');

// Stops: fit, then 50/100/200/400% where they are above fit and within the cap.
const max = Z.maxZoom(fitW, dpr, outW);
eq(Z.nextStop(1, 1, fitW, dpr, outW, max), Z.zoomForPixelRatio(0.5, fitW, dpr, outW), 'fit -> 50%');
eq(Z.nextStop(z100, 1, fitW, dpr, outW, max), Z.zoomForPixelRatio(2, fitW, dpr, outW), '100% -> 200%');
eq(Z.nextStop(Z.zoomForPixelRatio(2, fitW, dpr, outW), 1, fitW, dpr, outW, max), max, '200% -> 400% (the cap)');
eq(Z.nextStop(max, 1, fitW, dpr, outW, max), max, 'stays at the cap');
eq(Z.nextStop(z100, -1, fitW, dpr, outW, max), Z.zoomForPixelRatio(0.5, fitW, dpr, outW), '100% -> 50%');
eq(Z.nextStop(Z.zoomForPixelRatio(0.5, fitW, dpr, outW), -1, fitW, dpr, outW, max), 1, '50% -> fit');
eq(Z.nextStop(1, -1, fitW, dpr, outW, max), 1, 'stays at fit');

// Pan: offset of the image's left edge in the view; the image always covers the view.
eq(Z.panOffset(100, 1, 0.9), 0, 'no pan at fit');
eq(Z.panOffset(100, 4, 0.5), -150, 'centred');
eq(Z.panOffset(100, 4, 0.0), 0, 'clamped at the left edge');
eq(Z.panOffset(100, 4, 1.0), -300, 'clamped at the right edge');

// Zooming around the cursor keeps the image point under it fixed.
{
    const view = 100, z0 = 2, u0 = 0.5, cursor = 80, z1 = 4;
    const off0 = Z.panOffset(view, z0, u0);
    const point = (cursor - off0) / (view * z0);
    const u1 = Z.zoomAround(view, z0, u0, z1, cursor);
    const off1 = Z.panOffset(view, z1, u1);
    eq((cursor - off1) / (view * z1), point, 'point under the cursor is unchanged');
}
eq(Z.zoomAround(100, 1, 0.5, 4, 0), 0.125, 'zooming at the left edge keeps it pinned (u is the view centre)');

// Dragging moves the image with the pointer, and stops at the edge rather than overshooting.
{
    const u = Z.dragTo(100, 4, 0.5, 10);
    eq(Z.panOffset(100, 4, u), -140, 'drag right by 10 moves the image right by 10');
    const far = Z.dragTo(100, 4, 0.5, 10000);
    eq(Z.panOffset(100, 4, far), 0, 'overshoot is clamped');
    eq(Z.dragTo(100, 4, far, -10), Z.dragTo(100, 4, 0.125, -10), 'no dead zone after an overshoot');
}

console.log(failed ? `${failed} FAILED` : 'all passed');
process.exit(failed ? 1 : 0);
