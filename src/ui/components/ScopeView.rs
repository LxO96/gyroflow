// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2026 Gyroflow

#![allow(non_snake_case)]

//! Draws one video scope (waveform, histogram or vectorscope) of the *displayed* preview frame.
//!
//! The frames come from the player's observe hook (`MDKVideoItem::onObserveFrame`), which runs
//! on the render thread after the stabilization/colour/denoise pass - so the scope measures
//! exactly what is on screen. That thread only takes a subsampled copy (~1 MB); the scope
//! itself is computed on a rayon worker and handed back to the GUI thread. The worker keeps
//! just the newest frame, so a slow scope drops frames instead of queueing them.
//!
//! The observe hook costs a GPU readback, so it is switched off whenever the scope is hidden
//! (`active: false`), and limited to paused frames unless `live` is set.

use std::sync::{ Arc, atomic::{ AtomicBool, AtomicI32, Ordering::SeqCst } };
use parking_lot::Mutex;
use cpp::*;
use qmetaobject::*;
use qml_video_rs::video_item::{ MDKVideoItem, ObserveMode };
use gyroflow_core::scopes::{ self, Frame, Samples, ScopeImage, ScopeKind };
use crate::util;

/// Scope refresh interval while playing with `live` on (~10 per second).
const LIVE_INTERVAL_MS: u32 = 100;

type Deliver = Box<dyn Fn(ScopeImage) + Send + Sync>;

#[derive(Default)]
struct Shared {
    kind: AtomicI32,
    /// Newest samples, kept after computing so a change of `kind` can redraw without a new frame.
    latest: Mutex<Option<Arc<Samples>>>,
    dirty: AtomicBool,
    running: AtomicBool,
    deliver: Mutex<Option<Deliver>>,
}

impl Shared {
    fn submit(self: &Arc<Self>, samples: Samples) {
        *self.latest.lock() = Some(Arc::new(samples));
        self.dirty.store(true, SeqCst);
        self.kick();
    }

    /// Starts the worker unless it is already running. The worker loops while `dirty` is set,
    /// and re-checks after clearing `running` so a submit racing with its exit is not lost.
    fn kick(self: &Arc<Self>) {
        if self.running.swap(true, SeqCst) { return; }
        let me = self.clone();
        rayon::spawn(move || loop {
            if me.dirty.swap(false, SeqCst) {
                let samples = me.latest.lock().clone();
                if let (Some(samples), Some(kind)) = (samples, ScopeKind::from_index(me.kind.load(SeqCst))) {
                    let image = scopes::compute(kind, &samples);
                    if let Some(deliver) = me.deliver.lock().as_ref() { deliver(image); }
                }
                continue;
            }
            me.running.store(false, SeqCst);
            if me.dirty.load(SeqCst) && !me.running.swap(true, SeqCst) { continue; }
            break;
        });
    }
}

#[derive(Default, QObject)]
pub struct ScopeView {
    base: qt_base_class!(trait QQuickPaintedItem),

    /// 0 waveform, 1 histogram, 2 vectorscope - `ScopeKind`'s values.
    kind: qt_property!(i32; WRITE setKind),
    /// Also follow playback, throttled. Off: only paused frames.
    live: qt_property!(bool; WRITE setLive),
    /// Whether frames are observed at all. QML binds it to the overlay's visibility.
    active: qt_property!(bool; WRITE setActive),

    attach: qt_method!(fn(&mut self, player: QJSValue)),

    shared: Arc<Shared>,
    image: Option<QImage>,
    player: Option<QPointer<MDKVideoItem>>,
}

impl ScopeView {
    fn setKind(&mut self, v: i32) {
        if ScopeKind::from_index(v).is_none() { return; }
        self.kind = v;
        self.shared.kind.store(v, SeqCst);
        // The last image is of the old kind; redraw from the samples already held.
        self.image = None;
        (self as &dyn QQuickItem).update();
        if self.shared.latest.lock().is_some() {
            self.shared.dirty.store(true, SeqCst);
            self.shared.kick();
        }
    }
    fn setLive  (&mut self, v: bool) { self.live = v;   self.apply_mode(); }
    fn setActive(&mut self, v: bool) { self.active = v; self.apply_mode(); }

    fn apply_mode(&mut self) {
        let mode = match (self.active, self.live) {
            (false, _)    => ObserveMode::Off,
            (true, false) => ObserveMode::Paused,
            (true, true)  => ObserveMode::Always,
        };
        if let Some(vid) = self.player.as_ref().and_then(|p| p.as_pinned()) {
            let vid = unsafe { &mut *vid.as_ptr() }; // vid.borrow_mut(), as in controller.rs
            vid.setObserveFrameMode(mode, LIVE_INTERVAL_MS);
        }
    }

    fn attach(&mut self, player: QJSValue) {
        let Some(vid) = player.to_qobject::<MDKVideoItem>() else { return; };
        let vid = unsafe { &mut *vid.as_ptr() }; // vid.borrow_mut()
        self.player = Some(QPointer::from(&*vid));

        let deliver = util::qt_queued_callback_mut(QPointer::from(self as &Self), |this: &mut Self, image: ScopeImage| {
            this.image = Some(to_qimage(&image));
            (this as &dyn QQuickItem).update();
        });
        *self.shared.deliver.lock() = Some(Box::new(deliver));

        let shared = self.shared.clone();
        vid.onObserveFrame(Box::new(move |_frame, _timestamp, width, height, stride, pixels| {
            let frame = Frame { width: width as usize, height: height as usize, stride: stride as usize, data: pixels };
            if let Some(samples) = Samples::from_frame(&frame, scopes::DEFAULT_MAX_SAMPLES) {
                shared.submit(samples);
            }
        }));
        self.apply_mode();
    }
}

fn to_qimage(img: &ScopeImage) -> QImage {
    let (w, h, ptr) = (img.width as u32, img.height as u32, img.rgba.as_ptr());
    // `copy()` detaches from `img.rgba`, which does not outlive this call.
    cpp!(unsafe [w as "uint32_t", h as "uint32_t", ptr as "const uint8_t *"] -> QImage as "QImage" {
        return QImage(ptr, w, h, w * 4, QImage::Format_RGBA8888_Premultiplied).copy();
    })
}

fn line(p: &mut QPainter, x1: f64, y1: f64, x2: f64, y2: f64) {
    p.draw_line(QLineF { pt1: QPointF { x: x1, y: y1 }, pt2: QPointF { x: x2, y: y2 } });
}

impl QQuickItem for ScopeView {
    fn class_begin(&mut self) {
        self.shared.kind.store(self.kind, SeqCst);
    }
    fn geometry_changed(&mut self, _new: QRectF, _old: QRectF) {
        (self as &dyn QQuickItem).update();
    }
}

impl QQuickPaintedItem for ScopeView {
    fn paint(&mut self, p: &mut QPainter) {
        let rect = (self as &dyn QQuickItem).bounding_rect();
        if rect.width <= 0.0 || rect.height <= 0.0 { return; }
        let kind = ScopeKind::from_index(self.kind).unwrap_or(ScopeKind::Waveform);

        // The vectorscope is a circle: keep its plot square and centred.
        let plot = if kind == ScopeKind::Vectorscope {
            let s = rect.width.min(rect.height);
            QRectF { x: rect.x + (rect.width - s) / 2.0, y: rect.y + (rect.height - s) / 2.0, width: s, height: s }
        } else {
            rect
        };

        p.set_render_hint(QPainterRenderHint::Antialiasing, true);
        if let Some(image) = &self.image {
            p.draw_image_fit_rect(plot, image.clone());
        }

        let mut pen = QPen::from_color(QColor::from_rgba(255, 255, 255, 70));
        pen.set_width_f(1.0);
        p.set_pen(pen);
        let label = QColor::from_rgba(255, 255, 255, 150);

        match kind {
            ScopeKind::Waveform => {
                for pct in [0, 25, 50, 75, 100] {
                    let y = plot.y + plot.height * (1.0 - pct as f64 / 100.0);
                    line(p, plot.x, y, plot.x + plot.width, y);
                }
                p.set_pen(QPen::from_color(label));
                for pct in [25, 50, 75, 100] {
                    let y = plot.y + plot.height * (1.0 - pct as f64 / 100.0);
                    p.draw_text(QPointF { x: plot.x + 3.0, y: y + 11.0 }, QString::from(pct.to_string()));
                }
            }
            ScopeKind::Histogram => {
                for q in [1, 2, 3] {
                    let x = plot.x + plot.width * q as f64 / 4.0;
                    line(p, x, plot.y, x, plot.y + plot.height);
                }
            }
            ScopeKind::Vectorscope => {
                let (cx, cy) = (plot.x + plot.width / 2.0, plot.y + plot.height / 2.0);
                let r = plot.width / 2.0;
                p.draw_ellipse_with_center(QPointF { x: cx, y: cy }, r, r);
                line(p, cx - r, cy, cx + r, cy);
                line(p, cx, cy - r, cx, cy + r);

                // Skin-tone line: +Cb is right, +Cr is up, so the y component flips.
                let a = scopes::SKIN_TONE_ANGLE_DEG.to_radians();
                p.set_pen(QPen::from_color(QColor::from_rgba(255, 200, 150, 110)));
                line(p, cx, cy, cx + r * a.cos(), cy - r * a.sin());

                let box_half = plot.width * 0.03;
                for (name, (tx, ty)) in scopes::vectorscope_targets() {
                    let (x, y) = (plot.x + tx as f64 * plot.width, plot.y + ty as f64 * plot.height);
                    p.set_pen(QPen::from_color(QColor::from_rgba(255, 255, 255, 110)));
                    p.draw_rect(QRectF { x: x - box_half, y: y - box_half, width: box_half * 2.0, height: box_half * 2.0 });
                    p.set_pen(QPen::from_color(label));
                    // Labels sit outward from the centre so they never cover the target.
                    let (dx, dy) = (x - cx, y - cy);
                    let d = (dx * dx + dy * dy).sqrt().max(1.0);
                    p.draw_text(QPointF { x: x + dx / d * box_half * 2.2 - 6.0, y: y + dy / d * box_half * 2.2 + 4.0 }, QString::from(name));
                }
            }
        }
    }
}
