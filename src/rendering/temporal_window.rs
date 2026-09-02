// Copyright © 2026 Gyroflow

//! A fixed ring of decoded frames that hands back a centre frame plus its
//! temporal neighbours.
//!
//! The decoder is strictly forward-only, so a symmetric window is only possible
//! by delaying output: frame `N+1` does not exist when `N` arrives. Pushing
//! frame `N` therefore emits frame `N-k`, and the trailing `k` frames come out
//! of [`TemporalWindow::drain`] once decoding has finished.
//!
//! Frames at both ends of a clip get fewer references than the interior. That
//! is deliberate and standard for temporal filters — the alternative is either
//! trimming the clip or duplicating edge frames, and duplicating them biases the
//! average toward the edge frame's own noise.

use std::collections::VecDeque;

/// One frame ready to be processed, together with the neighbours it should be
/// denoised against.
pub struct Emission<'a, T> {
    pub centre_ts: i64,
    pub centre: &'a T,
    /// Neighbours in temporal order, excluding the centre. Between `0` and `2k`
    /// entries: fewer at the start and end of the clip.
    pub references: Vec<&'a T>,
    /// Timestamps of `references`, in the same order. Task 7 needs these to build
    /// one rotation-matrix block per reference frame, and the window is the only
    /// thing that knows which timestamps are currently in the ring.
    pub reference_timestamps: Vec<i64>,
}

pub struct TemporalWindow<T> {
    k: usize,
    ring: VecDeque<(i64, T)>,
    /// Absolute frame index (0-based, in push order) of `ring.front()`. The ring
    /// is a contiguous slice `[front_index, front_index + ring.len())` of the
    /// full frame sequence — frames older than that have already been trimmed.
    front_index: usize,
    /// Total number of frames pushed so far.
    total: usize,
    /// Absolute frame index of the next frame to emit, whether via `push` or
    /// `drain`.
    next_emit: usize,
    drained: bool,
}

impl<T> TemporalWindow<T> {
    pub fn new(k: usize) -> Self {
        Self { k, ring: VecDeque::with_capacity(2 * k + 1), front_index: 0, total: 0, next_emit: 0, drained: false }
    }

    pub fn radius(&self) -> usize { self.k }

    /// Push a newly decoded frame. Returns the frame that is now ready, which is
    /// `k` frames behind the one just pushed — or `None` while the window is
    /// still filling.
    pub fn push(&mut self, ts: i64, item: T) -> Option<Emission<'_, T>> {
        self.ring.push_back((ts, item));
        self.total += 1;
        // A new frame re-arms the drain. Without this the one-shot guard makes a window that
        // was drained at a trim-range boundary refuse to drain again at end of decode, so the
        // final range's held tail is silently dropped. The guard's real purpose - two
        // consecutive drains with nothing pushed between them yielding nothing the second
        // time - is unaffected.
        self.drained = false;

        // Frame `next_emit` is ready once `k` frames after it have arrived.
        if self.total <= self.next_emit + self.k {
            return None;
        }

        Some(self.emit_next())
    }

    /// Emit the frames still held after the decoder has finished, oldest first.
    /// Returns their timestamps in emission order.
    ///
    /// Callers that need the frames themselves should use [`TemporalWindow::drain_each`];
    /// this form exists because the render loop only needs to know how many
    /// frames remain and at what timestamps.
    pub fn drain(&mut self) -> Vec<i64> {
        let mut out = Vec::new();
        self.drain_each(|e| out.push(e.centre_ts));
        out
    }

    /// Emit each remaining frame in turn, oldest first. Idempotent: a second
    /// call after the window has been drained yields nothing.
    pub fn drain_each<F: FnMut(Emission<'_, T>)>(&mut self, mut f: F) {
        if self.drained { return; }
        self.drained = true;
        while self.next_emit < self.total {
            f(self.emit_next());
        }
    }

    /// Emit the frame at `next_emit`, trim frames that have fallen out of the
    /// backward half of the *next* window, and advance `next_emit`. Only valid
    /// to call while `next_emit < total`.
    fn emit_next(&mut self) -> Emission<'_, T> {
        // Frames older than `next_emit - k` are no longer needed as backward
        // references for this or any future emission.
        let keep_from = self.next_emit.saturating_sub(self.k);
        while self.front_index < keep_from {
            self.ring.pop_front();
            self.front_index += 1;
        }

        let centre_pos = self.next_emit - self.front_index;
        self.next_emit += 1;
        Self::emit(&self.ring, centre_pos)
    }

    fn emit(ring: &VecDeque<(i64, T)>, centre: usize) -> Emission<'_, T> {
        let (centre_ts, ref centre_item) = ring[centre];
        let references = ring.iter()
            .enumerate()
            .filter(|(i, _)| *i != centre)
            .map(|(_, (_, item))| item)
            .collect();
        let reference_timestamps = ring.iter()
            .enumerate()
            .filter(|(i, _)| *i != centre)
            .map(|(_, (ts, _))| *ts)
            .collect();
        Emission { centre_ts, centre: centre_item, references, reference_timestamps }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive the window with timestamps as payloads and collect what it emits.
    fn run(k: usize, count: i64) -> Vec<i64> {
        let mut w = TemporalWindow::new(k);
        let mut emitted = Vec::new();
        for ts in 0..count {
            if let Some(e) = w.push(ts, ts) { emitted.push(e.centre_ts); }
        }
        emitted.extend(w.drain());
        emitted
    }

    #[test]
    fn k_zero_emits_every_frame_immediately() {
        assert_eq!(run(0, 5), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn every_frame_is_emitted_exactly_once_and_in_order() {
        for k in 0..=3 {
            for count in 1..12 {
                let emitted = run(k, count);
                let expected: Vec<i64> = (0..count).collect();
                assert_eq!(emitted, expected, "k={k}, count={count}");
            }
        }
    }

    #[test]
    fn output_is_delayed_by_k_while_priming() {
        let mut w = TemporalWindow::new(2);
        // The first k pushes emit nothing at all.
        assert!(w.push(0, 0).is_none());
        assert!(w.push(1, 1).is_none());
        // The (k+1)th push emits the centre, which is frame 0.
        assert_eq!(w.push(2, 2).unwrap().centre_ts, 0);
        assert_eq!(w.push(3, 3).unwrap().centre_ts, 1);
    }

    #[test]
    fn drain_emits_exactly_the_trailing_k_frames() {
        let mut w = TemporalWindow::new(3);
        for ts in 0..10 { w.push(ts, ts); }
        // Frames 0..=6 came out during push; 7, 8, 9 are still held.
        assert_eq!(w.drain(), vec![7, 8, 9]);
    }

    #[test]
    fn drain_is_idempotent() {
        let mut w = TemporalWindow::new(2);
        for ts in 0..5 { w.push(ts, ts); }
        let first = w.drain();
        assert!(!first.is_empty());
        assert!(w.drain().is_empty(), "a second drain must not re-emit");
    }

    /// A multi-range export drains at every trim seam and again at end of decode. The
    /// one-shot guard must not let the seam drain disarm the final one, or the last range's
    /// held tail disappears from the output - which is exactly what happened: a two-range
    /// export came out 2k frames short with no error anywhere.
    #[test]
    fn draining_at_a_range_boundary_does_not_disarm_the_final_drain() {
        let mut w = TemporalWindow::new(1);
        let mut out = Vec::new();

        // Range 1: three frames, one emitted live, then the seam drain takes the rest.
        for i in 0..3 { if let Some(e) = w.push(i, i) { out.push(e.centre_ts); } }
        w.drain_each(|e| out.push(e.centre_ts));
        assert_eq!(out, vec![0, 1, 2], "range 1 must be complete at the seam");

        // Range 2: the same again. Every frame must still come out.
        for i in 10..13 { if let Some(e) = w.push(i, i) { out.push(e.centre_ts); } }
        w.drain_each(|e| out.push(e.centre_ts));
        assert_eq!(out, vec![0, 1, 2, 10, 11, 12], "range 2's tail was dropped");

        // The guard itself still holds: draining twice with nothing pushed between emits
        // nothing the second time.
        let before = out.len();
        w.drain_each(|e| out.push(e.centre_ts));
        assert_eq!(out.len(), before, "a repeated drain must not re-emit");
    }

    #[test]
    fn a_clip_shorter_than_the_window_still_emits_every_frame() {
        // Two frames with k=3 never fills the ring, so everything comes from drain.
        assert_eq!(run(3, 2), vec![0, 1]);
    }

    #[test]
    fn references_shrink_at_the_head_and_tail() {
        let mut w = TemporalWindow::new(1);
        // Frame 0 is emitted with only its forward neighbour available.
        let e = w.push(0, 0).map(|e| e.references.len());
        assert_eq!(e, None);
        let e = w.push(1, 1).unwrap();
        assert_eq!(e.centre_ts, 0);
        assert_eq!(e.references.len(), 1, "head frame has no backward reference");
        let e = w.push(2, 2).unwrap();
        assert_eq!(e.centre_ts, 1);
        assert_eq!(e.references.len(), 2, "interior frame sees both sides");
    }

    #[test]
    fn references_never_include_the_centre() {
        let mut w = TemporalWindow::new(1);
        w.push(0, 0);
        w.push(1, 1);
        let e = w.push(2, 2).unwrap();
        assert_eq!(*e.centre, 1);
        assert!(!e.references.iter().any(|r| **r == 1));
    }

    #[test]
    fn reference_timestamps_stay_aligned_with_references() {
        let mut w = TemporalWindow::new(2);
        let mut checked_any = false;
        for ts in 0..5 {
            if let Some(e) = w.push(ts, ts) {
                assert_eq!(e.reference_timestamps.len(), e.references.len());
                let expected_ts: Vec<i64> = e.references.iter().map(|r| **r).collect();
                assert_eq!(e.reference_timestamps, expected_ts);
                assert!(!e.reference_timestamps.contains(&e.centre_ts));
                checked_any = true;
            }
        }
        assert!(checked_any, "window of radius 2 driven with 0..5 must emit at least once");
    }
}
