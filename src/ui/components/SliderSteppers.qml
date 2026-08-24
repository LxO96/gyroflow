// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2026 Adrian <adrian.eddy at gmail>

import QtQuick

// A "-" / "+" pair that nudges a SliderWithField by a fixed step, with room for a note
// between them saying what the value is measured from.
//
// Separate from SliderWithField rather than built into it, because that component is a `Row`
// used in 39 places across five panels; turning it into a `Column` to make space underneath
// would change the height of every one of them. The caller wraps the slider and this in a
// `Column` instead - and note that `Label.qml` sizes its content from `children[0].height`
// alone, so that wrapping Column has to be the Label's only child or the row's height is wrong.
Row {
    id: root;

    // The SliderWithField to nudge. Its `field.value` is what moves, so the change travels the
    // same path a drag does: the field's own handler writes back to `value`, clamps through
    // `set_color_param` and records a keyframe if keyframing is on.
    property var target: null;

    // One click, in the field's *display* units - EV, Kelvin, or the -100..100 the panel shows,
    // never the core units underneath. `from`/`to` on SliderWithField are in those same units,
    // which is what makes the clamp below correct.
    property real step: 1;

    // Optional, shown between the buttons. The panel uses it to spell out the offset from the
    // camera's own reading, which is the whole point of the buttons on those two sliders.
    property string note: "";

    spacing: 8 * dpiScale;
    height: 16 * dpiScale;

    function nudge(direction: int): void {
        if (!target) return;
        const next = target.field.value + (direction * root.step);
        target.field.value = Math.max(target.from, Math.min(target.to, next));
    }

    LinkButton {
        text: "−"; // real minus sign, not a hyphen - it lines up with "+" optically
        font.underline: false;
        font.pixelSize: 14 * dpiScale;
        leftPadding: 4 * dpiScale; rightPadding: 4 * dpiScale;
        topPadding: 0; bottomPadding: 0;
        height: parent.height;
        onClicked: root.nudge(-1);
        // Repeat while held, so a long drag-free adjustment does not need forty clicks.
        autoRepeat: true;
    }
    BasicText {
        text: root.note;
        visible: text.length > 0;
        opacity: 0.6;
        font.pixelSize: 11 * dpiScale;
        anchors.verticalCenter: parent.verticalCenter;
    }
    LinkButton {
        text: "+";
        font.underline: false;
        font.pixelSize: 14 * dpiScale;
        leftPadding: 4 * dpiScale; rightPadding: 4 * dpiScale;
        topPadding: 0; bottomPadding: 0;
        height: parent.height;
        onClicked: root.nudge(1);
        autoRepeat: true;
    }
}
