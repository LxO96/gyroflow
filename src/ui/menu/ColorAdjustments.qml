// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2026 Adrian <adrian.eddy at gmail>

import QtQuick

import "../components/"

MenuItem {
    id: root;
    text: qsTr("Color adjustments");
    // There is no colour icon in the set, and the packaged subset (src/resources.rs) is
    // smaller still. "pencil" is the closest existing meaning - "adjust" - and is not used
    // as any other panel's icon. A dedicated colour icon would need design review.
    iconName: "pencil";
    innerItem.enabled: window.videoArea.vid.loaded;
    objectName: "colorAdjustments";

    // NOT called `enabled`: MenuItem derives from Item, which already has `enabled`, and
    // shadowing it here would silently disable the whole panel's interactivity.
    property bool colorEnabled: false;
    property bool bakeInRender: false;
    property string lutName: "";
    property var sourceExposure: ({});

    function applyEnabled(): void { controller.set_color_enabled(root.colorEnabled, root.bakeInRender); }

    // Every slider below uses SliderWithField's `scaler`: `from`/`to` are in *display* units
    // and `root.value` is in *core* units (SliderWithField.qml:118 divides by the scaler).
    // That matters because the keyframe machinery stores `root.value` verbatim
    // (SliderWithField.qml:94/122), and the formatters in core/keyframes.rs:57-62 expect core
    // units - contrast/saturation/tint as ±1.0, LUT strength as 0..1. Doing the /100 in
    // `onValueChanged` instead, as a plain slider would, would keyframe the display value and
    // the timeline would read 100x off.

    FileDialog {
        id: lutDialog;
        title: qsTr("Choose a LUT file");
        nameFilters: Qt.platform.os == "android"? undefined : [qsTr("LUT files") + " (*.cube *.CUBE)"];
        type: "lut";
        onAccepted: root.loadLut(selectedFile);
    }

    function loadLut(url: url): void {
        const err = controller.load_lut(url);
        if (err.length > 0) {
            // messageBox(type, text, buttons, ...) - App.qml:540. There is no separate title
            // parameter, so the heading and the detail are composed into the one `text`.
            messageBox(Modal.Error, qsTr("Failed to load the LUT file:") + "\n" + err, [ { text: qsTr("Ok"), accent: true } ]);
        } else {
            root.lutName = controller.get_lut_name();
        }
    }

    // Restores the panel from a loaded project or an applied preset. Mirrors the core
    // importer (core/lib.rs, the `obj.get("color")` arm): if the file carries no `color`
    // section the core leaves the current adjustments alone, so the panel does too.
    function loadGyroflow(obj: var): void {
        const c = obj.color;
        if (!c || typeof c !== "object") return;

        if (typeof c.exposure     !== "undefined") exposureSlider   .value = +c.exposure;
        if (typeof c.contrast     !== "undefined") contrastSlider   .value = +c.contrast;
        if (typeof c.saturation   !== "undefined") saturationSlider .value = +c.saturation;
        if (typeof c.tint         !== "undefined") tintSlider       .value = +c.tint;
        if (typeof c.lut_strength !== "undefined") lutStrengthSlider.value = +c.lut_strength;
        if (typeof c.target_kelvin !== "undefined") {
            const k = +c.target_kelvin;
            if (isFinite(k)) temperatureSlider.value = k;
        }
        if (typeof c.enabled         !== "undefined") enableCb.checked = !!c.enabled;
        if (typeof c.bake_in_render  !== "undefined") bakeCb  .checked = !!c.bake_in_render;

        // The controller is the authority on which LUT is actually loaded - the project JSON
        // only says which one it asked for. They differ when a preset references a .cube that
        // doesn't resolve on this machine; the core then keeps whatever the user already had
        // (core/lib.rs, the preset arm) rather than silently dropping it, and this is where
        // the user gets told about it.
        const actual = controller.get_lut_name();
        root.lutName = actual;
        if (c.lut_name && c.lut_name !== actual) {
            window.showNotification(Modal.Warning,
                actual.length > 0
                    ? qsTr("The LUT \"%1\" from this preset could not be loaded, so \"%2\" is still in use.").arg(c.lut_name).arg(actual)
                    : qsTr("The LUT \"%1\" from this preset could not be loaded.").arg(c.lut_name));
        }
    }

    // Called from VideoArea.qml's telemetry_loaded handler. Task 14's controller already put
    // the detected white balance into the core (`source_kelvin`, and `target_kelvin` too while
    // colour is off); this keeps the slider in step, under the same `!enabled` guard so a grade
    // the user has already dialled in survives loading a second file.
    //
    // `defaultValue` is moved unconditionally so the context menu's "Reset value" returns to
    // the *source* temperature rather than a hardcoded 5500 K. `value` needs its own explicit
    // assignment: SliderWithField declares `property real value: defaultValue` (:25), but the
    // slider below overrides that with `value: 5500`, and NumberField breaks the binding
    // outright the first time the user touches the field (:118). Neither a fresh panel nor a
    // touched one will therefore follow `defaultValue` on its own.
    function setTemperatureAnchor(kelvin: real): void {
        const k = +kelvin;
        if (!isFinite(k) || k <= 0) return;
        temperatureSlider.defaultValue = k;
        if (!root.colorEnabled) temperatureSlider.value = k;
    }

    CheckBox {
        id: enableCb;
        text: qsTr("Enable");
        checked: false;
        onCheckedChanged: { root.colorEnabled = checked; root.applyEnabled(); }
    }

    // What the camera recorded, for reference. Every field of SourceExposure is an Option and
    // util::serde_json_to_qt_object drops nulls (util.rs:24), so absent fields arrive as
    // `undefined`. `ev` is checked explicitly rather than for truthiness, because 0 EV is a
    // meaningful reading that a truthiness test would throw away.
    BasicText {
        width: parent.width;
        wrapMode: Text.WordWrap;
        visible: text.length > 0;
        opacity: 0.6;
        text: {
            const s = root.sourceExposure;
            if (!s || typeof s !== "object") return "";
            let parts = [];
            if (typeof s.white_balance_k !== "undefined") parts.push(Math.round(s.white_balance_k) + " K");
            else if (typeof s.wb_mode    !== "undefined") parts.push(s.wb_mode);
            if (typeof s.iso       !== "undefined") parts.push("ISO " + Math.round(s.iso));
            if (typeof s.shutter_s !== "undefined" && s.shutter_s > 0) parts.push("1/" + Math.round(1.0 / s.shutter_s));
            if (typeof s.aperture  !== "undefined") parts.push("f/" + s.aperture.toFixed(1));
            if (typeof s.ev        !== "undefined") parts.push((s.ev >= 0? "+" : "") + s.ev.toFixed(1) + " EV");
            if (parts.length === 0) return "";
            return qsTr("Source") + ": " + parts.join(" · ");
        }
    }

    Label {
        text: qsTr("LUT");
        Row {
            spacing: 5 * dpiScale;
            Button {
                text: root.lutName.length > 0? root.lutName : qsTr("Choose file…");
                iconName: "file-empty";
                onClicked: lutDialog.open2();
            }
            LinkButton {
                anchors.verticalCenter: parent.verticalCenter;
                visible: root.lutName.length > 0;
                text: qsTr("Remove");
                onClicked: { controller.clear_lut(); root.lutName = ""; }
            }
        }
    }

    Label {
        text: qsTr("LUT strength");
        visible: root.lutName.length > 0;
        SliderWithField {
            id: lutStrengthSlider;
            width: parent.width;
            from: 0; to: 100;              // core: 0 .. 1
            value: 1.0; defaultValue: 1.0;
            scaler: 100.0;
            unit: "%"; precision: 0;
            showOrigin: true;
            keyframe: "ColorLutStrength";
            onValueChanged: controller.set_color_param("lut_strength", value);
        }
    }

    Label {
        text: qsTr("Exposure");
        SliderWithField {
            id: exposureSlider;
            width: parent.width;
            from: -5; to: 5;               // matches ColorAdjustments::set_param
            value: 0; defaultValue: 0;
            unit: " EV"; precision: 2;
            showOrigin: true;
            keyframe: "ColorExposure";
            onValueChanged: controller.set_color_param("exposure", value);
        }
    }

    Label {
        text: qsTr("Contrast");
        SliderWithField {
            id: contrastSlider;
            width: parent.width;
            from: -100; to: 100;           // core: -1 .. 1
            value: 0; defaultValue: 0;
            scaler: 100.0;
            precision: 0;
            showOrigin: true;
            keyframe: "ColorContrast";
            onValueChanged: controller.set_color_param("contrast", value);
        }
    }

    Label {
        text: qsTr("Saturation");
        SliderWithField {
            id: saturationSlider;
            width: parent.width;
            from: -100; to: 100;           // core: -1 .. 1
            value: 0; defaultValue: 0;
            scaler: 100.0;
            precision: 0;
            showOrigin: true;
            keyframe: "ColorSaturation";
            onValueChanged: controller.set_color_param("saturation", value);
        }
    }

    Label {
        text: qsTr("Temperature");
        tooltip: qsTr("Absolute target white point. The correction applied is the difference between this and the source white balance.");
        // The full MIN_KELVIN..MAX_KELVIN range from core/color/mod.rs, not a narrower
        // "usable" range: a Slider clamps its value to [from, to], so a project legitimately
        // carrying 1800 K would be silently rewritten to the slider's floor on load.
        SliderWithField {
            id: temperatureSlider;
            width: parent.width;
            from: 1667; to: 25000;         // MIN_KELVIN .. MAX_KELVIN
            value: 5500; defaultValue: 5500;
            // Kelvin is a reciprocal-ish scale: 2000->3000 K is a large visible shift while
            // 20000->21000 K is invisible. On a linear track the entire range people actually
            // grade in (roughly 2500-8000 K) is crammed into the left fifth. Log spreads it
            // out. Strictly, mireds (10^6/K) are the perceptually even unit, but log is very
            // close over this range and keeps the axis readable in Kelvin.
            logarithmic: true;
            // The camera's own white balance, so the offset being applied is visible.
            showOrigin: true;
            originValue: defaultValue;
            unit: " K"; precision: 0;
            keyframe: "ColorTemperature";
            onValueChanged: controller.set_color_param("target_kelvin", value);
        }
    }

    Label {
        text: qsTr("Tint");
        SliderWithField {
            id: tintSlider;
            width: parent.width;
            from: -100; to: 100;           // core: -1 .. 1
            value: 0; defaultValue: 0;
            scaler: 100.0;
            precision: 0;
            showOrigin: true;
            keyframe: "ColorTint";
            onValueChanged: controller.set_color_param("tint", value);
        }
    }

    CheckBox {
        id: bakeCb;
        text: qsTr("Apply to rendered video");
        checked: false;
        onCheckedChanged: { root.bakeInRender = checked; root.applyEnabled(); }
    }

    InfoMessageSmall {
        show: root.colorEnabled && !root.bakeInRender;
        text: qsTr("Color adjustments affect the preview only. Enable \"Apply to rendered video\" to include them in the export.");
    }
}
