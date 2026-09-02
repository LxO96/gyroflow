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

    // Called by VideoArea once the picker has solved a patch. Writes through `field.value`, the
    // same path a drag takes, so the value is clamped, a keyframe is recorded if keyframing is on,
    // and the change reaches the core through the sliders' own `onValueChanged`.
    //
    // `tintSlider` is in *display* units here (-100..100, `scaler: 100`) while the solver works in
    // core units (-1..1), hence the multiply; `temperatureSlider` has no scaler and takes Kelvin
    // as-is. Getting this backwards would put the tint slider 100x off and look like a solver bug.
    function applyPickedWhiteBalance(kelvin: real, tint: real): void {
        temperatureSlider.field.value = kelvin;
        tintSlider.field.value = tint * 100.0;
    }

    // What the two metadata-anchored sliders are offset *from*. Exposure is already relative -
    // the grade is a gain, so +0.5 EV means half a stop brighter than this clip was, whatever
    // it was - and temperature is relative because the core keeps the offset when a new clip's
    // white balance arrives (ColorAdjustments::reanchor_source_kelvin). Spelling the offset out
    // here is what makes that legible, and it is the same number that travels to a batch.
    // Just the signed offset. What it is measured *from* is already on screen twice - the
    // "Source:" line above and the origin tick on the track - and spelling it out again made
    // the row wider than the panel, which pushed the buttons outside their parent and stopped
    // them receiving clicks at all.
    readonly property string evNote: {
        const v = exposureSlider.field.value;
        return (v >= 0? "+" : "") + v.toFixed(2) + " EV";
    }
    readonly property string kelvinNote: {
        const d = temperatureSlider.field.value - temperatureSlider.defaultValue;
        return (d >= 0? "+" : "") + Math.round(d) + " K";
    }

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
            outputSpace.syncFromCore();
            outputRange.syncFromCore();
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
        if (typeof c.shadows      !== "undefined") shadowsSlider    .value = +c.shadows;
        if (typeof c.highlights   !== "undefined") highlightsSlider .value = +c.highlights;
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
        outputSpace.syncFromCore();
        outputRange.syncFromCore();
        if (c.lut_name && c.lut_name !== actual) {
            window.showNotification(Modal.Warning,
                actual.length > 0
                    ? qsTr("The LUT \"%1\" from this preset could not be loaded, so \"%2\" is still in use.").arg(c.lut_name).arg(actual)
                    : qsTr("The LUT \"%1\" from this preset could not be loaded.").arg(c.lut_name));
        }
    }

    // Called from VideoArea.qml's telemetry_loaded handler. The controller has already put the
    // detected white balance into the core via `reanchor_source_kelvin`, which moves
    // `source_kelvin` to this clip's reading and carries the user's offset along with it. This
    // keeps the slider in step with both halves of that.
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
        const previous = temperatureSlider.defaultValue;
        temperatureSlider.defaultValue = k;
        if (!root.colorEnabled) {
            // No grade dialled in, so the offset is zero and the target sits on the source.
            temperatureSlider.value = k;
        } else if (isFinite(previous) && previous > 0) {
            // Mirror what the core just did: shift the target by the same distance the source
            // moved, so the offset the user chose is what survives. Mirrored rather than read
            // back because there is no getter for `target_kelvin`; the slider's own bounds are
            // MIN_KELVIN..MAX_KELVIN, the same clamp the core applies, so the two agree.
            temperatureSlider.value = temperatureSlider.value + (k - previous);
        }
    }

    CheckBox {
        id: enableCb;
        text: qsTr("Enable");
        checked: false;
        onCheckedChanged: {
            root.colorEnabled = checked;
            root.applyEnabled();
            // Turning the section on with no LUT loaded is almost always a prelude to picking
            // one, so offer the file dialog straight away. Cancelling costs nothing: the panel
            // stays enabled and every other control still works without a LUT.
            if (checked && root.lutName.length === 0) lutDialog.open2();
        }
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
        Column {
            width: parent.width;
            spacing: 3 * dpiScale;
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
            SliderSteppers {
                target: lutStrengthSlider;
                step: 5;
            }
        }
    }

    Label {
        text: qsTr("Exposure");
        Column {
            width: parent.width;
            spacing: 3 * dpiScale;
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
            SliderSteppers {
                target: exposureSlider;
                // A third of a stop, the increment every camera and light meter uses.
                step: 1/3;
                note: root.evNote;
            }
        }
    }

    Label {
        text: qsTr("Contrast");
        Column {
            width: parent.width;
            spacing: 3 * dpiScale;
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
            SliderSteppers {
                target: contrastSlider;
                step: 5;
            }
        }
    }

    Label {
        text: qsTr("Saturation");
        Column {
            width: parent.width;
            spacing: 3 * dpiScale;
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
            SliderSteppers {
                target: saturationSlider;
                step: 5;
            }
        }
    }

    // Placed after Saturation and before Temperature to match the order the grade is applied
    // in (core/color/mod.rs `apply_color`), so the panel reads top-to-bottom as the pipeline.
    Label {
        text: qsTr("Shadows");
        tooltip: qsTr("Lifts or deepens the darkest part of the image. The falloff is weighted to the toe, so mid grey moves about an eighth as far and the highlights barely move at all.");
        Column {
            width: parent.width;
            spacing: 3 * dpiScale;
            SliderWithField {
                id: shadowsSlider;
                width: parent.width;
                from: -100; to: 100;           // core: -1 .. 1
                value: 0; defaultValue: 0;
                scaler: 100.0;
                precision: 0;
                showOrigin: true;
                keyframe: "ColorShadows";
                onValueChanged: controller.set_color_param("shadows", value);
            }
            SliderSteppers {
                target: shadowsSlider;
                step: 5;
            }
        }
    }

    Label {
        text: qsTr("Highlights");
        tooltip: qsTr("Recovers or pushes the brightest part of the image. Weighted to the shoulder the same way Shadows is weighted to the toe, so the two do not fight each other.");
        Column {
            width: parent.width;
            spacing: 3 * dpiScale;
            SliderWithField {
                id: highlightsSlider;
                width: parent.width;
                from: -100; to: 100;           // core: -1 .. 1
                value: 0; defaultValue: 0;
                scaler: 100.0;
                precision: 0;
                showOrigin: true;
                keyframe: "ColorHighlights";
                onValueChanged: controller.set_color_param("highlights", value);
            }
            SliderSteppers {
                target: highlightsSlider;
                step: 5;
            }
        }
    }

    Label {
        text: qsTr("Temperature");
        tooltip: qsTr("Absolute target white point. The correction applied is the difference between this and the source white balance.");
        // The full MIN_KELVIN..MAX_KELVIN range from core/color/mod.rs. The floor is 2000 K
        // rather than the 1667 K the Planckian fit is defined over because sRGB cannot represent
        // the locus below ~1902 K; the core clamps there too, so a project carrying 1800 K is
        // corrected identically whichever way it arrives.
        Column {
            width: parent.width;
            spacing: 3 * dpiScale;
            SliderWithField {
                id: temperatureSlider;
                width: parent.width;
                from: 2000; to: 25000;         // MIN_KELVIN .. MAX_KELVIN
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
            // The steppers keep centring themselves on the track, so the space to the right -
            // under the number field - is free for the picker. An Item rather than a Row because
            // the two want different alignments: centred and right-anchored.
            Item {
                width: parent.width;
                height: 20 * dpiScale;
                SliderSteppers {
                    target: temperatureSlider;
                    step: 100;
                    note: root.kelvinNote;
                }
                LinkButton {
                    id: pickWbBtn;
                    anchors.right: parent.right;
                    anchors.verticalCenter: parent.verticalCenter;
                    height: parent.height;
                    iconName: "eyedropper";
                    leftPadding: 4 * dpiScale; rightPadding: 4 * dpiScale;
                    topPadding: 0; bottomPadding: 0;
                    // Picking is meaningless while the section is off - the solved values would
                    // be written to sliders nothing reads - so gate on it rather than letting the
                    // click quietly do nothing.
                    enabled: root.colorEnabled;
                    opacity: enabled? (checked? 1 : 0.7) : 0.3;
                    checked: window.videoArea.wbPickArmed;
                    textColor: checked? styleAccentColor : styleTextColor;
                    tooltip: checked? qsTr("Click a grey area of the video, or drag to average a region. Right-click, or click here again, to cancel")
                                    : qsTr("Pick white balance from a grey area of the video (click, or drag to average a region)");
                    onClicked: window.videoArea.wbPickArmed = !window.videoArea.wbPickArmed;
                }
            }
        }
    }

    Label {
        text: qsTr("Tint");
        Column {
            width: parent.width;
            spacing: 3 * dpiScale;
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
            SliderSteppers {
                target: tintSlider;
                step: 5;
            }
        }
    }

    CheckBox {
        id: bakeCb;
        text: qsTr("Apply to rendered video");
        checked: false;
        onCheckedChanged: { root.bakeInRender = checked; root.applyEnabled(); }
    }

    // Only shown when the grade is actually baked in: the tag describes what the LUT
    // produced, so declaring it on an export whose pixels were never converted would be a
    // worse lie than saying nothing. The render side applies the same rule.
    Label {
        visible: root.colorEnabled && root.bakeInRender;
        text: qsTr("Output color space");
        tooltip: qsTr("What the exported file declares itself to be. This writes tags only - it never changes a pixel. Guessed from the LUT's name; correct it here if the guess is wrong.");
        ComboBox {
            id: outputSpace;
            width: parent.width;
            font.pixelSize: 12 * dpiScale;
            currentIndex: 0;
            // Ids parallel to `model`, kept out of the visible labels so the labels stay
            // translatable and the ids stay stable. Both come from OutputColorSpace::ALL, so
            // the dropdown cannot drift out of step with the Rust enum.
            property var ids: [];
            property bool preventChange: false;

            function reload(): void {
                const list = controller.get_output_color_spaces();
                let newIds = [], labels = [];
                for (let i = 0; i < list.length; ++i) { newIds.push(list[i].id); labels.push(list[i].label); }
                ids = newIds;
                model = labels;
                syncFromCore();
            }
            // The core is the authority: the value moves on its own when a LUT is loaded
            // (the name-based guess) or a project is opened, and the dropdown follows.
            function syncFromCore(): void {
                const idx = ids.indexOf(controller.get_output_color_space());
                if (idx >= 0 && idx !== currentIndex) {
                    preventChange = true;
                    currentIndex = idx;
                    preventChange = false;
                }
            }
            onCurrentIndexChanged: {
                if (!preventChange && currentIndex >= 0 && currentIndex < ids.length) {
                    controller.set_output_color_space(ids[currentIndex]);
                }
            }
            Component.onCompleted: reload();
        }
    }

    Label {
        visible: root.colorEnabled && root.bakeInRender;
        text: qsTr("Output range");
        tooltip: qsTr("Limited range is what almost every player assumes for video, so it is the safe choice for a file you are going to watch or share directly. Full range keeps every code value - better if the file is going back into a grading suite.");
        ComboBox {
            id: outputRange;
            width: parent.width;
            font.pixelSize: 12 * dpiScale;
            currentIndex: 0;
            property var ids: [];
            property bool preventChange: false;

            function reload(): void {
                const list = controller.get_output_ranges();
                let newIds = [], labels = [];
                for (let i = 0; i < list.length; ++i) { newIds.push(list[i].id); labels.push(list[i].label); }
                ids = newIds;
                model = labels;
                syncFromCore();
            }
            // As with the colour space, the core is the authority: loading a display-referred
            // LUT moves this to Limited on its own, and the dropdown has to follow.
            function syncFromCore(): void {
                const idx = ids.indexOf(controller.get_output_range());
                if (idx >= 0 && idx !== currentIndex) {
                    preventChange = true;
                    currentIndex = idx;
                    preventChange = false;
                }
            }
            onCurrentIndexChanged: {
                if (!preventChange && currentIndex >= 0 && currentIndex < ids.length) {
                    controller.set_output_range(ids[currentIndex]);
                }
            }
            Component.onCompleted: reload();
        }
    }

    InfoMessageSmall {
        show: root.colorEnabled && !root.bakeInRender;
        text: qsTr("Color adjustments affect the preview only. Enable \"Apply to rendered video\" to include them in the export.");
    }
}
