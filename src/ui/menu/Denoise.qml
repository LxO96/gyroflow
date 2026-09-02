// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2026 Adrian <adrian.eddy at gmail>

import QtQuick

import "../components/"

MenuItem {
    id: root;
    text: qsTr("Denoise");
    // No grain/noise icon exists in the packaged subset (src/resources.rs). "grid" is the
    // closest existing meaning - a field of samples - and is not used by any other panel.
    iconName: "grid";
    innerItem.enabled: window.videoArea.vid.loaded;
    objectName: "denoise";
    opened: false;

    // NOT called `enabled`: MenuItem derives from Item, which already has `enabled`, and
    // shadowing it here would silently disable the whole panel's interactivity. Same reason
    // ColorAdjustments calls its flag `colorEnabled`.
    property bool denoiseEnabled: false;

    // Restores the panel from a loaded project or an applied preset. Mirrors the core
    // importer (core/lib.rs, the `obj.get("denoise")` arm): a file with no `denoise` section
    // leaves the core's current settings alone, so the panel leaves its controls alone too.
    //
    // Without this the core and the panel drift apart on every project load - the render
    // would denoise while the panel showed the previous clip's values.
    function loadGyroflow(obj: var): void {
        const d = obj.denoise;
        if (!d || typeof d !== "object") return;

        if (typeof d.luma_strength   !== "undefined") lumaSlider  .value = +d.luma_strength;
        if (typeof d.chroma_strength !== "undefined") chromaSlider.value = +d.chroma_strength;
        if (typeof d.enabled         !== "undefined") enableCb.checked = !!d.enabled;
        // Read back from the core rather than from `d.quality`: the core rejects an
        // unknown id and keeps its current value, and the dropdown must follow what was
        // actually applied, not what the file asked for.
        qualityCb.syncFromCore();
    }

    CheckBox {
        id: enableCb;
        text: qsTr("Enable");
        checked: false;
        onCheckedChanged: {
            root.denoiseEnabled = checked;
            controller.set_denoise_enabled(checked);
        }
    }

    // The preview cannot show denoising while the video is playing: the player owns decoding
    // and seeks arbitrarily, so no symmetric ±k window of neighbouring frames exists until
    // playback stops. Exports are unaffected. Saying so here is what stops a user concluding
    // the feature is broken when nothing changes during playback.
    BasicText {
        width: parent.width;
        wrapMode: Text.WordWrap;
        visible: root.denoiseEnabled;
        opacity: 0.6;
        text: qsTr("Denoising is visible in the preview only while playback is paused. Exports always include it.");
    }

    Label {
        text: qsTr("Quality");
        tooltip: qsTr("How many neighbouring frames are averaged together. Fast uses three, High uses seven - more frames remove more noise but cost proportionally more time to export.");
        ComboBox {
            id: qualityCb;
            width: parent.width;
            font.pixelSize: 12 * dpiScale;
            currentIndex: 0;
            // Ids parallel to `model`, kept out of the visible labels so the labels stay
            // translatable and the ids stay stable. They are the strings DenoiseQuality::from_id
            // accepts; anything else is rejected and warned about by the controller.
            readonly property var ids: ["fast", "high"];
            property bool preventChange: false;

            model: [qsTr("Fast"), qsTr("High")];

            // The core is the authority - a loaded project or a preset moves the value on its
            // own - so the dropdown reads it back rather than assuming its own default.
            function syncFromCore(): void {
                const idx = ids.indexOf(controller.get_denoise_quality());
                if (idx >= 0 && idx !== currentIndex) {
                    preventChange = true;
                    currentIndex = idx;
                    preventChange = false;
                }
            }
            onCurrentIndexChanged: {
                if (!preventChange && currentIndex >= 0 && currentIndex < ids.length) {
                    controller.set_denoise_quality(ids[currentIndex]);
                }
            }
            Component.onCompleted: syncFromCore();
        }
    }

    // Both sliders are shown in percent while the core takes 0..1, hence `scaler: 100`.
    // SliderWithField.qml divides `field.value` by the scaler to produce `root.value`, so
    // `onValueChanged` already carries core units and needs no conversion of its own.
    Label {
        text: qsTr("Luma strength");
        tooltip: qsTr("How readily a neighbouring frame's brightness is allowed to differ from this frame's and still be averaged in. Higher removes more noise; too high smears moving detail.");
        Column {
            width: parent.width;
            spacing: 3 * dpiScale;
            SliderWithField {
                id: lumaSlider;
                width: parent.width;
                from: 0; to: 100;              // core: 0 .. 1
                value: 0.5; defaultValue: 0.5;
                scaler: 100.0;
                unit: "%"; precision: 0;
                onValueChanged: controller.set_denoise_param("luma_strength", value);
            }
            SliderSteppers {
                target: lumaSlider;
                step: 5;
            }
        }
    }

    Label {
        text: qsTr("Chroma strength");
        tooltip: qsTr("The same threshold for the colour planes. Chroma noise is usually coarser than luma noise, so this often wants to sit higher than Luma strength.");
        Column {
            width: parent.width;
            spacing: 3 * dpiScale;
            SliderWithField {
                id: chromaSlider;
                width: parent.width;
                from: 0; to: 100;              // core: 0 .. 1
                value: 0.5; defaultValue: 0.5;
                scaler: 100.0;
                unit: "%"; precision: 0;
                onValueChanged: controller.set_denoise_param("chroma_strength", value);
            }
            SliderSteppers {
                target: chromaSlider;
                step: 5;
            }
        }
    }
}
