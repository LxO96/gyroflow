// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2021-2022 Adrian <adrian.eddy at gmail>

import QtQuick
import QtQuick.Controls as QQC

Row {
    id: root;
    spacing: 5 * dpiScale;
    height: 25 * dpiScale;
    property alias slider: slider;
    property alias field: field;
    property alias defaultValue: field.defaultValue;
    property real from: 0;
    property real to: 1;
    property alias live: slider.live;

    // A logarithmic track, for quantities where equal *ratios* matter rather than equal
    // differences - colour temperature being the case in hand: 2000-3000 K is a huge shift
    // and 20000-21000 K is imperceptible, so a linear track squeezes the entire usable range
    // into its left edge.
    //
    // Only the *track* changes. `value`, `field.value`, `defaultValue` and everything handed
    // to the keyframe machinery stay in real units, because those are stored verbatim
    // (SliderWithField.qml keyframe calls below) and core/keyframes.rs formats them as such.
    // Putting log units into `root.value` would silently corrupt every saved keyframe.
    property bool logarithmic: false;
    function toSlider(v: real): real   { return root.logarithmic? Math.log(Math.max(v, 1e-9)) : v; }
    function fromSlider(v: real): real { return root.logarithmic? Math.exp(v) : v; }

    // Where this slider's value started out, marked on the track. In real units; NaN falls
    // back to `defaultValue`.
    property bool showOrigin: false;
    property real originValue: NaN;
    property alias unit: field.unit;
    property alias precision: field.precision;
    property string keyframe: "";
    property bool keyframesEnabled: false;
    property real scaler: 1;

    property bool preventChange: false;

    property real value: defaultValue;

    property alias contextMenu: menuLoader.sourceComponent;

    onValueChanged: {
        if (!root.preventChange) {
            field.value = value * scaler;
        }
    }
    Connections {
        target: controller;
        enabled: root.keyframe.length > 0;
        function onKeyframe_value_updated(keyframe: string, value: real): void {
            if (keyframe == root.keyframe) {
                root.preventChange = true;
                field.value = value * root.scaler;
                root.preventChange = false;
            }
        }
        function onGyroflow_file_loaded(obj: var): void {
            if (obj.keyframes && typeof obj.keyframes[root.keyframe] !== "undefined") {
                root.keyframesEnabled = true;
            }
        }
    }

    Slider {
        id: slider;
        width: parent.width - field.width - root.spacing;
        anchors.verticalCenter: parent.verticalCenter;
        property bool preventChange: false;
        from: root.toSlider(root.from);
        to:   root.toSlider(root.to);
        onValueChanged: if (!preventChange) field.value = root.fromSlider(value);
        unit: field.unit;
        precision: field.precision;
        showOrigin: root.showOrigin;
        originValue: root.toSlider(isNaN(root.originValue)? root.defaultValue : root.originValue);
        displayTransform: root.logarithmic? root.fromSlider : null;

        ContextMenuMouseArea {
            underlyingItem: slider;
            onContextMenu: (isHold, x, y) => menuLoader.popup(slider, x, y);
        }

        Component {
            id: defaultMenu;
            Menu {
                font.pixelSize: 11.5 * dpiScale;
                Action {
                    iconName: "undo";
                    text: qsTr("Reset value");
                    enabled: field.value != defaultValue;
                    onTriggered: {
                        field.value = defaultValue;
                    }
                }
                Action {
                    iconName: "keyframe";
                    enabled: root.keyframe.length > 0;
                    text: qsTr("Enable keyframing");
                    checked: root.keyframesEnabled;
                    onTriggered: {
                        checked = !checked;
                        root.keyframesEnabled = checked;
                        if (!checked) {
                            controller.clear_keyframes_type(root.keyframe);
                        }
                    }
                }
                Action {
                    iconName: "plus";
                    enabled: root.keyframe.length > 0 && root.keyframesEnabled;
                    text: qsTr("Add keyframe");
                    onTriggered: controller.set_keyframe(root.keyframe, window.videoArea.timeline.getTimestampUs(), root.value);
                }
            }
        }
        ContextMenuLoader {
            id: menuLoader;
            sourceComponent: defaultMenu
        }
    }
    NumberField {
        id: field;
        width: 55 * dpiScale;
        height: 25 * dpiScale;
        precision: 3;
        font.pixelSize: 11 * dpiScale;
        anchors.verticalCenter: parent.verticalCenter;
        contextMenu: root.contextMenu;
        onValueChanged: {
            slider.preventChange = true;
            slider.value = root.toSlider(value);
            Qt.callLater(() => { if (slider) slider.preventChange = false; });

            if (!root.preventChange) {
                root.preventChange = true;
                root.value = value / root.scaler;
                root.preventChange = false;

                if (root.keyframe && root.keyframesEnabled) {
                    controller.set_keyframe(root.keyframe, window.videoArea.timeline.getTimestampUs(), root.value);
                }
            }
        }
    }
}
