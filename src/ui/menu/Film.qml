// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2026 Adrian <adrian.eddy at gmail>

import QtQuick

import "../components/"

MenuItem {
    id: root;
    text: qsTr("Film");
    // No film icon exists in the packaged subset (src/resources.rs); "grid" is shared with Denoise.
    iconName: "grid";
    innerItem.enabled: window.videoArea.vid.loaded;
    objectName: "film";
    opened: false;

    readonly property var stocks: controller.get_film_stocks();
    readonly property var formats: controller.get_film_formats();
    property bool filmOn: stockCb.currentIndex > 0;

    // Mirrors the core importer (`obj.get("film")`): a file with no `film` section leaves the panel alone.
    function loadGyroflow(obj: var): void {
        const f = obj.film;
        if (!f || typeof f !== "object") return;
        if (typeof f.stock !== "undefined") stockCb.selectId(f.stock);
        if (typeof f.format !== "undefined") formatCb.selectId(f.format);
        if (typeof f.tone !== "undefined") toneCb.checked = !!f.tone;
        if (typeof f.grain !== "undefined") grainCb.checked = !!f.grain;
        if (typeof f.halation !== "undefined") haloCb.checked = !!f.halation;
        if (typeof f.grain_amount !== "undefined") grainSlider.value = +f.grain_amount;
        if (typeof f.halation_amount !== "undefined") haloSlider.value = +f.halation_amount;
    }

    Label {
        text: qsTr("Stock");
        ComboBox {
            id: stockCb;
            width: parent.width;
            font.pixelSize: 12 * dpiScale;
            model: [qsTr("Off")].concat(root.stocks.map(s => s.label));
            currentIndex: 0;
            property bool preventChange: false;
            function selectId(id: string): void {
                const i = id === "off" ? 0 : root.stocks.findIndex(s => s.id === id) + 1;
                preventChange = true; currentIndex = Math.max(0, i); preventChange = false;
                controller.set_film_stock(currentIndex > 0 ? root.stocks[currentIndex - 1].id : "off");
            }
            onCurrentIndexChanged: if (!preventChange) controller.set_film_stock(currentIndex > 0 ? root.stocks[currentIndex - 1].id : "off");
        }
    }

    BasicText {
        id: gateText;
        width: parent.width;
        wrapMode: Text.WordWrap;
        visible: root.filmOn && text.length > 0;
        opacity: 0.6;
        text: "";
        // The gate depends on the colour panel and the output colour space, which change elsewhere.
        Timer { interval: 1000; running: root.filmOn && root.opened; repeat: true; triggeredOnStart: true; onTriggered: gateText.text = controller.get_film_gate_message(); }
    }

    Label {
        text: qsTr("Format");
        tooltip: qsTr("The film gauge the frame stands for. Smaller formats have coarser grain and wider halation at the same output size.");
        visible: root.filmOn;
        ComboBox {
            id: formatCb;
            width: parent.width;
            font.pixelSize: 12 * dpiScale;
            model: root.formats.map(f => f.label);
            currentIndex: Math.max(0, root.formats.findIndex(f => f.id === "35"));
            function selectId(id: string): void { const i = root.formats.findIndex(f => f.id === id); if (i >= 0) currentIndex = i; }
            onCurrentIndexChanged: if (currentIndex >= 0) controller.set_film_format(root.formats[currentIndex].id);
        }
    }

    CheckBox { id: toneCb; visible: root.filmOn; text: qsTr("Tone & colour"); checked: true; onCheckedChanged: controller.set_film_flag("tone", checked);
               tooltip: qsTr("Re-develop the grade with the stock's curve and colour. Off keeps your grade and adds only grain and halation."); }
    CheckBox { id: grainCb; visible: root.filmOn; text: qsTr("Grain"); checked: true; onCheckedChanged: controller.set_film_flag("grain", checked); }
    CheckBox { id: haloCb; visible: root.filmOn; text: qsTr("Halation"); checked: true; onCheckedChanged: controller.set_film_flag("halation", checked);
               tooltip: qsTr("A red glow around clipped highlights, from light reflecting off the film base."); }

    Label {
        text: qsTr("Grain amount");
        visible: root.filmOn && grainCb.checked;
        SliderWithField {
            id: grainSlider;
            width: parent.width;
            from: 0; to: 200; value: 1.0; defaultValue: 1.0; scaler: 100.0; unit: "%"; precision: 0;
            onValueChanged: controller.set_film_param("grain_amount", value);
        }
    }
    Label {
        text: qsTr("Halation");
        visible: root.filmOn && haloCb.checked;
        SliderWithField {
            id: haloSlider;
            width: parent.width;
            from: 0; to: 100; value: 0.5; defaultValue: 0.5; scaler: 100.0; unit: "%"; precision: 0;
            onValueChanged: controller.set_film_param("halation_amount", value);
        }
    }
}
