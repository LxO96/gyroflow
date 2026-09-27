// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2021-2022 Adrian <adrian.eddy at gmail>

import QtQuick

import "../components/"

MenuItem {
    id: root;
    text: qsTr("Advanced");
    iconName: "settings";
    opened: false;
    objectName: "advanced";

    Item {
        id: sett;
        property alias previewPipeline: previewPipeline.currentIndex;
        property alias renderBackground: renderBackground.text;
        property alias uiScaling: uiScaling.currentIndex;
        property alias safeAreaGuide: safeAreaGuide.checked;
        property alias gpudecode: gpudecode.checked;
        property alias backgroundMode: backgroundMode.currentIndex;
        property alias marginPixels: marginPixels.value;
        property alias featherPixels: featherPixels.value;
        property alias defaultSuffix: defaultSuffix.text;
        // `exportFolderMode` and `exportSubfolder` are deliberately NOT here: the control below
        // reads and writes them itself, because both have to be persisted synchronously for the
        // Rust-side resolver to see them, and this block saves through a queued connection.
        property alias autoPixelFormat: autoPixelFormat.checked;
        property alias playSounds: playSounds.checked;
        property alias r3dConvertFormat: r3dConvertFormat.currentIndex;
        property alias r3dColorMode: r3dColorMode.currentIndex;
        property alias r3dGammaCurve: r3dGammaCurve.currentIndex;
        property alias r3dColorSpace: r3dColorSpace.currentIndex;
        property alias r3dRedlineParams: r3dRedlineParams.text;
        property string lang: ui_tools.get_default_language();

        Component.onCompleted: {
            settings.init(sett);
            let selectedIndex = 0;
            let i = 0;
            langList.model = langList.langs.map((x) => { if (x[1] == sett.lang) { selectedIndex = i; } i++; return x[0]; });
            langList.currentIndex = selectedIndex;
        }
        function propChanged() { settings.propChanged(sett); }
    }
    property alias defaultSuffix: defaultSuffix;
    property alias autoPixelFormat: autoPixelFormat;
    property alias previewResolution: previewResolution.currentIndex;

    /// Point the loaded clip's output path at wherever the "Export to" setting now says.
    ///
    /// Without this a change only takes effect on the NEXT video opened, which reads as the
    /// setting having been ignored. Everything is guarded because this menu is restored from
    /// settings during startup, before there is a video or an output path field to speak of.
    function refreshOutputFolder(): void {
        if (!window.videoArea || !window.outputFile) return;
        const url = window.videoArea.loadedFileUrl.toString();
        if (!url) return;
        window.outputFile.setFolder(render_queue.get_default_output_folder(url));
    }
    property alias r3dConvertFormat: r3dConvertFormat;
    property alias gpudecode: gpudecode;

    function loadGyroflow(obj: var): void {
        if (obj.hasOwnProperty("background_mode")) backgroundMode.currentIndex = +obj.background_mode;
        if (obj.hasOwnProperty("background_margin")) marginPixels.value = +obj.background_margin;
        if (obj.hasOwnProperty("background_margin_feather")) featherPixels.value = +obj.background_margin_feather;
        if (obj.hasOwnProperty("background_color")) renderBackground.text = Qt.rgba(obj.background_color[0], obj.background_color[1], obj.background_color[2], obj.background_color[3]).toString();
    }
    Label {
        position: Label.LeftPosition;
        text: qsTr("Preview resolution");

        ComboBox {
            id: previewResolution;
            model: [QT_TRANSLATE_NOOP("Popup", "Full"), "4k", "1080p", "720p", "480p"];
            font.pixelSize: 12 * dpiScale;
            width: parent.width;
            currentIndex: 0;
            Component.onCompleted: {
                if (settings.value("previewResolution", -1) != -1)
                    currentIndex = +settings.value("previewResolution", -1);
            }
            onCurrentIndexChanged: {
                let target_height = -1; // Full
                switch (currentIndex) {
                    case 0: window.videoArea.vid.setProperty("scale", ""); break;
                    case 1: target_height = 2160; window.videoArea.vid.setProperty("scale", "3840x2160"); break;
                    case 2: target_height = 1080; window.videoArea.vid.setProperty("scale", "1920x1080"); break;
                    case 3: target_height = 720;  window.videoArea.vid.setProperty("scale", "1280x720");  break;
                    case 4: target_height = 480;  window.videoArea.vid.setProperty("scale", "640x480");   break;
                }

                controller.set_preview_resolution(target_height, window.videoArea.vid);
                settings.setValue("previewResolution", currentIndex);
            }
        }
    }

    Label {
        position: Label.LeftPosition;
        text: qsTr("Background mode");
        ComboBox {
            id: backgroundMode;
            model: [QT_TRANSLATE_NOOP("Popup", "Solid color"), QT_TRANSLATE_NOOP("Popup", "Repeat edge pixels"), QT_TRANSLATE_NOOP("Popup", "Mirror edge pixels"), QT_TRANSLATE_NOOP("Popup", "Margin with feather")];
            font.pixelSize: 12 * dpiScale;
            width: parent.width;
            currentIndex: 0;
            onCurrentIndexChanged: controller.background_mode = currentIndex;
        }
    }
    Column {
        width: parent.width;
        visible: backgroundMode.currentIndex == 3;
        Label {
            text: qsTr("Margin");
            SliderWithField {
                id: marginPixels;
                value: 0.20;
                defaultValue: 20;
                from: 0;
                to: 50;
                unit: "%";
                precision: 0;
                width: parent.width;
                keyframe: "BackgroundMargin";
                scaler: 100.0;
                onValueChanged: controller.background_margin = value;
            }
        }
        Label {
            text: qsTr("Feather");
            SliderWithField {
                id: featherPixels;
                value: 0.05;
                defaultValue: 5;
                from: 0;
                to: 50;
                unit: "%";
                precision: 0;
                width: parent.width;
                keyframe: "BackgroundFeather";
                scaler: 100.0;
                onValueChanged: controller.background_margin_feather = value;
            }
        }
    }
    Label {
        position: Label.LeftPosition;
        visible: backgroundMode.currentIndex == 0;
        text: qsTr("Render background");

        TextField {
            id: renderBackground;
            text: "#111111";
            width: parent.width;
            onTextChanged: controller.set_background_color(text, window.videoArea.vid);
        }
    }
    Label {
        position: Label.LeftPosition;
        text: qsTr("Theme");

        ComboBox {
            id: themeList;
            model: [];
            font.pixelSize: 12 * dpiScale;
            width: parent.width;
            Component.onCompleted: {
                const savedTheme = +settings.value("theme", 1);
                let m = [QT_TRANSLATE_NOOP("Popup", "Light"), QT_TRANSLATE_NOOP("Popup", "Dark")];
                if (!(isMobile && screenSize < 7.0)) {
                    m.push(QT_TRANSLATE_NOOP("Popup", "Mobile Light"));
                    m.push(QT_TRANSLATE_NOOP("Popup", "Mobile Dark"));
                }
                model = m;
                currentIndex = savedTheme;
            }
            onCurrentIndexChanged: {
                const themes = ["light", "dark", "mobile_light", "mobile_dark"];
                let theme = themes[currentIndex];
                ui_tools.set_theme(theme);
                settings.setValue("theme", currentIndex);
            }
        }
    }
    Label {
        position: Label.LeftPosition;
        text: qsTr("UI scaling");
        ComboBox {
            id: uiScaling;
            model: ["50%", "75%", "100%", "125%", "150%", "175%", "200%"];
            font.pixelSize: 12 * dpiScale;
            width: parent.width;
            currentIndex: 2;
            onCurrentIndexChanged: {
                ui_tools.set_scaling([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0][currentIndex]);
            }
        }
    }
    Label {
        position: Label.LeftPosition;
        text: qsTr("Language");

        ComboBox {
            id: langList;
            property var langs: [
                ["English",                      "en"],
                ["Chinese - Simplified (简体中文)",  "zh_CN"],
                ["Chinese - Traditional (繁體中文)", "zh_TW"],
                ["Czech (Čeština)",              "cs"],
                ["Danish (dansk)",               "da"],
                ["Finnish (suomi)",              "fi"],
                ["French (français)",            "fr"],
                ["Galician (Galego)",            "gl"],
                ["German (Deutsch)",             "de"],
                ["Greek (Ελληνικά)",             "el"],
                ["Indonesian (Bahasa Indonesia)","id"],
                ["Italian (italiano)",           "it"],
                ["Japanese (日本語)",             "ja"],
                ["Korean (한국어)",              "ko"],
                ["Norwegian (norsk)",            "no"],
                ["Polish (polski)",              "pl"],
                ["Portuguese - Brazilian (português brasileiro)", "pt_BR"],
                ["Portuguese (português)",       "pt"],
                ["Russian (русский)",            "ru"],
                ["Serbian (srpski)",             "sr_CS"],
                ["Slovak (slovenský)",           "sk"],
                ["Spanish (español)",            "es"],
                ["Turkish (Türkçe)",             "tr"],
                ["Ukrainian (Українська мова)",  "uk"]
            ];
            font.pixelSize: 12 * dpiScale;
            width: parent.width;
            function setLang(): void {
                sett.lang = langs[currentIndex][1];

                window.LayoutMirroring.enabled = sett.lang == "ar" || sett.lang == "fa" || sett.lang == "he";
                window.LayoutMirroring.childrenInherit = true;
                ui_tools.set_language(sett.lang);
            }
            onCurrentIndexChanged: Qt.callLater(setLang);
        }
    }
    CheckBox {
        id: safeAreaGuide;
        text: qsTr("Safe area guide");
        tooltip: qsTr("When FOV > 1, show an rectangle simulating FOV = 1 over the preview video.\nNote that this is only a visual indicator, it doesn't affect rendering.");
        checked: false;
        onCheckedChanged: controller.show_safe_area = checked;
        Component.onCompleted: Qt.callLater(checkedChanged);
    }
    CheckBox {
        id: gpudecode;
        text: qsTr("Use GPU decoding");
        checked: true;
        onCheckedChanged: controller.set_gpu_decoding(checked);
    }
    CheckBox {
        id: videoThumbnails;
        visible: Qt.platform.os == "windows";
        text: qsTr("Thumbnails for video Windows can't preview");
        tooltip: qsTr("Lets Windows Explorer show a preview frame for formats it cannot decode itself, such as 10-bit 4:2:2 video.

While this is on, Gyroflow generates the thumbnails for every .mp4, .mov, .mxf, .360, .insv and .exr file - Windows offers no way to handle only the ones it fails at.

Already-drawn thumbnails stay in Explorer's cache, so existing files may not change until that cache is cleared.");
        checked: false;
        property bool loading: true;
        Component.onCompleted: {
            checked = settings.value("videoThumbnailsEnabled", "false") == "true";
            loading = false;
        }
        onCheckedChanged: {
            // Skip the write while restoring the saved value, or opening this panel would
            // re-register on every launch.
            if (loading) return;
            if (!controller.set_video_thumbnails_enabled(checked)) {
                // The registry write failed, so the checkbox must not claim otherwise.
                loading = true;
                checked = !checked;
                loading = false;
                window.showNotification(Modal.Error, qsTr("Couldn't change the thumbnail setting."));
            }
        }
    }
    Label {
        id: r3dConvertFormatLabel;
        position: Label.LeftPosition;
        text: qsTr("Format for R3D conversion");
        visible: !!controller.find_redline();
        ComboBox {
            id: r3dConvertFormat;
            model: [
                "ProRes 422 HQ",
                "ProRes 422",
                "ProRes 422 LT",
                "ProRes 422 Proxy",
                "ProRes 4444",
                "ProRes 4444 XQ",
            ];
            font.pixelSize: 12 * dpiScale;
            width: parent.width;
        }
    }
    Label {
        position: Label.LeftPosition;
        text: "Colors for R3D conversion";
        visible: r3dConvertFormatLabel.visible;
        ComboBox {
            id: r3dColorMode;
            model: ["Fully graded in REDCINE-X", "Primary development only"];
            font.pixelSize: 12 * dpiScale;
            width: parent.width;
        }
    }
    Label {
        position: Label.LeftPosition;
        text: "Gamma curve for R3D conversion";
        visible: r3dConvertFormatLabel.visible;
        ComboBox {
            id: r3dGammaCurve;
            model: ["Linear", "BT.709", "sRGB", "REDlog", "PDLog985", "PDLog685", "PDLogCustom", "REDspace", "REDgamma", "REDLogFilm", "REDgamma2", "REDgamma3", "REDgamma4", "ST 2084", "BT.1886", "Log3G12", "Log3G10", "Hybrid Log-Gamma", "Gamma 2.2", "Gamma 2.6"];
            currentIndex: 7;
            font.pixelSize: 12 * dpiScale;
            width: parent.width;
        }
    }
    Label {
        position: Label.LeftPosition;
        text: "Color space for R3D conversion";
        visible: r3dConvertFormatLabel.visible;
        ComboBox {
            id: r3dColorSpace;
            model: [ "REDspace", "CameraRGB", "BT.709", "REDcolor", "sRGB", "Adobe1998", "REDcolor2", "REDcolor3", "DRAGONcolor", "XYZ", "REDcolor4", "DRAGONcolor2", "BT.2020", "REDWideGamutRGB", "DCI-P3", "DCI-P3 D65"];
            currentIndex: 0;
            font.pixelSize: 12 * dpiScale;
            width: parent.width;
        }
    }
    Label {
        position: Label.LeftPosition;
        text: "Additional REDline params";
        visible: r3dConvertFormatLabel.visible;
        TextField {
            id: r3dRedlineParams;
            width: parent.width;
        }
    }
    Label {
        position: Label.LeftPosition;
        text: qsTr("Preview pipeline");

        ComboBox {
            id: previewPipeline;
            model: ["Zero-copy Qt RHI", "Zero-copy OpenCL", "OpenCL/wgpu/CPU"];
            font.pixelSize: 12 * dpiScale;
            width: parent.width;
            currentIndex: 0;
            onCurrentIndexChanged: {
                if (currentIndex != 2) {
                    if (previewResolution.currentIndex == 3) {
                        previewResolution.currentIndex = 0;
                        Qt.callLater(window.exportSettings.notifySizeChanged);
                    }
                }
                controller.set_preview_pipeline(currentIndex);
                Qt.callLater(processingDevice.updateController);
                Qt.callLater(window.videoArea.vid.forceRedraw);
            }
            Component.onCompleted: Qt.callLater(currentIndexChanged);
        }
    }
    Label {
        position: Label.TopPosition;
        text: qsTr("Device for video processing");
        visible: processingDevice.model.length > 0;
        ComboBox {
            id: processingDevice;
            model: [];
            font.pixelSize: 12 * dpiScale;
            width: parent.width;
            currentIndex: 0;
            property bool preventChange: true;
            Connections {
                target: controller;
                function onGpu_list_loaded(list: list<string>): void {
                    const saved = settings.value("processingDevice", defaultInitializedDevice);
                    processingDevice.preventChange = true;
                    processingDevice.model = [...list, qsTr("CPU only")];
                    for (let i = 0; i < list.length; ++i) {
                        if (list[i] == saved) {
                            processingDevice.currentIndex = i;
                            break;
                        }
                    }
                    if (saved != defaultInitializedDevice) {
                        Qt.callLater(processingDevice.updateController);
                    }
                    processingDevice.preventChange = false;
                    if (saved == "cpu") {
                        processingDevice.currentIndex = processingDevice.model.length - 1;
                    }
                }
            }
            Component.onCompleted: controller.list_gpu_devices();
            onCurrentIndexChanged: {
                if (preventChange) return;
                Qt.callLater(processingDevice.updateController);
            }
            function updateController(): void {
                if (model.length == 0) return;
                if (currentIndex == model.length - 1) {
                    controller.set_device(-1);
                } else {
                    controller.set_device(currentIndex);
                }
                const text = currentIndex == model.length - 1? "cpu" : currentText;
                settings.setValue("processingDevice", text);
                settings.setValue("processingDeviceIndex", processingDevice.currentIndex);
            }
        }
    }
    BasicText {
        visible: text.length > 0;
        text: controller.processing_info;
        width: parent.width;
        wrapMode: Text.WordWrap;
        font.pixelSize: 11 * dpiScale;
    }
    Label {
        position: Label.LeftPosition;
        text: qsTr("Default file suffix");

        TextField {
            id: defaultSuffix;
            text: "_stabilized";
            width: parent.width;
            onTextChanged: render_queue.default_suffix = text;
        }
    }

    // Where exports go. Sits next to the file suffix because the two answer the same question
    // between them - what the output is called, and where it lands.
    //
    // Three mutually exclusive answers, so radio buttons rather than a checkbox each: "next to
    // the source" and "a fixed folder" cannot both be true, and two independent checkboxes would
    // let a user express that and then wonder which won.
    //
    // The RULE lives in Rust (`render_queue::resolve_output_folder`), not here. This is only the
    // control that writes the settings it reads, so the render queue and the output path field
    // cannot disagree about what "subfolder of the source" means.
    Label {
        position: Label.TopPosition;
        text: qsTr("Export to");

        Column {
            id: exportFolderMode;
            width: parent.width;
            spacing: 2 * dpiScale;

            // 0: next to the source, 1: subfolder of the source, 2: a fixed folder. Stored as a
            // number rather than the button's identity so the setting survives reordering.
            property int mode: 0;

            // False until the stored value has been read. Nothing here may WRITE settings before
            // then: creating the buttons assigns `checked`, which sets `mode`, which would
            // persist the default and overwrite the user's stored choice before it was ever read.
            // That is not hypothetical - it is what this flag was added to fix.
            property bool restored: false;

            function syncButtons(): void {
                nextToSource.checked = mode == 0;
                intoSubfolder.checked = mode == 1;
                intoFixedFolder.checked = mode == 2;
            }
            onModeChanged: {
                syncButtons();
                if (!restored) return;
                // Written explicitly rather than left to the `sett` alias block, which persists
                // through a QUEUED connection and so lands after this call stack. The resolver
                // reads the STORED value, and would otherwise answer with the mode just left.
                settings.setValue("exportFolderMode", mode);
                root.refreshOutputFolder();
            }
            Component.onCompleted: {
                // Read directly rather than relying on `settings.init` having run: these are
                // separate objects with no guaranteed completion order between them, and this
                // way the outcome does not depend on which went first.
                if (!settings.contains("exportFolderMode") && +settings.value("preserveOutputPath", 0)) {
                    // Carry over the checkbox this replaces, so anyone already exporting to a
                    // fixed folder keeps doing that rather than silently reverting.
                    mode = 2;
                } else {
                    mode = +settings.value("exportFolderMode", 0);
                }
                syncButtons();
                exportSubfolder.text = settings.value("exportSubfolder", "Stabilized") || "Stabilized";
                restored = true;
                // Persist the default the field is showing, rather than leaving the setting
                // absent while the UI implies a value. The resolver reads the stored name, so an
                // unwritten default meant picking "subfolder" did nothing at all: the mode was
                // saved, the name was not, and every export went beside the source anyway.
                if (!settings.contains("exportSubfolder")) settings.setValue("exportSubfolder", exportSubfolder.text);
            }

            RadioButton {
                id: nextToSource;
                text: qsTr("Next to the source file");
                checked: true;
                onCheckedChanged: if (checked) exportFolderMode.mode = 0;
            }
            RadioButton {
                id: intoSubfolder;
                // Hidden where it cannot work: under sandboxing an export folder is something
                // the user grants access to, not something the app invents a path inside.
                visible: !isSandboxed;
                text: qsTr("Subfolder of the source file");
                onCheckedChanged: if (checked) exportFolderMode.mode = 1;
            }
            TextField {
                id: exportSubfolder;
                visible: intoSubfolder.visible;
                enabled: intoSubfolder.checked;
                opacity: enabled? 1 : 0.5;
                x: 20 * dpiScale;
                width: parent.width - x;
                text: "Stabilized";
                onTextChanged: {
                    if (!exportFolderMode.restored) return; // don't overwrite the stored name with the default
                    settings.setValue("exportSubfolder", text); // ahead of the queued save, as above
                    if (intoSubfolder.checked) root.refreshOutputFolder();
                }
                tooltip: qsTr("Created on export if it doesn't exist. A path like \"Renders/2026\" nests.");
            }
            RadioButton {
                id: intoFixedFolder;
                text: qsTr("A fixed folder");
                onCheckedChanged: if (checked) exportFolderMode.mode = 2;
            }
            OutputPathField {
                id: exportFolder;
                folderOnly: true;
                enabled: intoFixedFolder.checked;
                opacity: enabled? 1 : 0.5;
                x: 20 * dpiScale;
                width: parent.width - x;

                // Restored by hand rather than through the `sett` alias block: that binds to a
                // plain property, and this one has to travel through `setFolder` to keep
                // `folderUrl`, `filename` and the displayed text consistent.
                Component.onCompleted: {
                    const saved = settings.value("preservedOutputPath", "");
                    if (saved) setFolder(saved);
                }
                onFolderUrlChanged: {
                    const folder = folderUrl.toString();
                    if (folder) settings.setValue("preservedOutputPath", folder);
                    if (intoFixedFolder.checked) root.refreshOutputFolder();
                }
            }
        }
    }

    // Standing answer to the pixel format warning in export settings.
    //
    // Ticked by accepting the conversion in that warning, so nobody has to find this to get the
    // behaviour - it is here to be seen and switched off. What it stores is the decision, not the
    // format: the check recomputes the right format for whatever encoder is selected.
    CheckBox {
        id: autoPixelFormat;
        text: qsTr("Automatically convert unsupported pixel formats");
        checked: false;
        width: parent.width;
        Component.onCompleted: contentItem.wrapMode = Text.WordWrap;
        tooltip: qsTr("When the selected encoder can't write the source pixel format, convert to the closest one it accepts instead of asking every time.");
        // Guarded: this menu and the export menu are separate lazily-created items, and the one
        // that holds the check may not exist yet while settings are being restored.
        onCheckedChanged: if (window.exportSettings) window.exportSettings.updatePixelFormatCheck();
    }

    CheckBox {
        id: playSounds;
        text: qsTr("Notification sounds");
        checked: true;
    }
    Item { width: 1; height: 10 * dpiScale; }
    LinkButton {
        text: qsTr("Reset all settings to default");
        textColor: "#f67575"
        anchors.horizontalCenter: parent.horizontalCenter;
        onClicked: {
            messageBox(Modal.Warning, qsTr("Are you sure you want to clear all settings and restore the defaults?"), [
                { text: qsTr("Yes"), clicked: () => {
                    // Preserve lens profile favorites
                    const lenses = settings.value("lensProfileFavorites", "");

                    settings.clear();

                    if (lenses) settings.setValue("lensProfileFavorites", lenses);

                    messageBox(Modal.Info, qsTr("Settings cleared, please restart Gyroflow for the changes to take effect."), [
                        { text: qsTr("Exit"), accent: true, clicked: Qt.quit},
                        { text: qsTr("Ok") },
                    ]);
                }},
                { text: qsTr("No"), accent: true },
            ]);
        }
    }
    LinkButton {
        visible: Qt.platform.os != "android" && Qt.platform.os != "ios";
        text: qsTr("Open data folder");
        anchors.horizontalCenter: parent.horizontalCenter;
        onClicked: filesystem.open_file_externally(controller.data_folder());
    }
}
