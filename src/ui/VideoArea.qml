// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2021-2022 Adrian <adrian.eddy at gmail>

import QtQuick
import QtQuick.Controls as QQC
import MDKVideo
import Gyroflow

import "components/"
import "menu/" as Menu
import "Util.js" as Util
import "Zoom.js" as Zoom

Item {
    id: root;
    width: parent.width;
    height: parent.height;
    anchors.horizontalCenter: parent.horizontalCenter;

    property alias vid: vid;
    property alias timeline: timeline;
    property alias durationMs: timeline.durationMs;
    property alias videoLoader: videoLoader;
    property alias stabEnabledBtn: stabEnabledBtn;
    property alias fovOverviewBtn: fovOverviewBtn;
    property alias scopesBtn: scopesBtn;

    // Preview zoom (Zoom.js). 1 = fit. The video item itself grows inside the clipped vidParent.
    // Its texture is always the "Preview resolution" size, so up to 100% at "Full" this shows
    // real source detail, and zooming costs no extra rendering.
    property real zoom: 1;
    property real zoomU: 0.5; // image point (0..1) at the centre of the view
    property real zoomV: 0.5;
    readonly property real zoomDpr: Screen.devicePixelRatio;
    readonly property real zoomMax: Zoom.maxZoom(vidParent.width, zoomDpr, vidParent.orgW);
    // Clamped at use, not on write: a larger window lowers the cap under an existing zoom.
    readonly property real zoomEff: Zoom.clampZoom(zoom, zoomMax);
    readonly property int zoomPercent: Math.round(Zoom.pixelRatio(zoomEff, vidParent.width, zoomDpr, vidParent.orgW) * 100);

    // (cx, cy): the view point to zoom around, in vidParent's coordinates.
    function setZoom(z: real, cx: real, cy: real): void {
        let z1 = Zoom.clampZoom(z, zoomMax);
        if (z1 < 1.02) z1 = 1; // snap, so scrolling back out lands exactly on fit
        zoomU = Zoom.zoomAround(vidParent.width,  zoomEff, zoomU, z1, cx);
        zoomV = Zoom.zoomAround(vidParent.height, zoomEff, zoomV, z1, cy);
        zoom = z1;
    }
    function zoomStep(direction: int): void {
        setZoom(Zoom.nextStop(zoomEff, direction, vidParent.width, zoomDpr, vidParent.orgW, zoomMax), vidParent.width / 2, vidParent.height / 2);
    }
    function zoom100(): void {
        setZoom(Zoom.zoomForPixelRatio(1, vidParent.width, zoomDpr, vidParent.orgW), vidParent.width / 2, vidParent.height / 2);
    }
    function zoomFit(): void { zoom = 1; zoomU = 0.5; zoomV = 0.5; }
    property alias queue: queue.item;
    property alias statistics: statistics;
    property alias infoMessages: infoMessages;
    property alias gridGuide: gridGuide;
    property alias secondPreview: secondPreview;
    property alias opticalPointEditor: opticalPointEditor;

    property int outWidth: window? window.exportSettings.outWidth : 0;
    property int outHeight: window? window.exportSettings.outHeight : 0;

    property alias dropRect: dropRect;
    property bool isCalibrator: false;

    // Armed state of the white balance eyedropper in Color adjustments. Lives here rather than in
    // the panel because the click that matters lands on the video, and both sides need to see it:
    // the panel to light its button, this file to put a crosshair on the picture.
    property bool wbPickArmed: false;

    property var pendingGyroflowData: null;
    property int pendingQueueJobId: 0;
    property url loadedFileUrl;

    property int fullScreen: 0;
    property string detectedCamera: "";
    property real additionalTopMargin: 0;
    // While the motion data loads, the video plays unstabilized instead of being covered until it's loaded
    readonly property bool showStabilized: stabEnabledBtn.checked && !controller.loading_gyro_in_progress;
    onShowStabilizedChanged: { controller.stab_enabled = showStabilized; vid.forceRedraw(); vid.fovChanged(); }
    Component.onCompleted: controller.stab_enabled = showStabilized; // The core starts with it enabled
    property var mergedFiles: [];

    property Menu.VideoInformation vidInfo: null;

    function loadGyroflowData(obj: var, queueJobId: var): void {
        root.pendingGyroflowData = null;
        root.pendingQueueJobId = 0;

        if (controller.loading_gyro_in_progress) {
            root.pendingGyroflowData = obj;
            root.pendingQueueJobId = +queueJobId;
            controller.cancel_current_operation();
            // we'll get called again from telemetry_loaded
            return;
        }

        let urls = null;
        let project_version = +obj.version;

        if (obj.toString() != '[object Object]') { // obj is url
            urls = controller.get_urls_from_gyroflow_file(obj);
            project_version = controller.get_version_from_gyroflow_file(obj);
        } else if (obj.project_file) {
            urls = controller.get_urls_from_gyroflow_file(obj.project_file);
            project_version = controller.get_version_from_gyroflow_file(obj.project_file);
        } else {
            urls = [
                obj.videofile,
                obj.gyro_source?.filepath || "",
                obj.gyro_source?.all_metadata? "1" : ""
            ];
        }
        if ((!urls || !urls[0]) && !vidInfo.filename) {
            messageBox(Modal.Error, qsTr("Preset can be applied only after loading a video."), [ { text: qsTr("Ok") } ]);
            return;
        }

        const isCorrectVideoLoaded = urls[0] && vidInfo.filename == filesystem.get_filename(urls[0]);
        const isCorrectGyroLoaded  = urls[1] && window.motionData.filename == filesystem.get_filename(urls[1]);
        console.log("Video path:", urls[0], "(" + (isCorrectVideoLoaded? "loaded" : "not loaded") + ")", "Gyro path:", urls[1], "(" + (isCorrectGyroLoaded? "loaded" : "not loaded") + ")");

        if (urls[0] && !isCorrectVideoLoaded) {
            root.pendingGyroflowData = obj;
            root.pendingQueueJobId = +queueJobId;
            console.log("Loading video file", urls[0]);
            loadFile(urls[0], false, +queueJobId);
            if (controller.image_sequence_fps > 0) {
                vid.setFrameRate(controller.image_sequence_fps);
            }
            return;
        }
        if (urls[1] && !isCorrectGyroLoaded && filesystem.exists(urls[1])) {
            root.pendingGyroflowData = obj;
            root.pendingQueueJobId = +queueJobId;
            console.log("Loading gyro file", urls[1]);
            window.motionData.lastSelectedFile = urls[1];
            // Loaded with all of its metadata if the project was (its frame timing stays, eg. a CRM on its proxy)
            window.motionData.setAllMetadata(urls[2] == "1");
            controller.load_telemetry(urls[1], urls[0] == urls[1] || window.motionData.allMetadata, window.videoArea.vid, -1, project_version);
            return;
        }

        controller.set_prevent_recompute(true);
        if (obj.toString() != '[object Object]') {
            // obj is url
            controller.import_gyroflow_file(obj);
        } else if (obj.project_file) {
            controller.import_gyroflow_file(obj.project_file);
        } else {
            controller.import_gyroflow_data(JSON.stringify(obj));
        }
        render_queue.editing_job_id = +queueJobId;
    }
    Connections {
        target: controller;
        function onGyroflow_file_loaded(obj: var): void {
            if (obj) {
                let duration_ms = videoArea.vid.duration;
                const info = obj.video_info || { };
                if (info && Object.keys(info).length > 0) {
                    if (info.hasOwnProperty("vfr_fps") && Math.round(+info.vfr_fps * 1000) != Math.round(+info.fps * 1000)) {
                        vidInfo.updateEntryWithTrigger("Frame rate", +info.vfr_fps);
                    }
                    if (info.hasOwnProperty("rotation")) {
                        vidInfo.updateEntryWithTrigger("Rotation", +info.rotation);
                    }
                    if (info.hasOwnProperty("duration_ms")) {
                        duration_ms = info.duration_ms;
                    }
                }

                for (const ts in obj.offsets) {
                    controller.set_offset(ts, obj.offsets[ts]);
                }
                if (obj.hasOwnProperty("trim_start")) {
                    timeline.setTrimRanges([[obj.trim_start, obj.trim_end]]);
                }
                if (obj.hasOwnProperty("trim_ranges_ms")) {
                    timeline.setTrimRanges(obj.trim_ranges_ms.map(x => [x[0] / duration_ms, (x[1] < 0? duration_ms + x[1] : x[1]) / duration_ms]));
                } else if (obj.hasOwnProperty("trim_ranges")) {
                    timeline.setTrimRanges(obj.trim_ranges);
                }
                window.motionData.loadGyroflow(obj);
                window.stab.loadGyroflow(obj);
                window.advanced.loadGyroflow(obj);
                window.sync.loadGyroflow(obj);
                window.lensProfile.loadGyroflow(obj);
                window.colorAdjustments.loadGyroflow(obj);
                window.denoise.loadGyroflow(obj);
                window.film.loadGyroflow(obj);
                Qt.callLater(window.exportSettings.loadGyroflow, obj);

                if (obj.hasOwnProperty("image_sequence_start") && +obj.image_sequence_start > 0) {
                    controller.image_sequence_start = +obj.image_sequence_start;
                }
                if (obj.hasOwnProperty("image_sequence_fps") && +obj.image_sequence_fps > 0.0) {
                    vid.setFrameRate(+obj.image_sequence_fps);
                    controller.image_sequence_fps = +obj.image_sequence_fps;
                }
                if (obj.hasOwnProperty("playback_speed")) {
                    let i = 0;
                    const speed = +obj.playback_speed;
                    for (const x of playbackRateCb.model) {
                        const rate = +x.replace("x", "");
                        if (Math.abs(rate - speed) < 0.01) {
                            playbackRateCb.currentIndex = i;
                            break;
                        }
                        ++i;
                    }
                }
                if (obj.hasOwnProperty("muted")) {
                    videoArea.vid.muted = !!obj.muted;
                }
            }
            controller.set_prevent_recompute(false);
            Qt.callLater(controller.recompute_gyro);
            Qt.callLater(controller.recompute_threaded);
            Qt.callLater(timeline.updateDurations);
        }
        function onExternal_sdk_progress(percent: real, sdk_name: string, error_string: string, url: string): void {
            if (externalSdkModal !== null && externalSdkModal.loader !== null) {
                externalSdkModal.loader.visible = percent < 1;
                externalSdkModal.loader.active = percent < 1;
                externalSdkModal.loader.progress = percent;
                externalSdkModal.loader.text = qsTr("Downloading %1 (%2)").arg(sdk_name);
                if (percent >= 1) {
                    externalSdkModal.close();
                    externalSdkModal = null;
                    window.isDialogOpened = false;
                    if (!error_string) {
                        if (url == "ffmpeg_gpl") {
                            messageBox(Modal.Success, qsTr("Component was installed successfully.\nYou need to restart Gyroflow for changes to take effect.\nYour render queue and current file is saved automatically."), [ { text: qsTr("Ok") } ]);
                        } else {
                            loadFile(url, false);
                        }
                    } else {
                        if (Qt.platform.os == "osx") {
                            error_string += "\n" + qsTr("This is often caused by read-only file system.\nMake sure you copied the Gyroflow app to your Applications folder, instead of running from the .dmg directly.");
                        }
                        if (Qt.platform.os == "windows") {
                            error_string += "\n" + qsTr("This is often caused by read-only file system.\nIf you have Gyroflow in %1, then you'll need to run Gyroflow as Administrator in order to extract the SDK to the Gyroflow folder.").arg("C:\\Program Files\\");
                        }
                        messageBox(Modal.Error, error_string, [ { text: qsTr("Ok") } ]);
                    }
                }
            }
        }

        function onMp4_merge_progress(percent: real, error_string: string, url: url): void {
            if (externalSdkModal !== null && externalSdkModal.loader !== null) {
                externalSdkModal.loader.visible = percent < 1;
                externalSdkModal.loader.active = percent < 1;
                externalSdkModal.loader.progress = percent;
                externalSdkModal.loader.text = qsTr("Merging files to %1 (%2)").arg("<b>" + filesystem.display_url(url) + "</b>");
                if (percent >= 1) {
                    externalSdkModal.close();
                    externalSdkModal = null;
                    window.isDialogOpened = false;
                    if (!error_string) {
                        loadFile(url, true);
                    } else {
                        root.mergedFiles = []; // Not merged: nothing to offer to delete when the next file is loaded
                        messageBox(Modal.Error, error_string, [ { text: qsTr("Ok") } ]);
                    }
                }
            }
        }
        function onTelemetry_loaded(is_main_video: bool, filename: string, camera: string, additional_data: var): void {
            console.log("Telemetry additional data:", JSON.stringify(additional_data));
            if (is_main_video) {
                root.detectedCamera = camera;
                vidInfo.updateEntry("Detected camera", camera || "---");

                let lens = "";
                if (additional_data.camera_identifier) {
                    const camera_id = additional_data.camera_identifier;
                    if (camera_id) {
                        if (camera_id.lens_model) { lens += camera_id.lens_model; }
                        if (camera_id.lens_info)  { lens += (lens? " " : "") + camera_id.lens_info; }
                    }
                }
                vidInfo.updateEntry("Detected lens", lens || "---");
                vidInfo.updateEntry("Contains gyro", additional_data.contains_motion? "Yes" : "No");
                // If source was detected, but gyro data is empty
                if (camera) {
                    if (!additional_data.contains_motion && !additional_data.contains_quats) {
                        messageBox(Modal.Warning, qsTr("File format was detected, but no motion data was found.\nThe camera probably doesn't record motion data in this particular shooting mode."), [ { "text": qsTr("Ok") } ]);
                    }
                    if (additional_data.unsupported_lens) {
                        messageBox(Modal.Warning, qsTr("This video cannot be stabilized, because this lens doesn't support OSS metadata.\nDisable lens stabilization (Optical SteadyShot) in order to use Gyroflow."), [ { "text": qsTr("Ok") } ]);
                    }
                    if (additional_data.contains_raw_gyro && !additional_data.contains_quats) timeline.setDisplayMode(0); // Switch to gyro view
                    if (!additional_data.contains_raw_gyro && additional_data.contains_quats) timeline.setDisplayMode(3); // Switch to quaternions view
                }

                if (additional_data.hasOwnProperty("cam_posture") && additional_data.camera_type == "Insta360 GO 3S") {
                    vidInfo.updateEntryWithTrigger("Rotation", 360 - (+additional_data.cam_posture.replace("CameraRotate", "") + 90));
                } else if (additional_data.hasOwnProperty("cam_posture") && Math.abs(+additional_data.cam_posture.replace("CameraRotate", "")) > 0) {
                    vidInfo.updateEntryWithTrigger("Rotation", +additional_data.cam_posture.replace("CameraRotate", ""));
                }
                if (additional_data.hasOwnProperty("realtime_fps") && +additional_data.realtime_fps > 0) {
                    vidInfo.updateEntryWithTrigger("Frame rate", +additional_data.realtime_fps);
                }
                if (additional_data.hasOwnProperty("recording_settings") && Object.keys(additional_data.recording_settings).length > 0) {
                    vidInfo.cleanupModel();
                    let model = vidInfo.infoList.model;
                    model[""] = " ";
                    for (const x in additional_data.recording_settings) {
                        model[x] = additional_data.recording_settings[x];
                    }
                    vidInfo.infoList.model = model;
                    vidInfo.infoList.modelChanged();
                }

                // Source exposure/white-balance readout, and the temperature anchor (spec R9).
                // Task 14's controller has already seeded `source_kelvin` in the core from the
                // same metadata; this only brings the UI into step. Absent fields arrive as
                // `undefined`, never `null` - `serde_json_to_qt_object` drops nulls.
                const se = controller.get_source_exposure();
                window.colorAdjustments.sourceExposure = se;
                window.colorAdjustments.setTemperatureAnchor(
                    typeof se.white_balance_k !== "undefined"? +se.white_balance_k : 5500);
            }
            if (camera && additional_data.lens_stabilizer_on) {
                // Also for a file loaded as the motion data only (a CRM with its proxy)
                messageBox(Modal.Warning, qsTr("This video was recorded with the lens image stabilizer (IS) on, which Gyroflow doesn't support for Canon cameras.\nTurn the image stabilizer off in order to use Gyroflow."), [ { "text": qsTr("Ok") } ]);
            }
            if (+additional_data.sample_rate > 0.0 && Math.round(+additional_data.sample_rate) < 50) {
                messageBox(Modal.Warning, qsTr("Motion data sampling rate is too low (%1 Hz).\n50 Hz is an absolute minimum and we recommend at least 200 Hz.").arg(additional_data.sample_rate.toFixed(0)), [ { "text": qsTr("Ok") } ]);
            }
            if (root.pendingGyroflowData) {
                Qt.callLater(loadGyroflowData, root.pendingGyroflowData, root.pendingQueueJobId);
            } else {
                Qt.callLater(controller.recompute_threaded);
                if (is_main_video) {
                    controller.load_default_preset();
                }
            }
            if (is_main_video && window.pendingLoadPreset) {
                Qt.callLater(loadGyroflowData, JSON.parse(window.pendingLoadPreset), 0);
                window.pendingLoadPreset = "";
            }
        }
        function onChart_data_changed(): void {
            timeline.triggerUpdateChart("");
        }
        function onZooming_data_changed(): void {
            timeline.triggerUpdateChart("8");
        }
        function updateKeyframesView(): void {
            controller.update_keyframes_view(timeline.getKeyframesView());
            controller.update_keyframe_values(vid.timestamp);
        }
        function onKeyframes_changed(): void {
            Qt.callLater(updateKeyframesView);
        }
        function onCompute_progress(id: real, progress: real): void {
            videoLoader.active = progress < 1;
            videoLoader.cancelable = false;
        }
        function onSync_progress(progress: real, ready: int, total: int): void {
            videoLoader.active = progress < 1;
            videoLoader.currentFrame = ready;
            videoLoader.totalFrames = total;
            videoLoader.additional = "";
            videoLoader.text = videoLoader.active? qsTr("Analyzing %1...") : "";
            videoLoader.progress = videoLoader.active? progress : -1;
            videoLoader.cancelable = true;
        }
        function onLoading_gyro_progress(progress: real): void {
            gyroLoading.progress = progress < 1? progress : 0;
        }
    }
    property Modal externalSdkModal: null;

    function loadFile(url: url, skip_detection: bool, queueJobId: int): void {
        let filename = filesystem.get_filename(url);
        let folder = filesystem.get_folder(url);

        if (filename.endsWith(".gyroflow")) {
            return loadGyroflowData(url, queueJobId);
        }
        if (filename.endsWith(".RDC")) {
            // Assumes regular filesystem
            let parts = url.toString().split("/");
            parts.push(filename.replace(".RDC", "_001.R3D"));
            url = parts.join("/");
            filename = filesystem.get_filename(url);
            folder = filesystem.get_folder(url);
        }

        if (isMobile || filename.toLowerCase().endsWith(".r3d") || filename.toLowerCase().endsWith(".nev") || filename.toLowerCase().endsWith(".braw") || filename.toLowerCase().endsWith(".crm")) {
            // Preview resolution to 1080p
            if (isCalibrator && calibrator_window.lensCalib) {
                if (calibrator_window.lensCalib.previewResolution == 0) {
                    calibrator_window.lensCalib.previewResolution = 2;
                }
            } else {
                if (settings.value("previewResolution", -1) == -1 && window.advanced.previewResolution == 0) {
                    window.advanced.previewResolution = 2;
                }
            }
        }

        stabEnabledBtn.checked = false;

        if (controller.check_external_sdk(filename)) {
            const dlg = messageBox(Modal.Info, qsTr("This format requires an external SDK. Do you want to download it now?"), [
                { text: qsTr("Yes"), accent: true, clicked: function() {
                    dlg.btnsRow.children[0].enabled = false;
                    controller.install_external_sdk(url.toString());
                    return false;
                } },
                { text: qsTr("Cancel"), clicked: function() {
                    externalSdkModal = null;
                } },
            ]);
            externalSdkModal = dlg;
            dlg.addLoader();
            return;
        }

        window.motionData.lastSelectedFile = "";
        if (!(/\.(png|jpg|exr|dng)$/i.test(filename) && filename.includes("%0"))) {
            root.loadedFileUrl = url;
        }

        if (isStorePackage && Qt.platform.os == "osx" && filename.toLowerCase().endsWith(".r3d") && folder.toString().length < 3) {
            messageBox(Modal.Info, qsTr("In order to load all R3D parts, you need to select the entire .RDC folder."), [
                { text: qsTr("OK"), accent: true, clicked: function() {
                    opf.selectFolder("", function(_) {
                        root.loadFile(root.loadedFileUrl);
                    });
                } },
            ], null, undefined, "open-rdc-folder");
            return;
        }

        if (!skip_detection) {
            let newUrl;
            if (newUrl = detectImageSequence(folder, filename)) {
                const dlg = messageBox(Modal.Info, qsTr("Image sequence has been detected.\nPlease provide frame rate: "), [
                    { text: qsTr("Ok"), accent: true, clicked: function() {
                        const fps = dlg.mainColumn.children[1].value;
                        settings.setValue("imageSequenceFps", fps);
                        controller.image_sequence_fps = fps;
                        loadFile(newUrl, true);
                        vid.setFrameRate(fps);
                    } },
                    { text: qsTr("Cancel") },
                ]);
                // The frame rate the images record (CinemaDNG), else the last one entered
                const recordedFps = controller.image_sequence_frame_rate(url);
                const nf = Qt.createComponent("components/NumberField.qml").createObject(dlg.mainColumn, { precision: 3, unit: "fps", value: recordedFps > 0 ? recordedFps : +settings.value("imageSequenceFps", "30") });
                nf.anchors.horizontalCenter = dlg.mainColumn.horizontalCenter;
                return;
            }
            let sequenceList;
            if (sequenceList = detectVideoSequence(folder, filename)) {
                const list = "<b>" + sequenceList.join(", ") + "</b>";
                const dlg = messageBox(Modal.Info, qsTr("Split recording has been detected, do you want to automatically join the files (%1) to create one full clip?").arg(list), [
                    { text: qsTr("Yes"), accent: true, clicked: function() {
                        dlg.btnsRow.children[0].enabled = false;
                        getOutputFile(folder, sequenceList[0], "_joined", "", true, function(outFolder, outFilename, outFullFileUrl) {
                            mergeFiles(sequenceList.map(x => filesystem.get_file_url(folder, x, false).toString()), outFolder, outFilename);
                        });
                        return false;
                    } },
                    { text: qsTr("No"), clicked: function() {
                        externalSdkModal = null;
                        loadFile(url, true);
                    } },
                ])
                externalSdkModal = dlg;
                dlg.addLoader();
                return;
            }
        }
        vidInfo.hasAccessToInputDirectory = folder.toString().length > 3;

        window.stab.fovSlider.value = 1.0;
        vid.loaded = false;
        videoLoader.active = true;
        vidInfo.loader = true;
        //vid.url = url;
        vid.errorShown = false;
        render_queue.editing_job_id = 0;
        controller.load_video(url, vid);
        if (!isCalibrator) {
            const suffix = window.advanced.defaultSuffix.text;
            window.outputFile.setFilename(filesystem.filename_with_suffix(filename, suffix).replace(/%0[0-9]+d/, ""));

            // The "Export to" rule is answered in Rust, by the same call the render queue makes
            // for each of its files, so a clip opened here and the same clip queued cannot end
            // up in different places.
            window.outputFile.setFolder(render_queue.get_default_output_folder(url));
            window.exportSettings.updateCodecParams();
        }
        if (!root.pendingGyroflowData) {
            const gfFilename = filesystem.filename_with_extension(filename, "gyroflow");
            if (filesystem.exists_in_folder(folder, gfFilename)) {
                messageBox(Modal.Question, qsTr("There's a %1 file associated with this video, do you want to load it?").arg("<b>" + gfFilename + "</b>"), [
                    { text: qsTr("Yes"), clicked: function() {
                        Qt.callLater(() => loadFile(filesystem.get_file_url(folder, gfFilename, false), true));
                    } },
                    { text: qsTr("No"), accent: true },
                ]);
            }
        }

        dropText.loadingFile = filename;
        vidInfo.cleanupModel();
        vidInfo.updateEntry("File name", filename);
        vidInfo.updateEntry("Detected camera", "---");
        vidInfo.updateEntry("Detected lens", "---");
        vidInfo.updateEntry("Contains gyro", "---");
        timeline.editingSyncPoint = false;
    }
    function loadMultipleFiles(urls: list<url>, skip_detection: bool): void {
        if (urls.length == 1) {
            root.loadFile(urls[0], skip_detection);
        } else if (urls.length > 1) {
            const urlsCopy = [...urls];
            if (urlsCopy[0].toString().toLowerCase().endsWith(".r3d")) {
                return root.loadFile(urlsCopy[0], true);
            }
            const dlg = messageBox(Modal.Question, qsTr("You have opened multiple files. What do you want to do?"), [
                { text: qsTr("Add to render queue"), clicked: () => {
                    queue.item.dt.loadFiles(urlsCopy);
                    queue.item.shown = true;
                } },
                { text: qsTr("Merge them into one video"), clicked: () => {
                    dlg.btnsRow.children[0].enabled = false;
                    dlg.btnsRow.children[1].enabled = false;
                    dlg.btnsRow.children[2].enabled = false;
                    const filename = filesystem.get_filename(urlsCopy[0]);
                    const folder = filesystem.get_folder(urlsCopy[0]);
                    getOutputFile(folder, filename, "_joined", "", true, function(outFolder, outFilename, outFullFileUrl) {
                        mergeFiles(urlsCopy.map(x => x.toString()), outFolder, outFilename);
                    });
                    return false;
                } },
                { text: qsTr("Open the first file"), clicked: () => {
                    root.loadFile(urlsCopy[0], skip_detection);
                } },
                { text: qsTr("Cancel") },
            ]);
            externalSdkModal = dlg;
            dlg.addLoader();
        }
    }

    // Merges `urls` into `outFilename` in `outFolder`. One of them is never the output (the controller refuses that too), and
    // replacing another file is asked first, as when rendering. Asks for the output path again when that's not done
    function mergeFiles(urls: var, outFolder: url, outFilename: string): void {
        const askAgain = () => askForOutputLocation(outFolder, outFilename, false, (folder, filename) => mergeFiles(urls, folder, filename));
        const outUrl = filesystem.get_file_url(outFolder, outFilename, false);
        if (urls.some(x => filesystem.is_same_file(outUrl, x))) {
            messageBox(Modal.Error, qsTr("The output file is one of the files being merged. Choose another name or folder for the output."), [
                { text: qsTr("Ok"), clicked: askAgain },
            ]);
            return;
        }
        // Uses this call's `urls` and `outFolder`. A JavaScript function: type annotations are allowed only on QML objects' functions
        const merge = (filename) => {
            root.mergedFiles = urls;
            controller.mp4_merge(urls, outFolder, filename);
        };
        if (filesystem.exists_in_folder(outFolder, outFilename)) {
            messageBox(Modal.Question, qsTr("Output file already exists, do you want to overwrite it?"), [
                { text: qsTr("Yes"), clicked: () => merge(outFilename) },
                { text: qsTr("Rename"), clicked: () => merge(window.renameOutput(outFilename, outFolder)) },
                { text: qsTr("No"), accent: true, clicked: askAgain },
            ]);
        } else {
            merge(outFilename);
        }
    }
    function askForOutputLocation(folder: url, filename: string, choice: bool, cb: var): void {
        const dlg = messageBox(Modal.Question, qsTr("Please enter the output path:"), [
            { text: qsTr("Ok"), accent: true, clicked: function() {
                if (choice) {
                    if (dlg.mainColumn.children[1].children[0].checked) { cb("", ""); }
                    if (dlg.mainColumn.children[1].children[1].checked) { const opf = dlg.mainColumn.children[1].children[3]; cb(opf.folderUrl, opf.filename, opf.fullFileUrl); }
                } else {
                    const opf = dlg.mainColumn.children[1];
                    if (!opf.folderUrl.toString() && !opf.fullFileUrl.toString()) {
                        opf.prompt();
                        return false;
                    }
                    cb(opf.folderUrl, opf.filename, opf.fullFileUrl);
                }
            } },
            { text: qsTr("Cancel") },
        ]);

        if (choice) {
            let col = Qt.createQmlObject(`import QtQuick; import "components/";
                Column {
                    width: parent.width;
                    RadioButton { checked: true; }
                    RadioButton { id: custom; }
                    Item { height: 10 * dpiScale; width: 1; }
                    OutputPathField { enabled: custom.checked; folderOnly: true; }
                }`, dlg.mainColumn, "dlgRadios");
            col.children[0].text = qsTr("Same as the original file");
            col.children[1].text = qsTr("Custom path");
            col.children[3].setFolder(folder);
        } else {
            const opf = Qt.createComponent("components/OutputPathField.qml").createObject(dlg.mainColumn, { });
            opf.setFolder(folder);
            opf.setFilename(filename);
        }
    }
    function getOutputFile(folder: url, filename: string, suffix: string, extension: string, ask: bool, cb: var): void {
        if (suffix) filename = filesystem.filename_with_suffix(filename, suffix);
        if (extension) filename = filesystem.filename_with_extension(filename, extension);
        if (ask) {
            askForOutputLocation(folder, filename, false, cb);
        } else {
            cb(folder, filename);
        }
    }

    function detectImageSequence(folder: url, filename: string): var {
        if (!filename.includes("%0")) {
            controller.image_sequence_start = 0;
            controller.image_sequence_fps = 0;
        }
        if (/\d+\.(png|jpg|exr|dng)$/i.test(filename)) {
            let firstNum = filename.match(/(\d+)\.(png|jpg|exr|dng)$/i);
            if (firstNum[1]) {
                const ext = firstNum[2];
                firstNum = firstNum[1];
                const firstNumNum = parseInt(firstNum, 10);
                for (let i = firstNumNum + 1; i < firstNumNum + 5; ++i) { // At least 5 frames
                    const newNum = i.toString().padStart(firstNum.length, '0');
                    const newName = filename.replace(firstNum + "." + ext, newNum + "." + ext);
                    if (!filesystem.exists_in_folder(folder, newName)) {
                        return false;
                    }
                }
                controller.image_sequence_start = firstNumNum;
                return filesystem.get_file_url(folder, filename.replace(`${firstNum}.${ext}`, `%0${firstNum.length}d.${ext}`), false);
            }
        }
        return false;
    }
    function detectVideoSequence(folder: url, filename: string): var {
        // url pattern, 1st file index, new path function
        const patterns = [
            // GoPro 1-5
            [/((?:GOPR|GP\d{2})(\d{4})\.MP4)$/i, 0, function(match, i) {
                return (i == 0 ? "GOPR" : "GP" + i.toString().padStart(2, '0')) + match.substring(4);
            }],
            // GoPro 6+
            [/(G[XH]\d{2}(\d{4})\.MP4)$/i, 1, function(match, i) {
                return match.substring(0, 2) + i.toString().padStart(2, '0') + match.substring(4);
            }],
            // DJI Action
            [/(DJI_\d+_(\d+)\.MP4)$/i, null, function(match, i) {
                return match.substring(0, 9) + i.toString().padStart(3, '0') + match.substring(12);
            }],
        ];
        for (const x of patterns) {
            let match = filename.match(x[0]);
            if (match && match[1]) {
                let list = [];
                const firstNum = (x[1] !== null ? x[1] : parseInt(match[2], 10));
                for (let i = firstNum; i < firstNum + 99; ++i) { // Max 99 parts
                    const newName = filename.replace(match[1], x[2](match[1], i));
                    if (filesystem.exists_in_folder(folder, newName)) {
                        list.push(newName);
                    } else {
                        break;
                    }
                }
                if (list.length > 1)
                    return list;
            }
        }
        return false;
    }
    OutputPathField { id: opf; visible: false; }

    Item {
        id: vidParentParent;
        width: parent.width;
        height: parent.height - (root.fullScreen || window.isMobileLayout? 0 : tlcol.height);

        Grid {
            readonly property bool vertical: vidParentParent.height - vidParent.height * 2 > vidParentParent.width - vidParent.width * 2;
            columns: secondPreview.visible? (vertical? 1 : 2) : 1;
            rows:    secondPreview.visible? (vertical? 2 : 1) : 1;
            anchors.centerIn: parent;
            spacing: 10 * dpiScale;
            Item {
                id: vidParent;
                readonly property real orgW: (root.showStabilized && root.outWidth > 0? root.outWidth : (vid.videoWidth * window.lensProfile.input_horizontal_stretch));
                readonly property real orgH: (root.showStabilized && root.outHeight > 0? root.outHeight : (vid.videoHeight * window.lensProfile.input_vertical_stretch));
                readonly property real ratio: orgW / Math.max(1, orgH);
                readonly property real w: vidParentParent.width  / parent.columns - (root.fullScreen? 0 : 20 * dpiScale);
                readonly property real h: vidParentParent.height / parent.rows    - (root.fullScreen? 0 : 20 * dpiScale);

                width:  (ratio * h) > w ? w : (ratio * h)
                height: (ratio * h) > w ? (w / ratio) : h
                opacity: da.containsDrag? 0.5 : 1.0;
                clip: root.zoomEff > 1;

                /*Image {
                    // Transparency grid
                    fillMode: Image.Tile;
                    anchors.fill: parent;
                    source: "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14'><rect fill='%23fff' x='0' y='0' width='7' height='7'/><rect fill='%23aaa' x='7' y='0' width='7' height='7'/><rect fill='%23aaa' x='0' y='7' width='7' height='7'/><rect fill='%23fff' x='7' y='7' width='7' height='7'/></svg>"
                }*/

                MDKVideo {
                    id: vid;
                    visible: opacity > 0;
                    opacity: loaded? 1 : 0;
                    Ease on opacity { }
                    x: Zoom.panOffset(vidParent.width,  root.zoomEff, root.zoomU);
                    y: Zoom.panOffset(vidParent.height, root.zoomEff, root.zoomV);
                    width:  vidParent.width  * root.zoomEff;
                    height: vidParent.height * root.zoomEff;
                    property bool loaded: false;

                    property bool stabEnabled: root.showStabilized;
                    transform: [
                        Scale {
                            readonly property real r: vidInfo.videoRotation * (Math.PI / 180);
                            readonly property real rotW: Math.abs(vidParent.width * Math.cos(r)) + Math.abs(vidParent.height * Math.sin(r));
                            readonly property real rotH: Math.abs(vidParent.width * Math.sin(r)) + Math.abs(vidParent.height * Math.cos(r));
                            origin.x: vid.width / 2; origin.y: vid.height / 2;
                            xScale: vid.stabEnabled? 1 : Math.min(vidParent.h / rotH, vidParent.w / rotW) * (fovOverviewBtn.checked? 0.5 : 1);
                            yScale: xScale;
                        },
                        Rotation {
                            origin.x: vid.width / 2; origin.y: vid.height / 2;
                            angle: vid.stabEnabled? 0 : -vidInfo.videoRotation;
                        }
                    ]

                    function fovChanged(): void {
                        const fov = controller.current_fov;
                        const focal_length = controller.current_focal_length;
                        const crop_factor = window.lensProfile?.cropFactor || 1.0;
                        // const ratio = controller.get_scaling_ratio(); // this shouldn't be called every frame because it locks the params mutex
                        currentFovText.text = qsTr("Zoom: %1").arg(fov > 0? (100 / fov).toFixed(2) + "%" : "---");

                        if (+focal_length > 0) {
                            const fl = +focal_length / fov;
                            currentFovText.text += "\n" + qsTr("Focal length: %1 mm").arg(fl.toFixed(2));
                            if (crop_factor && crop_factor != 1.0) {
                                currentFovText.text += " (" + qsTr("full frame equiv.: %1 mm").arg((fl * crop_factor).toFixed(2)) + ")";
                            }
                        }
                    }

                    function updateTurnSpeed(): void {
                        const turnSpeed = controller.get_turn_speed(vid.timestamp);
                        if (isNaN(turnSpeed)) {
                            turnSpeedValue.text = "---";
                        } else {
                            const xAngle = controller.get_x_angle(vid.timestamp);
                            turnSpeedValue.text = turnSpeed.toFixed(2) + "°/s (" + xAngle.toFixed(2) + "°)";
                        }
                    }

                    onCurrentFrameChanged: {
                        fovChanged();
                        controller.update_keyframe_values(timestamp);
                        window.motionData.orientationIndicator.updateOrientation(timeline.position * timeline.durationMs * 1000);
                        updateTurnSpeed();
                    }
                    onMetadataLoaded: (md) => {
                        Qt.callLater(fileLoaded, md);
                    }
                    function fileLoaded(md: var): void {
                        loaded = vid.videoWidth > 0;
                        videoLoader.active = false;
                        vidInfo.loader = false;
                        timeline.resetTrim();
                        timeline.resetZoom();
                        root.zoomFit();

                        controller.video_file_loaded(vid);
                        // Already now, not only once the gyro data is loaded: the video plays while it loads, and the playhead needs them
                        Qt.callLater(timeline.updateDurations);
                        window.motionData.filename = "";

                        if (root.pendingGyroflowData) {
                            Qt.callLater(root.loadGyroflowData, root.pendingGyroflowData, root.pendingQueueJobId);
                        } else {
                            controller.load_telemetry(root.loadedFileUrl, true, vid, -1, 0);
                        }
                        vidInfo.loadFromVideoMetadata(md, vid.videoWidth, vid.videoHeight);
                        window.sync.customSyncTimestamps = [];

                        if (root.mergedFiles.length > 1) {
                            if (loaded) {
                                const copy = [...root.mergedFiles];
                                messageBox(Modal.Question, qsTr("Files merged successfully, do you want to delete the original ones?"), [
                                    { text: qsTr("Yes"), clicked: function() {
                                        for (const x of copy) {
                                            filesystem.move_to_trash(x);
                                        }
                                        return true;
                                    } },
                                    { text: qsTr("No"), accent: true },
                                ], null, undefined, "delete-after-join");
                            }
                            root.mergedFiles = [];
                        }

                        window.lensProfile.selected_manually = false;

                        // for (var i in md) console.info(i, md[i]);
                    }
                    property bool errorShown: false;
                    onMetadataChanged: {
                        if (vid.videoWidth > 0) {
                            // Trigger seek to buffer the video frames
                            if (vid.duration == 0) {
                                vid.play();
                                Qt.callLater(function() {
                                    stabEnabledBtn.checked = true;
                                    vid.volume = volumeSlider.value / 100.0;
                                })
                            } else {
                                bufferTrigger.waited = 0;
                                bufferTrigger.start();
                            }
                        } else if (!errorShown) {
                            messageBox(Modal.Error, qsTr("Failed to load the selected file, it may be unsupported or invalid."), [ { "text": qsTr("Ok") } ]);
                            errorShown = true;
                            dropText.loadingFile = "";
                            root.pendingGyroflowData = null;
                            stabEnabledBtn.checked = true;
                        }
                    }
                    Timer {
                        id: bufferTrigger;
                        interval: 150;
                        property int waited: 0;
                        onTriggered: {
                            if (!vid.videoWidth) return; // The file failed to load, the next one starts it again in onMetadataChanged
                            // Only once the first frame is shown (which ends the loading screen), unless it doesn't show up on
                            // its own: seeking while it's decoded made the decoder start over at the seek target, which kept
                            // the loading screen up for seconds with videos decoded on the CPU (eg. 5.3K GoPro footage)
                            if (!vid.loaded && ++bufferTrigger.waited < 20) { bufferTrigger.start(); return; }
                            Qt.callLater(() => {
                                // Not `currentFrame++`: until the first frame is shown, it's still the previous video's frame
                                vid.currentFrame = 1;
                                Qt.callLater(() => vid.currentFrame = 0);
                                if (vid.videoWidth) {
                                    stabEnabledBtn.checked = true;
                                    vid.volume = volumeSlider.value / 100.0;
                                }
                            });
                        }
                    }

                    backgroundColor: "#111111";
                    Component.onCompleted: {
                        controller.init_player(this);
                    }
                    Rectangle {
                        border.color: styleVideoBorderColor;
                        border.width: 1 * dpiScale;
                        color: "transparent";
                        radius: 5 * dpiScale;
                        anchors.fill: parent;
                        anchors.margins: -border.width;
                    }
                    OpticalPointEditor {
                        id: opticalPointEditor;
                        anchors.fill: parent;
                        timestampUs: vid.timestamp * 1000;
                        stabilized: vid.stabEnabled;
                        onRedraw: vid.forceRedraw();
                    }
                }

                TapHandler {
                    onTapped: timeline.focus = true;
                    onDoubleTapped: root.fullScreen = root.fullScreen? 0 : 1;
                }
                GridGuide {
                    id: gridGuide;
                    anchors.fill: vid;
                    canShow: vid.loaded && !root.wbPickArmed;
                }

                // White balance picker. Armed from the eyedropper in Color adjustments; the next
                // click on the picture samples a patch and disarms.
                //
                // Sampling `vid` rather than the source frame is deliberate and is what makes the
                // whole feature cheap: `grabToImage` already exists here, and the solver in
                // core/color/picker.rs is built to work from the *graded* pixel - see its module
                // docs for why that is enough. The cost is that anything drawn over the video
                // (the grid guide, feature points) would be sampled too, so the guide is hidden
                // while armed rather than leaving a trap the user cannot see.
                MouseArea {
                    id: wbPicker;
                    anchors.fill: vid;
                    enabled: root.wbPickArmed && vid.loaded;
                    visible: enabled;
                    cursorShape: Qt.CrossCursor;
                    hoverEnabled: true;
                    // Right-click cancels. Escape would be the conventional key, but a QML
                    // Shortcut only ever fired here with `enabled` removed - and an
                    // always-registered Escape is one the whole app pays for, because shortcuts
                    // are matched before key events reach items and it would shadow the Escape
                    // that closes a combo box or a tooltip. Cancelling on a button this MouseArea
                    // already receives, and only while it is armed, steals nothing.
                    acceptedButtons: Qt.LeftButton | Qt.RightButton;

                    // Where the press landed, and where the cursor is now. Both in item pixels;
                    // they are only normalised at the moment of the pick, because the grab is
                    // taken at the window's device pixel ratio and is not this item's size.
                    property real startX: 0;
                    property real startY: 0;
                    property bool dragging: false;

                    // Below this a drag is a click. Fingers and mice both wobble a pixel or two
                    // on the way down, and a two-pixel rectangle is not what anyone meant.
                    readonly property real dragThreshold: 4 * dpiScale;

                    onPressed: mouse => {
                        if (mouse.button !== Qt.LeftButton) return;
                        startX = mouse.x; startY = mouse.y;
                        marquee.x = mouse.x; marquee.y = mouse.y;
                        marquee.width = 0; marquee.height = 0;
                        dragging = false;
                    }
                    onPositionChanged: mouse => {
                        if (!pressed) return;
                        if (!dragging && (Math.abs(mouse.x - startX) > dragThreshold || Math.abs(mouse.y - startY) > dragThreshold))
                            dragging = true;
                        marquee.x = Math.min(startX, mouse.x);
                        marquee.y = Math.min(startY, mouse.y);
                        marquee.width  = Math.abs(mouse.x - startX);
                        marquee.height = Math.abs(mouse.y - startY);
                    }
                    onReleased: mouse => {
                        if (mouse.button === Qt.RightButton) { root.wbPickArmed = false; dragging = false; return; }

                        // A click is sent as a rectangle with both corners at the same point;
                        // `average_patch` widens it to a small block, so there is one code path
                        // on the Rust side and clicking keeps behaving exactly as it did.
                        const x1 = dragging? mouse.x : startX;
                        const y1 = dragging? mouse.y : startY;
                        const nx0 = startX / Math.max(1, width), ny0 = startY / Math.max(1, height);
                        const nx1 = x1     / Math.max(1, width), ny1 = y1     / Math.max(1, height);

                        dragging = false;
                        root.wbPickArmed = false;
                        vid.grabToImage(function(result) {
                            const r = controller.pick_white_balance(result.image, nx0, ny0, nx1, ny1);
                            if (r.error) {
                                window.showNotification(Modal.Warning, r.error);
                            } else {
                                window.colorAdjustments.applyPickedWhiteBalance(r.kelvin, r.tint);
                            }
                        });
                    }

                    // Shows what is about to be sampled. Sits inside the MouseArea so it is torn
                    // down with it, and is `enabled: false` by inheritance - a plain Rectangle
                    // takes no input, so it cannot eat the release that ends the drag.
                    Rectangle {
                        id: marquee;
                        visible: wbPicker.dragging;
                        color: "#20ffffff";
                        border.color: "#ffffff";
                        border.width: 1 * dpiScale;
                    }
                }
                // Preview zoom: Ctrl+scroll zooms around the cursor, dragging pans while zoomed.
                // Ctrl is required so a plain scroll over the video stays free for everything else.
                WheelHandler {
                    target: null;
                    acceptedModifiers: Qt.ControlModifier;
                    enabled: vid.loaded;
                    onWheel: (event) => {
                        const steps = event.angleDelta.y / 120;
                        if (steps) root.setZoom(root.zoomEff * Math.pow(1.25, steps), point.position.x, point.position.y);
                    }
                }
                DragHandler {
                    target: null;
                    // Off while picking white balance: that drag draws the sampling rectangle.
                    enabled: vid.loaded && root.zoomEff > 1 && !root.wbPickArmed;
                    cursorShape: active? Qt.ClosedHandCursor : Qt.OpenHandCursor;
                    property point last: Qt.point(0, 0);
                    onActiveChanged: last = Qt.point(0, 0);
                    onTranslationChanged: {
                        const dx = translation.x - last.x, dy = translation.y - last.y;
                        last = translation;
                        root.zoomU = Zoom.dragTo(vidParent.width,  root.zoomEff, root.zoomU, dx);
                        root.zoomV = Zoom.dragTo(vidParent.height, root.zoomEff, root.zoomV, dy);
                    }
                }
                Rectangle {
                    visible: root.zoomEff > 1;
                    anchors.left: parent.left;
                    anchors.bottom: parent.bottom;
                    anchors.margins: 8 * dpiScale;
                    width: zoomBadgeText.width + 16 * dpiScale;
                    height: zoomBadgeText.height + 8 * dpiScale;
                    radius: 4 * dpiScale;
                    color: "#c0101010";
                    border.color: "#40ffffff";
                    border.width: 1;
                    BasicText {
                        id: zoomBadgeText;
                        anchors.centerIn: parent;
                        leftPadding: 0;
                        font.pixelSize: 11 * dpiScale;
                        text: qsTr("%1% · Fit").arg(root.zoomPercent);
                    }
                    MouseArea {
                        anchors.fill: parent;
                        cursorShape: Qt.PointingHandCursor;
                        preventStealing: true;
                        onClicked: root.zoomFit();
                    }
                }

                // Video scopes of the *displayed* frame - the graded, stabilized preview, which
                // is what the export will look like. See ScopeView.rs for where the pixels come
                // from. A sibling of `vid`, not a child, so `vid.grabToImage` (the white balance
                // picker) never samples the scope.
                Rectangle {
                    id: scopesOverlay;
                    visible: scopesBtn.checked && vid.loaded;
                    // 0 waveform, 1 histogram, 2 vectorscope - ScopeKind's values.
                    property int kind: 0;
                    property bool restored: false;
                    onKindChanged: if (restored) settings.setValue("scopeKind", kind);
                    Component.onCompleted: { kind = +settings.value("scopeKind", 0); restored = true; }

                    anchors.top: parent.top;
                    anchors.right: parent.right;
                    anchors.margins: 8 * dpiScale;
                    width: Math.min(parent.width * 0.45, 360 * dpiScale);
                    height: scopeHeader.height + scopeView.height + 8 * dpiScale;
                    color: "#c0101010";
                    border.color: "#40ffffff";
                    border.width: 1;
                    radius: 5 * dpiScale;

                    // Clicks on the scope are not clicks on the video: without this, a quick
                    // double click on the tabs would toggle full screen.
                    // `preventStealing`: the zoom pan's DragHandler would otherwise take over a
                    // drag that starts here.
                    MouseArea { anchors.fill: parent; acceptedButtons: Qt.AllButtons; preventStealing: true; }

                    Row {
                        id: scopeHeader;
                        x: 4 * dpiScale;
                        y: 2 * dpiScale;
                        component ScopeTab: LinkButton {
                            property int scopeKind: 0;
                            readonly property bool selected: scopesOverlay.kind == scopeKind;
                            font.pixelSize: 11 * dpiScale;
                            font.underline: false;
                            leftPadding: 5 * dpiScale; rightPadding: 5 * dpiScale;
                            topPadding: 3 * dpiScale; bottomPadding: 3 * dpiScale;
                            textColor: selected? styleAccentColor : styleTextColor;
                            opacity: selected? 1 : 0.6;
                            onClicked: scopesOverlay.kind = scopeKind;
                        }
                        ScopeTab { scopeKind: 0; text: qsTr("Waveform"); }
                        ScopeTab { scopeKind: 1; text: qsTr("Histogram"); }
                        ScopeTab { scopeKind: 2; text: qsTr("Vectorscope"); }
                        LinkButton {
                            id: scopeLive;
                            property bool on: false;
                            Component.onCompleted: on = settings.value("scopeLive", false);
                            onClicked: { on = !on; settings.setValue("scopeLive", on); }
                            text: qsTr("Live");
                            font.pixelSize: 11 * dpiScale;
                            font.underline: false;
                            leftPadding: 8 * dpiScale; rightPadding: 5 * dpiScale;
                            topPadding: 3 * dpiScale; bottomPadding: 3 * dpiScale;
                            textColor: on? styleAccentColor : styleTextColor;
                            opacity: on? 1 : 0.6;
                            tooltip: qsTr("Also update during playback, about 10 times a second. Off: only while paused.");
                        }
                    }
                    ScopeView {
                        id: scopeView;
                        x: 4 * dpiScale;
                        anchors.top: scopeHeader.bottom;
                        width: parent.width - 8 * dpiScale;
                        height: scopesOverlay.kind == 2? Math.min(width, 220 * dpiScale) : width * 0.5;
                        kind: scopesOverlay.kind;
                        live: scopeLive.on;
                        active: scopesOverlay.visible;
                        Component.onCompleted: attach(vid);
                    }
                }
            }
            Item {
                id: secondPreview;
                property bool show: false;
                onShowChanged: settings.setValue("stabOverviewSplit", show);
                Component.onCompleted: show = settings.value("stabOverviewSplit", false);
                visible: show && fovOverviewBtn.checked;
                readonly property real ratio: 1 + 1 / window.stab.fovSlider.value;
                onRatioChanged: {
                    if (visible) {
                        vid.forceRedraw();
                        vidParent.widthChanged();
                    }
                }
                width: vidParent.width;
                height: vidParent.height;
                ShaderEffectSource {
                    id: secondPreviewSource;
                    live: secondPreview.visible;
                    width: parent.width; height: parent.height;
                    sourceItem: vidParent;
                    sourceRect: Qt.rect((vidParent.width - (vidParent.width / secondPreview.ratio)) / 2, (vidParent.height - (vidParent.height / secondPreview.ratio)) / 2, vidParent.width / secondPreview.ratio, vidParent.height / secondPreview.ratio);
                }
                TapHandler {
                    onTapped: timeline.focus = true;
                    onDoubleTapped: root.fullScreen = root.fullScreen? 0 : 1;
                }
            }
        }

        Rectangle {
            id: dropRect;
            border.width: vid.loaded? 0 : (3 * dpiScale);
            border.color: style === "light"? Qt.darker(styleBackground, 1.3) : Qt.lighter(styleBackground, 2);
            anchors.fill: parent;
            anchors.margins: vid.loaded? 0 : (20 * dpiScale);
            anchors.topMargin: vid.loaded? 0 : (50 * dpiScale);
            anchors.bottomMargin: vid.loaded? 0 : (50 * dpiScale);
            color: styleBackground;
            radius: 5 * dpiScale;
            opacity: da.containsDrag? (vid.loaded? 0.8 : 0.3) : vid.loaded? 0 : 1.0;
            Ease on opacity { duration: 300; }
            visible: opacity > 0;
            onVisibleChanged: if (!visible) dropText.loadingFile = "";

            BasicText {
                id: dropText;
                property string loadingFile: "";
                text: loadingFile? qsTr("Loading %1...").arg(loadingFile) : (Qt.platform.os == "ios" || Qt.platform.os == "android"? qsTr("Click here to open a video file") : qsTr("Drop video file here"));
                font.pixelSize: (window.isMobileLayout? 23 : 30) * dpiScale;
                anchors.centerIn: parent;
                leftPadding: 0;
                scale: dropText.contentWidth > (parent.width - 50 * dpiScale)? (parent.width - 50 * dpiScale) / dropText.contentWidth : 1.0;
            }
            ItemLoader {
                anchors.fill: dropText;
                anchors.margins: -30 * dpiScale;
                visible: !dropText.loadingFile && !vid.loaded;
                scale: dropText.scale;
                sourceComponent: Component { DropTargetRect { } }
            }
            ItemLoader {
                anchors.fill: parent;
                anchors.margins: 5 * dpiScale;
                visible: !dropText.loadingFile && vid.loaded;
                sourceComponent: Component { DropTargetRect { } }
            }
            MouseArea {
                visible: !vid.loaded;
                anchors.fill: parent;
                cursorShape: Qt.PointingHandCursor;
                onClicked: vidInfo.selectFileRequest();
            }
        }
        DropArea {
            id: da;
            anchors.fill: dropRect;
            enabled: queue.item && !queue.item.shown && !queue.item.isDragging;

            onEntered: (drag) => {
                if (!drag.urls.length) return;
                const ext = drag.urls[0].toString().split(".").pop().toLowerCase();
                drag.accepted = fileDialog.extensions.indexOf(ext) > -1 || ext == "rdc";
            }
            onDropped: (drop) => {
                if (isCalibrator) {
                    calibrator_window.loadFiles(drop.urls);
                } else {
                    root.loadMultipleFiles(drop.urls, false);
                }
            }
        }
    }

    Column {
        id: tlcol;
        width: parent.width;
        anchors.horizontalCenter: parent.horizontalCenter;
        anchors.bottom: parent.bottom;
        anchors.bottomMargin: areButtonsUp? 0 : 5 * dpiScale;
        spacing: root.fullScreen || window.isMobileLayout? 0 : 10 * dpiScale;
        property bool areButtonsUp: !window.isMobileLayout;
        onAreButtonsUpChanged: {
            buttonsArea.parent = null;
            bottomPanel.parent = null;
            if (areButtonsUp) {
                buttonsArea.parent = tlcol;
                bottomPanel.parent = tlcol;
            } else {
                bottomPanel.parent = tlcol;
                buttonsArea.parent = tlcol;
            }
        }
        Component.onCompleted: areButtonsUpChanged();

        Item {
            id: buttonsArea;
            width: parent? parent.width : 0;
            height: 40 * dpiScale;
            visible: !root.fullScreen;

            Rectangle {
                visible: window.isMobileLayout || !middleButtons.willFit;
                color: styleBackground;
                opacity: 0.8;
                radius: 5 * dpiScale;
                anchors.fill: textCol;
                anchors.margins: -4 * dpiScale;
            }
            Column {
                id: textCol;
                enabled: vid.loaded;
                y: middleButtons.willFit? ((parent.height - height) / 2) : -buttonsArea.y - tlcol.y + 7 * dpiScale + ((main_window.safeAreaMargins.top || 0) * 0.8);
                anchors.left: parent.left;
                anchors.leftMargin: 10 * dpiScale;
                spacing: 3 * dpiScale;
                property real widthPadded: Math.ceil(width / (20 * dpiScale)) * (20 * dpiScale);
                Row {
                    BasicText {
                        text: timeline.timeAtPosition((vid.currentFrame + 1) / Math.max(1, vid.frameCount));
                        leftPadding: 0;
                        font.pixelSize: 14 * dpiScale;
                        anchors.verticalCenter: parent.verticalCenter;
                    }
                    BasicText {
                        text: `(${vid.currentFrame+1}/${vid.frameCount})`;
                        leftPadding: 5 * dpiScale;
                        font.pixelSize: 11 * dpiScale;
                        anchors.verticalCenter: parent.verticalCenter;
                    }
                }
                Row {
                    visible: window.stab.automaticHorizonLock;
                    BasicText {
                        text: qsTr("Turn Speed (Roll):");
                        leftPadding: 0;
                        font.pixelSize: 11 * dpiScale;
                        anchors.verticalCenter: parent.verticalCenter;
                    }
                    BasicText {
                        id: turnSpeedValue;
                        text: "---";
                        leftPadding: 5 * dpiScale;
                        font.pixelSize: 11 * dpiScale;
                        anchors.verticalCenter: parent.verticalCenter;
                    }
                }
                BasicText {
                    id: currentFovText;
                    font.pixelSize: 11 * dpiScale;
                    leftPadding: 0;
                }
            }

            Item {
                id: middleButtons;
                property real availableWidth: parent.width - textCol.widthPadded - rightButtons.width - 40 * dpiScale;
                width: parent.width - (willFit? textCol.widthPadded + rightButtons.width + 40 * dpiScale : 0);
                height: parent.height;
                x: willFit? textCol.x + textCol.widthPadded + 10 * dpiScale : 0;
                property bool willFit: availableWidth > children[0].width;
                Row {
                    anchors.centerIn: parent;
                    spacing: 5 * dpiScale;
                    enabled: vid.loaded;
                    Button { text: "["; font.bold: true; onClicked: timeline.setTrimStart(timeline.closestTrimRange(timeline.position, true), timeline.position); tooltip: qsTr("Trim start"); transparentOnMobile: true; }
                    Button {
                        iconName: "chevron-left";
                        tooltip: qsTr("Previous frame");
                        transparentOnMobile: true;
                        MouseArea {
                            anchors.fill: parent;
                            onClicked: mouse => {
                                if (mouse.modifiers & Qt.ShiftModifier) {
                                    timeline.jumpToPrevKeyframe("");
                                } else if (mouse.modifiers & Qt.ControlModifier) {
                                    vid.seekToFrameDelta(-10);
                                } else {
                                    vid.seekToFrameDelta(-1);
                                }
                            }
                        }
                    }
                    Button {
                        onClicked: { if (vid.playing) vid.pause(); else vid.play(); }
                        tooltip: vid.playing? qsTr("Pause") : qsTr("Play");
                        iconName: vid.playing? "pause" : "play";
                        transparentOnMobile: true;
                    }
                    Button {
                        iconName: "chevron-right";
                        tooltip: qsTr("Next frame");
                        transparentOnMobile: true;
                        MouseArea {
                            anchors.fill: parent;
                            onClicked: mouse => {
                                if (mouse.modifiers & Qt.ShiftModifier) {
                                    timeline.jumpToNextKeyframe("");
                                } else if (mouse.modifiers & Qt.ControlModifier) {
                                    vid.seekToFrameDelta(10);
                                } else {
                                    vid.seekToFrameDelta(1);
                                }
                            }
                        }
                    }
                    Button { text: "]"; font.bold: true; onClicked: timeline.setTrimEnd(timeline.closestTrimRange(timeline.position, false), timeline.position); tooltip: qsTr("Trim end"); transparentOnMobile: true; }
                    Button { visible: isMobile; iconName: "menu"; onClicked: timeline.toggleContextMenu(this); tooltip: qsTr("Show timeline menu"); transparentOnMobile: true; leftPadding: 10 * dpiScale; rightPadding: 10 * dpiScale; }
                }
            }
            Rectangle {
                visible: window.isMobileLayout || !middleButtons.willFit;
                color: styleBackground;
                opacity: 0.8;
                radius: 5 * dpiScale;
                anchors.fill: rightButtons;
                anchors.margins: -4 * dpiScale;
            }
            Row {
                id: rightButtons;
                enabled: vid.loaded;
                spacing: 5 * dpiScale;
                y: middleButtons.willFit? ((parent.height - height) / 2) : -buttonsArea.y - tlcol.y + ((main_window.safeAreaMargins.top || 0) * 0.8);
                onYChanged: root.additionalTopMargin = middleButtons.willFit? 0 : Math.max(height, textCol.height) + 2*4 * dpiScale + ((main_window.safeAreaMargins.top || 0) * 0.8);
                anchors.right: parent.right;
                anchors.rightMargin: 10 * dpiScale;
                height: parent.height;

                component SmallLinkButton: LinkButton {
                    height: Math.round(parent.height);
                    anchors.verticalCenter: parent.verticalCenter;
                    textColor: !checked? styleTextColor : styleAccentColor;
                    onClicked: checked = !checked;
                    opacity: checked? 1 : 0.5;
                    checked: true;
                    leftPadding: 6 * dpiScale;
                    rightPadding: 6 * dpiScale;
                    topPadding: 8 * dpiScale;
                    bottomPadding: 8 * dpiScale;
                }

                SmallLinkButton {
                    id: fovOverviewBtn;
                    iconName: "fov-overview";
                    checked: false;
                    onCheckedChanged: { controller.fov_overview = checked; vid.forceRedraw(); }
                    tooltip: qsTr("Toggle stabilization overview");
                    TapHandler {
                        acceptedModifiers: Qt.ControlModifier
                        onTapped: { if (fovOverviewBtn.checked) { secondPreview.show = !secondPreview.show; fovOverviewBtn.checked = false; vid.forceRedraw(); } }
                    }
                }

                SmallLinkButton {
                    id: stabEnabledBtn;
                    iconName: "gyroflow";
                    tooltip: qsTr("Toggle stabilization");
                }

                // A/B the grade on its own. Deliberately beside the stabilization toggle and
                // deliberately separate from it: turning stabilization off used to skip the whole
                // render pass, taking the colour with it, so there was no way to see one without
                // the other.
                //
                // Both are preview-only. `set_render_params` forces them back on for every job,
                // so a toggle left off here can never silently ship a wrong-looking export.
                SmallLinkButton {
                    id: colorEnabledBtn;
                    iconName: "pencil-outline";
                    onCheckedChanged: { controller.preview_color_enabled = checked; vid.forceRedraw(); }
                    tooltip: qsTr("Toggle color adjustments");
                }

                SmallLinkButton {
                    id: scopesBtn;
                    iconName: "chart";
                    checked: false;
                    onCheckedChanged: settings.setValue("scopesShown", checked);
                    Component.onCompleted: checked = settings.value("scopesShown", false);
                    tooltip: qsTr("Toggle video scopes (Shift+S)");
                }

                SmallLinkButton {
                    id: muteBtn;
                    iconName: checked? "sound" : "sound-mute";
                    // Suppressed while the volume popup is open: the tooltip is anchored to this
                    // button and lands on top of the slider, which is unreadable and, worse,
                    // looks like part of it.
                    tooltip: volumePopup.opened? "" : (checked? qsTr("Mute") : qsTr("Unmute"));
                    checked: !vid.muted;

                    ContextMenuMouseArea {
                        underlyingItem: muteBtn;
                        cursorShape: Qt.PointingHandCursor;
                        onContextMenu: (isHold, x, y) => { volumePopup.open(); if (isHold) vid.muted = !vid.muted; }
                    }
                    onClicked: () => { vid.muted = !vid.muted; }

                    // Hovering the speaker reveals the volume slider; clicking still mutes.
                    //
                    // The popup and its slider are unchanged - they have been here all along, but
                    // only reachable by right-clicking the icon (or long-pressing it on touch,
                    // which is still the only route where there is no hover), so in practice
                    // nobody found them.
                    //
                    // Two delays, and both earn their place. Opening waits, because this button
                    // sits between the colour toggle and the playback-rate box: without a delay
                    // the popup flashes every time the cursor crosses it on the way somewhere
                    // else. Closing waits, because the cursor has to leave the button to reach
                    // the slider - close on the first hover-out and the slider is unreachable.
                    onHoveredChanged: {
                        if (hovered) { volumeCloseTimer.stop(); volumeOpenTimer.restart(); }
                        else         { volumeOpenTimer.stop();  volumeCloseTimer.restart(); }
                    }
                    Timer {
                        id: volumeOpenTimer;
                        interval: 250;
                        onTriggered: if (muteBtn.hovered) volumePopup.open();
                    }
                    Timer {
                        id: volumeCloseTimer;
                        interval: 400;
                        // `pressed` keeps it open through a drag that wanders outside the popup,
                        // which is easy to do when the handle is already at one end of the track.
                        onTriggered: if (!muteBtn.hovered && !volumeHover.hovered && !volumeSlider.pressed) volumePopup.close();
                    }

                    // Scroll over the speaker to set the level without aiming at anything. The
                    // popup comes up with it so the change is visible rather than silent.
                    WheelHandler {
                        onWheel: event => {
                            volumeSlider.value = Math.max(0, Math.min(100, volumeSlider.value + (event.angleDelta.y > 0? 5 : -5)));
                            volumePopup.open();
                            volumeCloseTimer.restart();
                        }
                    }

                    // Declarative rather than assigning `vid.volume` from the slider's
                    // `onValueChanged`: a stored level that happens to equal the Slider's own
                    // initial value fires no change signal, so a saved volume of 0 was silently
                    // never applied and audio came back at full level on the next launch.
                    Binding { target: vid; property: "volume"; value: volumeSlider.value / 100.0; }

                    Popup {
                        id: volumePopup;
                        width: volumeRow.width + 20 * dpiScale;
                        height: 30 * dpiScale;
                        x: -width + muteBtn.width;
                        y: -height;

                        Row {
                            id: volumeRow;
                            anchors.centerIn: parent;
                            spacing: 8 * dpiScale;

                            // Keeps the popup alive while the cursor is inside it, which is the
                            // other half of the close rule above. On the Row rather than on the
                            // Popup because this component's `contentItem` is a ListView (it was
                            // written for combo boxes), so a handler declared on the Popup lands
                            // somewhere less predictable than the item actually under the cursor.
                            HoverHandler {
                                id: volumeHover;
                                margin: 10 * dpiScale;   // cover the popup's padding too
                                onHoveredChanged: {
                                    if (hovered) volumeCloseTimer.stop();
                                    else         volumeCloseTimer.restart();
                                }
                            }
                            BasicText {
                                text: qsTr("Volume");
                                anchors.verticalCenter: parent.verticalCenter;
                            }
                            Slider {
                                id: volumeSlider;
                                width: 150 * dpiScale;
                                anchors.verticalCenter: parent.verticalCenter;
                                unit: "%";
                                from: 0;
                                to: 100;
                                value: settings.value("volume", 100);
                                precision: 0;
                                onValueChanged: settings.setValue("volume", value);
                            }
                            BasicText {
                                // Fixed width, so the row does not shuffle sideways as the number
                                // goes from "5 %" to "100 %" while you drag.
                                width: 36 * dpiScale;
                                horizontalAlignment: Text.AlignRight;
                                anchors.verticalCenter: parent.verticalCenter;
                                text: Math.round(volumeSlider.value) + " %";
                                opacity: 0.7;
                            }
                        }
                    }
                }

                ComboBox {
                    id: playbackRateCb;
                    model: ["0.13x", "0.25x", "0.5x", "1x", "2x", "4x", "5x", "8x", "10x", "20x", "50x"];
                    width: 60 * dpiScale;
                    currentIndex: 3;
                    height: 25 * dpiScale;
                    itemHeight: 25 * dpiScale;
                    font.pixelSize: 11 * dpiScale;
                    anchors.verticalCenter: parent.verticalCenter;
                    onCurrentTextChanged: {
                        const rate = +currentText.replace("x", ""); // hacky but simple and it works
                        vid.playbackRate = rate;
                    }
                    tooltip: qsTr("Playback speed");
                }
            }
        }

        ResizablePanel {
            id: bottomPanel;
            direction: ResizablePanel.HandleUp;
            width: parent? parent.width : 0;
            color: "transparent";
            hr.height: 30 * dpiScale;
            hr.opacity: root.fullScreen || window.isMobileLayout? 0.1 : 1.0;
            additionalHeight: timeline.additionalHeight;
            defaultHeight: (window.isMobileLayout? 50 : 165) * dpiScale;
            minHeight: (root.fullScreen || window.isMobileLayout? 50 : 100) * dpiScale;
            lastHeight: settings.value("bottomPanelSize" + (root.fullScreen? "-full" : ""), defaultHeight);
            onHeightAdjusted: settings.setValue("bottomPanelSize" + (root.fullScreen? "-full" : ""), height);
            Connections {
                target: root;
                function onFullScreenChanged(): void {
                    bottomPanel.lastHeight = settings.value("bottomPanelSize" + (root.fullScreen? "-full" : ""), bottomPanel.defaultHeight);
                    if (root.fullScreen == 2) {
                        main_window.visibility = Window.FullScreen;
                    } else {
                        if (main_window.visibility == Window.FullScreen) main_window.visibility = Window.Windowed;
                    }
                }
            }
            visible: root.fullScreen != 2;
            maxHeight: root.height - 50 * dpiScale;
            Timeline {
                id: timeline;
                durationMs: vid.duration;
                scaledFps: vid.frameRate;
                anchors.fill: parent;
                fullScreen: root.fullScreen;
                visible: vid.loaded || !window.isMobileLayout;
                property bool prevRestrictTrim: false;
                Component.onCompleted: prevRestrictTrim = restrictTrim;

                onTrimRangesChanged: {
                    controller.set_trim_ranges(timeline.trimRanges.map(x => x[0] + ":" + x[1]).join(";"));
                    restrictTrimChanged();
                }
                onRestrictTrimChanged: {
                    if (restrictTrim) {
                        const ranges = timeline.getTrimRanges();
                        vid.setPlaybackRange(ranges[0][0] * vid.duration, ranges[ranges.length - 1][1] * vid.duration);
                    } else if (prevRestrictTrim != restrictTrim) {
                        vid.setPlaybackRange(0, -1);
                    }
                    prevRestrictTrim = restrictTrim;
                }
            }
        }
    }
    Item {
        width: vidParentParent.width;
        height: vidParentParent.height;
        LoaderOverlay {
            id: videoLoader;
            background: styleBackground;
            verticalOffset: window.isMobileLayout? -bottomPanel.height / 2 : 0;
            onActiveChanged: { vid.forceRedraw(); vid.fovChanged(); }
            canHide: render_queue.main_job_id > 0;
            onCancel: {
                if (render_queue.main_job_id > 0) {
                    render_queue.cancel_job(render_queue.main_job_id);
                } else {
                    controller.cancel_current_operation();
                }
            }
            onHide: {
                render_queue.main_job_id = 0;
                videoLoader.active = false;
            }
        }
        Column {
            id: infoMessages;
            width: parent.width;
            spacing: 5 * dpiScale;
            visible: children.length > 0;
            y: root.additionalTopMargin;
            InfoMessage {
                type: InfoMessage.Warning;
                visible: vid.loaded && !controller.lens_loaded && !isCalibrator;
                text: qsTr("Lens profile is not loaded, the results will not look correct. Please load a lens profile for your camera.");
            }
        }
        Item {
            id: gyroLoading;
            property real progress: 0;
            visible: opacity > 0;
            opacity: controller.loading_gyro_in_progress? 1 : 0;
            Ease on opacity { }
            anchors.right: parent.right;
            anchors.bottom: parent.bottom;
            anchors.rightMargin: 20 * dpiScale;
            // In full screen and in the mobile layout the timeline is drawn over the bottom of the video area
            anchors.bottomMargin: Math.max(0, vidParentParent.height - tlcol.y) + 20 * dpiScale;
            width: gyroLoadingCol.width + 2 * 14 * dpiScale;
            height: gyroLoadingCol.height + 2 * 12 * dpiScale;
            Rectangle {
                anchors.fill: parent;
                color: styleBackground;
                opacity: 0.8;
                radius: 6 * dpiScale;
            }
            Column {
                id: gyroLoadingCol;
                anchors.centerIn: parent;
                spacing: 4 * dpiScale;
                BasicText {
                    leftPadding: 0;
                    text: qsTr("Loading gyro data %1...").arg("<b>" + (gyroLoading.progress * 100).toFixed(0) + "%</b>");
                }
                BasicText {
                    leftPadding: 0;
                    bottomPadding: 8 * dpiScale;
                    font.pixelSize: 11 * dpiScale;
                    opacity: 0.7;
                    text: qsTr("The video plays unstabilized until it's loaded.");
                }
                Row {
                    spacing: 12 * dpiScale;
                    QQC.ProgressBar {
                        width: 200 * dpiScale;
                        anchors.verticalCenter: parent.verticalCenter;
                        value: gyroLoading.progress;
                    }
                    LinkButton {
                        transparent: true;
                        text: qsTr("Cancel");
                        leftPadding: 0; rightPadding: 0; topPadding: 0; bottomPadding: 0;
                        anchors.verticalCenter: parent.verticalCenter;
                        onClicked: controller.cancel_current_operation();
                    }
                }
            }
        }
    }
    Loader {
        id: queue;
        asynchronous: true;
        anchors.fill: vidParentParent;
        anchors.margins: 10 * dpiScale;
        sourceComponent: Component {
            RenderQueue {
                onShownChanged: if (statistics.item) statistics.item.shown &= !shown;
            }
        }
    }
    Loader {
        id: statistics;
        asynchronous: true;
        active: false;
        anchors.fill: vidParentParent;
        anchors.margins: 10 * dpiScale;
        onStatusChanged: if (status == Loader.Ready) statistics.item.shown = true;
        sourceComponent: Component {
            Statistics {
                onShownChanged: queue.item.shown &= !shown;
            }
        }
    }
}
