import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import Qt.labs.platform 1.1
import Qt5Compat.GraphicalEffects
import QtQuick.Layouts 1.15

Window {
    visible: true
    color: "#2e2f30"
    width: 1400
    height: 700
    title: "ASI Camera Controller"
    Component.onCompleted: cameraController.initialize_controls()

    // File Paths
    property string standardCaptureSavePath: "path/to/data/file.hdf5"
    property string liveCaptureSavePath: "path/to/data/file.hdf5"
    property string backgroundLoadPath: "path/to/data/file.npy"
    property string nistReferenceFilePath: "path/to/data/file.npy"
    property string heNeFilePath: "path/to/data/file.npy"
    property string neAnchorFilePath: "path/to/data/file.npy"
    property string neCalibrationFilePath: "path/to/data/file.npy"
    property string saveCachePath: "path/to/cache/file.csv"
    property string loadCachePath: "path/to/cache/file.csv"

    Row {
        id: root
        anchors.fill: parent
        spacing: 10

    // Left side: image in a scrollable area
    Flickable {
        id: imageFlick
        width: parent.width * 0.70
        height: parent.height

        contentWidth: liveFeed.width
        contentHeight: liveFeed.height

        interactive: true
        clip: true

        boundsBehavior: Flickable.StopAtBounds
        flickDeceleration: 3000

        ScrollBar.horizontal: ScrollBar { }
        ScrollBar.vertical: ScrollBar { }

        Image {
            id: liveFeed
            source: "image://camera/live"
            width: implicitWidth
            height: implicitHeight
            fillMode: Image.Pad   // VERY important
            smooth: false
            asynchronous: false
            cache: false
        }

        TapHandler {
            acceptedButtons: Qt.LeftButton
            onDoubleTapped: cameraController.find_brightest_centroid_requested()
        }

        Connections {
            target: cameraController
            function onFrameReady(frame) {
                liveFeed.source = "image://camera/live?" + Date.now()
            }

            function onBrightestCentroidFound(x, y) {
                imageFlick.contentX = Math.max(0, x - imageFlick.width / 2)
                imageFlick.contentY = Math.max(0, y - imageFlick.height / 2)
            }
        }
    }

    ScrollView {
        id: imageScroll_right
        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff
        width: parent.width * 0.295   // take ~70% of window width
        height: parent.height       // full height
        clip: true  // ensure only visible portion is shown

        Column {
            id: masterControlColumn
            anchors.right: root.right
            width: root.width * 0.30
            leftPadding: 10
            spacing: 10

            TapHandler {
                onTapped: root.forceActiveFocus()
            }

            Column{
                id: cameraControlColumn
                spacing: 10

                Rectangle{
                    anchors.left: masterControlColumn.left
                    anchors.right: masterControlColumn.right
                    height: 1
                    color: "#cdcdcd"
                }

                Text {
                    text: "Camera Control"
                    font.pixelSize: 32       // makes it large
                    //font.bold: true
                    color: "#D21404"             // red text
                    anchors.left: parent.leftPadding

                    layer.enabled: true
                    layer.effect: DropShadow {
                        color: "#000000"
                        radius: 4
                        samples: 8
                        spread: 0.2
                        verticalOffset: 1
                    }
                }

                Text {
                    text: "Exposure (ms):"
                    font.bold: true
                    width: parent.width
                    color: "#cdcdcd"
                    font.pixelSize: 14
                }

                Row {
                    spacing: masterControlColumn.width/8
                    width: masterControlColumn.width

                    Slider {
                        id: exposureSlider
                        anchors.left: masterControlColumn.leftPadding
                        stepSize: 1
                        onValueChanged: {
                            exposureSpinBox.value = value
                            cameraController.setExposure(value)
                        }
                    }

                    SpinBox {
                        id: exposureSpinBox
                        font.pixelSize: 14
                        anchors.right: masterControlColumn.rightPadding
                        stepSize: 1
                        editable: true
                        onValueChanged: {
                            exposureSlider.value = value
                            cameraController.setExposure(value)
                        }
                    }

                    Connections {
                        target: cameraController
                        function onExposureRangeChanged(min, max) {
                            exposureSlider.from = min
                            exposureSlider.to = max
                            exposureSpinBox.from = min
                            exposureSpinBox.to = max
                            exposureSlider.value = 300
                            exposureSpinBox.value = 300
                        }
                    }

                    Connections {
                        target: cameraController
                        function onCanEditControls(status) {
                            exposureSlider.enabled = status
                            exposureSpinBox.enabled = status
                            exposureSpinBox.font.bold = !status
                            exposureSpinBox.opacity = status ? 1.0 : 0.9
                        }
                    }
                }

                Text {
                    text: "Gain:"
                    font.bold: true
                    width: masterControlColumn.width
                    color: "#cdcdcd"
                    font.pixelSize: 14
                }

                Row {
                    width: masterControlColumn.width
                    spacing: masterControlColumn.width / 8

                    Slider {
                        id: gainSlider
                        stepSize: 1
                        onValueChanged: {
                            gainSpinBox.value = value
                            cameraController.setGain(value)
                        }
                    }

                    SpinBox {
                        id: gainSpinBox
                        font.pixelSize: 14
                        stepSize: 1
                        editable: true
                        onValueChanged: {
                            gainSlider.value = value
                            cameraController.setGain(value)
                        }
                    }

                    Connections {
                        target: cameraController
                        function onGainRangeChanged(min, max) {
                            gainSlider.from = min
                            gainSlider.to = max
                            gainSpinBox.from = min
                            gainSpinBox.to = max
                            gainSlider.value = (min + max) / 2
                            gainSpinBox.value = (min + max) / 2
                        }
                    }

                    Connections {
                        target: cameraController
                        function onCanEditControls(status) {
                            gainSlider.enabled = status
                            gainSpinBox.enabled = status
                            gainSpinBox.font.bold = !status
                            gainSpinBox.opacity = status ? 1.0 : 0.9
                        }
                    }
                }
            }

            Column{
                id: temperatureControlColumn
                spacing: 10

                Rectangle{
                    anchors.left: masterControlColumn.left
                    width: masterControlColumn.width * 0.9
                    height: 1
                    color: "#cdcdcd"
                }

                Text {
                    text: "Temperature Control"
                    font.pixelSize: 32       // makes it large
                    color: "#D21404"             // red text
                    anchors.left: masterControlColumn.leftPadding

                    layer.enabled: true
                    layer.effect: DropShadow {
                        color: "#000000"
                        radius: 4
                        samples: 8
                        spread: 0.2
                        verticalOffset: 1
                    }
                }

                Row {
                    spacing: masterControlColumn.width/6

                    CheckBox {
                        id: coolerCheck
                        text: "Cooler"
                        contentItem: Text {
                            text: coolerCheck.text
                            color: "#cdcdcd"
                            font.pixelSize: 14
                            font.bold: true
                            verticalAlignment: Text.AlignVCenter
                            leftPadding: coolerCheck.indicator.width + 6
                        }
                        onToggled: cameraController.setCooler(checked)
                    }

                    CheckBox {
                        id: dewCheck
                        text: "Anti-Dew Heater"
                        contentItem: Text {
                            text: dewCheck.text
                            color: "#cdcdcd"
                            font.pixelSize: 14
                            font.bold: true
                            verticalAlignment: Text.AlignVCenter
                            leftPadding: dewCheck.indicator.width + 6
                        }
                        onToggled: cameraController.setAntiDewHeater(checked)
                    }

                    Connections {
                        target: cameraController
                        function onCanEditControls(status) {
                            coolerCheck.checkable = status
                            dewCheck.checkable = status
                        }
                    }
                }

                Text {
                    text: "Target Temperature:"
                    font.bold: true
                    color: "#cdcdcd"
                    font.pixelSize: 14
                }

                Row {
                    width: masterControlColumn.width
                    spacing: masterControlColumn.width / 8

                    Slider {
                        id: targetTempSlider
                        stepSize: 1
                        onValueChanged: {
                            targetTempSpinBox.value = value
                            cameraController.setTargetTemp(value)
                        }
                    }

                    SpinBox {
                        id: targetTempSpinBox
                        font.pixelSize: 14
                        stepSize: 1
                        editable: true
                        onValueChanged: {
                            targetTempSlider.value = value
                            cameraController.setTargetTemp(value)
                        }
                    }

                    Connections {
                        target: cameraController
                        function onTempRangeChanged(min, max) {
                            targetTempSlider.from = min
                            targetTempSlider.to = max
                            targetTempSpinBox.from = min
                            targetTempSpinBox.to = max
                            targetTempSpinBox.value = (min + max) / 2
                            targetTempSpinBox.value = (min + max) / 2
                        }
                    }

                    Connections {
                        target: cameraController
                        function onCanEditControls(status) {
                            targetTempSlider.enabled = status
                            targetTempSpinBox.enabled =  status
                            targetTempSpinBox.font.bold = !status
                            targetTempSpinBox.opacity = status ? 1.0 : 0.9
                        }
                    }
                }

                Text{
                    text: "Current Temperature: "
                    font.bold: true
                    color: "#cdcdcd"
                    font.pixelSize: 14
                }

                Rectangle{
                    id: temperatureBox
                    color: "#222021"
                    width: 200
                    height: 30

                    Text {
                        id: tempLabel
                        anchors.horizontalCenter: temperatureBox.horizontalCenter
                        anchors.verticalCenter: temperatureBox.verticalCenter
                        text: "-- °C"
                        font.pixelSize: 18
                        font.bold: true
                        color: "#cdcdcd"

                        Connections {
                            target: cameraController
                            function onTemperatureChanged(temp) {
                                tempLabel.text = temp.toFixed(1) + " °C"
                            }
                        }
                    }
                }
            }

            Column{
                id: captureFramesColumn
                spacing: 10

                Rectangle{
                    anchors.left: masterControlColumn.left
                    width: masterControlColumn.width * 0.9
                    height: 1
                    color: "#cdcdcd"
                }

                Text {
                    text: "Data Acquisition"
                    font.pixelSize: 32       // makes it large
                    color: "#D21404"             // red text
                    anchors.left: masterControlColumn.leftPadding

                    layer.enabled: true
                    layer.effect: DropShadow {
                        color: "#000000"
                        radius: 4
                        samples: 8
                        spread: 0.2
                        verticalOffset: 1
                    }
                }

                // Choice: Capture or Load
                Row {
                    spacing: masterControlColumn.width/12
                    RadioButton {
                        id: standardCaptureMode
                        checked: true

                        indicator: Rectangle {
                            id: standardCaptureCircle
                            anchors.verticalCenter: standardCaptureMode.verticalCenter
                            implicitWidth: 20
                            implicitHeight: 20
                            radius: width / 2
                            border.width: 2
                            border.color: "#555555"
                            color: "transparent"

                            Rectangle {
                                anchors.centerIn: standardCaptureCircle
                                width: standardCaptureCircle.width / 2
                                height: standardCaptureCircle.height / 2
                                radius: width / 2
                                color: standardCaptureMode.checked ? "#cdcdcd" : "transparent"
                            }
                        }

                        contentItem: Text {
                            text: "Standard Capture"
                            color: "#cdcdcd"
                            font.bold: true
                            font.pixelSize: 14
                            verticalAlignment: Text.AlignVCenter
                            leftPadding: standardCaptureMode.indicator.width + 6
                        }
                    }

                    RadioButton {
                        id: liveCaptureMode

                        indicator: Rectangle {
                            id: liveCaptureCircle
                            anchors.verticalCenter: liveCaptureMode.verticalCenter
                            implicitWidth: 20
                            implicitHeight: 20
                            radius: width / 2
                            border.width: 2
                            border.color: "#555555"
                            color: "transparent"

                            Rectangle {
                                anchors.centerIn: liveCaptureCircle
                                width: liveCaptureCircle.width / 2
                                height: liveCaptureCircle.height / 2
                                radius: width / 2
                                color: liveCaptureMode.checked ? "#cdcdcd" : "transparent"
                            }
                        }

                        contentItem: Text {
                            text: "Live Capture"
                            color: "#cdcdcd"
                            font.bold: true
                            font.pixelSize: 14
                            verticalAlignment: Text.AlignVCenter
                            leftPadding: liveCaptureMode.indicator.width + 6
                        }
                    }
                }

                RowLayout{
                    width: masterControlColumn.width * 0.9
                    visible: standardCaptureMode.checked

                    Button {
                        id: saveFramesForStandardCaptureButton
                        text: "Save Data To"
                        onClicked: saveFramesForStandardCaptureFileDialog.open()
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        height: saveFramesForStandardCaptureButton.height
                        radius: 4
                        border.color: "#888"
                        color: "#f5f5f5"

                        Text {
                            anchors.fill: parent
                            anchors.margins: 4
                            verticalAlignment: Text.AlignVCenter

                            text: standardCaptureSavePath
                            elide: Text.ElideLeft     // Clip path nicely
                            clip: true
                        }
                    }
                }

                RowLayout{
                    width: masterControlColumn.width * 0.9
                    visible: standardCaptureMode.checked
                    spacing: 10

                    Button {
                        id: captureFrameBtn
                        enabled: false
                        text: "Capture Frame(s)"
                        Layout.fillWidth: true
                        onClicked: cameraController.request_standard_capture()

                        Connections {
                            target: cameraController
                            function onCanStandardCaptureChanged(canCapture) {
                                captureFrameBtn.enabled = canCapture
                            }
                        }
                    }

                    Text{
                        text: "Number of Frames:"
                        color: "#cdcdcd"
                        font.bold: true
                        font.pixelSize: 14
                        anchors.verticalCenter: captureFrameBtn.verticalCenter
                    }

                    SpinBox {
                        id: n_frame_capture_spin_box
                        font.pixelSize: 14
                        stepSize: 1
                        editable: true
                        from: 1
                        to: 100

                        onValueChanged: {
                            n_frame_capture_spin_box.value = value
                            cameraController.setNStandardCaptureFrames(value)
                        }
                    }
                }

                RowLayout{
                    width: masterControlColumn.width * 0.9
                    visible: liveCaptureMode.checked

                    Button {
                        id: saveFramesForLiveCaptureButton
                        enabled: true
                        text: "Save Data To"
                        onClicked: saveFramesForLiveCaptureFileDialog.open()

                        Connections {
                            target: cameraController
                            function onCanEditLiveCaptureFileChanged(canSave) {
                                saveFramesForLiveCaptureButton.enabled = canSave
                            }
                        }
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        height: saveFramesForLiveCaptureButton.height
                        radius: 4
                        border.color: "#888"
                        color: "#f5f5f5"

                        Text {
                            anchors.fill: parent
                            anchors.margins: 4
                            verticalAlignment: Text.AlignVCenter

                            text: liveCaptureSavePath
                            elide: Text.ElideLeft     // Clip path nicely
                            clip: true
                        }
                    }
                }

                RowLayout{
                    width: masterControlColumn.width * 0.9
                    visible: liveCaptureMode.checked
                    spacing: 10

                    Button {
                        id: startLiveCaptureBtn
                        enabled: false
                        text: "Start Live Capture"
                        Layout.fillWidth: true
                        onClicked: cameraController.request_live_capture_start()

                        Connections {
                            target: cameraController
                            function onCanStartLiveCaptureChanged(canStart) {
                                startLiveCaptureBtn.enabled = canStart
                            }
                        }
                    }

                    Button {
                        id: endLiveCaptureBtn
                        enabled: false
                        text: "End Live Capture"
                        Layout.fillWidth: true
                        onClicked: {
                            cameraController.request_live_capture_end()
                            endLiveCaptureBtn.enabled = false
                            saveFramesForLiveCaptureButton.enabled = true
                        }

                        Connections {
                            target: cameraController
                            function onCanEndLiveCaptureChanged(canEnd) {
                                endLiveCaptureBtn.enabled = canEnd
                            }
                        }
                    }

                }

            RowLayout{
                id: dataToCollectRow

                Text{
                    text: "Data to Collect: "
                    font.bold: true
                    color: "#cdcdcd"
                    font.pixelSize: 14
                }

                    Text{
                        text: "Raw Data"
                        font.bold: true
                        color: "#cdcdcd"
                        font.pixelSize: 14
                    }

                    CheckBox {
                        id: rawCheckBox
                        checked: false
                        height: 30
                        width: masterControlColumn.width*0.6

                        onCheckedChanged: {
                            cameraController.request_can_save_raw(checked)
                        }
                    }

                    Text{
                        id: test
                        text: "Wavelength Data"
                        font.bold: true
                        color: "#cdcdcd"
                        font.pixelSize: 14
                    }

                    CheckBox {
                        id: wavelengthCheckBox
                        checked: false
                        height: 30
                        width: masterControlColumn.width*0.6

                        onCheckedChanged: {
                            cameraController.request_can_save_wavelength(checked)
                        }

                        Connections {
                            target: cameraController
                            function onCanEnableWavelengthCheckBox(enabled) {
                                wavelengthCheckBox.checked = enabled
                            }
                        }
                    }
                }
            }

            Column{
                id: backgroundSubtractionColumn
                spacing: 10

                Rectangle{
                    anchors.left: masterControlColumn.left
                    width: masterControlColumn.width * 0.9
                    height: 1
                    color: "#cdcdcd"
                }

                Text {
                    text: "Background Subtraction"
                    font.pixelSize: 32       // makes it large
                    color: "#D21404"             // red text
                    anchors.left: masterControlColumn.leftPadding

                    layer.enabled: true
                    layer.effect: DropShadow {
                        color: "#000000"
                        radius: 4
                        samples: 8
                        spread: 0.2
                        verticalOffset: 1
                    }
                }

                // Choice: Capture or Load
                Row {
                    spacing: masterControlColumn.width/12
                    RadioButton {
                        id: captureMode
                        checked: true

                        indicator: Rectangle {
                            id: captureCircle
                            anchors.verticalCenter: captureMode.verticalCenter
                            implicitWidth: 20
                            implicitHeight: 20
                            radius: width / 2
                            border.width: 2
                            border.color: "#555555"
                            color: "transparent"

                            Rectangle {
                                anchors.centerIn: captureCircle
                                width: captureCircle.width / 2
                                height: captureCircle.height / 2
                                radius: width / 2
                                color: captureMode.checked ? "#cdcdcd" : "transparent"
                            }
                        }

                        contentItem: Text {
                            text: "Capture Background"
                            color: "#cdcdcd"
                            font.bold: true
                            font.pixelSize: 14
                            verticalAlignment: Text.AlignVCenter
                            leftPadding: captureMode.indicator.width + 6
                        }
                    }

                    RadioButton {
                        id: loadMode

                        indicator: Rectangle {
                            id: loadCircle
                            anchors.verticalCenter: loadMode.verticalCenter
                            implicitWidth: 20
                            implicitHeight: 20
                            radius: width / 2
                            border.width: 2
                            border.color: "#555555"
                            color: "transparent"

                            Rectangle {
                                anchors.centerIn: loadCircle
                                width: loadCircle.width / 2
                                height: loadCircle.height / 2
                                radius: width / 2
                                color: loadMode.checked ? "#cdcdcd" : "transparent"
                            }
                        }

                        contentItem: Text {
                            text: "Load Background"
                            color: "#cdcdcd"
                            font.bold: true
                            font.pixelSize: 14
                            verticalAlignment: Text.AlignVCenter
                            leftPadding: loadMode.indicator.width + 6
                        }
                    }
                }

                GridLayout {
                    visible: captureMode.checked
                    columns: 3
                    columnSpacing: masterControlColumn.width / 20
                    rowSpacing: 10
                    anchors.left: masterControlColumn.leftPadding

                    Button {
                        id: captureBtn
                        text: "Capture Background"
                        onClicked: {
                            cameraController.request_capture_background()
                            subtractBtn2.enabled = false
                            resetBtn2.enabled = false
                        }
                        Layout.fillWidth: true
                    }

                        Text{
                            text: "Number of Frames:"
                            color: "#cdcdcd"
                            font.bold: true
                            font.pixelSize: 14
                            anchors.verticalCenter: n_frame_background_spin_box.verticalCenter
                        }

                        SpinBox {
                            id: n_frame_background_spin_box
                            font.pixelSize: 14
                            stepSize: 1
                            editable: true
                            from: 1
                            to: 100
                            onValueChanged: {
                                n_frame_background_spin_box.value = value
                                cameraController.setNBackgroundFrames(value)
                            }
                        }

                    Button {
                        id: saveButton
                        text: "Save Background"
                        enabled: false
                        onClicked: saveDialog.open()
                        Layout.fillWidth: true

                        Connections {
                            target: cameraController
                            function onCanSaveBackgroundChanged(canSave) {
                                saveButton.enabled = canSave
                            }
                        }
                    }

                    Button {
                        id: subtractBtn1
                        text: "Subtract Background"
                        enabled: false
                        onClicked: cameraController.request_subtract_background()

                        Connections {
                            target: cameraController
                            function onCanSubtractBackgroundChanged(captured) {
                                subtractBtn1.enabled = captured
                            }
                        }
                    }

                    Button {
                        id: resetBtn1
                        text: "Reset"
                        enabled: false
                        onClicked: cameraController.reset_background()

                        Connections {
                            target: cameraController
                            function onCanResetBackgroundChanged(captured) {
                                resetBtn1.enabled = captured
                            }
                        }
                    }
                }

                // Load Background controls
                Column {
                    spacing: 10
                    visible: loadMode.checked

                    RowLayout{
                        spacing: masterControlColumn.width*0.02
                        width: masterControlColumn.width * 0.9

                        Button {
                            id: openBackgroundButton
                            text: "Load Background"
                            onClicked: openBackgroundDialog.open()
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            height: openBackgroundButton.height
                            radius: 4
                            border.color: "#888"
                            color: "#f5f5f5"

                            Text {
                                anchors.fill: parent
                                anchors.margins: 4
                                verticalAlignment: Text.AlignVCenter

                                text: backgroundLoadPath
                                elide: Text.ElideLeft     // Clip path nicely
                                clip: true
                            }
                        }
                    }

                    RowLayout{
                        spacing: masterControlColumn.width*0.01
                        width: masterControlColumn.width * 0.4

                        Button {
                            id: subtractBtn2
                            text: "Subtract Background"
                            enabled: false
                            onClicked: cameraController.request_subtract_background()

                        Connections {
                            target: cameraController
                            function onCanSubtractBackgroundChanged(captured) {
                            subtractBtn2.enabled = captured
                            }
                        }
                    }

                        Button{
                            id: resetBtn2
                            text: "Reset"
                            enabled: false
                            onClicked: cameraController.reset_background()

                            Connections{
                                target: cameraController
                                function onCanResetBackgroundChanged(captured){
                                    resetBtn2.enabled = captured
                                }
                            }
                        }
                    }
                }
            }

            Column{
                id: snapshotColumn
                spacing: 10
                width: masterControlColumn.width

                Rectangle{
                    anchors.left: masterControlColumn.left
                    width: masterControlColumn.width * 0.9
                    height: 1
                    color: "#cdcdcd"
                }

                Text {
                    text: "Snapshot"
                    font.pixelSize: 32       // makes it large
                    //font.bold: true
                    color: "#D21404"             // red text
                    anchors.left: parent.leftPadding

                    layer.enabled: true
                    layer.effect: DropShadow {
                        color: "#000000"
                        radius: 4
                        samples: 8
                        spread: 0.2
                        verticalOffset: 1
                    }
                }

                RowLayout{
                    width: masterControlColumn.width * 0.9

                    Button {
                        id: snapshotCaptureFrameButton
                        text: "Capture Frame"
                        Layout.fillWidth: true
                        onClicked: {
                            cameraController.request_snapshot()
                            snapshotSaveFrameButton.enabled = true
                        }
                    }

                    Button {
                        id: snapshotSaveFrameButton
                        enabled: false
                        text: "Save Frame"
                        Layout.fillWidth: true
                        onClicked: snapshotFileDialog.open()
                    }

                }
            }

            Column {
                id: wavelengthCalibrationColumn
                spacing: 10
                width: masterControlColumn.width

                Rectangle{
                    anchors.left: masterControlColumn.left
                    width: masterControlColumn.width * 0.9
                    height: 1
                    color: "#cdcdcd"
                }

                Text {
                    text: "Wavelength Calibration"
                    font.pixelSize: 32       // makes it large
                    //font.bold: true
                    color: "#D21404"             // red text
                    anchors.left: parent.leftPadding

                    layer.enabled: true
                    layer.effect: DropShadow {
                        color: "#000000"
                        radius: 4
                        samples: 8
                        spread: 0.2
                        verticalOffset: 1
                    }
                }

                RowLayout{
                    Text{
                        text: "Estimated Central Wavelength (nm):"
                        font.bold: true
                        color: "#cdcdcd"
                        font.pixelSize: 14
                    }

                    SpinBox {
                        id: centralWavelengthSpinBox
                        font.pixelSize: 14
                        stepSize: 1.0
                        editable: true
                        from: 400
                        to: 1000
                        value: 700

                        onValueChanged: {
                            centralWavelengthSpinBox.value = value
                            cameraController.setCentralWavelength(value)
                        }
                    }
                }

                RowLayout {
                    spacing: 20
                    width: masterControlColumn.width * 0.9

                    Button {
                        id: nist_reference_button
                        text: "Load NIST Reference Data"
                        onClicked: nist_reference_fileDialog.open()
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        height: nist_reference_button.height
                        radius: 4
                        border.color: "#888"
                        color: "#f5f5f5"

                        Text {
                            anchors.fill: parent
                            anchors.margins: 4
                            verticalAlignment: Text.AlignVCenter

                            text: nistReferenceFilePath
                            elide: Text.ElideLeft     // Clip path nicely
                            clip: true
                        }
                    }
                }

                RowLayout {
                    spacing: 20
                    width: masterControlColumn.width * 0.9

                    Button {
                        id: ne_anchor_button
                        text: "Load 633 Centered Ne I Data"
                        onClicked: ne_anchor_fileDialog.open()
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        height: ne_anchor_button.height
                        radius: 4
                        border.color: "#888"
                        color: "#f5f5f5"

                        Text {
                            anchors.fill: parent
                            anchors.margins: 4
                            verticalAlignment: Text.AlignVCenter

                            text: neAnchorFilePath
                            elide: Text.ElideLeft     // Clip path nicely
                            clip: true
                        }
                    }
                }

                RowLayout {
                    spacing: 20
                    width: masterControlColumn.width * 0.9

                    Button {
                        id: ne_spectrum_button
                        text: "Load Ne I Data to Calibrate"
                        onClicked: ne_spectrum_fileDialog.open()
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        height: ne_spectrum_button.height
                        radius: 4
                        border.color: "#888"
                        color: "#f5f5f5"

                        Text {
                            anchors.fill: parent
                            anchors.margins: 4
                            verticalAlignment: Text.AlignVCenter

                            text: neCalibrationFilePath
                            elide: Text.ElideLeft     // Clip path nicely
                            clip: true
                        }
                    }
                }

                // Calibration controls
                RowLayout {
                    spacing: masterControlColumn.spacing * 0.1
                    width: masterControlColumn.width * 0.9

                    Button {
                        id: calibrationBtn
                        text: "Calibrate"
                        enabled: false
                        Layout.fillWidth: true
                        onClicked: cameraController.calibrateCamera()

                        Connections {
                            target: cameraController
                            function onCanCalibrate(isReady) {
                                calibrationBtn.enabled = isReady
                            }
                        }
                    }

                    Button{
                        id: saveCalibration
                        text: "Save Calibration"
                        enabled: false
                        Layout.fillWidth: true
                        onClicked: saveCalibrationDialog.open()

                        Connections {
                            target: cameraController
                            function onCanSaveCalibrationChanged(canSave) {
                                saveCalibration.enabled = canSave
                            }
                        }
                    }

                    Button {
                        id: loadCalibration
                        text: "Load Calibration File"
                        enabled: true
                        Layout.fillWidth: true
                        onClicked: loadCalibrationDialog.open()
                    }
                }

                RowLayout {
                    spacing: masterControlColumn.spacing * 0.1
                    width: masterControlColumn.width * 0.9

                    Button{
                        id: openSpectrumPlotButton
                        text: "Plot Spectrum"
                        enabled: false
                        Layout.fillWidth: true
                        onClicked: cameraController.show_spectrum_plot()

                        Connections {
                            target: cameraController
                            function onCanShowWavelengthPlotChanged(canShow) {
                                openSpectrumPlotButton.enabled = canShow
                            }
                        }
                    }
                }

                RowLayout{
                    spacing: wavelengthCalibrationColumn.width*0.02

                    Text{
                        text: "Calibrated Central Wavelength:"
                        font.bold: true
                        color: "#cdcdcd"
                        font.pixelSize: 14
                    }

                    Rectangle{
                        id: centralWavelengthBox
                        color: "#222021"
                        Layout.preferredWidth:  wavelengthCalibrationColumn.width * 0.35
                        height: 30

                        Text {
                            id: calibrationWavelengthLabel
                            anchors.horizontalCenter: centralWavelengthBox.horizontalCenter
                            anchors.verticalCenter: centralWavelengthBox.verticalCenter
                            text: "--- nm"
                            font.pixelSize: 18
                            font.bold: true
                            color: "#cdcdcd"

                            Connections {
                                target: cameraController
                                function onCentralWavelengthChanged(wavelength) {
                                    calibrationWavelengthLabel.text = wavelength.toFixed(2) + " nm"
                                }
                            }
                        }
                    }
            }

                RowLayout{
                    spacing: wavelengthCalibrationColumn.width*0.02

                    Text{
                        text: "Wavelength Range:"
                        font.bold: true
                        color: "#cdcdcd"
                        font.pixelSize: 14
                    }

                    Rectangle{
                        id: wavelengthRangeBox
                        color: "#222021"
                        Layout.preferredWidth: wavelengthCalibrationColumn.width * 0.5
                        height: 30

                        Text {
                            id: wavelengthRangeLabel
                            anchors.horizontalCenter: wavelengthRangeBox.horizontalCenter
                            anchors.verticalCenter: wavelengthRangeBox.verticalCenter
                            text: "--- nm"
                            font.pixelSize: 18
                            font.bold: true
                            color: "#cdcdcd"

                            Connections {
                                target: cameraController
                                function onWavelengthRangeChanged(minWavelength, maxWavelength) {
                                    wavelengthRangeLabel.text = minWavelength.toFixed(2) + " - " + maxWavelength.toFixed(2) + " nm"
                                }
                            }
                        }
                    }
            }

                RowLayout{
                    spacing: wavelengthCalibrationColumn.width*0.02

                    Text{
                        text: "Max Residual:"
                        font.bold: true
                        color: "#cdcdcd"
                        font.pixelSize: 14
                    }

                    Rectangle{
                        id: maxResidualBox
                        color: "#222021"
                        Layout.preferredWidth: wavelengthCalibrationColumn.width * 0.45
                        height: 30

                        Text {
                            id: maxResidualLabel
                            anchors.horizontalCenter: maxResidualBox.horizontalCenter
                            anchors.verticalCenter: maxResidualBox.verticalCenter
                            text: "--- nm"
                            font.pixelSize: 18
                            font.bold: true
                            color: "#cdcdcd"

                            Connections {
                                target: cameraController
                                function onResidualCalculated(residual) {
                                    maxResidualLabel.text = residual.toFixed(4) + " nm"
                                }
                                function onNotApplicableResidual(){
                                    maxResidualLabel.text = "N/A"
                                }
                            }
                        }
                    }
            }

                RowLayout{
                    spacing: wavelengthCalibrationColumn.width*0.05

                    Text{
                        text: "Calibration Status:"
                        font.bold: true
                        color: "#cdcdcd"
                        font.pixelSize: 14
                    }

                    Rectangle{
                        id: calibrationStatusBox
                        color: "#222021"
                        Layout.preferredWidth: wavelengthCalibrationColumn.width * 0.5
                        height: 30

                        Text {
                            id: calibrationStatusLabel
                            anchors.horizontalCenter: calibrationStatusBox.horizontalCenter
                            anchors.verticalCenter: calibrationStatusBox.verticalCenter
                            text: "Uncalibrated"
                            font.pixelSize: 18
                            font.bold: true
                            color: "#808080"

                            Connections {
                                target: cameraController
                                function onIsCalibrated(status) {
                                    if (status === "Uncalibrated") {
                                        calibrationStatusLabel.text = "Uncalibrated"
                                        calibrationStatusLabel.color = "#FF0000"
                                    }
                                    else if(status === "Calibrated") {
                                        calibrationStatusLabel.text = "Calibrated"
                                        calibrationStatusLabel.color = "#00FF00"
                                    }
                                    else if (status === "Loaded"){
                                        calibrationStatusLabel.text = "Loaded"
                                        calibrationStatusLabel.color = "#87CEEB"
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Rectangle{
                color: "#2e2f30"
                height: 1
                width: masterControlColumn.width
            }
        }


            FileDialog{
                id: saveFramesForStandardCaptureFileDialog
                title: "Save Data To"
                fileMode: FileDialog.SaveFile
                folder: defaultPaths["data"]
                nameFilters: ["HDF5 File (*.hdf5)"]

                onAccepted: {
                    standardCaptureSavePath = saveFramesForStandardCaptureFileDialog.file.toString()
                    cameraController.setStandardCaptureSavePath(standardCaptureSavePath)
                    captureFrameBtn.enabled = true
                    }
            }

            FileDialog{
                id: saveFramesForLiveCaptureFileDialog
                title: "Save Data To"
                fileMode: FileDialog.SaveFile
                folder: defaultPaths["data"]
                nameFilters: ["HDF5 File (*.hdf5)"]

                onAccepted: {
                    liveCaptureSavePath = saveFramesForLiveCaptureFileDialog.file.toString()
                    cameraController.setLiveCaptureSavePath(liveCaptureSavePath)
                    startLiveCaptureBtn.enabled = true
                    }
            }

            FileDialog {
                id: saveDialog
                title: "Save as"
                fileMode: FileDialog.SaveFile
                folder: defaultPaths["background"]
                nameFilters: ["Numpy Matrix (*.npy)", "TIFF (*.tiff)", "TIF (*.tif)", "PNG (*.png)"]

                onAccepted: {
                    cameraController.request_save_background(saveDialog.file)
                    saveButton.enabled = false
                }
            }

            FileDialog {
                id: openBackgroundDialog
                title: "Select a file"
                fileMode: FileDialog.OpenFile
                folder: defaultPaths["background"]
                nameFilters: ["Numpy Matrix (*.npy)", "HDF5 File (*.hdf5)", "H5 File (*.h5)", "TIFF (*.tiff)", "TIF (*.tif)"]

                onAccepted: {
                    backgroundLoadPath = openBackgroundDialog.file.toString()
                    cameraController.load_background_requested(backgroundLoadPath)

                    subtractBtn1.enabled = false
                    resetBtn1.enabled = false
                    saveButton.enabled = false
                }
            }

            FileDialog{
                id: snapshotFileDialog
                title: "Save Frame To"
                fileMode: FileDialog.SaveFile
                folder: defaultPaths["snapshots"]
                nameFilters: ["Numpy Matrix (*.npy)", "TIFF (*.tiff)", "TIF (*.tif)", "PNG (*.png)"]

                onAccepted: {
                    cameraController.request_save_snapshot(snapshotFileDialog.file.toString())
                    snapshotSaveFrameButton.enabled = false
                    }
            }

            FileDialog {
                id: nist_reference_fileDialog
                title: "Select a file"
                fileMode: FileDialog.OpenFile
                folder: defaultPaths["calibration"]
                nameFilters: ["CSV (*.csv)"]

                onAccepted: {
                    nistReferenceFilePath = nist_reference_fileDialog.file.toString()
                    cameraController.setNistReferencePath(nistReferenceFilePath)
                }
            }

            FileDialog {
                id: ne_anchor_fileDialog
                title: "Select a file"
                fileMode: FileDialog.OpenFile
                folder: defaultPaths["calibration"]
                nameFilters: ["Numpy Matrix (*.npy)", "TIFF (*.tiff)", "TIF (*.tif)"]

                onAccepted: {
                    neAnchorFilePath = ne_anchor_fileDialog.file.toString()
                    cameraController.setNe633AnchorPath(neAnchorFilePath)
                }
            }

            FileDialog {
                id: ne_spectrum_fileDialog
                title: "Select a file"
                fileMode: FileDialog.OpenFile
                folder: defaultPaths["calibration"]
                nameFilters: ["Numpy Matrix (*.npy)", "TIFF (*.tiff)", "TIF (*.tif)"]

                onAccepted: {
                    neCalibrationFilePath = ne_spectrum_fileDialog.file.toString()
                    cameraController.setNeCalibrationPath(neCalibrationFilePath)
                }
            }

              FileDialog {
                  id: saveCalibrationDialog
                  title: "Save as"
                  fileMode: FileDialog.SaveFile
                  folder: defaultPaths["calibration"]
                  nameFilters: ["CSV File (*.csv)"]

                  onAccepted: {
                      console.log("Saving to:", saveCalibrationDialog.file)
                      cameraController.saveCalibrationFile(saveCalibrationDialog.file)
                  }
              }

              FileDialog {
                  id: loadCalibrationDialog
                  title: "Select a file"
                  fileMode: FileDialog.OpenFile
                  folder: defaultPaths["calibration"]
                  nameFilters: ["CSV File (*.csv)", "HDF5 File (*.hdf5)", "H5 File (*.h5)"]

                  onAccepted: {
                      cameraController.loadCalibrationFile( loadCalibrationDialog.file)
                      saveCalibration.enabled = false
                  }
              }

              MessageDialog {
                  id: errorDialog
                  title: "Error"
                  text: ""

                  Connections {
                      target: cameraController
                      function onErrorOccurred(message) {
                          errorDialog.text = message
                          errorDialog.open()
                      }
                  }

              }
            }
    }
}
