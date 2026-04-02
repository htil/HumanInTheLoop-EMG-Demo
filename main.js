const { useEffect, useMemo, useRef, useState } = React;

const LABELS = ["relax", "lightGrip", "tightGrip"];
const LABEL_DISPLAY = {
  relax: "Relax",
  lightGrip: "Light Grip",
  tightGrip: "Tight Grip"
};

const ACTION_MAP = {
  relax: "Freeze / Stay",
  lightGrip: "Move Up",
  tightGrip: "Strong Action"
};

const SAMPLE_WINDOW = 40;
const MAX_RAW_BUFFER = 320;
const MAX_DRAW_BUFFER = 200;

const SELECTED_CHANNEL_INDEX = 0; // Channel 1
const BASELINE_WINDOW = 100;
const ENVELOPE_WINDOW = 12;

// Ganglion data scale from counts -> volts is documented by OpenBCI.
// For display/processing here, convert to microvolts.
const GANGLION_VOLTS_PER_COUNT = 1.2 / (8388607 * 1.5 * 51);
const GANGLION_MICROVOLTS_PER_COUNT = GANGLION_VOLTS_PER_COUNT * 1e6;

// OpenBCI Ganglion BLE UUIDs
const GANGLION_SERVICE = 0xfe84;
const GANGLION_RECEIVE_UUID = "2d30c082-f39f-4ce6-923f-3484ea480596";
const GANGLION_SEND_UUID = "2d30c083-f39f-4ce6-923f-3484ea480596";

function formatNum(value, digits = 4) {
  return Number.isFinite(value) ? value.toFixed(digits) : "--";
}

function rms(signal) {
  if (!signal.length) return 0;
  let sum = 0;
  for (const x of signal) sum += x * x;
  return Math.sqrt(sum / signal.length);
}

function mav(signal) {
  if (!signal.length) return 0;
  let sum = 0;
  for (const x of signal) sum += Math.abs(x);
  return sum / signal.length;
}

function waveformLength(signal) {
  if (signal.length < 2) return 0;
  let sum = 0;
  for (let i = 1; i < signal.length; i++) {
    sum += Math.abs(signal[i] - signal[i - 1]);
  }
  return sum;
}

function meanCenter(signal) {
  if (!signal.length) return signal;
  const avg = signal.reduce((a, b) => a + b, 0) / signal.length;
  return signal.map(x => x - avg);
}

function rectify(signal) {
  return signal.map(x => Math.abs(x));
}

function getFeatureVector(windowData) {
  const centered = meanCenter(windowData);
  const rectified = rectify(centered);
  return [
    rms(rectified),
    mav(rectified),
    waveformLength(rectified)
  ];
}

function normalizeFeatures(features) {
  if (!features || features.length === 0) return features;
  // Normalize to 0-1 range based on typical EMG feature ranges
  const ranges = [
    { min: 0, max: 200 },   // RMS: 0-200 µV typical
    { min: 0, max: 100 },   // MAV: 0-100 µV typical
    { min: 0, max: 500 }    // WL: 0-500 typical
  ];
  return features.map((val, i) => {
    const range = ranges[i] || { min: 0, max: 1 };
    return Math.max(0, Math.min(1, (val - range.min) / (range.max - range.min)));
  });
}

function oneHot(index, length) {
  const arr = new Array(length).fill(0);
  arr[index] = 1;
  return arr;
}

function labelToIndex(label) {
  return LABELS.indexOf(label);
}

function indexToLabel(index) {
  return LABELS[index] ?? "relax";
}

function getStatusClass(kind) {
  if (kind === "good") return "status-good";
  if (kind === "bad") return "status-bad";
  if (kind === "accent") return "status-accent";
  return "status-warn";
}

function signed24BitToInt(b0, b1, b2) {
  let value = (b0 << 16) | (b1 << 8) | b2;
  if (value & 0x800000) {
    value |= 0xff000000;
  }
  return value | 0;
}

function extractUnsignedBits(bytes, startBit, bitLength) {
  let out = 0;

  for (let i = 0; i < bitLength; i++) {
    const absoluteBit = startBit + i;
    const byteIndex = Math.floor(absoluteBit / 8);
    const bitIndexInByte = 7 - (absoluteBit % 8);
    const bit = (bytes[byteIndex] >> bitIndexInByte) & 1;
    out = (out << 1) | bit;
  }

  return out;
}

// Ganglion compressed numbers are special because the sign is stored in bit0/LSB,
// not in the MSB. This follows the OpenBCI Ganglion docs.
function convertGanglion18BitToInt(raw18) {
  const signBit = raw18 & 0x01;
  const prefix = signBit ? 0b11111111111111 : 0;
  return (prefix << 18) | raw18;
}

function convertGanglion19BitToInt(raw19) {
  const signBit = raw19 & 0x01;
  const prefix = signBit ? 0b1111111111111 : 0;
  return (prefix << 19) | raw19;
}

// Returns an array of frames. Each frame is [ch1, ch2, ch3, ch4] in microvolts.
function parseGanglionPacket(dataView, decompressionState) {
  const bytes = new Uint8Array(
    dataView.buffer,
    dataView.byteOffset,
    dataView.byteLength
  );

  if (bytes.length < 20) {
    return { packetId: null, frames: [], kind: "short" };
  }

  const packetId = bytes[0];
  const payload = bytes.slice(1, 20);

  // Raw uncompressed packet: packet id 0, 4 channels x 24-bit signed = 12 bytes.
  if (packetId === 0) {
    const counts = [
      signed24BitToInt(payload[0], payload[1], payload[2]),
      signed24BitToInt(payload[3], payload[4], payload[5]),
      signed24BitToInt(payload[6], payload[7], payload[8]),
      signed24BitToInt(payload[9], payload[10], payload[11])
    ];

    decompressionState.lastCounts = [...counts];

    const frameMicrovolts = counts.map(
      count => count * GANGLION_MICROVOLTS_PER_COUNT
    );

    return {
      packetId,
      frames: [frameMicrovolts],
      kind: "raw"
    };
  }

  // 18-bit compression with accelerometer
  if (packetId >= 1 && packetId <= 100) {
    if (!decompressionState.lastCounts) {
      return { packetId, frames: [], kind: "18bit-no-seed" };
    }

    const packed = [];
    for (let i = 0; i < 8; i++) {
      const raw18 = extractUnsignedBits(payload, i * 18, 18);
      packed.push(convertGanglion18BitToInt(raw18));
    }

    const delta1 = packed.slice(0, 4);
    const delta2 = packed.slice(4, 8);

    const prev = decompressionState.lastCounts;
    const sample1 = prev.map((oldVal, ch) => oldVal - delta1[ch]);
    const sample2 = sample1.map((oldVal, ch) => oldVal - delta2[ch]);

    decompressionState.lastCounts = [...sample2];

    return {
      packetId,
      frames: [
        sample1.map(count => count * GANGLION_MICROVOLTS_PER_COUNT),
        sample2.map(count => count * GANGLION_MICROVOLTS_PER_COUNT)
      ],
      kind: "18bit"
    };
  }

  // 19-bit compression without accelerometer
  if (packetId >= 101 && packetId <= 200) {
    if (!decompressionState.lastCounts) {
      return { packetId, frames: [], kind: "19bit-no-seed" };
    }

    const packed = [];
    for (let i = 0; i < 8; i++) {
      const raw19 = extractUnsignedBits(payload, i * 19, 19);
      packed.push(convertGanglion19BitToInt(raw19));
    }

    const delta1 = packed.slice(0, 4);
    const delta2 = packed.slice(4, 8);

    const prev = decompressionState.lastCounts;
    const sample1 = prev.map((oldVal, ch) => oldVal - delta1[ch]);
    const sample2 = sample1.map((oldVal, ch) => oldVal - delta2[ch]);

    decompressionState.lastCounts = [...sample2];

    return {
      packetId,
      frames: [
        sample1.map(count => count * GANGLION_MICROVOLTS_PER_COUNT),
        sample2.map(count => count * GANGLION_MICROVOLTS_PER_COUNT)
      ],
      kind: "19bit"
    };
  }

  // Ignore impedance / ascii / unknown packets for now
  return {
    packetId,
    frames: [],
    kind: "other"
  };
}

function App() {
  const [connectionState, setConnectionState] = useState("Demo Mode Ready");
  const [connectionKind, setConnectionKind] = useState("accent");

  const [rawBuffer, setRawBuffer] = useState([]);
  const [processedBuffer, setProcessedBuffer] = useState([]);
  const [latestRaw, setLatestRaw] = useState(null);
  const [latestProcessed, setLatestProcessed] = useState(null);

  const [dataset, setDataset] = useState([]);
  const [model, setModel] = useState(null);
  const [modelStatus, setModelStatus] = useState("Not trained");
  const [modelStatusKind, setModelStatusKind] = useState("warn");
  const [trainAccuracy, setTrainAccuracy] = useState(null);

  const [prediction, setPrediction] = useState(null);
  const [predictionConfidence, setPredictionConfidence] = useState(null);
  const [predictionAction, setPredictionAction] = useState("--");
  const [predictionStatus, setPredictionStatus] = useState("Waiting");
  const [predictionStatusKind, setPredictionStatusKind] = useState("warn");

  const [correctionCount, setCorrectionCount] = useState(0);
  const [pendingCorrection, setPendingCorrection] = useState(null);
  const [selectedCorrectionLabel, setSelectedCorrectionLabel] = useState("relax");
  const [samplesPerClick, setSamplesPerClick] = useState(5);

  const [currentDemoPhase, setCurrentDemoPhase] = useState("Stopped");
  const [showElectrodeInfo, setShowElectrodeInfo] = useState(false);
  const [showRawChannel, setShowRawChannel] = useState(true);

  const canvasRef = useRef(null);
  const demoIntervalRef = useRef(null);
  const lastPredictionRef = useRef(null);
  const trainingRef = useRef(false);

  const baselineWindowRef = useRef([]);
  const envelopeWindowRef = useRef([]);
  const ganglionStateRef = useRef({ lastCounts: null });

  const bleDeviceRef = useRef(null);
  const bleServerRef = useRef(null);
  const bleReceiveCharRef = useRef(null);
  const bleSendCharRef = useRef(null);
  const bleNotificationHandlerRef = useRef(null);

  const windowReady = processedBuffer.length >= SAMPLE_WINDOW;
  const currentWindow = windowReady
    ? processedBuffer.slice(processedBuffer.length - SAMPLE_WINDOW)
    : null;
  const features = currentWindow ? getFeatureVector(currentWindow) : null;

  const counts = useMemo(() => {
    const out = { relax: 0, lightGrip: 0, tightGrip: 0 };
    for (const row of dataset) out[row.label] += 1;
    return out;
  }, [dataset]);

  function addLog(msg) {
    console.log(msg);
  }

  function resetSignalState() {
    baselineWindowRef.current = [];
    envelopeWindowRef.current = [];
    ganglionStateRef.current = { lastCounts: null };

    setRawBuffer([]);
    setProcessedBuffer([]);
    setLatestRaw(null);
    setLatestProcessed(null);
  }

  function processEmgSample(sampleValue) {
    const numeric = Number(sampleValue);
    if (!Number.isFinite(numeric)) return null;

    const baselineWindow = baselineWindowRef.current;
    baselineWindow.push(numeric);
    if (baselineWindow.length > BASELINE_WINDOW) baselineWindow.shift();

    let baseline = 0;
    for (let i = 0; i < baselineWindow.length; i++) {
      baseline += baselineWindow[i];
    }
    baseline /= baselineWindow.length || 1;

    const centered = numeric - baseline;
    const rectified = Math.abs(centered);

    const envelopeWindow = envelopeWindowRef.current;
    envelopeWindow.push(rectified);
    if (envelopeWindow.length > ENVELOPE_WINDOW) envelopeWindow.shift();

    let smoothed = 0;
    for (let i = 0; i < envelopeWindow.length; i++) {
      smoothed += envelopeWindow[i];
    }
    smoothed /= envelopeWindow.length || 1;

    return {
      raw: numeric,
      processed: smoothed
    };
  }

  function handleIncomingSample(sampleValue) {
    const result = processEmgSample(sampleValue);
    if (!result) return;

    setLatestRaw(result.raw);
    setLatestProcessed(result.processed);

    setRawBuffer(prev => {
      const next = [...prev, result.raw];
      if (next.length > MAX_RAW_BUFFER) next.shift();
      return next;
    });

    setProcessedBuffer(prev => {
      const next = [...prev, result.processed];
      if (next.length > MAX_RAW_BUFFER) next.shift();
      return next;
    });
  }

  function startDemoSignal() {
    stopDemoSignal();
    resetSignalState();
    setModel(null);
    setModelStatus("Not trained");
    setModelStatusKind("warn");
    setTrainAccuracy(null);

    let t = 0;

    demoIntervalRef.current = setInterval(() => {
      const phase = t % 360;

      // Demo signal: realistic EMG-like waveform in microvolt range (10-100 µV typical)
      let baseSignal = 30 * Math.sin(t * 0.08) + 15 * Math.sin(t * 0.25);
      let sample = baseSignal + 10 * (Math.random() - 0.5);
      let phaseLabel = "Relax";

      if (phase > 60 && phase < 130) {
        sample = baseSignal * 2.5 + 15 * (Math.random() - 0.5);
        phaseLabel = "Light Grip";
      } else if (phase > 200 && phase < 270) {
        sample = baseSignal * 4.5 + 20 * (Math.random() - 0.5);
        phaseLabel = "Tight Grip";
      }

      setCurrentDemoPhase(phaseLabel);
      handleIncomingSample(sample);
      t += 1;
    }, 20);

    setConnectionState("Demo Signal Running");
    setConnectionKind("accent");
    addLog("Demo signal started.");
  }

  function stopDemoSignal() {
    if (demoIntervalRef.current) {
      clearInterval(demoIntervalRef.current);
      demoIntervalRef.current = null;
      addLog("Demo signal stopped.");
    }
    setCurrentDemoPhase("Stopped");
  }

  async function writeGanglionCommand(commandChar) {
    const sendChar = bleSendCharRef.current;
    if (!sendChar) {
      throw new Error("Ganglion send characteristic is not ready.");
    }

    const encoder = new TextEncoder();
    const payload = encoder.encode(commandChar);

    if (sendChar.properties.write) {
      await sendChar.writeValueWithResponse(payload);
    } else if (sendChar.properties.writeWithoutResponse) {
      await sendChar.writeValueWithoutResponse(payload);
    } else {
      throw new Error("Ganglion send characteristic is not writable.");
    }
  }

  async function disconnectGanglionBLE() {
    try {
      const receiveChar = bleReceiveCharRef.current;
      const device = bleDeviceRef.current;
      const handler = bleNotificationHandlerRef.current;

      try {
        if (receiveChar && handler) {
          receiveChar.removeEventListener("characteristicvaluechanged", handler);
        }
      } catch (err) {
        console.warn("Could not remove handler:", err);
      }

      try {
        if (receiveChar) {
          await receiveChar.stopNotifications();
          addLog("Notifications stopped.");
        }
      } catch (err) {
        console.warn("Could not stop notifications:", err);
      }

      try {
        if (bleSendCharRef.current) {
          await writeGanglionCommand("s");
          addLog("Sent stop stream command: s");
        }
      } catch (err) {
        console.warn("Could not send stop command:", err);
      }

      if (device?.gatt?.connected) {
        device.gatt.disconnect();
      }
    } finally {
      bleDeviceRef.current = null;
      bleServerRef.current = null;
      bleReceiveCharRef.current = null;
      bleSendCharRef.current = null;
      bleNotificationHandlerRef.current = null;

      setConnectionState("Disconnected");
      setConnectionKind("warn");
    }
  }

  async function connectGanglionBLE() {
    stopDemoSignal();
    resetSignalState();
    setModel(null);
    setModelStatus("Not trained");
    setModelStatusKind("warn");
    setTrainAccuracy(null);

    try {
      if (!navigator.bluetooth) {
        throw new Error("Web Bluetooth is not available in this browser.");
      }

      if (bleDeviceRef.current?.gatt?.connected) {
        await disconnectGanglionBLE();
      }

      setConnectionState("Requesting Ganglion...");
      setConnectionKind("warn");
      addLog("Opening Bluetooth device picker for Ganglion...");

      const device = await navigator.bluetooth.requestDevice({
        filters: [{ namePrefix: "Ganglion" }],
        optionalServices: [GANGLION_SERVICE]
      });

      bleDeviceRef.current = device;

      device.addEventListener("gattserverdisconnected", () => {
        addLog("Ganglion disconnected.");
        setConnectionState("Ganglion Disconnected");
        setConnectionKind("bad");
        bleServerRef.current = null;
        bleReceiveCharRef.current = null;
        bleSendCharRef.current = null;
        bleNotificationHandlerRef.current = null;
      });

      setConnectionState("Connecting...");
      addLog(`Selected device: ${device.name || "Ganglion"}`);

      const server = await device.gatt.connect();
      bleServerRef.current = server;

      const service = await server.getPrimaryService(GANGLION_SERVICE);
      const receiveChar = await service.getCharacteristic(GANGLION_RECEIVE_UUID);
      const sendChar = await service.getCharacteristic(GANGLION_SEND_UUID);

      bleReceiveCharRef.current = receiveChar;
      bleSendCharRef.current = sendChar;

      const handleNotification = (event) => {
        const parsed = parseGanglionPacket(
          event.target.value,
          ganglionStateRef.current
        );

        if (!parsed.frames.length) return;

        for (const frame of parsed.frames) {
          if (frame.length <= SELECTED_CHANNEL_INDEX) continue;

          const sample = frame[SELECTED_CHANNEL_INDEX];
          handleIncomingSample(sample);
        }
      };

      bleNotificationHandlerRef.current = handleNotification;

      receiveChar.addEventListener("characteristicvaluechanged", handleNotification);
      await receiveChar.startNotifications();
      addLog("Notifications started.");

      await writeGanglionCommand("b");
      addLog("Sent start stream command: b");

      setConnectionState(`Ganglion BLE Streaming (Ch ${SELECTED_CHANNEL_INDEX + 1})`);
      setConnectionKind("good");
    } catch (err) {
      console.error(err);
      addLog(`BLE connect failed: ${err.message}`);
      setConnectionState("BLE Failed");
      setConnectionKind("bad");
    }
  }

  function collectSamples(label) {
    if (!features) {
      addLog("Cannot collect sample: feature window not ready.");
      return;
    }

    const rows = [];
    for (let i = 0; i < samplesPerClick; i++) {
      rows.push({
        features: [...features],
        label
      });
    }

    setDataset(prev => [...prev, ...rows]);
    addLog(`Collected ${samplesPerClick} sample(s) for ${LABEL_DISPLAY[label]}.`);
  }

  function clearDataset() {
    setDataset([]);
    setCorrectionCount(0);
    setPendingCorrection(null);
    addLog("Dataset cleared.");
  }

  function exportDataset() {
    const blob = new Blob([JSON.stringify(dataset, null, 2)], {
      type: "application/json"
    });

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "emg_dataset.json";
    a.click();
    URL.revokeObjectURL(url);

    addLog("Dataset exported.");
  }

  async function trainModel() {
    if (trainingRef.current) {
      addLog("Already training. Please wait for it to complete.");
      return;
    }

    if (dataset.length < 6) {
      addLog("Need more samples before training.");
      return;
    }

    const uniqueLabels = new Set(dataset.map(x => x.label));
    if (uniqueLabels.size < 2) {
      addLog("Need at least 2 classes before training.");
      return;
    }

    trainingRef.current = true;
    setModelStatus("Training...");
    setModelStatusKind("warn");
    setTrainAccuracy(null);
    addLog("Starting model training...");

    let xTensor, yTensor, nextModel;

    try {
      const xs = dataset.map(row => normalizeFeatures(row.features));
      const ys = dataset.map(row => oneHot(labelToIndex(row.label), LABELS.length));

      xTensor = tf.tensor2d(xs);
      yTensor = tf.tensor2d(ys);

      if (model) {
        model.dispose();
      }

      nextModel = tf.sequential();
      nextModel.add(
        tf.layers.dense({
          units: 16,
          activation: "relu",
          inputShape: [3],
          kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
        })
      );
      nextModel.add(tf.layers.dropout({ rate: 0.3 }));
      nextModel.add(
        tf.layers.dense({
          units: 16,
          activation: "relu",
          kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
        })
      );
      nextModel.add(tf.layers.dropout({ rate: 0.3 }));
      nextModel.add(
        tf.layers.dense({
          units: LABELS.length,
          activation: "softmax"
        })
      );

      nextModel.compile({
        optimizer: tf.train.adam(0.001),
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
      });

      let finalAcc = 0;
      const history = await nextModel.fit(xTensor, yTensor, {
        epochs: 15,
        batchSize: 2,
        shuffle: true,
        verbose: 0,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            finalAcc = logs.acc || logs.accuracy || 0;
            setModelStatus(`Training... Epoch ${epoch + 1}/15`);
          }
        }
      });

      setModel(nextModel);
      setModelStatus("Trained");
      setModelStatusKind("good");
      setTrainAccuracy(finalAcc);
      addLog(`Model trained with ${dataset.length} samples (${(finalAcc * 100).toFixed(1)}% accuracy).`);
    } catch (err) {
      console.error("Training error:", err);
      addLog("Training failed: " + err.message);
      setModelStatus("Training failed");
      setModelStatusKind("bad");
      if (nextModel) nextModel.dispose();
    } finally {
      if (xTensor) xTensor.dispose();
      if (yTensor) yTensor.dispose();
      trainingRef.current = false;
    }
  }

  function confirmPredictionRight() {
    const lastPrediction = lastPredictionRef.current;
    if (!lastPrediction) {
      addLog("No prediction available to confirm.");
      return;
    }

    setDataset(prev => [
      ...prev,
      {
        features: [...lastPrediction.features],
        label: lastPrediction.predictedLabel
      }
    ]);

    addLog(`Confirmed prediction as correct: ${LABEL_DISPLAY[lastPrediction.predictedLabel]}.`);
  }

  function markPredictionWrong() {
    const lastPrediction = lastPredictionRef.current;
    if (!lastPrediction) {
      addLog("No prediction available to correct.");
      return;
    }

    setPendingCorrection(lastPrediction);
    addLog(
      `Prediction marked wrong. Select the correct class and apply correction. ` +
        `Predicted: ${LABEL_DISPLAY[lastPrediction.predictedLabel]}.`
    );
  }

  function applyCorrection() {
    if (!pendingCorrection) {
      addLog("No pending correction to apply.");
      return;
    }

    setDataset(prev => [
      ...prev,
      {
        features: [...pendingCorrection.features],
        label: selectedCorrectionLabel
      }
    ]);

    setCorrectionCount(prev => prev + 1);
    addLog(
      `Correction added. Predicted ${LABEL_DISPLAY[pendingCorrection.predictedLabel]}, ` +
        `correct label is ${LABEL_DISPLAY[selectedCorrectionLabel]}.`
    );
    setPendingCorrection(null);
  }

  useEffect(() => {
    if (!model || !features) {
      setPredictionStatus("Waiting");
      setPredictionStatusKind("warn");
      return;
    }

    tf.tidy(() => {
      const normalizedFeatures = normalizeFeatures(features);
      const input = tf.tensor2d([normalizedFeatures]);
      const output = model.predict(input);
      const probs = output.dataSync();

      let bestIndex = 0;
      let bestProb = probs[0];
      let secondProb = -1;

      for (let i = 1; i < probs.length; i++) {
        if (probs[i] > bestProb) {
          secondProb = bestProb;
          bestProb = probs[i];
          bestIndex = i;
        } else if (probs[i] > secondProb) {
          secondProb = probs[i];
        }
      }

      const predictedLabel = indexToLabel(bestIndex);
      
      // Use margin-based confidence: how much better is the top prediction vs second?
      const margin = bestProb - secondProb;
      const confidence = Math.min(0.95, margin * 0.7 + bestProb * 0.3);

      setPrediction(predictedLabel);
      setPredictionConfidence(confidence);
      setPredictionAction(ACTION_MAP[predictedLabel]);
      setPredictionStatus("Running");
      setPredictionStatusKind(confidence > 0.6 ? "good" : "warn");

      lastPredictionRef.current = {
        features: [...features],
        predictedLabel,
        confidence
      };
    });
  }, [features, model]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "#dddddd";
    ctx.lineWidth = 1;

    for (let i = 1; i <= 4; i++) {
      const y = (height / 5) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    const rawDrawData = rawBuffer.slice(-MAX_DRAW_BUFFER);
    const processedDrawData = processedBuffer.slice(-MAX_DRAW_BUFFER);
    const longestLength = Math.max(rawDrawData.length, processedDrawData.length);
    if (longestLength < 2) return;

    const combined = [...rawDrawData, ...processedDrawData];
    let min = Math.min(...combined);
    let max = Math.max(...combined);

    if (min === max) {
      min -= 1;
      max += 1;
    }

    const drawSeries = (data, strokeStyle, lineWidth) => {
      if (data.length < 2) return;
      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();

      data.forEach((value, i) => {
        const x = (i / (data.length - 1)) * width;
        const y = height - ((value - min) / (max - min)) * height;

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });

      ctx.stroke();
    };

    if (showRawChannel) {
      drawSeries(rawDrawData, "#9e1b32", 1.4);
    }
    drawSeries(processedDrawData, "#111111", 2.2);
  }, [rawBuffer, processedBuffer, showRawChannel]);

  useEffect(() => {
    return () => {
      stopDemoSignal();
      if (bleDeviceRef.current?.gatt?.connected) {
        bleDeviceRef.current.gatt.disconnect();
      }
    };
  }, []);

  return (
    <div className="app-shell">
      <div className="hero">
        <h1>Human In The Loop EMG Project</h1>
        <p>A early demo of human in the loop machine learnign with EMG data</p>
      </div>

      <div className="main-dashboard">
        <div className="left-panel">
          <div className="left-top">
            <section className="card">
              <h2>Live Stream Control</h2>

              <div className="row">
                <button onClick={startDemoSignal}>Use Demo Signal</button>
                <button onClick={stopDemoSignal} className="secondary">Stop Demo</button>
                <button onClick={connectGanglionBLE} className="secondary">Connect Ganglion BLE</button>
                <button onClick={() => writeGanglionCommand("b")} className="secondary">Start Stream</button>
                <button onClick={() => writeGanglionCommand("s")} className="secondary">Stop Stream</button>
                <button onClick={disconnectGanglionBLE} className="secondary">Disconnect BLE</button>
                <button
                  onClick={() => setShowElectrodeInfo(!showElectrodeInfo)}
                  className="secondary"
                >
                  ℹ️ Electrode Setup
                </button>
                <button
                  onClick={() => setShowRawChannel(!showRawChannel)}
                  className="secondary"
                >
                  {showRawChannel ? "✓" : "○"} Raw Channel
                </button>
              </div>

              {showElectrodeInfo && (
                <div
                  style={{
                    marginTop: "12px",
                    padding: "12px",
                    background: "#fffbf0",
                    border: "2px solid var(--accent)",
                    borderRadius: "10px",
                    fontSize: "13px",
                    lineHeight: "1.6"
                  }}
                >
                  <strong style={{ color: "var(--accent)", fontSize: "14px" }}>
                    📌 Electrode Placement:
                  </strong>
                  <div style={{ marginTop: "8px" }}>
                    <div>
                      🟫 <strong>Black (Reference):</strong> Back of hand between middle and pinky fingers
                    </div>
                    <div style={{ marginTop: "6px" }}>
                      🔴 <strong>Red (Channel 1):</strong> 4 inches below brown on forearm
                    </div>
                    <div style={{ marginTop: "6px" }}>
                      🟫 <strong>Brown (Channel 2):</strong> Middle of forearm, palm facing up
                    </div>
                    <div style={{ marginTop: "10px", color: "var(--muted)", fontSize: "12px" }}>
                      ✓ This setup picks up hand/forearm muscle activation
                    </div>
                  </div>
                </div>
              )}

              <div className="metric-grid">
                <div className="metric-box">
                  <div className="metric-label">Connection</div>
                  <div className={`metric-value ${getStatusClass(connectionKind)}`}>
                    {connectionState}
                  </div>
                </div>

                <div className="metric-box">
                  <div className="metric-label">Current Demo Phase</div>
                  <div className="metric-value">{currentDemoPhase}</div>
                </div>

                <div className="metric-box">
                  <div className="metric-label">Latest Raw Sample (µV)</div>
                  <div className="metric-value">{formatNum(latestRaw, 3)}</div>
                </div>

                <div className="metric-box">
                  <div className="metric-label">Latest Processed (µV)</div>
                  <div className="metric-value">{formatNum(latestProcessed, 3)}</div>
                </div>

                <div className="metric-box">
                  <div className="metric-label">Buffer Size</div>
                  <div className="metric-value">{processedBuffer.length}</div>
                </div>

                <div className="metric-box">
                  <div className="metric-label">Window Ready</div>
                  <div className={`metric-value ${windowReady ? "status-good" : "status-warn"}`}>
                    {windowReady ? "Yes" : "No"}
                  </div>
                </div>

                <div className="metric-box">
                  <div className="metric-label">RMS</div>
                  <div className="metric-value">{features ? formatNum(features[0], 3) : "--"}</div>
                </div>

                <div className="metric-box">
                  <div className="metric-label">MAV</div>
                  <div className="metric-value">{features ? formatNum(features[1], 3) : "--"}</div>
                </div>

                <div className="metric-box">
                  <div className="metric-label">Waveform Length</div>
                  <div className="metric-value">{features ? formatNum(features[2], 3) : "--"}</div>
                </div>

                <div className="metric-box">
                  <div className="metric-label">Selected Channel</div>
                  <div className="metric-value">Channel {SELECTED_CHANNEL_INDEX + 1}</div>
                </div>
              </div>
            </section>
          </div>

          <div className="left-scroll">
            <section className="card">
              <h2>Live EMG Signal</h2>
              <div className="subtle" style={{ marginBottom: "8px" }}>
                Crimson = raw selected channel, black = processed envelope used for features.
              </div>
              <div className="chart-wrap">
                <canvas ref={canvasRef} width="920" height="600" />
              </div>
            </section>
          </div>
        </div>

        <div className="right-panel">
          <div className="right-stack">
            <section className="card">
              <h2>Data Collection</h2>

              <div className="row">
                <button onClick={() => collectSamples("relax")}>Collect Relax</button>
                <button onClick={() => collectSamples("lightGrip")}>Collect Light Grip</button>
                <button onClick={() => collectSamples("tightGrip")}>Collect Tight Grip</button>
              </div>

              <div className="row">
                <span className="subtle">Samples per click:</span>
                <select
                  value={samplesPerClick}
                  onChange={(e) => setSamplesPerClick(Number(e.target.value))}
                >
                  <option value={1}>1</option>
                  <option value={5}>5</option>
                  <option value={10}>10</option>
                </select>
                <button onClick={clearDataset} className="secondary">Clear</button>
                <button onClick={exportDataset} className="secondary">Export</button>
              </div>

              <div className="pill-wrap">
                <div className="pill">Relax: <strong>{counts.relax}</strong></div>
                <div className="pill">Light Grip: <strong>{counts.lightGrip}</strong></div>
                <div className="pill">Tight Grip: <strong>{counts.tightGrip}</strong></div>
                <div className="pill">Total: <strong>{dataset.length}</strong></div>
              </div>

              <div className="subtle" style={{ marginTop: "10px" }}>
                Use the Current Demo Phase on the left to know when to click each collection button.
              </div>
            </section>

            <section className="card">
              <h2>Live Prediction</h2>

              <div className="big-value">
                {prediction ? LABEL_DISPLAY[prediction] : "--"}
              </div>

              <div className="metric-grid" style={{ marginTop: "10px" }}>
                <div className="metric-box">
                  <div className="metric-label">Confidence</div>
                  <div className="metric-value">
                    {predictionConfidence !== null
                      ? `${(predictionConfidence * 100).toFixed(1)}%`
                      : "--"}
                  </div>
                </div>

                <div className="metric-box">
                  <div className="metric-label">Prediction Status</div>
                  <div className={`metric-value ${getStatusClass(predictionStatusKind)}`}>
                    {predictionStatus}
                  </div>
                </div>
              </div>
            </section>

            <section className="card">
              <h2>Feedback </h2>

              <div className="row">
                <button onClick={confirmPredictionRight}>That Was Right</button>
                <button onClick={markPredictionWrong}>That Was Wrong</button>
              </div>

              <div className="row">
                <select
                  value={selectedCorrectionLabel}
                  onChange={(e) => setSelectedCorrectionLabel(e.target.value)}
                >
                  <option value="relax">Relax</option>
                  <option value="lightGrip">Light Grip</option>
                  <option value="tightGrip">Tight Grip</option>
                </select>

                <button onClick={applyCorrection}>Apply Correction</button>
              </div>

              <div className="metric-grid">
                <div className="metric-box">
                  <div className="metric-label">Pending Correction</div>
                  <div className="metric-value">
                    {pendingCorrection ? LABEL_DISPLAY[pendingCorrection.predictedLabel] : "None"}
                  </div>
                </div>

                <div className="metric-box">
                  <div className="metric-label">Corrections Added</div>
                  <div className="metric-value">{correctionCount}</div>
                </div>
              </div>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);