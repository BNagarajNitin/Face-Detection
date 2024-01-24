const video = document.getElementById("video");

// Load face detection models
Promise.all([
  faceapi.nets.ssdMobilenetv1.loadFromUri("/assets/models"),
  faceapi.nets.faceRecognitionNet.loadFromUri("/assets/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/assets/models"),
]).then(startWebcam);

// Function to start the webcam
function startWebcam() {
  navigator.mediaDevices
    .getUserMedia({
      video: true,
      audio: false,
    })
    .then((stream) => {
      video.srcObject = stream;
      video.onloadedmetadata = () => {
        // Set dimensions after video metadata has loaded
        video.width = video.videoWidth;
        video.height = video.videoHeight;
        startFaceDetection();
      };
    })
    .catch((error) => {
      console.error("Error accessing webcam:", error);
    });
}

// Function to get labeled face descriptions
async function getLabeledFaceDescriptions() {
  const labels = ["Felipe", "Nithin"];
  return Promise.all(
    labels.map(async (label) => {
      const descriptions = [];
      for (let i = 1; i <= 2; i++) {
        try {
          console.log(`Loading image: ./labels/${label}/${i}.jpg`);
          const img = await faceapi.fetchImage(`./labels/${label}/${i}.jpg`);

          const detections = await faceapi
            .detectSingleFace(img, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }))
            .withFaceLandmarks()
            .withFaceDescriptor();

          if (detections) {
            descriptions.push(detections.descriptor);
          } else {
            console.log(`No face detected in ./labels/${label}/${i}.jpg`);
          }
        } catch (error) {
          console.error(`Error loading image ./labels/${label}/${i}.jpg:`, error);
        }
      }
      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}

// Function to start face detection
async function startFaceDetection() {
  const labeledFaceDescriptors = await getLabeledFaceDescriptions();
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors);

  const canvas = faceapi.createCanvasFromMedia(video);
  document.body.append(canvas);

  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);

  // Continuously detect faces and draw bounding boxes
  setInterval(async () => {
    const detections = await faceapi
      .detectAllFaces(video, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }))
      .withFaceLandmarks()
      .withFaceDescriptors();

    console.log("Detections:", detections);

    const resizedDetections = faceapi.resizeResults(detections, displaySize);

    // Clear the canvas before drawing new bounding boxes
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

    // Draw bounding boxes based on face matching results
    const results = resizedDetections.map((d) => {
      return faceMatcher.findBestMatch(d.descriptor);
    });

    console.log("Matched Results:", results);

    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box;
      const drawBox = new faceapi.draw.DrawBox(box, {
        label: result.label,
      });
      drawBox.draw(canvas);
    });
  }, 100);
}

// Event listener for when the video starts playing
video.addEventListener("play", startFaceDetection);
