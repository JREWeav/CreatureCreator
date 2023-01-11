const SIZE = 512;

let inputCanvas,
  outputCanvas,
  htmlMessage,
  redButton,
  blueButton,
  slider,
  runOnce = false,
  modelReady = false,
  isProcessing = false;

// Create pix2pix model
let p2p = pix2pix("./models/catndog_BtoA_v2.pict", modelLoaded);

function setup() {
  // Create canvas
  inputCanvas = createCanvas(SIZE, SIZE);
  inputCanvas.parent("canvasContainer");
  background(255);

  // Select all HTML Elements (no need for eraser because it never changes)
  outputCanvas = select("#modelOutput");
  htmlMessage = select("#modelStatus");
  redButton = select("#red");
  blueButton = select("#blue");
  slider = select("#strokeWeightSlider");

  // Set stroke to red
  stroke(255, 0, 0);
  strokeWeight(20);
}

// Draw on the canvas when mouse is pressed
function draw() {
  if (mouseIsPressed) {
    // Check if mouse is inside canvas
    if (mouseX > 0 && mouseX < SIZE && mouseY > 0 && mouseY < SIZE) {
      // Set model status
      htmlMessage.html("Transferring...");
      // Draw line
      line(mouseX, mouseY, pmouseX, pmouseY);
    }
  }
}

//
function mouseReleased() {
  // Run transfer function if model is ready
  if (modelReady && !isProcessing) {
    transfer();
  }
}

// Run the model on the canvas
function transfer() {
  isProcessing = true;
  console.log("Transferring...");
  // Get canvas element
  let canvasElement = document.getElementById("defaultCanvas0");
  // Transfer
  p2p.transfer(canvasElement, (result) => {
    outputCanvas.html("");
    // Create an image from the result and display it
    createImg(result.src).parent("modelOutput");
    // Set model status
    isProcessing = false;
    htmlMessage.html("Ready for use!");
  });
}

function modelLoaded() {
  // Run transfer function once to initialize model
  if (!runOnce) {
    transfer();
    runOnce = true;
  }
  // Set model status
  modelReady = true;
  htmlMessage.html("Ready for use!");
}

// Clear the canvas
function clearCanvas() {
  background(255);
}

// Select eraser
function selectEraser() {
  stroke(255);
  redButton.class("unselected");
  blueButton.class("unselected");
}

// Select red
function selectRed() {
  stroke(255, 0, 0);
  redButton.class("selected");
  blueButton.class("unselected");
}

// Select blue
function selectBlue() {
  stroke(0, 0, 255);
  redButton.class("unselected");
  blueButton.class("selected");
}

// Change stroke weight
function changeStrokeWeight() {
  strokeWeight(slider.value());
}
