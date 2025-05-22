function decibelsToProbability(dB) {
  const odds = Math.pow(10, dB / 10);
  return odds / (1 + odds);
}

function drawSpinner(probability) {
  const canvas = document.getElementById("spinnerCanvas");
  const ctx = canvas.getContext("2d");
  const totalSlices = 8;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const blueSlices = Math.round(probability * totalSlices);
  const whiteSlices = totalSlices - blueSlices;

  const drawSlice = (startAngle, endAngle, color) => {
    ctx.beginPath();
    ctx.moveTo(100, 100);
    ctx.arc(100, 100, 100, startAngle, endAngle);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  };

  let angle = 0;
  for (let i = 0; i < totalSlices; i++) {
    const sliceAngle = (2 * Math.PI) / totalSlices;
    drawSlice(angle, angle + sliceAngle, i < blueSlices ? "#2b6cb0" : "#ffffff");
    angle += sliceAngle;
  }

  ctx.strokeStyle = "#000000";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(100, 100, 100, 0, 2 * Math.PI);
  ctx.stroke();
}

document.addEventListener("DOMContentLoaded", function () {
  const slider = document.getElementById("decibelSlider");
  const dBDisplay = document.getElementById("dBValue");

  function update() {
    const dB = parseInt(slider.value, 10);
    dBDisplay.textContent = dB;
    const prob = decibelsToProbability(dB);
    drawSpinner(prob);
  }

  slider.addEventListener("input", update);
  update(); // Initial draw
});
