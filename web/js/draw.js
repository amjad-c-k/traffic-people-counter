// Drawing on the canvas: video frame, counting line, boxes with IDs, trails, count panel.
export const COLORS = { car: "#2d9cff", truck: "#ff7a28", bus: "#c83cc8", motorcycle: "#e0b000", person: "#2fbf4f" };
const COUNTED_COLOR = "#ff3c3c"; // object already counted
const LINE_COLOR = "#ffe600";

function label(ctx, text, x, y, color, size = 13) {
  ctx.font = `${size}px system-ui, sans-serif`;
  const w = ctx.measureText(text).width + 6;
  const top = Math.max(0, y - size - 4);
  ctx.fillStyle = color;
  ctx.fillRect(x, top, w, size + 4);
  ctx.fillStyle = "#fff";
  ctx.fillText(text, x + 3, top + size);
}

// Counting line with small arrows showing which side is IN (down) and OUT (up).
export function drawLine(ctx, a, b) {
  ctx.strokeStyle = LINE_COLOR;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(...a);
  ctx.lineTo(...b);
  ctx.stroke();
  // Perpendicular direction pointing to the IN side (positive side of the cross product).
  const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy) || 1;
  const nx = -dy / len, ny = dx / len;
  const mx = a[0] + dx * 0.05, my = a[1] + dy * 0.05; // near the left end, away from traffic
  for (const [sign, text, color] of [[1, "IN", "#1b8a2e"], [-1, "OUT", "#b3261e"]]) {
    const tx = mx + sign * nx * 32, ty = my + sign * ny * 32;
    ctx.beginPath();
    ctx.moveTo(mx, my);
    ctx.lineTo(tx, ty);
    ctx.stroke();
    label(ctx, text, tx + 6, ty + 8, color);
  }
}

export function drawFrame(ctx, video, tracks, counter, trails) {
  const { width, height } = ctx.canvas;
  ctx.drawImage(video, 0, 0, width, height);
  drawLine(ctx, counter.a, counter.b);

  for (const t of tracks) {
    const color = counter.counted.has(t.trackId) ? COUNTED_COLOR : COLORS[t.className] || "#ccc";
    const pts = trails.get(t.trackId) || []; // recent centre points: shows the direction of motion
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.stroke();
    const [x1, y1, x2, y2] = t.box;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    label(ctx, `#${t.trackId} ${t.className}`, x1, y1, color);
  }

  const { IN, OUT } = counter.totals();
  ctx.fillStyle = "rgba(20,20,20,0.65)";
  ctx.fillRect(10, 10, 180, 42);
  ctx.fillStyle = LINE_COLOR;
  ctx.font = "bold 22px system-ui, sans-serif";
  ctx.fillText(`IN ${IN}   OUT ${OUT}`, 22, 40);
}

// Before processing: the current video frame with the counting line and its drag handles
// (two ends + middle). `active` = the handle under the pointer, drawn bigger.
export function drawPreview(ctx, video, a, b, active = null) {
  ctx.drawImage(video, 0, 0, ctx.canvas.width, ctx.canvas.height);
  drawLine(ctx, a, b);
  const r = handleRadius(ctx.canvas);
  const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  for (const [name, [x, y]] of [["a", a], ["b", b], ["mid", mid]]) {
    ctx.beginPath();
    ctx.arc(x, y, name === active ? r * 1.35 : r, 0, 2 * Math.PI);
    ctx.fillStyle = LINE_COLOR;
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#222";
    ctx.stroke();
  }
}

// Handle size in canvas pixels, so it looks about the same on screen for any video size.
export const handleRadius = (canvas) => Math.max(8, canvas.width / 90);
