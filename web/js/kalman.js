// Kalman filter for one bounding box, ported from Ultralytics' KalmanFilterXYAH (ByteTrack).
//
// State (8 numbers): [cx, cy, a, h, vcx, vcy, va, vh]
//   cx, cy = box centre, a = aspect ratio (w / h), h = height, v* = their speeds per frame.
// "Predict" moves the box by its speed (constant-velocity model) and grows the uncertainty;
// "update" blends that prediction with the new detection, trusting whichever is more certain.

const STD_POS = 1 / 20;   // position noise, relative to box height
const STD_VEL = 1 / 160;  // velocity noise, relative to box height

// --- tiny matrix helpers (matrices are arrays of rows) ---
const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
const diag = (v) => v.map((x, i) => v.map((_, j) => (i === j ? x : 0)));
const transpose = (A) => A[0].map((_, j) => A.map((row) => row[j]));
const matmul = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)));
const add = (A, B) => A.map((row, i) => row.map((v, j) => v + B[i][j]));
const sub = (A, B) => A.map((row, i) => row.map((v, j) => v - B[i][j]));

// Inverse of a small matrix by Gauss-Jordan elimination (used on 4x4 matrices only).
function inverse(M) {
  const n = M.length;
  const A = M.map((row, i) => [...row, ...row.map((_, j) => (i === j ? 1 : 0))]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    const div = A[c][c];
    for (let j = 0; j < 2 * n; j++) A[c][j] /= div;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = A[r][c];
      for (let j = 0; j < 2 * n; j++) A[r][j] -= f * A[c][j];
    }
  }
  return A.map((row) => row.slice(n));
}

// Motion matrix F: new position = old position + velocity (dt = 1 frame).
const F = diag(new Array(8).fill(1));
for (let i = 0; i < 4; i++) F[i][4 + i] = 1;
const FT = transpose(F);

export function initiate(xyah) {
  const h = xyah[3];
  const mean = [...xyah, 0, 0, 0, 0]; // start with zero speed
  const std = [2 * STD_POS * h, 2 * STD_POS * h, 1e-2, 2 * STD_POS * h,
               10 * STD_VEL * h, 10 * STD_VEL * h, 1e-5, 10 * STD_VEL * h];
  return { mean, cov: diag(std.map((s) => s * s)) };
}

export function predict(mean, cov) {
  const h = mean[3];
  const std = [STD_POS * h, STD_POS * h, 1e-2, STD_POS * h, STD_VEL * h, STD_VEL * h, 1e-5, STD_VEL * h];
  const Q = diag(std.map((s) => s * s)); // extra uncertainty added each frame
  const newMean = mean.map((_, i) => F[i].reduce((s, v, k) => s + v * mean[k], 0));
  return { mean: newMean, cov: add(matmul(matmul(F, cov), FT), Q) };
}

export function update(mean, cov, xyah) {
  const h = mean[3];
  const std = [STD_POS * h, STD_POS * h, 1e-1, STD_POS * h];
  // Projected (measurement-space) covariance S = cov[:4,:4] + R
  const S = add(cov.slice(0, 4).map((row) => row.slice(0, 4)), diag(std.map((s) => s * s)));
  const PHt = cov.map((row) => row.slice(0, 4));        // cov[:, :4]   (8x4)
  const K = matmul(PHt, inverse(S));                    // Kalman gain  (8x4)
  const innovation = xyah.map((z, i) => z - mean[i]);   // detection minus prediction
  const newMean = mean.map((m, i) => m + K[i].reduce((s, v, k) => s + v * innovation[k], 0));
  const newCov = sub(cov, matmul(matmul(K, S), transpose(K)));
  return { mean: newMean, cov: newCov };
}

export { zeros };
