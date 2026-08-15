/**
 * Self-contained N-Dimensional Array (NDArray) linear algebra component.
 */
export class NDArray {
  public dtype: string;
  public data: Float32Array;
  public shape: number[];
  public strides: number[];
  public offset: number;
  public order: string;

  constructor(
    dtype: string,
    data: Float32Array,
    shape: number[],
    strides: number[],
    offset = 0,
    order = 'row-major'
  ) {
    this.dtype = dtype;
    this.data = data;
    this.shape = shape;
    this.strides = strides;
    this.offset = offset;
    this.order = order;
  }

  /**
   * Get element at multi-dimensional indices.
   */
  get(...indices: number[]): number {
    let idx = this.offset;
    for (let i = 0; i < indices.length; i++) {
      idx += indices[i] * this.strides[i];
    }
    return this.data[idx];
  }

  /**
   * Set element at multi-dimensional indices to given value.
   * Last argument is value, preceding arguments are indices.
   */
  set(...args: number[]): void {
    const val = args[args.length - 1];
    let idx = this.offset;
    for (let i = 0; i < args.length - 1; i++) {
      idx += args[i] * this.strides[i];
    }
    this.data[idx] = val;
  }
}

/**
 * Compute row-major strides for a given shape.
 */
export function computeRowMajorStrides(shape: number[]): number[] {
  if (shape.length === 0) return [1];
  const strides = new Array(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

/**
 * Get total number of elements in a shape.
 */
export function getShapeSize(shape: number[]): number {
  if (shape.length === 0) return 1;
  return shape.reduce((acc, val) => acc * val, 1);
}

/**
 * Resolve a reshape target shape: validate total element count and
 * auto-derive at most one `-1` dimension from the original shape.
 */
export function resolveReshapeShape(originalShape: number[], newShape: number[]): number[] {
  for (const s of newShape) {
    if (!Number.isInteger(s) || s < -1) {
      throw new Error(`Invalid reshape shape: ${JSON.stringify(newShape)}`);
    }
  }
  const origSize = getShapeSize(originalShape);
  const inferredCount = newShape.filter((s) => s === -1).length;
  if (inferredCount > 1) {
    throw new Error('Reshape shape can contain at most one -1 dimension');
  }
  if (inferredCount === 1) {
    const knownSize = newShape.reduce((acc, s) => (s === -1 ? acc : acc * s), 1);
    if (knownSize === 0 || origSize % knownSize !== 0) {
      throw new Error(`Cannot infer -1 dimension: ${origSize} elements are not divisible by ${knownSize}`);
    }
    return newShape.map((s) => (s === -1 ? origSize / knownSize : s));
  }
  if (getShapeSize(newShape) !== origSize) {
    throw new Error(`Reshape size mismatch: ${origSize} elements cannot be reshaped to ${JSON.stringify(newShape)}`);
  }
  return newShape;
}

/**
 * Create an NDArray from a Float32Array buffer and shape.
 */
export function createNdArray(buffer: Float32Array, shape: number[]): NDArray {
  const strides = computeRowMajorStrides(shape);
  return new NDArray('float64', buffer, shape, strides, 0, 'row-major');
}

/**
 * Create an all-zeros ndarray with the given shape.
 */
export function zerosNdArray(shape: number[]): any {
  const size = getShapeSize(shape);
  return createNdArray(new Float32Array(size), shape);
}

/**
 * Create an all-ones ndarray with the given shape.
 */
export function onesNdArray(shape: number[]): any {
  const size = getShapeSize(shape);
  const buffer = new Float32Array(size);
  buffer.fill(1.0);
  return createNdArray(buffer, shape);
}

/**
 * Create a random normal distribution ndarray (Box-Muller approximation or standard random).
 */
export function randomNormalNdArray(shape: number[], mean = 0, std = 1): any {
  const size = getShapeSize(shape);
  const buffer = new Float32Array(size);
  for (let i = 0; i < size; i += 2) {
    const u1 = Math.max(1e-15, Math.random());
    const u2 = Math.random();
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    const z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);
    buffer[i] = mean + z0 * std;
    if (i + 1 < size) {
      buffer[i + 1] = mean + z1 * std;
    }
  }
  return createNdArray(buffer, shape);
}

/**
 * Convert a nested JS Array or scalar into an NDArray and inferred shape.
 */
export function fromJsArray(input: number | any[]): { ndarray: NDArray; shape: number[] } {
  if (typeof input === 'number') {
    const buffer = new Float32Array([input]);
    return { ndarray: createNdArray(buffer, []), shape: [] };
  }

  // Infer shape
  const shape: number[] = [];
  let curr: any = input;
  while (Array.isArray(curr)) {
    shape.push(curr.length);
    curr = curr[0];
  }

  const flatList: number[] = [];
  function flatten(arr: any[]) {
    for (const item of arr) {
      if (Array.isArray(item)) {
        flatten(item);
      } else {
        flatList.push(Number(item));
      }
    }
  }
  flatten(input);

  const buffer = new Float32Array(flatList);
  return { ndarray: createNdArray(buffer, shape), shape };
}

/**
 * Convert an NDArray back to a JS nested Array (or scalar array).
 */
export function toJsArray(nd: any): any {
  const shape: number[] = nd.shape;
  if (shape.length === 0) {
    return nd.data[nd.offset];
  }
  return toNestedArray(nd.data, shape, nd.offset, nd.strides, 0, []);
}

function toNestedArray(data: Float32Array, shape: number[], offset: number, strides: number[], dim: number, pos: number[]): any {
  const size = shape[dim];
  const result: any[] = [];
  for (let i = 0; i < size; i++) {
    const newPos = [...pos, i];
    if (dim === shape.length - 1) {
      let idx = offset;
      for (let j = 0; j < newPos.length; j++) {
        idx += newPos[j] * strides[j];
      }
      result.push(data[idx]);
    } else {
      result.push(toNestedArray(data, shape, offset, strides, dim + 1, newPos));
    }
  }
  return result;
}

/**
 * Clone an ndarray.
 */
export function cloneNdArray(nd: any): any {
  const newBuf = new Float32Array(nd.data);
  return createNdArray(newBuf, Array.from(nd.shape));
}

/**
 * Element-wise addition of two ndarrays (supports scalar/shape broadcasting or same shape).
 */
export function addNdArray(a: any, b: any): any {
  const sizeA = a.data.length;
  const sizeB = b.data.length;

  if (sizeA === sizeB && arraysEqual(a.shape, b.shape)) {
    const outBuf = new Float32Array(sizeA);
    for (let i = 0; i < sizeA; i++) {
      outBuf[i] = a.data[i] + b.data[i];
    }
    return createNdArray(outBuf, Array.from(a.shape));
  }

  // Handle scalar broadcast (e.g. bias shape [1])
  if (a.shape.length === 1 && a.shape[0] === 1) {
    const outBuf = new Float32Array(sizeB);
    const val = a.data[0];
    for (let i = 0; i < sizeB; i++) {
      outBuf[i] = val + b.data[i];
    }
    return createNdArray(outBuf, Array.from(b.shape));
  }

  if (b.shape.length === 1 && b.shape[0] === 1) {
    const outBuf = new Float32Array(sizeA);
    const val = b.data[0];
    for (let i = 0; i < sizeA; i++) {
      outBuf[i] = a.data[i] + val;
    }
    return createNdArray(outBuf, Array.from(a.shape));
  }

  // Handle general broadcast by expanding b to a's shape
  const bBroadcast = broadcastNdArray(b, a.shape);
  return addNdArray(a, bBroadcast);
}

/**
 * Element-wise subtraction of two ndarrays.
 */
export function subNdArray(a: any, b: any): any {
  const sizeA = a.data.length;
  const sizeB = b.data.length;

  if (sizeA === sizeB && arraysEqual(a.shape, b.shape)) {
    const outBuf = new Float32Array(sizeA);
    for (let i = 0; i < sizeA; i++) {
      outBuf[i] = a.data[i] - b.data[i];
    }
    return createNdArray(outBuf, Array.from(a.shape));
  }

  if (a.shape.length === 1 && a.shape[0] === 1) {
    const outBuf = new Float32Array(sizeB);
    const val = a.data[0];
    for (let i = 0; i < sizeB; i++) {
      outBuf[i] = val - b.data[i];
    }
    return createNdArray(outBuf, Array.from(b.shape));
  }

  if (b.shape.length === 1 && b.shape[0] === 1) {
    const outBuf = new Float32Array(sizeA);
    const val = b.data[0];
    for (let i = 0; i < sizeA; i++) {
      outBuf[i] = a.data[i] - val;
    }
    return createNdArray(outBuf, Array.from(a.shape));
  }

  // Handle general broadcast by expanding b to a's shape
  const bBroadcast = broadcastNdArray(b, a.shape);
  return subNdArray(a, bBroadcast);
}

/**
 * Element-wise multiplication of two ndarrays.
 */
export function mulNdArray(a: any, b: any): any {
  const sizeA = a.data.length;
  const sizeB = b.data.length;

  if (sizeA === sizeB && arraysEqual(a.shape, b.shape)) {
    const outBuf = new Float32Array(sizeA);
    for (let i = 0; i < sizeA; i++) {
      outBuf[i] = a.data[i] * b.data[i];
    }
    return createNdArray(outBuf, Array.from(a.shape));
  }

  if (a.shape.length === 1 && a.shape[0] === 1) {
    const outBuf = new Float32Array(sizeB);
    const val = a.data[0];
    for (let i = 0; i < sizeB; i++) {
      outBuf[i] = val * b.data[i];
    }
    return createNdArray(outBuf, Array.from(b.shape));
  }

  if (b.shape.length === 1 && b.shape[0] === 1) {
    const outBuf = new Float32Array(sizeA);
    const val = b.data[0];
    for (let i = 0; i < sizeA; i++) {
      outBuf[i] = a.data[i] * val;
    }
    return createNdArray(outBuf, Array.from(a.shape));
  }

  // Handle general broadcast by expanding b to a's shape
  const bBroadcast = broadcastNdArray(b, a.shape);
  return mulNdArray(a, bBroadcast);
}

/**
 * Element-wise division of two ndarrays.
 */
export function divNdArray(a: any, b: any): any {
  const sizeA = a.data.length;
  const sizeB = b.data.length;

  if (sizeA === sizeB && arraysEqual(a.shape, b.shape)) {
    const outBuf = new Float32Array(sizeA);
    for (let i = 0; i < sizeA; i++) {
      outBuf[i] = a.data[i] / b.data[i];
    }
    return createNdArray(outBuf, Array.from(a.shape));
  }

  if (a.shape.length === 1 && a.shape[0] === 1) {
    const outBuf = new Float32Array(sizeB);
    const val = a.data[0];
    for (let i = 0; i < sizeB; i++) {
      outBuf[i] = val / b.data[i];
    }
    return createNdArray(outBuf, Array.from(b.shape));
  }

  if (b.shape.length === 1 && b.shape[0] === 1) {
    const outBuf = new Float32Array(sizeA);
    const val = b.data[0];
    for (let i = 0; i < sizeA; i++) {
      outBuf[i] = a.data[i] / val;
    }
    return createNdArray(outBuf, Array.from(a.shape));
  }

  // Handle general broadcast by expanding b to a's shape
  const bBroadcast = broadcastNdArray(b, a.shape);
  return divNdArray(a, bBroadcast);
}


/**
 * Matrix multiplication supporting PyTorch torch.matmul semantics.
 * Supports: 1D x 1D, 2D x 1D, 1D x 2D, 2D x 2D, and arbitrary batched
 * broadcasting for tensors of rank >= 2 (e.g. [B,M,K] x [K,N]).
 */
export function matMulNdArray(a: any, b: any): any {
  const rankA = a.shape.length;
  const rankB = b.shape.length;

  // 1D x 1D: dot product scalar
  if (rankA === 1 && rankB === 1) {
    const K = a.shape[0];
    let sum = 0;
    for (let k = 0; k < K; k++) {
      sum += a.get(k) * b.get(k);
    }
    return createNdArray(new Float32Array([sum]), []);
  }

  // 2D x 1D: matrix-vector -> 1D [M]
  if (rankA === 2 && rankB === 1) {
    const M = a.shape[0];
    const K = a.shape[1];
    const outBuf = new Float32Array(M);
    for (let i = 0; i < M; i++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += a.get(i, k) * b.get(k);
      }
      outBuf[i] = sum;
    }
    return createNdArray(outBuf, [M]);
  }

  // 1D x 2D: vector-matrix -> 1D [N]
  if (rankA === 1 && rankB === 2) {
    const K = a.shape[0];
    const N = b.shape[1];
    const outBuf = new Float32Array(N);
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += a.get(k) * b.get(k, j);
      }
      outBuf[j] = sum;
    }
    return createNdArray(outBuf, [N]);
  }

  // General batched case: both tensors rank >= 2 with broadcasting.
  return matMulBroadcastNdArray(a, b);
}

function broadcastShapes(shapeA: number[], shapeB: number[]): number[] {
  const maxLen = Math.max(shapeA.length, shapeB.length);
  const result: number[] = [];
  for (let i = 0; i < maxLen; i++) {
    const idxA = shapeA.length - 1 - i;
    const idxB = shapeB.length - 1 - i;
    const dimA = idxA >= 0 ? shapeA[idxA] : 1;
    const dimB = idxB >= 0 ? shapeB[idxB] : 1;
    if (dimA !== dimB && dimA !== 1 && dimB !== 1) {
      throw new Error(`Cannot broadcast matmul batch shapes ${JSON.stringify(shapeA)} and ${JSON.stringify(shapeB)}`);
    }
    result.unshift(Math.max(dimA, dimB));
  }
  return result;
}

function posFromFlatIndex(flat: number, shape: number[]): number[] {
  const pos: number[] = new Array(shape.length).fill(0);
  let tmp = flat;
  for (let i = shape.length - 1; i >= 0; i--) {
    pos[i] = tmp % shape[i];
    tmp = Math.floor(tmp / shape[i]);
  }
  return pos;
}

function getBroadcastPos(outPos: number[], inShape: number[]): number[] {
  const pos: number[] = [];
  const offset = outPos.length - inShape.length;
  for (let i = 0; i < inShape.length; i++) {
    const outIdx = outPos[offset + i];
    pos.push(inShape[i] === 1 ? 0 : outIdx);
  }
  return pos;
}

function getFlatOffset(pos: number[], strides: number[]): number {
  let offset = 0;
  for (let i = 0; i < pos.length; i++) {
    offset += pos[i] * strides[i];
  }
  return offset;
}

function matMulBroadcastNdArray(a: any, b: any): any {
  const aShape: number[] = Array.from(a.shape);
  const bShape: number[] = Array.from(b.shape);
  const rankA = aShape.length;
  const rankB = bShape.length;
  if (rankA < 2 || rankB < 2) {
    throw new Error(`matMulBroadcastNdArray requires inputs of rank >= 2, got ${rankA} and ${rankB}`);
  }
  const M = aShape[rankA - 2];
  const K = aShape[rankA - 1];
  const N = bShape[rankB - 1];
  const Kb = bShape[rankB - 2];
  if (K !== Kb) {
    throw new Error(`Incompatible matmul shapes: ${JSON.stringify(aShape)} x ${JSON.stringify(bShape)}`);
  }
  const batchShapeA = aShape.slice(0, -2);
  const batchShapeB = bShape.slice(0, -2);
  const batchShape = broadcastShapes(batchShapeA, batchShapeB);
  const batchSize = getShapeSize(batchShape);
  const outShape = [...batchShape, M, N];
  const outBuf = new Float32Array(batchSize * M * N).fill(0);

  const aStrides = a.strides;
  const bStrides = b.strides;

  for (let batch = 0; batch < batchSize; batch++) {
    const outPos = posFromFlatIndex(batch, batchShape);
    const aPos = getBroadcastPos(outPos, batchShapeA);
    const bPos = getBroadcastPos(outPos, batchShapeB);
    const aBatchOffset = a.offset + getFlatOffset(aPos, aStrides);
    const bBatchOffset = b.offset + getFlatOffset(bPos, bStrides);
    for (let i = 0; i < M; i++) {
      for (let k = 0; k < K; k++) {
        const aIdx = aBatchOffset + i * aStrides[rankA - 2] + k * aStrides[rankA - 1];
        for (let j = 0; j < N; j++) {
          const bIdx = bBatchOffset + k * bStrides[rankB - 2] + j * bStrides[rankB - 1];
          outBuf[batch * M * N + i * N + j] += a.data[aIdx] * b.data[bIdx];
        }
      }
    }
  }
  return createNdArray(outBuf, outShape);
}

function matMul2D(a: any, b: any): any {
  const shapeA = a.shape;
  const shapeB = b.shape;
  const M = shapeA[0];
  const K = shapeA[1];
  const N = shapeB[1];
  const outBuf = new Float32Array(M * N).fill(0);
  for (let i = 0; i < M; i++) {
    for (let k = 0; k < K; k++) {
      const aik = a.get(i, k);
      for (let j = 0; j < N; j++) {
        outBuf[i * N + j] += aik * b.get(k, j);
      }
    }
  }
  return createNdArray(outBuf, [M, N]);
}

/**
 * Transpose a 2D matrix.
 */
/**
 * ReLU activation forward: max(0, x).
 */
export function reluNdArray(a: any): any {
  const size = a.data.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    outBuf[i] = Math.max(0, a.data[i]);
  }
  return createNdArray(outBuf, Array.from(a.shape));
}

/**
 * ReLU activation backward.
 */
export function reluGradNdArray(gradOut: any, forwardInput: any): any {
  const size = gradOut.data.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    outBuf[i] = forwardInput.data[i] > 0 ? gradOut.data[i] : 0;
  }
  return createNdArray(outBuf, Array.from(gradOut.shape));
}

/**
 * Mean Squared Error loss between pred and target.
 */
export function mseLossNdArray(pred: any, target: any): { lossValue: number; gradPred: any } {
  const size = pred.data.length;
  let sumSq = 0;
  const gradBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    const diff = pred.data[i] - target.data[i];
    sumSq += diff * diff;
    gradBuf[i] = (2.0 * diff) / size;
  }
  const lossValue = sumSq / size;
  return {
    lossValue,
    gradPred: createNdArray(gradBuf, Array.from(pred.shape))
  };
}

// ============================================================================
// Advanced Tensor Structure Operations
// ============================================================================

function normalizeDim(dim: number, ndim: number): number {
  let d = dim < 0 ? dim + ndim : dim;
  if (d < 0 || d >= ndim) {
    throw new Error(`Dimension ${dim} out of range for ${ndim}-D tensor`);
  }
  return d;
}

function isCompatibleBroadcast(srcShape: number[], targetShape: number[]): boolean {
  if (srcShape.length > targetShape.length) return false;
  const offset = targetShape.length - srcShape.length;
  for (let i = 0; i < srcShape.length; i++) {
    const s = srcShape[i];
    const t = targetShape[offset + i];
    if (s !== 1 && s !== t) return false;
  }
  return true;
}

/**
 * Insert a size-1 dimension at the specified axis (supports negative indexing and append).
 * PyTorch-compatible: dim in [-ndim-1, ndim].
 */
export function unsqueezeNdArray(nd: any, dim: number): any {
  const ndim = nd.shape.length;
  let d = dim < 0 ? dim + ndim + 1 : dim;
  if (d < 0 || d > ndim) {
    throw new Error(`Dimension ${dim} out of range for ${ndim}-D tensor`);
  }
  const newShape = [...nd.shape.slice(0, d), 1, ...nd.shape.slice(d)];
  return createNdArray(nd.data, newShape);
}

/**
 * Remove all size-1 dimensions, or a specific dimension if dim is provided.
 */
export function squeezeNdArray(nd: any, dim?: number): any {
  if (dim !== undefined) {
    const ndim = nd.shape.length;
    const d = normalizeDim(dim, ndim);
    if (nd.shape[d] !== 1) {
      throw new Error(`Squeeze target dim ${d} has size ${nd.shape[d]}, not 1`);
    }
    const newShape = [...nd.shape.slice(0, d), ...nd.shape.slice(d + 1)];
    return createNdArray(nd.data, newShape);
  }
  const newShape = nd.shape.filter((s: number) => s !== 1);
  return createNdArray(nd.data, newShape);
}

/**
 * Broadcast an ndarray to a target shape (row-major, element-wise copy).
 */
export function expandNdArray(nd: any, targetShape: number[]): any {
  if (!isCompatibleBroadcast(nd.shape, targetShape)) {
    throw new Error(`Cannot broadcast shape ${JSON.stringify(nd.shape)} to ${JSON.stringify(targetShape)}`);
  }
  const srcShape = nd.shape;
  const outSize = getShapeSize(targetShape);
  const outBuf = new Float32Array(outSize);
  const outStrides = computeRowMajorStrides(targetShape);
  const srcStrides = computeRowMajorStrides(srcShape);

  const paddedSrcShape = new Array(targetShape.length).fill(1);
  for (let i = 0; i < srcShape.length; i++) {
    paddedSrcShape[targetShape.length - srcShape.length + i] = srcShape[i];
  }

  function iterate(pos: number[], d: number) {
    if (d === targetShape.length) {
      let srcIdx = 0;
      for (let i = 0; i < targetShape.length; i++) {
        const srcDim = paddedSrcShape[i];
        if (srcDim !== 1) {
          const srcStrideIdx = i - (targetShape.length - srcShape.length);
          srcIdx += pos[i] * srcStrides[srcStrideIdx];
        }
      }
      let outIdx = 0;
      for (let i = 0; i < targetShape.length; i++) {
        outIdx += pos[i] * outStrides[i];
      }
      outBuf[outIdx] = nd.data[srcIdx + nd.offset];
      return;
    }
    for (let i = 0; i < targetShape[d]; i++) {
      pos.push(i);
      iterate(pos, d + 1);
      pos.pop();
    }
  }
  iterate([], 0);
  return createNdArray(outBuf, targetShape);
}

/**
 * Sum over dimensions that were expanded from originalShape to expandedShape.
 * Used in the backward of expandNdArray.
 */
function squeezeToRank(shape: number[], targetRank: number): number[] | null {
  if (shape.length < targetRank) return null;
  if (shape.length === targetRank) return shape;
  const extra = shape.length - targetRank;

  let result = [...shape];
  let removed = 0;
  for (let i = 0; i < result.length && removed < extra; i++) {
    if (result[i] === 1) {
      result.splice(i, 1);
      removed++;
      i--;
    }
  }
  if (removed === extra) return result;

  result = [...shape];
  removed = 0;
  for (let i = result.length - 1; i >= 0 && removed < extra; i--) {
    if (result[i] === 1) {
      result.splice(i, 1);
      removed++;
    }
  }
  if (removed === extra) return result;

  return null;
}

export function broadcastNdArray(nd: any, targetShape: number[]): any {
  if (arraysEqual(nd.shape, targetShape)) return nd;
  if (!isCompatibleBroadcast(nd.shape, targetShape)) {
    if (nd.shape.length > targetShape.length) {
      const squeezedShape = squeezeToRank(nd.shape, targetShape.length);
      if (squeezedShape && isCompatibleBroadcast(squeezedShape, targetShape)) {
        const squeezed = createNdArray(nd.data, squeezedShape);
        return expandNdArray(squeezed, targetShape);
      }
    }
    throw new Error(`Cannot broadcast shape ${JSON.stringify(nd.shape)} to ${JSON.stringify(targetShape)}`);
  }
  return expandNdArray(nd, targetShape);
}

export function arraysEqual(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  return a.every((v, i) => v === b[i]);
}

/**
 * Sum an ndarray along the specified dimension.
 */
export function sumAlongDimNdArray(nd: any, dim?: number | null, keepdim = false): any {
  if (dim === null || dim === undefined) {
    let total = 0;
    for (let i = 0; i < nd.data.length; i++) {
      total += nd.data[i + nd.offset];
    }
    const scalar = createNdArray(new Float32Array([total]), []);
    if (keepdim) {
      return createNdArray(scalar.data, new Array(nd.shape.length).fill(1));
    }
    return scalar;
  }
  const d = normalizeDim(dim, nd.shape.length);
  const newShape: number[] = Array.from(nd.shape);
  newShape[d] = 1;
  const outSize = getShapeSize(newShape);
  const outBuf = new Float32Array(outSize).fill(0);
  const outStrides = computeRowMajorStrides(newShape);

  function iterate(pos: number[], cd: number) {
    if (cd === newShape.length) {
      let outIdx = 0;
      for (let i = 0; i < pos.length; i++) {
        outIdx += pos[i] * outStrides[i];
      }
      for (let k = 0; k < nd.shape[d]; k++) {
        const srcPos = [...pos];
        srcPos[d] = k;
        let srcIdx = nd.offset;
        for (let i = 0; i < srcPos.length; i++) {
          srcIdx += srcPos[i] * nd.strides[i];
        }
        outBuf[outIdx] += nd.data[srcIdx];
      }
      return;
    }
    for (let i = 0; i < newShape[cd]; i++) {
      pos.push(i);
      iterate(pos, cd + 1);
      pos.pop();
    }
  }
  iterate([], 0);
  const result = createNdArray(outBuf, newShape);
  if (keepdim) return result;
  if (nd.shape.length === 1) return createNdArray(result.data, []);
  return squeezeNdArray(result, d);
}

/**
 * Concatenate a list of ndarrays along the given dimension.
 */
export function concatNdArray(nds: any[], dim: number): any {
  if (nds.length === 0) throw new Error('concatNdArray requires at least one ndarray');
  const refShape = nds[0].shape;
  const ndim = refShape.length;
  const d = normalizeDim(dim, ndim);
  for (let i = 1; i < nds.length; i++) {
    const shape = nds[i].shape;
    if (shape.length !== ndim) throw new Error('All ndarrays must have the same rank for concat');
    for (let j = 0; j < ndim; j++) {
      if (j !== d && shape[j] !== refShape[j]) {
        throw new Error(`Concat shape mismatch at dim ${j}: ${shape[j]} vs ${refShape[j]}`);
      }
    }
  }
  const outShape = [...refShape];
  outShape[d] = nds.reduce((sum, nd) => sum + nd.shape[d], 0);
  const outSize = getShapeSize(outShape);
  const outBuf = new Float32Array(outSize).fill(0);
  const outStrides = computeRowMajorStrides(outShape);

  let offset = 0;
  for (const nd of nds) {
    copyNdArrayInto(nd, outBuf, outShape, outStrides, d, offset);
    offset += nd.shape[d];
  }
  return createNdArray(outBuf, outShape);
}

function copyNdArrayInto(
  nd: any,
  outBuf: Float32Array,
  outShape: number[],
  outStrides: number[],
  dim: number,
  offset: number
): void {
  function iterate(pos: number[], d: number) {
    if (d === nd.shape.length) {
      const outPos = [...pos];
      outPos[dim] = offset + outPos[dim];
      let outIdx = 0;
      for (let i = 0; i < outPos.length; i++) {
        outIdx += outPos[i] * outStrides[i];
      }
      let srcIdx = 0;
      for (let i = 0; i < pos.length; i++) {
        srcIdx += pos[i] * nd.strides[i];
      }
      outBuf[outIdx] = nd.data[srcIdx + nd.offset];
      return;
    }
    for (let i = 0; i < nd.shape[d]; i++) {
      pos.push(i);
      iterate(pos, d + 1);
      pos.pop();
    }
  }
  iterate([], 0);
}

/**
 * Slice an ndarray along a dimension [start, end).
 */
export function sliceNdArray(nd: any, dim: number, start: number, end: number): any {
  const d = normalizeDim(dim, nd.shape.length);
  if (start < 0 || end > nd.shape[d] || start >= end) {
    throw new Error(`Invalid slice range [${start}, ${end}) on dim ${d} of size ${nd.shape[d]}`);
  }
  const newShape = [...nd.shape];
  newShape[d] = end - start;
  const outSize = getShapeSize(newShape);
  const outBuf = new Float32Array(outSize);
  const outStrides = computeRowMajorStrides(newShape);

  function iterate(pos: number[], cd: number) {
    if (cd === newShape.length) {
      const srcPos = [...pos];
      srcPos[d] = start + pos[d];
      let srcIdx = 0;
      for (let i = 0; i < srcPos.length; i++) {
        srcIdx += srcPos[i] * nd.strides[i];
      }
      let outIdx = 0;
      for (let i = 0; i < pos.length; i++) {
        outIdx += pos[i] * outStrides[i];
      }
      outBuf[outIdx] = nd.data[srcIdx + nd.offset];
      return;
    }
    for (let i = 0; i < newShape[cd]; i++) {
      pos.push(i);
      iterate(pos, cd + 1);
      pos.pop();
    }
  }
  iterate([], 0);
  return createNdArray(outBuf, newShape);
}

/**
 * Split an ndarray evenly into n parts along a dimension.
 */
export function splitNdArray(nd: any, dim: number, n: number): any[] {
  const d = normalizeDim(dim, nd.shape.length);
  const dimSize = nd.shape[d];
  if (dimSize % n !== 0) {
    throw new Error(`Cannot split dim ${d} of size ${dimSize} into ${n} equal parts`);
  }
  const chunk = dimSize / n;
  const results: any[] = [];
  for (let i = 0; i < n; i++) {
    const start = i * chunk;
    results.push(sliceNdArray(nd, d, start, start + chunk));
  }
  return results;
}

/**
 * Stack a list of ndarrays along a new dimension.
 */
export function stackNdArray(nds: any[], dim: number): any {
  if (nds.length === 0) throw new Error('stackNdArray requires at least one ndarray');
  const ndim = nds[0].shape.length;
  const d = dim < 0 ? dim + ndim + 1 : dim;
  if (d < 0 || d > ndim) {
    throw new Error(`Dimension ${dim} out of range for ${ndim}-D tensor`);
  }
  const unsqueezed = nds.map(nd => {
    const newShape = [...nd.shape.slice(0, d), 1, ...nd.shape.slice(d)];
    return createNdArray(nd.data, newShape);
  });
  return concatNdArray(unsqueezed, d);
}

export function computeBroadcastShape(shapes: number[][]): number[] {
  const maxRank = Math.max(...shapes.map(s => s.length));
  const result = new Array(maxRank).fill(1);
  for (const shape of shapes) {
    const offset = maxRank - shape.length;
    for (let i = 0; i < shape.length; i++) {
      const s = shape[i];
      const t = result[offset + i];
      if (s === 1) continue;
      if (t === 1) {
        result[offset + i] = s;
      } else if (t !== s) {
        throw new Error(`Cannot broadcast shapes ${JSON.stringify(shapes)}`);
      }
    }
  }
  return result;
}

/**
 * Sum a broadcasted gradient back to a target shape, matching PyTorch's sum_to_size.
 */
export function sumToShapeNdArray(grad: any, targetShape: number[]): any {
  if (arraysEqual(grad.shape, targetShape)) return grad;
  const offset = grad.shape.length - targetShape.length;
  let result = grad;
  for (let i = 0; i < targetShape.length; i++) {
    const expDim = grad.shape[offset + i];
    const targetDim = targetShape[i];
    if (targetDim === 1 && expDim !== 1) {
      result = sumAlongDimNdArray(result, offset + i, true);
    }
  }
  for (let i = 0; i < offset; i++) {
    result = sumAlongDimNdArray(result, 0, true);
  }
  if (targetShape.length < result.shape.length) {
    const extra = result.shape.length - targetShape.length;
    for (let i = 0; i < extra; i++) {
      result = squeezeNdArray(result, 0);
    }
  }
  return result;
}

/**
 * Element-wise masked selection: cond ? a : b, with broadcasting.
 */
export function whereNdArray(cond: any, a: any, b: any): any {
  const shape = computeBroadcastShape([cond.shape, a.shape, b.shape]);
  const condB = broadcastNdArray(cond, shape);
  const aB = broadcastNdArray(a, shape);
  const bB = broadcastNdArray(b, shape);
  const size = aB.data.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    outBuf[i] = condB.data[i] ? aB.data[i] : bB.data[i];
  }
  return createNdArray(outBuf, shape);
}

/**
 * Element-wise clamp to [min, max].
 */
export function clampNdArray(nd: any, min: number, max: number): any {
  const size = nd.data.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    outBuf[i] = Math.min(max, Math.max(min, nd.data[i]));
  }
  return createNdArray(outBuf, Array.from(nd.shape));
}

// ============================================================================
// Element-wise Math Functions
// ============================================================================

export function powNdArray(nd: any, p: number): any {
  const size = nd.data.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    outBuf[i] = Math.pow(nd.data[i], p);
  }
  return createNdArray(outBuf, Array.from(nd.shape));
}

export function expNdArray(nd: any): any {
  const size = nd.data.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    outBuf[i] = Math.exp(nd.data[i]);
  }
  return createNdArray(outBuf, Array.from(nd.shape));
}

export function logNdArray(nd: any): any {
  const size = nd.data.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    outBuf[i] = Math.log(nd.data[i]);
  }
  return createNdArray(outBuf, Array.from(nd.shape));
}

export function sqrtNdArray(nd: any): any {
  const size = nd.data.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    outBuf[i] = Math.sqrt(nd.data[i]);
  }
  return createNdArray(outBuf, Array.from(nd.shape));
}

export function absNdArray(nd: any): any {
  const size = nd.data.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    outBuf[i] = Math.abs(nd.data[i]);
  }
  return createNdArray(outBuf, Array.from(nd.shape));
}

// ============================================================================
// Activation Functions
// ============================================================================

export function geluNdArray(nd: any): any {
  const size = nd.data.length;
  const outBuf = new Float32Array(size);
  const sqrt2OverPi = Math.sqrt(2.0 / Math.PI);
  for (let i = 0; i < size; i++) {
    const x = nd.data[i];
    const c = sqrt2OverPi * (x + 0.044715 * x * x * x);
    outBuf[i] = 0.5 * x * (1.0 + Math.tanh(c));
  }
  return createNdArray(outBuf, Array.from(nd.shape));
}

export function geluGradNdArray(gradOut: any, x: any): any {
  const size = gradOut.data.length;
  const outBuf = new Float32Array(size);
  const sqrt2OverPi = Math.sqrt(2.0 / Math.PI);
  for (let i = 0; i < size; i++) {
    const xv = x.data[i];
    const c = sqrt2OverPi * (xv + 0.044715 * xv * xv * xv);
    const tanhC = Math.tanh(c);
    const sech2 = 1.0 - tanhC * tanhC;
    const dCdx = sqrt2OverPi * (1.0 + 3.0 * 0.044715 * xv * xv);
    const dgelu = 0.5 * (1.0 + tanhC) + 0.5 * xv * sech2 * dCdx;
    outBuf[i] = gradOut.data[i] * dgelu;
  }
  return createNdArray(outBuf, Array.from(gradOut.shape));
}

export function sigmoidNdArray(nd: any): any {
  const size = nd.data.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    outBuf[i] = 1.0 / (1.0 + Math.exp(-nd.data[i]));
  }
  return createNdArray(outBuf, Array.from(nd.shape));
}

export function sigmoidGradNdArray(gradOut: any, x: any): any {
  const size = gradOut.data.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    const s = 1.0 / (1.0 + Math.exp(-x.data[i]));
    outBuf[i] = gradOut.data[i] * s * (1.0 - s);
  }
  return createNdArray(outBuf, Array.from(gradOut.shape));
}

export function tanhNdArray(nd: any): any {
  const size = nd.data.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    outBuf[i] = Math.tanh(nd.data[i]);
  }
  return createNdArray(outBuf, Array.from(nd.shape));
}

export function tanhGradNdArray(gradOut: any, x: any): any {
  const size = gradOut.data.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    const t = Math.tanh(x.data[i]);
    outBuf[i] = gradOut.data[i] * (1.0 - t * t);
  }
  return createNdArray(outBuf, Array.from(gradOut.shape));
}

function posFromOuterIndex(outer: number, shape: number[], d: number): number[] {
  const pos: number[] = new Array(shape.length).fill(0);
  let tmp = outer;
  for (let i = shape.length - 1; i >= 0; i--) {
    if (i === d) continue;
    pos[i] = tmp % shape[i];
    tmp = Math.floor(tmp / shape[i]);
  }
  return pos;
}

export function softmaxNdArray(nd: any, dim: number): any {
  const d = normalizeDim(dim, nd.shape.length);
  const outShape: number[] = Array.from(nd.shape);
  const outSize = getShapeSize(outShape);
  const outBuf = new Float32Array(outSize);
  const outStrides = computeRowMajorStrides(outShape);

  const outerSize = outSize / outShape[d];
  const dimSize = outShape[d];

  for (let outer = 0; outer < outerSize; outer++) {
    const pos = posFromOuterIndex(outer, outShape, d);
    let maxVal = -Infinity;
    for (let k = 0; k < dimSize; k++) {
      pos[d] = k;
      let srcIdx = nd.offset;
      for (let i = 0; i < pos.length; i++) srcIdx += pos[i] * nd.strides[i];
      maxVal = Math.max(maxVal, nd.data[srcIdx]);
    }
    let sumExp = 0;
    const expVals = new Float32Array(dimSize);
    for (let k = 0; k < dimSize; k++) {
      pos[d] = k;
      let srcIdx = nd.offset;
      for (let i = 0; i < pos.length; i++) srcIdx += pos[i] * nd.strides[i];
      const e = Math.exp(nd.data[srcIdx] - maxVal);
      expVals[k] = e;
      sumExp += e;
    }
    for (let k = 0; k < dimSize; k++) {
      pos[d] = k;
      let outIdx = 0;
      for (let i = 0; i < pos.length; i++) outIdx += pos[i] * outStrides[i];
      outBuf[outIdx] = expVals[k] / sumExp;
    }
  }
  return createNdArray(outBuf, outShape);
}

export function softmaxGradNdArray(gradOutput: any, out: any, dim: number): any {
  const d = normalizeDim(dim, out.shape.length);
  const outShape: number[] = Array.from(out.shape);
  const outSize = getShapeSize(outShape);
  const outBuf = new Float32Array(outSize);
  const outStrides = computeRowMajorStrides(outShape);

  const outerSize = outSize / outShape[d];
  const dimSize = outShape[d];

  const weightedSum = new Float32Array(outerSize).fill(0);
  for (let outer = 0; outer < outerSize; outer++) {
    const pos = posFromOuterIndex(outer, outShape, d);
    let sum = 0;
    for (let k = 0; k < dimSize; k++) {
      pos[d] = k;
      let outIdx = 0;
      let gradIdx = 0;
      for (let i = 0; i < pos.length; i++) {
        outIdx += pos[i] * outStrides[i];
        gradIdx += pos[i] * gradOutput.strides[i];
      }
      sum += out.data[outIdx + out.offset] * gradOutput.data[gradIdx + gradOutput.offset];
    }
    weightedSum[outer] = sum;
  }

  for (let outer = 0; outer < outerSize; outer++) {
    const pos = posFromOuterIndex(outer, outShape, d);
    for (let k = 0; k < dimSize; k++) {
      pos[d] = k;
      let outIdx = 0;
      let gradIdx = 0;
      let resIdx = 0;
      for (let i = 0; i < pos.length; i++) {
        outIdx += pos[i] * outStrides[i];
        gradIdx += pos[i] * gradOutput.strides[i];
        resIdx += pos[i] * outStrides[i];
      }
      outBuf[resIdx] = out.data[outIdx + out.offset] * (gradOutput.data[gradIdx + gradOutput.offset] - weightedSum[outer]);
    }
  }
  return createNdArray(outBuf, outShape);
}

// ============================================================================
// Loss Functions
// ============================================================================

export function crossEntropyLossNdArray(logits: any, labels: any): { lossValue: number; gradLogits: any } {
  const shape: number[] = logits.shape;
  if (shape.length !== 2) {
    throw new Error('CrossEntropyLoss expects 2D logits [N, C]');
  }
  const N = shape[0];
  const C = shape[1];
  const logitsFlat = logits.data;
  const labelsFlat = labels.data;
  const eps = 1e-12;

  const softmax = new Float32Array(N * C);
  let loss = 0;
  for (let i = 0; i < N; i++) {
    let maxVal = -Infinity;
    for (let j = 0; j < C; j++) {
      maxVal = Math.max(maxVal, logitsFlat[i * C + j]);
    }
    let sumExp = 0;
    for (let j = 0; j < C; j++) {
      const e = Math.exp(logitsFlat[i * C + j] - maxVal);
      softmax[i * C + j] = e;
      sumExp += e;
    }
    for (let j = 0; j < C; j++) {
      softmax[i * C + j] /= sumExp;
    }
    const label = Math.round(labelsFlat[i]);
    loss -= Math.log(Math.max(softmax[i * C + label], eps));
  }
  loss /= N;

  const gradBuf = new Float32Array(N * C);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < C; j++) {
      gradBuf[i * C + j] = softmax[i * C + j] / N;
    }
    const label = Math.round(labelsFlat[i]);
    gradBuf[i * C + label] -= 1.0 / N;
  }
  return { lossValue: loss, gradLogits: createNdArray(gradBuf, shape) };
}

export function bceLossNdArray(pred: any, target: any): { lossValue: number; gradPred: any } {
  const size = pred.data.length;
  const eps = 1e-12;
  let loss = 0;
  const gradBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    const p = Math.min(Math.max(pred.data[i], eps), 1 - eps);
    const t = target.data[i];
    loss -= (t * Math.log(p) + (1 - t) * Math.log(1 - p));
    gradBuf[i] = (p - t) / (p * (1 - p) * size);
  }
  loss /= size;
  return { lossValue: loss, gradPred: createNdArray(gradBuf, Array.from(pred.shape)) };
}

export function l1LossNdArray(pred: any, target: any): { lossValue: number; gradPred: any } {
  const size = pred.data.length;
  let loss = 0;
  const gradBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    const diff = pred.data[i] - target.data[i];
    loss += Math.abs(diff);
    gradBuf[i] = (diff > 0 ? 1 : diff < 0 ? -1 : 0) / size;
  }
  loss /= size;
  return { lossValue: loss, gradPred: createNdArray(gradBuf, Array.from(pred.shape)) };
}

// ============================================================================
// Normalization Operations
// ============================================================================

/**
 * Compute mean and inverse standard deviation over the last n dimensions.
 * Input shape: [..., d_1, ..., d_n]; output shape keeps leading dims and sets last n dims to 1.
 */
function computeMomentsOverLastDims(nd: any, n: number): { mean: any; invStd: any } {
  const shape: number[] = Array.from(nd.shape);
  const rank = shape.length;
  if (n <= 0 || n > rank) {
    throw new Error(`Cannot normalize last ${n} dims of a ${rank}-D tensor`);
  }
  const leadingSize = shape.slice(0, rank - n).reduce((a: number, b: number) => a * b, 1);
  const normalizedSize = shape.slice(rank - n).reduce((a: number, b: number) => a * b, 1);

  const x2d = createNdArray(nd.data, [leadingSize, normalizedSize]);
  const mean2d = meanNdArray(x2d, 1, true);
  const centered2d = subNdArray(x2d, expandNdArray(mean2d, [leadingSize, normalizedSize]));
  const var2d = meanNdArray(mulNdArray(centered2d, centered2d), 1, true);

  const invStdBuf = new Float32Array(leadingSize);
  for (let i = 0; i < leadingSize; i++) {
    invStdBuf[i] = 1.0 / Math.sqrt(var2d.data[i] + 1e-5);
  }
  const invStd2d = createNdArray(invStdBuf, [leadingSize, 1]);

  const meanShape: number[] = shape.map((s: number, i: number) => (i < rank - n ? s : 1));
  return {
    mean: createNdArray(mean2d.data, meanShape),
    invStd: createNdArray(invStd2d.data, meanShape)
  };
}

/**
 * LayerNorm forward: normalize over the last n dimensions.
 */
export function layerNormNdArray(nd: any, n: number): { out: any; mean: any; invStd: any } {
  const shape: number[] = Array.from(nd.shape);
  const rank = shape.length;
  const leadingSize = shape.slice(0, rank - n).reduce((a: number, b: number) => a * b, 1);
  const normalizedSize = shape.slice(rank - n).reduce((a: number, b: number) => a * b, 1);

  const { mean, invStd } = computeMomentsOverLastDims(nd, n);
  const centered2d = subNdArray(
    createNdArray(nd.data, [leadingSize, normalizedSize]),
    expandNdArray(createNdArray(mean.data, [leadingSize, 1]), [leadingSize, normalizedSize])
  );
  const out2d = mulNdArray(centered2d, expandNdArray(createNdArray(invStd.data, [leadingSize, 1]), [leadingSize, normalizedSize]));
  return { out: createNdArray(out2d.data, shape), mean, invStd };
}

/**
 * LayerNorm backward.
 */
export function layerNormGradNdArray(gradOutput: any, x: any, mean: any, invStd: any, n: number): any {
  const shape: number[] = Array.from(x.shape);
  const rank = shape.length;
  const leadingSize = shape.slice(0, rank - n).reduce((a: number, b: number) => a * b, 1);
  const normalizedSize = shape.slice(rank - n).reduce((a: number, b: number) => a * b, 1);

  const x2d = createNdArray(x.data, [leadingSize, normalizedSize]);
  const mean2d = createNdArray(mean.data, [leadingSize, 1]);
  const invStd2d = createNdArray(invStd.data, [leadingSize, 1]);
  const out2d = mulNdArray(
    subNdArray(x2d, expandNdArray(mean2d, [leadingSize, normalizedSize])),
    expandNdArray(invStd2d, [leadingSize, normalizedSize])
  );
  const gradOutput2d = createNdArray(gradOutput.data, [leadingSize, normalizedSize]);

  const meanGrad = meanNdArray(gradOutput2d, 1, true);
  const meanGradOut = meanNdArray(mulNdArray(gradOutput2d, out2d), 1, true);
  const term1 = subNdArray(gradOutput2d, expandNdArray(meanGrad, [leadingSize, normalizedSize]));
  const term2 = mulNdArray(out2d, expandNdArray(meanGradOut, [leadingSize, normalizedSize]));
  const dx2d = mulNdArray(
    expandNdArray(invStd2d, [leadingSize, normalizedSize]),
    subNdArray(term1, term2)
  );
  return createNdArray(dx2d.data, shape);
}

// ============================================================================
// Reduction Operations
// ============================================================================

function reduceAlongDimNdArray(
  nd: any,
  dim: number,
  identity: number,
  reducer: (a: number, b: number) => number
): any {
  const d = normalizeDim(dim, nd.shape.length);
  const newShape: number[] = Array.from(nd.shape);
  newShape[d] = 1;
  const outSize = getShapeSize(newShape);
  const outBuf = new Float32Array(outSize).fill(identity);
  const outStrides = computeRowMajorStrides(newShape);

  function iterate(pos: number[], cd: number) {
    if (cd === newShape.length) {
      let outIdx = 0;
      for (let i = 0; i < pos.length; i++) {
        outIdx += pos[i] * outStrides[i];
      }
      for (let k = 0; k < nd.shape[d]; k++) {
        const srcPos = [...pos];
        srcPos[d] = k;
        let srcIdx = nd.offset;
        for (let i = 0; i < srcPos.length; i++) {
          srcIdx += srcPos[i] * nd.strides[i];
        }
        outBuf[outIdx] = reducer(outBuf[outIdx], nd.data[srcIdx]);
      }
      return;
    }
    for (let i = 0; i < newShape[cd]; i++) {
      pos.push(i);
      iterate(pos, cd + 1);
      pos.pop();
    }
  }
  iterate([], 0);
  return createNdArray(outBuf, newShape);
}

export function maxNdArray(nd: any, dim?: number | null, keepdim = false): any {
  if (dim === null || dim === undefined) {
    let maxVal = -Infinity;
    for (let i = 0; i < nd.data.length; i++) {
      maxVal = Math.max(maxVal, nd.data[i + nd.offset]);
    }
    const scalar = createNdArray(new Float32Array([maxVal]), []);
    if (keepdim) {
      return createNdArray(scalar.data, new Array(nd.shape.length).fill(1));
    }
    return scalar;
  }
  const d = normalizeDim(dim, nd.shape.length);
  const result = reduceAlongDimNdArray(nd, d, -Infinity, (a, b) => Math.max(a, b));
  if (keepdim) return result;
  if (nd.shape.length === 1) return createNdArray(result.data, []);
  return squeezeNdArray(result, d);
}

export function minNdArray(nd: any, dim?: number | null, keepdim = false): any {
  if (dim === null || dim === undefined) {
    let minVal = Infinity;
    for (let i = 0; i < nd.data.length; i++) {
      minVal = Math.min(minVal, nd.data[i + nd.offset]);
    }
    const scalar = createNdArray(new Float32Array([minVal]), []);
    if (keepdim) {
      return createNdArray(scalar.data, new Array(nd.shape.length).fill(1));
    }
    return scalar;
  }
  const d = normalizeDim(dim, nd.shape.length);
  const result = reduceAlongDimNdArray(nd, d, Infinity, (a, b) => Math.min(a, b));
  if (keepdim) return result;
  if (nd.shape.length === 1) return createNdArray(result.data, []);
  return squeezeNdArray(result, d);
}

export function meanNdArray(nd: any, dim?: number | null, keepdim = false): any {
  if (dim === null || dim === undefined) {
    let total = 0;
    for (let i = 0; i < nd.data.length; i++) {
      total += nd.data[i + nd.offset];
    }
    const scalar = createNdArray(new Float32Array([total / nd.data.length]), []);
    if (keepdim) {
      return createNdArray(scalar.data, new Array(nd.shape.length).fill(1));
    }
    return scalar;
  }
  const d = normalizeDim(dim, nd.shape.length);
  const summed = sumAlongDimNdArray(nd, d, true);
  const size = nd.shape[d];
  const outBuf = new Float32Array(summed.data.length);
  for (let i = 0; i < outBuf.length; i++) {
    outBuf[i] = summed.data[i] / size;
  }
  const result = createNdArray(outBuf, Array.from(summed.shape));
  if (keepdim) return result;
  if (nd.shape.length === 1) return createNdArray(result.data, []);
  return squeezeNdArray(result, d);
}

// ============================================================================
// Transpose and BMM
// ============================================================================

export function transposeNdArray(nd: any, dim1: number, dim2: number): any {
  const d1 = normalizeDim(dim1, nd.shape.length);
  const d2 = normalizeDim(dim2, nd.shape.length);
  const outShape: number[] = Array.from(nd.shape);
  outShape[d1] = nd.shape[d2];
  outShape[d2] = nd.shape[d1];
  const outSize = getShapeSize(outShape);
  const outBuf = new Float32Array(outSize);
  const outStrides = computeRowMajorStrides(outShape);

  function iterate(pos: number[], cd: number) {
    if (cd === outShape.length) {
      const srcPos = [...pos];
      srcPos[d1] = pos[d2];
      srcPos[d2] = pos[d1];
      let srcIdx = nd.offset;
      for (let i = 0; i < srcPos.length; i++) {
        srcIdx += srcPos[i] * nd.strides[i];
      }
      let outIdx = 0;
      for (let i = 0; i < pos.length; i++) {
        outIdx += pos[i] * outStrides[i];
      }
      outBuf[outIdx] = nd.data[srcIdx];
      return;
    }
    for (let i = 0; i < outShape[cd]; i++) {
      pos.push(i);
      iterate(pos, cd + 1);
      pos.pop();
    }
  }
  iterate([], 0);
  return createNdArray(outBuf, outShape);
}

export function bmmNdArray(a: any, b: any): any {
  const shapeA = a.shape;
  const shapeB = b.shape;
  if (shapeA.length !== 3 || shapeB.length !== 3) {
    throw new Error('BMM expects 3D inputs [B, M, K] and [B, K, N]');
  }
  const B = shapeA[0];
  const M = shapeA[1];
  const K = shapeA[2];
  const N = shapeB[2];
  if (shapeB[0] !== B || shapeB[1] !== K) {
    throw new Error(`BMM shape mismatch: ${JSON.stringify(shapeA)} vs ${JSON.stringify(shapeB)}`);
  }
  const outBuf = new Float32Array(B * M * N).fill(0);
  for (let batch = 0; batch < B; batch++) {
    for (let i = 0; i < M; i++) {
      for (let k = 0; k < K; k++) {
        const aik = a.get(batch, i, k);
        for (let j = 0; j < N; j++) {
          outBuf[batch * M * N + i * N + j] += aik * b.get(batch, k, j);
        }
      }
    }
  }
  return createNdArray(outBuf, [B, M, N]);
}

// ============================================================================
// Dropout
// ============================================================================

export function dropoutNdArray(nd: any, p: number, isTraining: boolean): { out: any; mask: Float32Array } {
  const size = nd.data.length;
  const mask = new Float32Array(size).fill(1);
  if (!isTraining || p <= 0) {
    return { out: cloneNdArray(nd), mask };
  }
  if (p >= 1) {
    mask.fill(0);
    return { out: createNdArray(new Float32Array(size), Array.from(nd.shape)), mask };
  }
  const outBuf = new Float32Array(size);
  const scale = 1.0 / (1.0 - p);
  for (let i = 0; i < size; i++) {
    const keep = Math.random() >= p;
    mask[i] = keep ? 1 : 0;
    outBuf[i] = nd.data[i] * mask[i] * scale;
  }
  return { out: createNdArray(outBuf, Array.from(nd.shape)), mask };
}

// ============================================================================
// Scaled Dot Product Attention
// ============================================================================

function transposeLastTwoNdArray(nd: any): any {
  const rank = nd.shape.length;
  if (rank < 2) throw new Error('transposeLastTwoNdArray requires at least 2D tensor');
  return transposeNdArray(nd, rank - 2, rank - 1);
}

function applyCausalMaskNdArray(scores: any): any {
  const shape: number[] = Array.from(scores.shape);
  const rank = shape.length;
  if (rank < 2) throw new Error('applyCausalMaskNdArray requires at least 2D tensor');
  const Lq = shape[rank - 2] as number;
  const Lk = shape[rank - 1] as number;
  const outerSize = scores.data.length / (Lq * Lk);
  const outBuf = new Float32Array(scores.data.length);
  outBuf.set(scores.data);
  for (let outer = 0; outer < outerSize; outer++) {
    const base = outer * Lq * Lk;
    for (let i = 0; i < Lq; i++) {
      for (let j = i + 1; j < Lk; j++) {
        outBuf[base + i * Lk + j] = -Infinity;
      }
    }
  }
  return createNdArray(outBuf, shape);
}

function mulScalarNdArray(nd: any, scalar: number): any {
  const size = nd.data.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) outBuf[i] = nd.data[i] * scalar;
  return createNdArray(outBuf, Array.from(nd.shape));
}

function flattenBatchHeadNdArray(nd: any): any {
  const shape: number[] = Array.from(nd.shape);
  if (shape.length < 4) throw new Error('flattenBatchHeadNdArray requires 4D tensor');
  const B = shape[0] as number;
  const H = shape[1] as number;
  const rest = shape.slice(2);
  return createNdArray(nd.data, [B * H, ...rest]);
}

function unflattenBatchHeadNdArray(nd: any, B: number, H: number): any {
  const shape: number[] = Array.from(nd.shape);
  if (shape.length < 3 || shape[0] !== B * H) {
    throw new Error('unflattenBatchHeadNdArray: first dim must equal B*H');
  }
  const newShape = [B, H, ...shape.slice(1)];
  return createNdArray(nd.data, newShape);
}

export function scaledDotProductAttentionNdArray(
  q: any,
  k: any,
  v: any,
  isTraining: boolean,
  dropoutP: number,
  isCausal: boolean,
  scale?: number,
  attnMask?: any
): { out: any; scores: any; attn: any; dropoutMask: Float32Array; scale: number } {
  const qShape: number[] = Array.from(q.shape);
  const kShape: number[] = Array.from(k.shape);
  const vShape: number[] = Array.from(v.shape);
  if (qShape.length !== 4 || kShape.length !== 4 || vShape.length !== 4) {
    throw new Error('Scaled dot product attention expects 4D inputs [B, H, L, D]');
  }
  const B = qShape[0] as number;
  const H = qShape[1] as number;
  const Lq = qShape[2] as number;
  const Dq = qShape[3] as number;
  const Lk = kShape[2] as number;
  const Dk = kShape[3] as number;
  const Dv = vShape[3] as number;
  if (B !== kShape[0] || B !== vShape[0] || H !== kShape[1] || H !== vShape[1]) {
    throw new Error('Scaled dot product attention batch/head mismatch');
  }
  if (Lk !== vShape[2]) {
    throw new Error('Scaled dot product attention key/value sequence length mismatch');
  }
  if (Dk !== Dq) {
    throw new Error('Scaled dot product attention query/key head dimension mismatch');
  }

  const effScale = scale ?? 1.0 / Math.sqrt(Dq);

  const q3d = flattenBatchHeadNdArray(q);
  const k3d = flattenBatchHeadNdArray(k);
  const v3d = flattenBatchHeadNdArray(v);

  const kt3d = transposeLastTwoNdArray(k3d);
  const scores3d = bmmNdArray(q3d, kt3d);
  let scores3dScaled = mulScalarNdArray(scores3d, effScale);

  if (attnMask) {
    let mask3d = attnMask;
    if (mask3d.shape.length === 4) {
      mask3d = flattenBatchHeadNdArray(mask3d);
    }
    mask3d = broadcastNdArray(mask3d, [B * H, Lq, Lk]);
    scores3dScaled = addNdArray(scores3dScaled, mask3d);
  }

  if (isCausal) {
    scores3dScaled = applyCausalMaskNdArray(scores3dScaled);
  }

  const attn3d = softmaxNdArray(scores3dScaled, -1);

  let dropoutMask = new Float32Array(attn3d.data.length).fill(1);
  let attn3dDropped = attn3d;
  if (isTraining && dropoutP > 0) {
    const { out, mask } = dropoutNdArray(attn3d, dropoutP, true);
    attn3dDropped = out;
    dropoutMask = mask;
  }

  const output3d = bmmNdArray(attn3dDropped, v3d);

  const out = unflattenBatchHeadNdArray(output3d, B, H);
  const scores = unflattenBatchHeadNdArray(scores3dScaled, B, H);
  const attn = unflattenBatchHeadNdArray(attn3dDropped, B, H);

  return { out, scores, attn, dropoutMask, scale: effScale };
}

export function scaledDotProductAttentionGradNdArray(
  gradOutput: any,
  q: any,
  k: any,
  v: any,
  attn: any,
  dropoutMask: Float32Array,
  scale: number,
  isTraining: boolean,
  dropoutP: number
): { dQ: any; dK: any; dV: any } {
  const qShape: number[] = Array.from(q.shape);
  const B = qShape[0] as number;
  const H = qShape[1] as number;
  const Lq = qShape[2] as number;
  const Lk = k.shape[2] as number;
  const Dv = v.shape[3] as number;

  const dO3d = flattenBatchHeadNdArray(gradOutput);
  const v3d = flattenBatchHeadNdArray(v);
  const k3d = flattenBatchHeadNdArray(k);
  const q3d = flattenBatchHeadNdArray(q);
  const attn3d = flattenBatchHeadNdArray(attn);

  const dV3d = bmmNdArray(transposeLastTwoNdArray(attn3d), dO3d);

  const dAttn3d = bmmNdArray(dO3d, transposeLastTwoNdArray(v3d));
  if (isTraining && dropoutP > 0) {
    const dropoutScale = 1.0 / (1.0 - dropoutP);
    const size = dAttn3d.data.length;
    const buf = new Float32Array(size);
    for (let i = 0; i < size; i++) buf[i] = dAttn3d.data[i] * dropoutMask[i] * dropoutScale;
    dAttn3d.data = buf;
  }

  const dScores3d = softmaxGradNdArray(dAttn3d, attn3d, -1);
  // Apply causal mask gradient: zero out positions that were masked in forward
  // Forward causal mask zeros the upper triangular part of scores
  const LqLocal = Lq;
  const LkLocal = Lk;
  const outerSize = dScores3d.data.length / (LqLocal * LkLocal);
  for (let outer = 0; outer < outerSize; outer++) {
    const base = outer * LqLocal * LkLocal;
    for (let i = 0; i < LqLocal; i++) {
      for (let j = i + 1; j < LkLocal; j++) {
        dScores3d.data[base + i * LkLocal + j] = 0;
      }
    }
  }

  dScores3d.data = mulScalarNdArray(dScores3d, scale).data;

  const dQ3d = bmmNdArray(dScores3d, k3d);
  const dK3d = bmmNdArray(transposeLastTwoNdArray(dScores3d), q3d);

  return {
    dQ: unflattenBatchHeadNdArray(dQ3d, B, H),
    dK: unflattenBatchHeadNdArray(dK3d, B, H),
    dV: unflattenBatchHeadNdArray(dV3d, B, H)
  };
}
