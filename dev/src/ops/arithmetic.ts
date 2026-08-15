import { TensorNode, nextId, registerTensor } from '../tensor.js';
import { OpNode } from '../types.js';
import { isGraphTracking } from '../autograd.js';
import {
  addNdArray,
  subNdArray,
  mulNdArray,
  divNdArray,
  matMulNdArray,
  transposeNdArray,
  sumToShapeNdArray,
  broadcastNdArray,
  createNdArray
} from '../ndarray-utils.js';

export class MatMulOp implements OpNode {
  opName = 'MatMul';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(A: TensorNode, B: TensorNode, outName: string) {
    this.inputs = [A, B];
    const outData = matMulNdArray(A.data, B.data);
    const requiresGrad = (A.requiresGrad || B.requiresGrad) && isGraphTracking();
    this.output = new TensorNode(nextId(), outName, outData, requiresGrad);
    this.output.isLeaf = false;
    if (requiresGrad) {
      this.output.creatorOp = this;
    }
    registerTensor(this.output);
  }

  backwardFn(gradOutput: any): void {
    const [A, B] = this.inputs;
    const rankA = A.shape.length;
    const rankB = B.shape.length;

    if (rankA === 1 && rankB === 1) {
      const scale = gradOutput.data[gradOutput.offset];
      if (A.requiresGrad) {
        const buf = new Float32Array(A.data.data.length);
        for (let i = 0; i < buf.length; i++) {
          buf[i] = scale * B.data.data[i + B.data.offset];
        }
        A.addGrad(createNdArray(buf, A.shape));
      }
      if (B.requiresGrad) {
        const buf = new Float32Array(B.data.data.length);
        for (let i = 0; i < buf.length; i++) {
          buf[i] = scale * A.data.data[i + A.data.offset];
        }
        B.addGrad(createNdArray(buf, B.shape));
      }
      return;
    }

    // 2D x 1D: matrix-vector
    if (rankA === 2 && rankB === 1) {
      if (A.requiresGrad) {
        const M = A.shape[0];
        const K = A.shape[1];
        const buf = new Float32Array(M * K);
        for (let i = 0; i < M; i++) {
          const g = gradOutput.data[i + gradOutput.offset];
          for (let k = 0; k < K; k++) {
            buf[i * K + k] = g * B.data.data[k + B.data.offset];
          }
        }
        A.addGrad(createNdArray(buf, A.shape));
      }
      if (B.requiresGrad) {
        const M = A.shape[0];
        const K = A.shape[1];
        const buf = new Float32Array(K).fill(0);
        for (let i = 0; i < M; i++) {
          const g = gradOutput.data[i + gradOutput.offset];
          for (let k = 0; k < K; k++) {
            buf[k] += A.data.data[i * A.data.strides[0] + k * A.data.strides[1] + A.data.offset] * g;
          }
        }
        B.addGrad(createNdArray(buf, B.shape));
      }
      return;
    }

    // 1D x 2D: vector-matrix
    if (rankA === 1 && rankB === 2) {
      if (A.requiresGrad) {
        const K = A.shape[0];
        const N = B.shape[1];
        const buf = new Float32Array(K).fill(0);
        for (let k = 0; k < K; k++) {
          let sum = 0;
          for (let j = 0; j < N; j++) {
            sum += B.data.data[k * B.data.strides[0] + j * B.data.strides[1] + B.data.offset] * gradOutput.data[j + gradOutput.offset];
          }
          buf[k] = sum;
        }
        A.addGrad(createNdArray(buf, A.shape));
      }
      if (B.requiresGrad) {
        const K = A.shape[0];
        const N = B.shape[1];
        const buf = new Float32Array(K * N);
        for (let k = 0; k < K; k++) {
          const a = A.data.data[k + A.data.offset];
          for (let j = 0; j < N; j++) {
            buf[k * N + j] = a * gradOutput.data[j + gradOutput.offset];
          }
        }
        B.addGrad(createNdArray(buf, B.shape));
      }
      return;
    }

    // General batched case: both >= 2D with broadcasting.
    if (A.requiresGrad) {
      const rankB = B.data.shape.length;
      const Bt = transposeNdArray(B.data, rankB - 2, rankB - 1);
      A.addGrad(sumToShapeNdArray(matMulNdArray(gradOutput, Bt), A.shape));
    }
    if (B.requiresGrad) {
      const rankA = A.data.shape.length;
      const At = transposeNdArray(A.data, rankA - 2, rankA - 1);
      B.addGrad(sumToShapeNdArray(matMulNdArray(At, gradOutput), B.shape));
    }
  }
}

function createNdArrayFromShape(buf: Float32Array, shape: number[]): any {
  return createNdArray(buf, shape);
}

export class AddOp implements OpNode {
  opName = 'Add';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(A: TensorNode, B: TensorNode, outName: string) {
    this.inputs = [A, B];
    const outData = addNdArray(A.data, B.data);
    const requiresGrad = (A.requiresGrad || B.requiresGrad) && isGraphTracking();
    this.output = new TensorNode(nextId(), outName, outData, requiresGrad);
    this.output.isLeaf = false;
    if (requiresGrad) {
      this.output.creatorOp = this;
    }
    registerTensor(this.output);
  }

  backwardFn(gradOutput: any): void {
    const [A, B] = this.inputs;
    if (A.requiresGrad) {
      A.addGrad(sumToShapeNdArray(gradOutput, A.shape));
    }
    if (B.requiresGrad) {
      B.addGrad(sumToShapeNdArray(gradOutput, B.shape));
    }
  }
}

export class SubOp implements OpNode {
  opName = 'Sub';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(A: TensorNode, B: TensorNode, outName: string) {
    this.inputs = [A, B];
    const outData = subNdArray(A.data, B.data);
    const requiresGrad = (A.requiresGrad || B.requiresGrad) && isGraphTracking();
    this.output = new TensorNode(nextId(), outName, outData, requiresGrad);
    this.output.isLeaf = false;
    if (requiresGrad) {
      this.output.creatorOp = this;
    }
    registerTensor(this.output);
  }

  backwardFn(gradOutput: any): void {
    const [A, B] = this.inputs;
    if (A.requiresGrad) {
      A.addGrad(sumToShapeNdArray(gradOutput, A.shape));
    }
    if (B.requiresGrad) {
      const negBuf = new Float32Array(gradOutput.data.length);
      for (let i = 0; i < negBuf.length; i++) {
        negBuf[i] = -gradOutput.data[i];
      }
      const negGrad = createNdArrayFromShape(negBuf, Array.from(gradOutput.shape));
      B.addGrad(sumToShapeNdArray(negGrad, B.shape));
    }
  }
}

export class MulOp implements OpNode {
  opName = 'Mul';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(A: TensorNode, B: TensorNode, outName: string) {
    this.inputs = [A, B];
    const outData = mulNdArray(A.data, B.data);
    const requiresGrad = (A.requiresGrad || B.requiresGrad) && isGraphTracking();
    this.output = new TensorNode(nextId(), outName, outData, requiresGrad);
    this.output.isLeaf = false;
    if (requiresGrad) {
      this.output.creatorOp = this;
    }
    registerTensor(this.output);
  }

  backwardFn(gradOutput: any): void {
    const [A, B] = this.inputs;
    if (A.requiresGrad) {
      const bBroadcast = broadcastNdArray(B.data, gradOutput.shape);
      A.addGrad(sumToShapeNdArray(mulNdArray(gradOutput, bBroadcast), A.shape));
    }
    if (B.requiresGrad) {
      const aBroadcast = broadcastNdArray(A.data, gradOutput.shape);
      B.addGrad(sumToShapeNdArray(mulNdArray(gradOutput, aBroadcast), B.shape));
    }
  }
}

export class DivOp implements OpNode {
  opName = 'Div';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(A: TensorNode, B: TensorNode, outName: string) {
    this.inputs = [A, B];
    const outData = divNdArray(A.data, B.data);
    const requiresGrad = (A.requiresGrad || B.requiresGrad) && isGraphTracking();
    this.output = new TensorNode(nextId(), outName, outData, requiresGrad);
    this.output.isLeaf = false;
    if (requiresGrad) {
      this.output.creatorOp = this;
    }
    registerTensor(this.output);
  }

  backwardFn(gradOutput: any): void {
    const [A, B] = this.inputs;
    if (A.requiresGrad) {
      const bBroadcast = broadcastNdArray(B.data, gradOutput.shape);
      A.addGrad(sumToShapeNdArray(divNdArray(gradOutput, bBroadcast), A.shape));
    }
    if (B.requiresGrad) {
      const aBroadcast = broadcastNdArray(A.data, gradOutput.shape);
      const bBroadcast = broadcastNdArray(B.data, gradOutput.shape);
      const size = gradOutput.data.length;
      const negBuf = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        const b = bBroadcast.data[i];
        negBuf[i] = -gradOutput.data[i] * aBroadcast.data[i] / (b * b);
      }
      const negGrad = createNdArrayFromShape(negBuf, Array.from(gradOutput.shape));
      B.addGrad(sumToShapeNdArray(negGrad, B.shape));
    }
  }
}

export function matmul(A: TensorNode, B: TensorNode, outName: string): TensorNode {
  const op = new MatMulOp(A, B, outName);
  return op.output;
}

export function add(A: TensorNode, B: TensorNode, outName: string): TensorNode {
  const op = new AddOp(A, B, outName);
  return op.output;
}

export function sub(A: TensorNode, B: TensorNode, outName: string): TensorNode {
  const op = new SubOp(A, B, outName);
  return op.output;
}

export function mul(A: TensorNode, B: TensorNode, outName: string): TensorNode {
  const op = new MulOp(A, B, outName);
  return op.output;
}

export function div(A: TensorNode, B: TensorNode, outName: string): TensorNode {
  const op = new DivOp(A, B, outName);
  return op.output;
}
