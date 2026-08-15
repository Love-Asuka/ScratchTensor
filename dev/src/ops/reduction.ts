import { TensorNode, nextId, registerTensor } from '../tensor.js';
import { OpNode } from '../types.js';
import { isGraphTracking } from '../autograd.js';
import {
  sumAlongDimNdArray,
  meanNdArray,
  maxNdArray,
  minNdArray,
  unsqueezeNdArray,
  expandNdArray,
  broadcastNdArray,
  computeRowMajorStrides,
  createNdArray
} from '../ndarray-utils.js';

export class SumOp implements OpNode {
  opName = 'Sum';
  inputs: TensorNode[];
  output: TensorNode;
  private dim: number | null;
  private keepdim: boolean;

  constructor(X: TensorNode, dim: number | null, outName: string, keepdim = false) {
    this.inputs = [X];
    this.dim = dim;
    this.keepdim = keepdim;
    const outData = sumAlongDimNdArray(X.data, dim, keepdim);
    const requiresGrad = X.requiresGrad && isGraphTracking();
    this.output = new TensorNode(nextId(), outName, outData, requiresGrad);
    this.output.isLeaf = false;
    if (requiresGrad) this.output.creatorOp = this;
    registerTensor(this.output);
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      if (this.dim === null || this.dim === undefined) {
        X.addGrad(broadcastNdArray(gradOutput, X.shape));
      } else {
        const d = this.dim < 0 ? this.dim + X.shape.length : this.dim;
        if (this.keepdim) {
          X.addGrad(expandNdArray(gradOutput, X.shape));
        } else {
          X.addGrad(expandNdArray(unsqueezeNdArray(gradOutput, d), X.shape));
        }
      }
    }
  }
}

export class MeanOp implements OpNode {
  opName = 'Mean';
  inputs: TensorNode[];
  output: TensorNode;
  private dim: number | null;
  private keepdim: boolean;

  constructor(X: TensorNode, dim: number | null, outName: string, keepdim = false) {
    this.inputs = [X];
    this.dim = dim;
    this.keepdim = keepdim;
    const outData = meanNdArray(X.data, dim, keepdim);
    const requiresGrad = X.requiresGrad && isGraphTracking();
    this.output = new TensorNode(nextId(), outName, outData, requiresGrad);
    this.output.isLeaf = false;
    if (requiresGrad) this.output.creatorOp = this;
    registerTensor(this.output);
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      let scaled;
      if (this.dim === null || this.dim === undefined) {
        const scale = 1.0 / X.data.data.length;
        const buf = new Float32Array(gradOutput.data.length);
        for (let i = 0; i < buf.length; i++) buf[i] = gradOutput.data[i] * scale;
        scaled = createNdArray(buf, Array.from(gradOutput.shape));
      } else {
        const dimSize = X.shape[this.dim];
        const buf = new Float32Array(gradOutput.data.length);
        for (let i = 0; i < buf.length; i++) buf[i] = gradOutput.data[i] / dimSize;
        scaled = createNdArray(buf, Array.from(gradOutput.shape));
      }
      if (this.dim === null || this.dim === undefined) {
        X.addGrad(broadcastNdArray(scaled, X.shape));
      } else {
        const d = this.dim < 0 ? this.dim + X.shape.length : this.dim;
        if (this.keepdim) {
          X.addGrad(expandNdArray(scaled, X.shape));
        } else {
          X.addGrad(expandNdArray(unsqueezeNdArray(scaled, d), X.shape));
        }
      }
    }
  }
}

export class MaxOp implements OpNode {
  opName = 'Max';
  inputs: TensorNode[];
  output: TensorNode;
  private dim: number | null;
  private keepdim: boolean;

  constructor(X: TensorNode, dim: number | null, outName: string, keepdim = false) {
    this.inputs = [X];
    this.dim = dim;
    this.keepdim = keepdim;
    const outData = maxNdArray(X.data, dim, keepdim);
    const requiresGrad = X.requiresGrad && isGraphTracking();
    this.output = new TensorNode(nextId(), outName, outData, requiresGrad);
    this.output.isLeaf = false;
    if (requiresGrad) this.output.creatorOp = this;
    registerTensor(this.output);
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      const d = this.dim === null ? null : (this.dim < 0 ? this.dim + X.shape.length : this.dim);
      let expanded = gradOutput;
      if (d !== null && !this.keepdim) {
        expanded = unsqueezeNdArray(gradOutput, d);
      }

      const gradBuf = new Float32Array(X.data.data.length).fill(0);

      if (d === null) {
        let maxVal = -Infinity;
        for (let i = 0; i < X.data.data.length; i++) {
          maxVal = Math.max(maxVal, X.data.data[i]);
        }
        for (let i = 0; i < X.data.data.length; i++) {
          if (X.data.data[i] === maxVal) {
            gradBuf[i] = gradOutput.data[gradOutput.offset];
            break;
          }
        }
      } else {
        const dval = d as number;
        const newShape = [...X.shape];
        newShape[dval] = 1;
        const outStrides = computeRowMajorStrides(newShape);
        const xStrides = computeRowMajorStrides(Array.from(X.shape));

        function iterate(pos: number[], cd: number) {
          if (cd === newShape.length) {
            let maxVal = -Infinity;
            for (let k = 0; k < X.shape[dval]; k++) {
              const xPos = [...pos];
              xPos[dval] = k;
              let xIdx = 0;
              for (let i = 0; i < xPos.length; i++) xIdx += xPos[i] * xStrides[i];
              maxVal = Math.max(maxVal, X.data.data[xIdx]);
            }
            for (let k = 0; k < X.shape[dval]; k++) {
              const xPos = [...pos];
              xPos[dval] = k;
              let xIdx = 0;
              let gradIdx = 0;
              for (let i = 0; i < xPos.length; i++) {
                xIdx += xPos[i] * xStrides[i];
                gradIdx += xPos[i] * outStrides[i];
              }
              if (X.data.data[xIdx] === maxVal) {
                gradBuf[xIdx] = expanded.data[gradIdx + expanded.offset];
                break;
              }
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
      }
      X.addGrad(createNdArray(gradBuf, Array.from(X.shape)));
    }
  }
}

export class MinOp implements OpNode {
  opName = 'Min';
  inputs: TensorNode[];
  output: TensorNode;
  private dim: number | null;
  private keepdim: boolean;

  constructor(X: TensorNode, dim: number | null, outName: string, keepdim = false) {
    this.inputs = [X];
    this.dim = dim;
    this.keepdim = keepdim;
    const outData = minNdArray(X.data, dim, keepdim);
    const requiresGrad = X.requiresGrad && isGraphTracking();
    this.output = new TensorNode(nextId(), outName, outData, requiresGrad);
    this.output.isLeaf = false;
    if (requiresGrad) this.output.creatorOp = this;
    registerTensor(this.output);
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      const d = this.dim === null ? null : (this.dim < 0 ? this.dim + X.shape.length : this.dim);
      let expanded = gradOutput;
      if (d !== null && !this.keepdim) {
        expanded = unsqueezeNdArray(gradOutput, d);
      }

      const gradBuf = new Float32Array(X.data.data.length).fill(0);

      if (d === null) {
        let minVal = Infinity;
        for (let i = 0; i < X.data.data.length; i++) {
          minVal = Math.min(minVal, X.data.data[i]);
        }
        for (let i = 0; i < X.data.data.length; i++) {
          if (X.data.data[i] === minVal) {
            gradBuf[i] = gradOutput.data[gradOutput.offset];
            break;
          }
        }
      } else {
        const dval = d as number;
        const newShape = [...X.shape];
        newShape[dval] = 1;
        const outStrides = computeRowMajorStrides(newShape);
        const xStrides = computeRowMajorStrides(Array.from(X.shape));

        function iterate(pos: number[], cd: number) {
          if (cd === newShape.length) {
            let minVal = Infinity;
            for (let k = 0; k < X.shape[dval]; k++) {
              const xPos = [...pos];
              xPos[dval] = k;
              let xIdx = 0;
              for (let i = 0; i < xPos.length; i++) xIdx += xPos[i] * xStrides[i];
              minVal = Math.min(minVal, X.data.data[xIdx]);
            }
            for (let k = 0; k < X.shape[dval]; k++) {
              const xPos = [...pos];
              xPos[dval] = k;
              let xIdx = 0;
              let gradIdx = 0;
              for (let i = 0; i < xPos.length; i++) {
                xIdx += xPos[i] * xStrides[i];
                gradIdx += xPos[i] * outStrides[i];
              }
              if (X.data.data[xIdx] === minVal) {
                gradBuf[xIdx] = expanded.data[gradIdx + expanded.offset];
                break;
              }
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
      }
      X.addGrad(createNdArray(gradBuf, Array.from(X.shape)));
    }
  }
}

export function sum(X: TensorNode, dim: number | null, outName: string, keepdim = false): TensorNode {
  return new SumOp(X, dim, outName, keepdim).output;
}

export function mean(X: TensorNode, dim: number | null, outName: string, keepdim = false): TensorNode {
  return new MeanOp(X, dim, outName, keepdim).output;
}

export function max(X: TensorNode, dim: number | null, outName: string, keepdim = false): TensorNode {
  return new MaxOp(X, dim, outName, keepdim).output;
}

export function min(X: TensorNode, dim: number | null, outName: string, keepdim = false): TensorNode {
  return new MinOp(X, dim, outName, keepdim).output;
}
