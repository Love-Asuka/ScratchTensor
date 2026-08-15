import { TensorNode, nextId, registerTensor } from '../tensor.js';
import { OpNode } from '../types.js';
import { isGraphTracking } from '../autograd.js';
import { markRequiresGrad } from './helpers.js';
import {
  whereNdArray,
  computeBroadcastShape,
  broadcastNdArray,
  clampNdArray,
  sumToShapeNdArray,
  createNdArray
} from '../ndarray-utils.js';

export class WhereOp implements OpNode {
  opName = 'Where';
  inputs: TensorNode[];
  output: TensorNode;
  private broadcastShape: number[];

  constructor(cond: TensorNode, a: TensorNode, b: TensorNode, outName: string) {
    this.inputs = [cond, a, b];
    this.broadcastShape = computeBroadcastShape([cond.shape, a.shape, b.shape]);
    const outData = whereNdArray(cond.data, a.data, b.data);
    this.output = new TensorNode(nextId(), outName, outData, false);
    markRequiresGrad(this, [a, b]);
  }

  backwardFn(gradOutput: any): void {
    const [cond, a, b] = this.inputs;
    const condBroadcast = broadcastNdArray(cond.data, this.broadcastShape);
    const size = gradOutput.data.length;
    if (a.requiresGrad) {
      const dA = createNdArray(new Float32Array(size), Array.from(gradOutput.shape));
      for (let i = 0; i < size; i++) {
        dA.data[i] = condBroadcast.data[i] ? gradOutput.data[i] : 0;
      }
      a.addGrad(sumToShapeNdArray(dA, a.shape));
    }
    if (b.requiresGrad) {
      const dB = createNdArray(new Float32Array(size), Array.from(gradOutput.shape));
      for (let i = 0; i < size; i++) {
        dB.data[i] = condBroadcast.data[i] ? 0 : gradOutput.data[i];
      }
      b.addGrad(sumToShapeNdArray(dB, b.shape));
    }
  }
}

export class ClampOp implements OpNode {
  opName = 'Clamp';
  inputs: TensorNode[];
  output: TensorNode;
  private min: number;
  private max: number;

  constructor(X: TensorNode, min: number, max: number, outName: string) {
    this.inputs = [X];
    this.min = min;
    this.max = max;
    const outData = clampNdArray(X.data, min, max);
    this.output = new TensorNode(nextId(), outName, outData, false);
    markRequiresGrad(this, this.inputs);
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      const size = gradOutput.data.length;
      const dXBuf = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        const v = X.data.data[i];
        dXBuf[i] = (v >= this.min && v <= this.max) ? gradOutput.data[i] : 0;
      }
      X.addGrad(createNdArray(dXBuf, Array.from(gradOutput.shape)));
    }
  }
}

export function where(cond: TensorNode, a: TensorNode, b: TensorNode, outName: string): TensorNode {
  return new WhereOp(cond, a, b, outName).output;
}

export function clamp(X: TensorNode, min: number, max: number, outName: string): TensorNode {
  return new ClampOp(X, min, max, outName).output;
}
