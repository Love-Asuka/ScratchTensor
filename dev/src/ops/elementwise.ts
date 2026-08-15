import { TensorNode, nextId, registerTensor } from '../tensor.js';
import { OpNode } from '../types.js';
import { isGraphTracking } from '../autograd.js';
import {
  powNdArray,
  expNdArray,
  logNdArray,
  sqrtNdArray,
  absNdArray,
  createNdArray,
  mulNdArray
} from '../ndarray-utils.js';

export class PowOp implements OpNode {
  opName = 'Pow';
  inputs: TensorNode[];
  output: TensorNode;
  private p: number;

  constructor(X: TensorNode, p: number, outName: string) {
    this.inputs = [X];
    this.p = p;
    const outData = powNdArray(X.data, p);
    const requiresGrad = X.requiresGrad && isGraphTracking();
    this.output = new TensorNode(nextId(), outName, outData, requiresGrad);
    this.output.isLeaf = false;
    if (requiresGrad) this.output.creatorOp = this;
    registerTensor(this.output);
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      const size = gradOutput.data.length;
      const buf = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        buf[i] = gradOutput.data[i] * this.p * Math.pow(X.data.data[i], this.p - 1);
      }
      X.addGrad(createNdArray(buf, Array.from(gradOutput.shape)));
    }
  }
}

export class ExpOp implements OpNode {
  opName = 'Exp';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(X: TensorNode, outName: string) {
    this.inputs = [X];
    const outData = expNdArray(X.data);
    const requiresGrad = X.requiresGrad && isGraphTracking();
    this.output = new TensorNode(nextId(), outName, outData, requiresGrad);
    this.output.isLeaf = false;
    if (requiresGrad) this.output.creatorOp = this;
    registerTensor(this.output);
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) X.addGrad(mulNdArray(gradOutput, this.output.data));
  }
}

export class LogOp implements OpNode {
  opName = 'Log';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(X: TensorNode, outName: string) {
    this.inputs = [X];
    const outData = logNdArray(X.data);
    const requiresGrad = X.requiresGrad && isGraphTracking();
    this.output = new TensorNode(nextId(), outName, outData, requiresGrad);
    this.output.isLeaf = false;
    if (requiresGrad) this.output.creatorOp = this;
    registerTensor(this.output);
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      const size = gradOutput.data.length;
      const buf = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        buf[i] = gradOutput.data[i] / X.data.data[i];
      }
      X.addGrad(createNdArray(buf, Array.from(gradOutput.shape)));
    }
  }
}

export class SqrtOp implements OpNode {
  opName = 'Sqrt';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(X: TensorNode, outName: string) {
    this.inputs = [X];
    const outData = sqrtNdArray(X.data);
    const requiresGrad = X.requiresGrad && isGraphTracking();
    this.output = new TensorNode(nextId(), outName, outData, requiresGrad);
    this.output.isLeaf = false;
    if (requiresGrad) this.output.creatorOp = this;
    registerTensor(this.output);
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      const size = gradOutput.data.length;
      const buf = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        buf[i] = gradOutput.data[i] / (2.0 * Math.sqrt(X.data.data[i]));
      }
      X.addGrad(createNdArray(buf, Array.from(gradOutput.shape)));
    }
  }
}

export class AbsOp implements OpNode {
  opName = 'Abs';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(X: TensorNode, outName: string) {
    this.inputs = [X];
    const outData = absNdArray(X.data);
    const requiresGrad = X.requiresGrad && isGraphTracking();
    this.output = new TensorNode(nextId(), outName, outData, requiresGrad);
    this.output.isLeaf = false;
    if (requiresGrad) this.output.creatorOp = this;
    registerTensor(this.output);
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      const size = gradOutput.data.length;
      const buf = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        const v = X.data.data[i];
        buf[i] = gradOutput.data[i] * (v > 0 ? 1 : v < 0 ? -1 : 0);
      }
      X.addGrad(createNdArray(buf, Array.from(gradOutput.shape)));
    }
  }
}

export function pow(X: TensorNode, p: number, outName: string): TensorNode {
  return new PowOp(X, p, outName).output;
}

export function exp(X: TensorNode, outName: string): TensorNode {
  return new ExpOp(X, outName).output;
}

export function log(X: TensorNode, outName: string): TensorNode {
  return new LogOp(X, outName).output;
}

export function sqrt(X: TensorNode, outName: string): TensorNode {
  return new SqrtOp(X, outName).output;
}

export function abs(X: TensorNode, outName: string): TensorNode {
  return new AbsOp(X, outName).output;
}
