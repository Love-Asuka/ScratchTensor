import { TensorNode, nextId, registerTensor } from '../tensor.js';
import { OpNode } from '../types.js';
import { isGraphTracking } from '../autograd.js';
import {
  mseLossNdArray,
  crossEntropyLossNdArray,
  bceLossNdArray,
  l1LossNdArray,
  fromJsArray,
  createNdArray
} from '../ndarray-utils.js';

export class MSELossOp implements OpNode {
  opName = 'MSELoss';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(pred: TensorNode, target: TensorNode, outName: string) {
    this.inputs = [pred, target];
    const { lossValue, gradPred } = mseLossNdArray(pred.data, target.data);
    const { ndarray: lossNdArray } = fromJsArray(lossValue);
    const requiresGrad = pred.requiresGrad && isGraphTracking();
    this.output = new TensorNode(nextId(), outName, lossNdArray, requiresGrad);
    this.output.isLeaf = false;
    if (requiresGrad) {
      this.output.creatorOp = this;
      (this as any)._gradPred = gradPred;
    }
    registerTensor(this.output);
  }

  backwardFn(gradOutput: any): void {
    const [pred] = this.inputs;
    if (pred.requiresGrad) {
      const scale = gradOutput.data[0];
      const gradPred = (this as any)._gradPred;
      if (scale === 1.0) {
        pred.addGrad(gradPred);
      } else {
        const scaledBuf = new Float32Array(gradPred.data.length);
        for (let i = 0; i < scaledBuf.length; i++) {
          scaledBuf[i] = gradPred.data[i] * scale;
        }
        pred.addGrad(createNdArray(scaledBuf, Array.from(gradPred.shape)));
      }
    }
  }
}

export class CrossEntropyLossOp implements OpNode {
  opName = 'CrossEntropyLoss';
  inputs: TensorNode[];
  output: TensorNode;
  private _gradLogits: any;

  constructor(logits: TensorNode, labels: TensorNode, outName: string) {
    this.inputs = [logits, labels];
    const { lossValue, gradLogits } = crossEntropyLossNdArray(logits.data, labels.data);
    const { ndarray: lossNdArray } = fromJsArray(lossValue);
    const requiresGrad = logits.requiresGrad && isGraphTracking();
    this.output = new TensorNode(nextId(), outName, lossNdArray, requiresGrad);
    this.output.isLeaf = false;
    if (requiresGrad) {
      this.output.creatorOp = this;
      this._gradLogits = gradLogits;
    }
    registerTensor(this.output);
  }

  backwardFn(gradOutput: any): void {
    const [logits] = this.inputs;
    if (logits.requiresGrad) {
      const scale = gradOutput.data[0];
      if (scale === 1.0) {
        logits.addGrad(this._gradLogits);
      } else {
        const scaledBuf = new Float32Array(this._gradLogits.data.length);
        for (let i = 0; i < scaledBuf.length; i++) {
          scaledBuf[i] = this._gradLogits.data[i] * scale;
        }
        logits.addGrad(createNdArray(scaledBuf, Array.from(this._gradLogits.shape)));
      }
    }
  }
}

export class BCELossOp implements OpNode {
  opName = 'BCELoss';
  inputs: TensorNode[];
  output: TensorNode;
  private _gradPred: any;

  constructor(pred: TensorNode, target: TensorNode, outName: string) {
    this.inputs = [pred, target];
    const { lossValue, gradPred } = bceLossNdArray(pred.data, target.data);
    const { ndarray: lossNdArray } = fromJsArray(lossValue);
    const requiresGrad = pred.requiresGrad && isGraphTracking();
    this.output = new TensorNode(nextId(), outName, lossNdArray, requiresGrad);
    this.output.isLeaf = false;
    if (requiresGrad) {
      this.output.creatorOp = this;
      this._gradPred = gradPred;
    }
    registerTensor(this.output);
  }

  backwardFn(gradOutput: any): void {
    const [pred] = this.inputs;
    if (pred.requiresGrad) {
      const scale = gradOutput.data[0];
      if (scale === 1.0) {
        pred.addGrad(this._gradPred);
      } else {
        const scaledBuf = new Float32Array(this._gradPred.data.length);
        for (let i = 0; i < scaledBuf.length; i++) {
          scaledBuf[i] = this._gradPred.data[i] * scale;
        }
        pred.addGrad(createNdArray(scaledBuf, Array.from(this._gradPred.shape)));
      }
    }
  }
}

export class L1LossOp implements OpNode {
  opName = 'L1Loss';
  inputs: TensorNode[];
  output: TensorNode;
  private _gradPred: any;

  constructor(pred: TensorNode, target: TensorNode, outName: string) {
    this.inputs = [pred, target];
    const { lossValue, gradPred } = l1LossNdArray(pred.data, target.data);
    const { ndarray: lossNdArray } = fromJsArray(lossValue);
    const requiresGrad = pred.requiresGrad && isGraphTracking();
    this.output = new TensorNode(nextId(), outName, lossNdArray, requiresGrad);
    this.output.isLeaf = false;
    if (requiresGrad) {
      this.output.creatorOp = this;
      this._gradPred = gradPred;
    }
    registerTensor(this.output);
  }

  backwardFn(gradOutput: any): void {
    const [pred] = this.inputs;
    if (pred.requiresGrad) {
      const scale = gradOutput.data[0];
      if (scale === 1.0) {
        pred.addGrad(this._gradPred);
      } else {
        const scaledBuf = new Float32Array(this._gradPred.data.length);
        for (let i = 0; i < scaledBuf.length; i++) {
          scaledBuf[i] = this._gradPred.data[i] * scale;
        }
        pred.addGrad(createNdArray(scaledBuf, Array.from(this._gradPred.shape)));
      }
    }
  }
}

export function mseLoss(pred: TensorNode, target: TensorNode, outName: string): TensorNode {
  return new MSELossOp(pred, target, outName).output;
}

export function crossEntropyLoss(logits: TensorNode, labels: TensorNode, outName: string): TensorNode {
  return new CrossEntropyLossOp(logits, labels, outName).output;
}

export function bceLoss(pred: TensorNode, target: TensorNode, outName: string): TensorNode {
  return new BCELossOp(pred, target, outName).output;
}

export function l1Loss(pred: TensorNode, target: TensorNode, outName: string): TensorNode {
  return new L1LossOp(pred, target, outName).output;
}
