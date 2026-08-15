import { TensorNode, nextId, registerTensor } from '../tensor.js';
import { OpNode } from '../types.js';
import { isGraphTracking } from '../autograd.js';
import { markRequiresGrad } from './helpers.js';
import {
  createNdArray,
  zerosNdArray,
  resolveReshapeShape,
  unsqueezeNdArray,
  squeezeNdArray,
  expandNdArray,
  sumToShapeNdArray,
  concatNdArray,
  splitNdArray,
  stackNdArray,
  sliceNdArray,
  transposeNdArray,
  computeRowMajorStrides
} from '../ndarray-utils.js';

export class ReshapeOp implements OpNode {
  opName = 'Reshape';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(X: TensorNode, newShape: number[], outName: string) {
    this.inputs = [X];
    const resolvedShape = resolveReshapeShape(X.shape, newShape);
    const outData = createNdArray(X.data.data, resolvedShape);
    const requiresGrad = X.requiresGrad && isGraphTracking();
    this.output = new TensorNode(nextId(), outName, outData, requiresGrad);
    this.output.isLeaf = false;
    if (requiresGrad) {
      this.output.creatorOp = this;
    }
    registerTensor(this.output);
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      const dX = createNdArray(gradOutput.data, X.shape);
      X.addGrad(dX);
    }
  }
}

export class TransposeOp implements OpNode {
  opName = 'Transpose';
  inputs: TensorNode[];
  output: TensorNode;
  private dim1: number;
  private dim2: number;

  constructor(X: TensorNode, dim1: number, dim2: number, outName: string) {
    this.inputs = [X];
    this.dim1 = dim1;
    this.dim2 = dim2;
    const outData = transposeNdArray(X.data, dim1, dim2);
    const requiresGrad = X.requiresGrad && isGraphTracking();
    this.output = new TensorNode(nextId(), outName, outData, requiresGrad);
    this.output.isLeaf = false;
    if (requiresGrad) this.output.creatorOp = this;
    registerTensor(this.output);
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) X.addGrad(transposeNdArray(gradOutput, this.dim1, this.dim2));
  }
}

export class UnsqueezeOp implements OpNode {
  opName = 'Unsqueeze';
  inputs: TensorNode[];
  output: TensorNode;
  private dim: number;

  constructor(X: TensorNode, dim: number, outName: string) {
    this.dim = dim;
    this.inputs = [X];
    const outData = unsqueezeNdArray(X.data, dim);
    this.output = new TensorNode(nextId(), outName, outData, false);
    markRequiresGrad(this, this.inputs);
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      X.addGrad(squeezeNdArray(gradOutput, this.dim));
    }
  }
}

export class SqueezeOp implements OpNode {
  opName = 'Squeeze';
  inputs: TensorNode[];
  output: TensorNode;
  private dim?: number;
  private originalShape: number[];

  constructor(X: TensorNode, outName: string, dim?: number) {
    this.dim = dim;
    this.inputs = [X];
    this.originalShape = Array.from(X.shape);
    const outData = squeezeNdArray(X.data, dim);
    this.output = new TensorNode(nextId(), outName, outData, false);
    markRequiresGrad(this, this.inputs);
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      if (this.dim !== undefined) {
        X.addGrad(unsqueezeNdArray(gradOutput, this.dim));
      } else {
        let expanded = gradOutput;
        for (let i = 0; i < this.originalShape.length; i++) {
          if (this.originalShape[i] === 1) {
            expanded = unsqueezeNdArray(expanded, i);
          }
        }
        X.addGrad(expanded);
      }
    }
  }
}

export class ExpandOp implements OpNode {
  opName = 'Expand';
  inputs: TensorNode[];
  output: TensorNode;
  private originalShape: number[];

  constructor(X: TensorNode, targetShape: number[], outName: string) {
    this.inputs = [X];
    this.originalShape = Array.from(X.shape);
    const outData = expandNdArray(X.data, targetShape);
    this.output = new TensorNode(nextId(), outName, outData, false);
    markRequiresGrad(this, this.inputs);
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      X.addGrad(sumToShapeNdArray(gradOutput, this.originalShape));
    }
  }
}

export class ConcatOp implements OpNode {
  opName = 'Concat';
  inputs: TensorNode[];
  output: TensorNode;
  private dim: number;

  constructor(inputs: TensorNode[], dim: number, outName: string) {
    this.inputs = inputs;
    this.dim = dim;
    const outData = concatNdArray(inputs.map(n => n.data), dim);
    this.output = new TensorNode(nextId(), outName, outData, false);
    markRequiresGrad(this, this.inputs);
  }

  backwardFn(gradOutput: any): void {
    let offset = 0;
    for (const input of this.inputs) {
      if (input.requiresGrad) {
        const start = offset;
        const end = offset + input.shape[this.dim];
        input.addGrad(sliceNdArray(gradOutput, this.dim, start, end));
      }
      offset += input.shape[this.dim];
    }
  }
}

export class StackOp implements OpNode {
  opName = 'Stack';
  inputs: TensorNode[];
  output: TensorNode;
  private dim: number;

  constructor(inputs: TensorNode[], dim: number, outName: string) {
    this.inputs = inputs;
    this.dim = dim;
    const outData = stackNdArray(inputs.map(n => n.data), dim);
    this.output = new TensorNode(nextId(), outName, outData, false);
    markRequiresGrad(this, this.inputs);
  }

  backwardFn(gradOutput: any): void {
    const slices = splitNdArray(gradOutput, this.dim, this.inputs.length);
    for (let i = 0; i < this.inputs.length; i++) {
      const input = this.inputs[i];
      if (input.requiresGrad) {
        input.addGrad(squeezeNdArray(slices[i], this.dim));
      }
    }
  }
}

function computeStrides(shape: number[]): number[] {
  if (shape.length === 0) return [1];
  const strides = new Array(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

export class SliceOp implements OpNode {
  opName = 'Slice';
  inputs: TensorNode[];
  output: TensorNode;
  private dim: number;
  private start: number;
  private end: number;
  private originalShape: number[];

  constructor(X: TensorNode, dim: number, start: number, end: number, outName: string) {
    this.inputs = [X];
    this.dim = dim;
    this.start = start;
    this.end = end;
    this.originalShape = Array.from(X.shape);
    const outData = sliceNdArray(X.data, dim, start, end);
    this.output = new TensorNode(nextId(), outName, outData, false);
    markRequiresGrad(this, this.inputs);
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      const dim = this.dim;
      const start = this.start;
      const dX = zerosNdArray(this.originalShape);
      const sliceShape = Array.from(this.originalShape);
      sliceShape[dim] = this.end - this.start;
      const outStrides = computeStrides(sliceShape);

      function iterate(pos: number[], cd: number) {
        if (cd === sliceShape.length) {
          const srcPos = [...pos];
          srcPos[dim] = start + pos[dim];
          let srcIdx = 0;
          for (let i = 0; i < srcPos.length; i++) {
            srcIdx += srcPos[i] * X.data.strides[i];
          }
          let gradIdx = 0;
          for (let i = 0; i < pos.length; i++) {
            gradIdx += pos[i] * outStrides[i];
          }
          dX.data[srcIdx] += gradOutput.data[gradIdx];
          return;
        }
        for (let i = 0; i < sliceShape[cd]; i++) {
          pos.push(i);
          iterate(pos, cd + 1);
          pos.pop();
        }
      }
      iterate([], 0);
      X.addGrad(dX);
    }
  }
}

export function split(X: TensorNode, dim: number, n: number, prefix: string): TensorNode[] {
  const ndim = X.shape.length;
  const d = dim < 0 ? dim + ndim : dim;
  if (d < 0 || d >= ndim) {
    throw new Error(`Dimension ${dim} out of range for ${ndim}-D tensor`);
  }
  const dimSize = X.shape[d];
  if (dimSize % n !== 0) {
    throw new Error(`Cannot split dim ${d} of size ${dimSize} into ${n} equal parts`);
  }
  const chunk = dimSize / n;
  const results: TensorNode[] = [];
  for (let i = 0; i < n; i++) {
    const start = i * chunk;
    const end = start + chunk;
    const outName = `${prefix}_${i}`;
    const op = new SliceOp(X, d, start, end, outName);
    results.push(op.output);
  }
  return results;
}

export function reshape(X: TensorNode, newShape: number[], outName: string): TensorNode {
  return new ReshapeOp(X, newShape, outName).output;
}

export function transpose(X: TensorNode, dim1: number, dim2: number, outName: string): TensorNode {
  return new TransposeOp(X, dim1, dim2, outName).output;
}

export function unsqueeze(X: TensorNode, dim: number, outName: string): TensorNode {
  return new UnsqueezeOp(X, dim, outName).output;
}

export function squeeze(X: TensorNode, outName: string, dim?: number): TensorNode {
  return new SqueezeOp(X, outName, dim).output;
}

export function expand(X: TensorNode, targetShape: number[], outName: string): TensorNode {
  return new ExpandOp(X, targetShape, outName).output;
}

export function concat(inputs: TensorNode[], dim: number, outName: string): TensorNode {
  return new ConcatOp(inputs, dim, outName).output;
}

export function slice(X: TensorNode, dim: number, start: number, end: number, outName: string): TensorNode {
  return new SliceOp(X, dim, start, end, outName).output;
}

export function stack(inputs: TensorNode[], dim: number, outName: string): TensorNode {
  return new StackOp(inputs, dim, outName).output;
}
