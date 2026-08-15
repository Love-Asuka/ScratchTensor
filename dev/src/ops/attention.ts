import { TensorNode, nextId, registerTensor } from '../tensor.js';
import { OpNode } from '../types.js';
import { isGraphTracking } from '../autograd.js';
import {
  scaledDotProductAttentionNdArray,
  scaledDotProductAttentionGradNdArray
} from '../ndarray-utils.js';

export class ScaledDotProductAttentionOp implements OpNode {
  opName = 'ScaledDotProductAttention';
  inputs: TensorNode[];
  output: TensorNode;
  private attn: any;
  private dropoutMask: Float32Array;
  private scale: number;
  private isTraining: boolean;
  private dropoutP: number;
  private isCausal: boolean;

  constructor(
    Q: TensorNode,
    K: TensorNode,
    V: TensorNode,
    outName: string,
    isCausal = false,
    dropoutP = 0,
    scale?: number,
    attnMask?: TensorNode
  ) {
    this.inputs = [Q, K, V];
    this.isCausal = isCausal;
    this.dropoutP = dropoutP;
    this.isTraining = isGraphTracking();
    const maskData = attnMask ? attnMask.data : undefined;
    const { out, attn, dropoutMask, scale: effScale } = scaledDotProductAttentionNdArray(
      Q.data, K.data, V.data, this.isTraining, dropoutP, isCausal, scale, maskData
    );
    this.attn = attn;
    this.dropoutMask = dropoutMask;
    this.scale = effScale;
    const requiresGrad = (Q.requiresGrad || K.requiresGrad || V.requiresGrad) && this.isTraining;
    this.output = new TensorNode(nextId(), outName, out, requiresGrad);
    this.output.isLeaf = false;
    if (requiresGrad) this.output.creatorOp = this;
    registerTensor(this.output);
  }

  backwardFn(gradOutput: any): void {
    const [Q, K, V] = this.inputs;
    const { dQ, dK, dV } = scaledDotProductAttentionGradNdArray(
      gradOutput, Q.data, K.data, V.data, this.attn, this.dropoutMask, this.scale, this.isTraining, this.dropoutP
    );
    if (Q.requiresGrad) Q.addGrad(dQ);
    if (K.requiresGrad) K.addGrad(dK);
    if (V.requiresGrad) V.addGrad(dV);
  }
}

export function scaledDotProductAttention(
  Q: TensorNode,
  K: TensorNode,
  V: TensorNode,
  outName: string,
  isCausal = false,
  dropoutP = 0,
  scale?: number,
  attnMask?: TensorNode
): TensorNode {
  return new ScaledDotProductAttentionOp(Q, K, V, outName, isCausal, dropoutP, scale, attnMask).output;
}
