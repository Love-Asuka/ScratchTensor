import { OpNode } from '../types.js';
import { TensorNode, nextId, registerTensor } from '../tensor.js';
import { isGraphTracking } from '../autograd.js';

export function requiresGradAny(nodes: TensorNode[]): boolean {
  return nodes.some(n => n.requiresGrad);
}

export function markRequiresGrad(op: OpNode, nodes: TensorNode[]): void {
  const grad = requiresGradAny(nodes) && isGraphTracking();
  op.output.requiresGrad = grad;
  op.output.isLeaf = false;
  if (grad) {
    op.output.creatorOp = op;
  }
  registerTensor(op.output);
}
