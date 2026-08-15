import {
  TensorNode,
  getAllLeafParameters,
  getTensorByName,
  createTensor
} from './tensor.js';
import { fromJsArray, arraysEqual } from './ndarray-utils.js';

export interface SerializedTensor {
  name: string;
  id: number;
  shape: number[];
  requires_grad: boolean;
  data: any;
}

export interface StateDict {
  framework: string;
  tensors: SerializedTensor[];
}

/**
 * [获取当前模型权重字典 JSON 字符串]
 * Exports all leaf model parameters to StateDict JSON string.
 */
export function exportStateDict(): string {
  const leaves = getAllLeafParameters();
  const serializedList: SerializedTensor[] = leaves.map(tensor => ({
    name: tensor.name,
    id: tensor.id,
    shape: tensor.shape,
    requires_grad: tensor.requiresGrad,
    data: tensor.toArray()
  }));

  const stateDict: StateDict = {
    framework: 'ScratchTensor',
    tensors: serializedList
  };

  return JSON.stringify(stateDict, null, 2);
}

/**
 * [从 StateDict 文本 <JSON文本字符串> 导入并恢复模型权重]
 * Name-First alignment: updates existing tensor by name or creates a new one.
 */
export function loadStateDict(jsonStr: string): void {
  const stateDict: StateDict = JSON.parse(jsonStr);
  if (!stateDict || !Array.isArray(stateDict.tensors)) {
    throw new Error('Invalid StateDict format: missing tensors array');
  }

  for (const item of stateDict.tensors) {
    const existing = getTensorByName(item.name);
    if (existing) {
      // Overwrite data and clear gradient cache
      if (!arraysEqual(existing.shape, item.shape)) {
        throw new Error(
          `Shape mismatch for tensor "${item.name}": existing ${JSON.stringify(existing.shape)} vs checkpoint ${JSON.stringify(item.shape)}`
        );
      }
      const { ndarray } = fromJsArray(item.data);
      existing.data = ndarray;
      existing.clearGrad();
    } else {
      createTensor(item.name, item.data, item.requires_grad);
    }
  }
}
