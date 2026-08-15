# ScratchTensor

[![TurboWarp](https://img.shields.io/badge/Made%20with-TurboWarp-FFAB19?style=flat-square)](https://turbowarp.org/)
[![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![npm](https://img.shields.io/badge/npm-CB3837?style=flat-square&logo=npm&logoColor=white)](https://www.npmjs.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

一款面向 Scratch 图形化编程的轻量级深度学习张量引擎，提供类 PyTorch 的动态计算图与自动求导能力，并打包为 Scratch 扩展积木。

## 核心特性

- **动态 Autograd**：支持 `requiresGrad` 叶子参数、反向传播与计算图自动清理，对齐 PyTorch `backward(retain_graph=False)` 内存模型；`backward` 后会自动从注册表释放非叶子中间张量。
- **Scratch 扩展**：将张量创建、算子、损失函数、优化器等封装为 Scratch 积木，可直接在 Scratch 3 中拖拽使用。
- **PyTorch 风格 API**：`sum` / `mean` / `max` / `min` 支持 `dim` 与 `keepdim`，`matmul` 支持 1D 点积、2D 矩阵乘、3D 及以上批量矩阵乘与 batch 广播，`where` 支持广播，`scaled_dot_product_attention` 支持 causal mask 与 dropout。
- **模型序列化**：通过 JSON 文本导出 / 导入 StateDict，方便在 Scratch 变量或外部文件中保存权重。
- **内存管理**：提供 `删除 Tensor` / `删除所有 Tensor` 积木，避免长期运行或循环训练时的内存泄漏。

## 快速开始

进入 `dev` 目录：

```bash
cd dev
npm install
npm run build
npm test
```

构建产物位于 `dev/dist/`：

- `scratch-tensor-extension.js`：Scratch 3 扩展包。

## 在 Scratch 中使用

1. 将 `dist/scratch-tensor-extension.js` 作为“自定义扩展”加载到 Scratch 3（或 Turbowarp）。
2. 扩展加载后，积木栏会出现 **ScratchTensor** 分类，包含：
   - 创建 / 查询 / 删除 Tensor
   - 矩阵乘法与四则运算
   - 激活函数与损失函数
   - 维度操作（Reshape、Unsqueeze、Squeeze、Expand、Concat、Stack、Slice、Where、Clamp）
   - 归约操作（Sum、Mean、Max、Min）
   - 归一化与正则化（LayerNorm、Dropout）
   - 注意力机制（Scaled Dot Product Attention）
   - 优化器（SGD、Adam、AdamW）
   - 模型权重导出 / 导入

3. 典型训练流程：
   ```scratch
   [创建 Tensor 命名为 "X" 并赋值 [[1.0, 2.0], [3.0, 4.0]] 需梯度: false]
   [创建 Tensor 命名为 "W" 并赋值 [[2.0], [1.0]] 需梯度: true]
   [张量 "X" 与 "W" 进行 [@] 运算 赋值给 "Y"]
   [MSELoss 预测: "Y" 真实标签: "label" 赋值给 "Loss"]
   [对 Tensor "Loss" 进行反向传播]
   [执行 SGD 优化器 学习率: 0.01 动量: 0.9 权重衰减: 0]
   [清空所有参数梯度]
   [删除 Tensor "Y"]
   [删除 Tensor "Loss"]
   ```
   说明：反向传播完成后，中间非叶子张量（如 `Y`）会自动从注册表释放；用户也可以显式使用 `删除 Tensor` 或 `删除所有 Tensor` 管理长期存在的临时张量。

## 核心概念

### Tensor 与 requiresGrad

- 所有 Tensor 在系统中通过**全局名称**注册，积木层只传递字符串名称。
- 创建时 `需梯度: true` 表示该张量为**可学习叶子参数**，反向传播时会在其 `.grad` 中累积梯度。
- 中间运算产出的张量会自动标记为非叶子节点，并在 `backward` 后清理。

### Autograd

- 开启 `[设置计算图追踪: true]` 后，前向运算会构建动态 DAG。
- 对损失标量调用 `[对 Tensor ... 进行反向传播]` 即可自动回传梯度。
- 反向传播完成后，中间节点的计算图会被立即释放，非叶子张量也会从全局注册表中移除，避免内存泄漏。

### 优化器

- 支持 SGD（含 Momentum、L2 权重衰减）、Adam、AdamW。
- 优化器会自动遍历所有 `requiresGrad = true` 的叶子参数并完成单步更新。
- `[对所有参数执行梯度裁剪 最大阈值: ...]` 应在反向传播之后、优化器之前调用。

## 开发测试

```bash
cd dev
npm test 
npm run build
```

## 文档索引

- `docs/tensor.md`：Tensor 创建、查询、结构变形、序列化
- `docs/autograd.md`：计算图追踪、反向传播、梯度清空
- `docs/fun.md`：算子、损失函数、激活函数、维度聚合
- `docs/optim.md`：SGD / Adam / AdamW 优化器、梯度裁剪、学习率缩放
- `docs/demo.md`：Linear 层封装与完整训练循环示例