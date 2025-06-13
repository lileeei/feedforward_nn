# 神经网络训练与反向传播详解

本文件结合项目代码，系统讲解前馈神经网络的训练过程，重点说明反向传播、delta 计算、权重更新等核心原理，并配合关键代码片段逐步解释。

---

## 1. 训练流程总览

一次训练（`train` 方法）包括：
- **前向传播（Forward Pass）**：依次通过每层，记录每层输入和线性输出。
- **损失计算**：如均方误差（MSE）。
- **反向传播（Backward Pass）**：逐层计算误差信号（delta），并用梯度下降法更新参数。

---

## 2. 前向传播

### 2.1 代码片段
```rust
let mut out = input.clone();
layer_inputs.push(out.clone());
for (layer, act) in self.hidden_layers.iter().zip(self.hidden_activations.iter()) {
    out = layer.forward(&out);         // 线性变换
    layer_outputs.push(out.clone());   // 保存未激活前的输出
    out = activate_vec(&out, act);     // 激活
    layer_inputs.push(out.clone());    // 保存激活后的输出
}
// 输出层
let output_raw = self.output.forward(&out);
layer_outputs.push(output_raw.clone());
let output = activate_vec(&output_raw, &self.output_activation);
layer_inputs.push(output.clone());
```

### 2.2 说明
- `layer.forward(&out)`：对输入做线性变换（加权求和+偏置）。
- `activate_vec(&out, act)`：对线性输出应用激活函数。
- `layer_inputs` 记录每层激活后的输出（包括输入层）。
- `layer_outputs` 记录每层线性变换（未激活前的输出）。
- 这些记录为反向传播做准备。

---

## 3. 损失函数

以均方误差（MSE）为例：

\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (o_i - t_i)^2
\]

### 3.1 代码片段
```rust
let loss = Network::mse_loss(&output, target);
```

---

## 4. 反向传播与参数更新

### 4.1 输出层 delta 计算

#### 代码片段
```rust
let delta = output
    .iter()
    .zip(target.iter())
    .zip(activate_derivative_vec(&output_raw, &self.output_activation))
    .map(|((o, t), d_act)| (o - t) * d_act)
    .collect::<Vec<f64>>();
```

#### 说明
- `output`：输出层激活值
- `target`：目标值
- `activate_derivative_vec`：输出层激活函数的导数
- 每个神经元的 delta = (输出 - 目标) × 激活函数导数

### 4.2 输出层参数更新

#### 代码片段
```rust
for i in 0..self.output.weights.len() {
    for j in 0..self.output.weights[0].len() {
        self.output.weights[i][j] -= lr * delta[i] * layer_inputs[layer_inputs.len() - 2][j];
    }
    self.output.biases[i] -= lr * delta[i];
}
```

#### 说明
- `delta[i]`：输出层第 i 个神经元的误差信号
- `layer_inputs[layer_inputs.len() - 2][j]`：输出层的输入（上一层激活输出）
- 权重更新公式：
  \[
  w_{ij} := w_{ij} - \eta \cdot \delta_i \cdot a_j
  \]
- 偏置更新公式：
  \[
  b_i := b_i - \eta \cdot \delta_i
  \]

### 4.3 反向传播到隐藏层

#### 代码片段
```rust
let mut next_layer = &self.output;
let mut next_delta = delta;
for l in (0..self.hidden_layers.len()).rev() {
    let z = &layer_outputs[l];
    let d_act = activate_derivative_vec(z, &self.hidden_activations[l]);
    let mut new_delta = vec![0.0; self.hidden_layers[l].weights.len()];
    for i in 0..self.hidden_layers[l].weights.len() {
        let mut sum = 0.0;
        for j in 0..next_layer.weights.len() {
            sum += next_layer.weights[j][i] * next_delta[j];
        }
        new_delta[i] = sum * d_act[i];
    }
    // 更新当前层权重和偏置
    for i in 0..self.hidden_layers[l].weights.len() {
        for j in 0..self.hidden_layers[l].weights[0].len() {
            self.hidden_layers[l].weights[i][j] -= lr * new_delta[i] * layer_inputs[l][j];
        }
        self.hidden_layers[l].biases[i] -= lr * new_delta[i];
    }
    next_layer = &self.hidden_layers[l];
    next_delta = new_delta;
}
```

#### 详细说明
- **delta 递推公式**：
  \[
  \delta^l_i = \left( \sum_j w^{l+1}_{ji} \delta^{l+1}_j \right) \cdot f'(z^l_i)
  \]
  - `sum += next_layer.weights[j][i] * next_delta[j];` 计算了来自上一层的误差信号加权和。
  - `d_act[i]` 是本层激活函数的导数。
- **权重更新**：
  - `self.hidden_layers[l].weights[i][j] -= lr * new_delta[i] * layer_inputs[l][j];`
  - 其中 `layer_inputs[l][j]` 是本层输入（上一层激活输出），即 \(a^{l-1}_j\)。
- **偏置更新**：
  - `self.hidden_layers[l].biases[i] -= lr * new_delta[i];`
- **递推**：
  - `next_layer` 和 `next_delta` 递推到前一层，继续反向传播。

#### 为什么要乘以 `layer_inputs[l][j]`？
- 这是权重梯度的来源：
  \[
  \frac{\partial L}{\partial w_{ij}} = \delta^l_i \cdot a^{l-1}_j
  \]
- 只有这样，权重的更新方向才是"让损失变小"的方向。

---

## 5. 总结

- 反向传播的本质是链式法则，delta 递推，参数按梯度下降更新
- 权重更新必须乘以本层输入（上一层激活），即 `layer_inputs[l][j]`
- 偏置更新只需乘以 delta
- 代码实现严格遵循数学推导，支持多层网络和多种激活函数 