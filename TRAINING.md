# 神经网络 `train` 过程详解笔记

## 1. 训练步骤总览

一次训练（`train` 方法）包括：
- **前向传播（Forward Pass）**：依次通过每层，记录每层输入和线性输出。
- **损失计算**：如均方误差（MSE）。
- **反向传播（Backward Pass）**：逐层计算误差信号（delta），并用梯度下降法更新参数。

---

## 2. 前向传播

- 依次通过每个隐藏层和输出层，先做线性变换（加权求和+偏置），再做激活函数。
- 记录每层激活后的输出（`layer_inputs`，包括输入层）和每层线性变换的输出（`layer_outputs`，未激活前）。
- 这些记录为反向传播做准备。

**代码片段：**
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

---

## 3. 损失函数

以均方误差（MSE）为例：

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (o_i - t_i)^2
$$

---

## 4. 反向传播与参数更新

### 4.1 输出层 delta 计算

输出层每个神经元的误差信号（delta）：

$$
\delta^L_i = (a^L_i - t_i) \cdot f'(z^L_i)
$$

- $a^L_i$：输出层激活值
- $t_i$：目标值
- $f'(z^L_i)$：激活函数对线性输出的导数

**代码片段：**
```rust
let delta = output
    .iter()
    .zip(target.iter())
    .zip(activate_derivative_vec(&output_raw, &self.output_activation))
    .map(|((o, t), d_act)| (o - t) * d_act)
    .collect::<Vec<f64>>();
```

---

### 4.2 输出层参数更新

权重和偏置的梯度推导如下：

$$
\frac{\partial L}{\partial w_{ij}^L} = \delta^L_i \cdot a^{L-1}_j
$$
$$
\frac{\partial L}{\partial b_i^L} = \delta^L_i
$$

- $a^{L-1}_j$：输出层的输入（上一层激活输出）

权重和偏置的更新公式：

$$
w_{ij}^L := w_{ij}^L - \eta \cdot \delta^L_i \cdot a^{L-1}_j
$$
$$
b_i^L := b_i^L - \eta \cdot \delta^L_i
$$

**代码片段：**
```rust
for i in 0..self.output.weights.len() {
    for j in 0..self.output.weights[0].len() {
        self.output.weights[i][j] -= lr * delta[i] * layer_inputs[layer_inputs.len() - 2][j];
    }
    self.output.biases[i] -= lr * delta[i];
}
```

---

### 4.3 反向传播到隐藏层

#### 数学公式

隐藏层 delta 的递推公式：

$$
\delta^l_i = \left(\sum_j w^{l+1}_{ji} \delta^{l+1}_j \right) \cdot f'(z^l_i)
$$

- $w^{l+1}_{ji}$：下一层权重
- $\delta^{l+1}_j$：下一层 delta
- $f'(z^l_i)$：本层激活函数导数

权重和偏置的梯度：

$$
\frac{\partial L}{\partial w_{ij}^l} = \delta^l_i \cdot a^{l-1}_j
$$
$$
\frac{\partial L}{\partial b_i^l} = \delta^l_i
$$

权重和偏置的更新：

$$
w_{ij}^l := w_{ij}^l - \eta \cdot \delta^l_i \cdot a^{l-1}_j
$$
$$
b_i^l := b_i^l - \eta \cdot \delta^l_i
$$

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

---

### 4.4 为什么权重更新要乘以 `layer_inputs[l][j]`？

#### 数学推导

对于第 l 层第 i 个神经元的第 j 个权重 $w_{ij}^l$：

$$
z_i^l = \sum_j w_{ij}^l a_j^{l-1} + b_i^l
$$
$$
a_i^l = f(z_i^l)
$$
$$
\frac{\partial L}{\partial w_{ij}^l} = \frac{\partial L}{\partial z_i^l} \cdot \frac{\partial z_i^l}{\partial w_{ij}^l} = \delta_i^l \cdot a_j^{l-1}
$$

- $\delta_i^l$：当前神经元的误差信号
- $a_j^{l-1}$：本层输入（上一层激活输出）

#### 直观理解

- delta 反映了该神经元对损失的敏感度
- 输入越大，权重的变化对输出影响越大，梯度也越大
- 两者相乘，反映了"当前权重的微小变化会对损失造成多大影响"

#### 代码映射

```rust
self.hidden_layers[l].weights[i][j] -= lr * new_delta[i] * layer_inputs[l][j];
```
- `new_delta[i]` 就是 $\delta_i^l$
- `layer_inputs[l][j]` 就是 $a_j^{l-1}$

#### 举例说明

假设：
- 某一层有 2 个输入（上一层输出为 [x_1, x_2]）
- 当前神经元的 delta 为 0.5
- 学习率为 0.1

则第一个权重的更新为：

$$
w_{i1} := w_{i1} - 0.1 \times 0.5 \times x_1
$$

#### 物理/几何直观

- 权重的作用是"放大/缩小"输入信号
- 只有输入信号大时，权重的调整才会显著影响输出
- 反向传播时，只有那些"真正参与"了输出的输入，才会对权重产生显著的梯度

---

## 5. 总结

- 反向传播的本质是链式法则，delta 递推，参数按梯度下降更新
- 权重更新必须乘以本层输入（上一层激活），即 `layer_inputs[l][j]`
- 偏置更新只需乘以 delta
- 代码实现严格遵循数学推导，支持多层网络和多种激活函数
