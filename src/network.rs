use crate::activation::{Activation, activate_derivative_vec, activate_vec};
use crate::layer::Layer;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{Write, Read};

/// Network 表示一个支持多层隐藏层和激活函数模块化的前馈神经网络
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Network {
    /// 输入层神经元数
    pub input_size: usize,
    /// 每个隐藏层的神经元数
    pub hidden_sizes: Vec<usize>,
    /// 输出层神经元数
    pub output_size: usize,
    /// 多个隐藏层
    pub hidden_layers: Vec<Layer>,
    /// 每个隐藏层的激活函数
    pub hidden_activations: Vec<Activation>,
    /// 输出层
    pub output: Layer,
    /// 输出层激活函数
    pub output_activation: Activation,
}

impl Network {
    /// 创建一个新的前馈神经网络
    /// input_size: 输入层神经元数
    /// hidden_sizes: 每个隐藏层的神经元数（Vec）
    /// output_size: 输出层神经元数
    /// hidden_activations: 每个隐藏层的激活函数（Vec）
    /// output_activation: 输出层激活函数
    pub fn new(
        input_size: usize,
        hidden_sizes: Vec<usize>,
        output_size: usize,
        hidden_activations: Vec<Activation>,
        output_activation: Activation,
    ) -> Self {
        assert_eq!(
            hidden_sizes.len(),
            hidden_activations.len(),
            "每个隐藏层都需要指定激活函数"
        );
        let mut hidden_layers = Vec::new();
        let mut prev_size = input_size;
        for &size in &hidden_sizes {
            hidden_layers.push(Layer::new(prev_size, size));
            prev_size = size;
        }
        let output = Layer::new(prev_size, output_size);
        Self {
            input_size,
            hidden_sizes,
            output_size,
            hidden_layers,
            hidden_activations,
            output,
            output_activation,
        }
    }

    /// 前向传播：输入 input，返回网络输出
    pub fn forward(&self, input: &Vec<f64>) -> Vec<f64> {
        // 依次通过所有隐藏层及其激活函数
        let mut out = input.clone();
        for (layer, act) in self
            .hidden_layers
            .iter()
            .zip(self.hidden_activations.iter())
        {
            out = activate_vec(&layer.forward(&out), act);
        }
        // 输出层
        let output_out = activate_vec(&self.output.forward(&out), &self.output_activation);
        output_out
    }

    /// 均方误差损失函数
    pub fn mse_loss(output: &Vec<f64>, target: &Vec<f64>) -> f64 {
        output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| (o - t).powi(2))
            .sum::<f64>()
            / output.len() as f64
    }

    /// 训练一次：前向传播、反向传播、参数更新（SGD）
    /// lr: 学习率
    pub fn train(&mut self, input: &Vec<f64>, target: &Vec<f64>, lr: f64) -> f64 {
        // 前向传播，记录每层的输入和输出
        let mut layer_inputs = Vec::new();
        let mut layer_outputs = Vec::new();
        let mut out = input.clone();
        layer_inputs.push(out.clone());
        for (layer, act) in self
            .hidden_layers
            .iter()
            .zip(self.hidden_activations.iter())
        {
            out = layer.forward(&out);
            layer_outputs.push(out.clone());
            out = activate_vec(&out, act);
            layer_inputs.push(out.clone());
        }
        // 输出层
        let output_raw = self.output.forward(&out);
        layer_outputs.push(output_raw.clone());
        let output = activate_vec(&output_raw, &self.output_activation);
        layer_inputs.push(output.clone());

        // 计算损失
        let loss = Network::mse_loss(&output, target);

        // 反向传播
        let delta = output
            .iter()
            .zip(target.iter())
            .zip(activate_derivative_vec(
                &output_raw,
                &self.output_activation,
            ))
            .map(|((o, t), d_act)| (o - t) * d_act)
            .collect::<Vec<f64>>();

        // 更新输出层权重和偏置
        for i in 0..self.output.weights.len() {
            for j in 0..self.output.weights[0].len() {
                self.output.weights[i][j] -=
                    lr * delta[i] * layer_inputs[layer_inputs.len() - 2][j];
            }
            self.output.biases[i] -= lr * delta[i];
        }

        // 反向传播到隐藏层
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
        loss
    }

    /// 推理接口：输入单个样本，返回预测结果
    pub fn predict(&self, input: &Vec<f64>) -> Vec<f64> {
        self.forward(input)
    }

    /// 批量训练接口：输入多组样本，指定轮数和学习率
    /// inputs: Vec<输入样本>
    /// targets: Vec<目标输出>
    /// epochs: 训练轮数
    /// lr: 学习率
    /// 返回每轮的平均损失
    pub fn fit(&mut self, inputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>, epochs: usize, lr: f64) -> Vec<f64> {
        assert_eq!(inputs.len(), targets.len(), "输入和目标数量必须一致");
        let mut losses = Vec::new();
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            for (input, target) in inputs.iter().zip(targets.iter()) {
                total_loss += self.train(input, target, lr);
            }
            let avg_loss = total_loss / inputs.len() as f64;
            losses.push(avg_loss);
        }
        losses
    }

    /// 保存模型到指定路径（JSON 格式）
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self).unwrap();
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// 从指定路径加载模型（JSON 格式）
    pub fn load(path: &str) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut json = String::new();
        file.read_to_string(&mut json)?;
        let net: Self = serde_json::from_str(&json).unwrap();
        Ok(net)
    }
}
