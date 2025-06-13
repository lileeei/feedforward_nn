use crate::layer::Layer;
use crate::activation::{Activation, activate_vec};

/// Network 表示一个支持多层隐藏层和激活函数模块化的前馈神经网络
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
    pub fn new(input_size: usize, hidden_sizes: Vec<usize>, output_size: usize, hidden_activations: Vec<Activation>, output_activation: Activation) -> Self {
        assert_eq!(hidden_sizes.len(), hidden_activations.len(), "每个隐藏层都需要指定激活函数");
        let mut hidden_layers = Vec::new();
        let mut prev_size = input_size;
        for &size in &hidden_sizes {
            hidden_layers.push(Layer::new(prev_size, size));
            prev_size = size;
        }
        let output = Layer::new(prev_size, output_size);
        Self { input_size, hidden_sizes, output_size, hidden_layers, hidden_activations, output, output_activation }
    }

    /// 前向传播：输入 input，返回网络输出
    pub fn forward(&self, input: &Vec<f64>) -> Vec<f64> {
        // 依次通过所有隐藏层及其激活函数
        let mut out = input.clone();
        for (layer, act) in self.hidden_layers.iter().zip(self.hidden_activations.iter()) {
            out = activate_vec(&layer.forward(&out), act);
        }
        // 输出层
        let output_out = activate_vec(&self.output.forward(&out), &self.output_activation);
        output_out
    }
} 