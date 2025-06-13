use crate::layer::Layer;
use crate::activation::sigmoid_vec;

/// Network 表示一个支持多层隐藏层的前馈神经网络
pub struct Network {
    /// 输入层神经元数
    pub input_size: usize,
    /// 每个隐藏层的神经元数
    pub hidden_sizes: Vec<usize>,
    /// 输出层神经元数
    pub output_size: usize,
    /// 多个隐藏层
    pub hidden_layers: Vec<Layer>,
    /// 输出层
    pub output: Layer,
}

impl Network {
    /// 创建一个新的前馈神经网络
    /// input_size: 输入层神经元数
    /// hidden_sizes: 每个隐藏层的神经元数（Vec）
    /// output_size: 输出层神经元数
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut hidden_layers = Vec::new();
        let mut prev_size = input_size;
        for &size in &hidden_sizes {
            hidden_layers.push(Layer::new(prev_size, size));
            prev_size = size;
        }
        let output = Layer::new(prev_size, output_size);
        Self { input_size, hidden_sizes, output_size, hidden_layers, output }
    }

    /// 前向传播：输入 input，返回网络输出
    pub fn forward(&self, input: &Vec<f64>) -> Vec<f64> {
        // 依次通过所有隐藏层
        let mut out = input.clone();
        for layer in &self.hidden_layers {
            out = sigmoid_vec(&layer.forward(&out));
        }
        // 输出层
        let output_out = sigmoid_vec(&self.output.forward(&out));
        output_out
    }
} 