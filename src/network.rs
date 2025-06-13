use crate::layer::Layer;
use crate::activation::sigmoid_vec;

/// Network 表示一个简单的前馈神经网络（单隐藏层）
pub struct Network {
    /// 输入层神经元数
    pub input_size: usize,
    /// 隐藏层神经元数
    pub hidden_size: usize,
    /// 输出层神经元数
    pub output_size: usize,
    /// 隐藏层
    pub hidden: Layer,
    /// 输出层
    pub output: Layer,
}

impl Network {
    /// 创建一个新的前馈神经网络
    /// input_size: 输入层神经元数
    /// hidden_size: 隐藏层神经元数
    /// output_size: 输出层神经元数
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let hidden = Layer::new(input_size, hidden_size);
        let output = Layer::new(hidden_size, output_size);
        Self { input_size, hidden_size, output_size, hidden, output }
    }

    /// 前向传播：输入 input，返回网络输出
    pub fn forward(&self, input: &Vec<f64>) -> Vec<f64> {
        // 1. 输入经过隐藏层线性变换和激活
        let hidden_out = sigmoid_vec(&self.hidden.forward(input));
        // 2. 隐藏层输出经过输出层线性变换和激活
        let output_out = sigmoid_vec(&self.output.forward(&hidden_out));
        output_out
    }
} 