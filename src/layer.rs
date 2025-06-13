use serde::{Serialize, Deserialize};

// Layer 表示神经网络中的一层，包括权重和偏置
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Layer {
    /// 权重矩阵，形状为 [输出神经元数][输入神经元数]
    pub weights: Vec<Vec<f64>>,
    /// 偏置向量，长度为输出神经元数
    pub biases: Vec<f64>,
}

impl Layer {
    /// 创建一个新的 Layer，随机初始化权重和偏置
    /// input_size: 输入神经元数
    /// output_size: 输出神经元数
    pub fn new(input_size: usize, output_size: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let weights = (0..output_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        let biases = (0..output_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Self { weights, biases }
    }

    /// 前向传播：对输入向量 input 进行线性变换并加上偏置，返回输出向量
    pub fn forward(&self, input: &Vec<f64>) -> Vec<f64> {
        self.weights.iter().zip(self.biases.iter()).map(|(w_row, b)| {
            w_row.iter().zip(input.iter()).map(|(w, i)| w * i).sum::<f64>() + b
        }).collect()
    }
} 