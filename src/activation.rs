/// Sigmoid 激活函数，将输入压缩到 (0,1) 区间
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// 对向量每个元素应用 Sigmoid 激活函数
pub fn sigmoid_vec(v: &Vec<f64>) -> Vec<f64> {
    v.iter().map(|&x| sigmoid(x)).collect()
} 