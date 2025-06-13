use serde::{Serialize, Deserialize};

/// 激活函数枚举，支持多种激活方式
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Activation {
    Sigmoid,
    ReLU,
    Tanh,
}

/// 对单个值应用激活函数
pub fn activate(x: f64, act: &Activation) -> f64 {
    match act {
        Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Activation::ReLU => x.max(0.0),
        Activation::Tanh => x.tanh(),
    }
}

/// 对向量每个元素应用激活函数
pub fn activate_vec(v: &Vec<f64>, act: &Activation) -> Vec<f64> {
    v.iter().map(|&x| activate(x, act)).collect()
}

/// 计算激活函数的导数
pub fn activate_derivative(x: f64, act: &Activation) -> f64 {
    match act {
        Activation::Sigmoid => {
            let s = activate(x, act);
            s * (1.0 - s)
        },
        Activation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
        Activation::Tanh => 1.0 - x.tanh().powi(2),
    }
}

/// 对向量每个元素计算激活函数的导数
pub fn activate_derivative_vec(v: &Vec<f64>, act: &Activation) -> Vec<f64> {
    v.iter().map(|&x| activate_derivative(x, act)).collect()
} 