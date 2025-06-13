mod layer;
mod activation;
mod network;

use activation::Activation;

fn main() {
    // 创建一个包含2个输入、2个隐藏层（每层3个神经元）、1个输出的前馈神经网络
    // 第一隐藏层用 ReLU，第二隐藏层用 Tanh，输出层用 Sigmoid
    let mut net = network::Network::new(
        2,
        vec![3, 3],
        1,
        vec![Activation::ReLU, Activation::Tanh],
        Activation::Sigmoid,
    );
    // 构造输入和目标输出
    let input = vec![0.5, -0.2];
    let target = vec![1.0];
    // 训练前的输出
    let output_before = net.forward(&input);
    println!("Before training: Output = {:?}", output_before);
    // 训练若干步
    for epoch in 0..1000 {
        let loss = net.train(&input, &target, 0.1);
        if epoch % 200 == 0 {
            println!("Epoch {}: loss = {}", epoch, loss);
        }
    }
    // 训练后的输出
    let output_after = net.forward(&input);
    println!("After training: Output = {:?}", output_after);
}
