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
    // 构造批量输入和目标输出
    let inputs = vec![vec![0.5, -0.2], vec![1.0, 0.0], vec![0.0, 1.0]];
    let targets = vec![vec![1.0], vec![0.0], vec![0.0]];
    // 训练前的推理
    for (i, input) in inputs.iter().enumerate() {
        let pred = net.predict(input);
        println!("Sample {} before training: input = {:?}, pred = {:?}", i, input, pred);
    }
    // 批量训练
    let losses = net.fit(&inputs, &targets, 1000, 0.1);
    println!("训练过程损失（每200轮）：");
    for (epoch, loss) in losses.iter().enumerate() {
        if epoch % 200 == 0 {
            println!("Epoch {}: loss = {}", epoch, loss);
        }
    }
    // 训练后的推理
    for (i, input) in inputs.iter().enumerate() {
        let pred = net.predict(input);
        println!("Sample {} after training: input = {:?}, pred = {:?}", i, input, pred);
    }
}
