mod layer;
mod activation;
mod network;

fn main() {
    // 创建一个包含2个输入、3个隐藏神经元、1个输出的前馈神经网络
    let net = network::Network::new(2, 3, 1);
    // 构造输入向量
    let input = vec![0.5, -0.2];
    // 执行前向传播，获得输出
    let output = net.forward(&input);
    // 打印输入和输出
    println!("Input: {:?}", input);
    println!("Output: {:?}", output);
}