from Generator_Discriminator import Generator
from utils import *
import matplotlib.pyplot as plt
import numpy as np



if __name__ == '__main__':
    seed_everything(seed=11)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    # 创建生成器和判别器
    generator = Generator(input_size=100, output_size=28*28).to(device)
    
    generator_log_path = 'log/G_epoch40.pth'
    
    if len(generator_log_path) != 0 :
        generator.load_state_dict(torch.load(generator_log_path))

    # 生成新样本并显示
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(16, 100).to(device)
        generated_samples = generator(noise).view(-1, 28, 28).cpu().detach().numpy()
    
    # 还原数据
    generated_data = (generated_samples * 0.5 + 0.5) * 255 / 7 # 还原数据
    generated_data = np.mean(generated_data,axis=1) # 按列求平均
    generated_data_final = np.zeros((generated_data.shape[0],7))
    
    for i in range(generated_data.shape[0]):
        for j in range(7):
            generated_data_final[i,j] = (generated_data[i,j]+generated_data[i,j+7]+generated_data[i,j+14]+generated_data[i,j+21]) / 4
        #sigma原则
    
    print('the final data is in the generated_data_final, plase check!')
    
    plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(generated_samples[i], cmap='gray')
        plt.axis('off')
    plt.show()