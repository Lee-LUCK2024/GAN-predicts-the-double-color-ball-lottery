import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Generator_Discriminator import Generator, Discriminator
from utils import *

# 定义训练过程
def train_gan(generator, discriminator, dataloader, device, num_epochs=40, lr=0.0002):
    criterion = nn.BCELoss().to(device)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for real_data, _ in dataloader:
            # print(real_data)
            batch_size = real_data.size(0)
            real_data = real_data.view(-1, 28 * 28).to(device)

            # 训练判别器
            optimizer_D.zero_grad()
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            real_output = discriminator(real_data)
            real_loss = criterion(real_output, real_labels)
            real_loss.backward()

            noise = torch.randn(batch_size, 100).to(device)
            fake_data = generator(noise)
            fake_output = discriminator(fake_data.detach())
            fake_loss = criterion(fake_output, fake_labels)
            fake_loss.backward()

            optimizer_D.step()

            # 训练生成器
            optimizer_G.zero_grad()
            output = discriminator(fake_data)
            gen_loss = criterion(output, real_labels)
            gen_loss.backward()
            optimizer_G.step()

        # 打印损失
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Generator Loss: {gen_loss.item():.4f}, "
              f"Discriminator Loss: {real_loss.item() + fake_loss.item():.4f}")
        
        # save model
        if ((epoch+1) % 5 == 0):
            torch.save(generator.state_dict(), 'log/G_epoch{}.pth'.format(epoch+1))
            torch.save(discriminator.state_dict(), 'log/D_epoch{}.pth'.format(epoch+1))

if __name__ == '__main__':
    
    seed_everything(seed=11)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据集
    print('-------load dataset-------')
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # mnist_dataset = datasets.MNIST(root='mnist', train=True, transform=transform, download=True)
    # dataloader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)
    input_path = 'data'
    dataloader = load_dataset(input_path)
    print('-------load dataset finish-------')
    
    # 创建生成器和判别器
    generator = Generator(input_size=100, output_size=28*28).to(device)
    discriminator = Discriminator(input_size=28*28).to(device)
    
    generator_log_path = 'log/G_epoch50_ssq.pth'
    discriminator_log_path = 'log/D_epoch50_ssq.pth'
    
    if len(generator_log_path) != 0 :
        generator.load_state_dict(torch.load(generator_log_path))
    
    if len(discriminator_log_path) != 0 :
        discriminator.load_state_dict(torch.load(discriminator_log_path))
    
    
    # 训练GAN
    print('-------start training-------')
    train_gan(generator, discriminator, dataloader, device)
    print('-------training finish-------')