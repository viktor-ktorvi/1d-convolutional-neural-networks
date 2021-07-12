import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from network_model import Generator, Discriminator
from utils.signals import generateSignalData

if __name__ == "__main__":
    # %% Signal parameters

    # ["sin", "square", "saw"]
    classes = ["sin"]

    # [np.sin, signal.square, signal.sawtooth]
    waves = [np.sin]

    Fs = 2000
    signal_len = 200
    t = np.linspace(0, (signal_len - 1) / Fs, signal_len)
    amp_max = 10
    amp_min = 9
    freq_max = 50
    freq_min = 45

    noise_std_percent = 0.1
    # %% Training parameters
    noise_len = 10
    num_signals = 1000
    num_epochs = 100
    batch_size = 64
    lrg = 0.006
    lrd = lrg/50
    discriminate_every_n_batches = 10

    # %% Generate data
    ground_truth, signal_labels, signal_data = generateSignalData(num_signals=num_signals,
                                                                  signal_len=signal_len,
                                                                  classes=classes,
                                                                  waves=waves,
                                                                  amp_max=amp_max,
                                                                  amp_min=amp_min,
                                                                  freq_max=freq_max,
                                                                  freq_min=freq_min,
                                                                  t=t,
                                                                  noise_std_percent=noise_std_percent)

    data_std = np.std(signal_data)

    # %% Setting up the data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")  # CPU because I have a weak GPU
    print(device)

    dataset = TensorDataset(torch.tensor(ground_truth), torch.ones(num_signals))

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # %% Training
    torch.manual_seed(26)

    generator = Generator(signal_len=signal_len, noise_len=noise_len)
    generator = generator.to(device)

    discriminator = Discriminator(signal_len=signal_len)
    discriminator = discriminator.to(device)

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lrg)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lrd)

    criterion = nn.BCELoss()

    generator_loss_array = []
    discriminator_loss_array = []
    for epoch in range(num_epochs):
        running_generator_loss = 0
        running_discriminator_loss = 0

        for i, data in enumerate(train_dataloader, 0):
            train_signals, _ = data[0].to(device, dtype=torch.float), data[1].to(device, dtype=torch.float)

            # generator
            generator_optimizer.zero_grad()

            noise = torch.randn((batch_size, noise_len)).to(device)
            generated_data = generator(noise.unsqueeze(1))
            generator_discriminator_out = discriminator(generated_data)

            generator_loss = criterion(generator_discriminator_out, torch.ones_like(generator_discriminator_out))
            generator_loss.backward()
            generator_optimizer.step()

            running_generator_loss += generator_loss.item()

            # discriminator

            if i % discriminate_every_n_batches == 0:
                discriminator_optimizer.zero_grad()

                true_discriminator_out = discriminator(train_signals.unsqueeze(1))
                true_discriminator_loss = criterion(true_discriminator_out, torch.ones_like(true_discriminator_out))

                generator_discriminator_out = discriminator(generated_data.detach())
                generator_discriminator_loss = criterion(generator_discriminator_out,
                                                         torch.zeros_like(generator_discriminator_out))

                discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2

                discriminator_loss.backward()
                discriminator_optimizer.step()

                running_discriminator_loss += discriminator_loss.item()

        print("epoch: %d\n\tG loss: %0.10f \t D loss: %0.10f" % (
            epoch, running_generator_loss, running_discriminator_loss))
        generator_loss_array.append(running_generator_loss)
        discriminator_loss_array.append(running_discriminator_loss)

    # %%
    plt.figure()
    plt.plot([x / max(generator_loss_array) for x in generator_loss_array], color='r', label='generator loss')
    plt.plot([x / max(discriminator_loss_array) for x in discriminator_loss_array], color='b', label='discriminator loss')
    plt.title("Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

    # %%

    with torch.no_grad():
        noise = torch.randn((1, noise_len)).to(device)
        generated_data = generator(noise.unsqueeze(1))

        generated_data = generated_data.detach().cpu().numpy().squeeze()

        plt.figure()
        plt.plot(generated_data, color='b')
        plt.title("Generated data")
        plt.xlabel("n [sample]")
        plt.ylabel("x(n) [unit]")
        plt.show()
