import torch
import matplotlib.pyplot as plt

class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas  = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1. - self.alpha_cum_prod)

    def add_noise(self, original, noise, t):
        original_shape = original.shape # -----------> (b, c, h, w)
        batch_size = original_shape[0] 

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].reshape(batch_size)

        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        return sqrt_alpha_cum_prod*original + sqrt_one_minus_alpha_cum_prod*noise
    
    def sample_prev_timestep(self, xt, noise_pred, t):
        x0 = (xt - (self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred)) / self.sqrt_alpha_cum_prod[t]
        x0 = torch.clamp(x0, -1., 1.)
        
        mean = xt - ((self.betas[t] * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod[t]))
        mean = mean / torch.sqrt(self.alphas[t])

        if t==0:
            return mean, x0
        else:
            variance = (1. - self.alpha_cum_prod[t-1]) / (1. - self.alpha_cum_prod[t])
            variance = variance * self.betas[t]
            sigma =  variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma*z, x0
        
batch_size = 1
channels = 1
height = 100
width = 100

original_data = torch.randn(batch_size, channels, height, width)
noise = torch.randn(batch_size, channels, height, width)

# Define the parameters for noise scaling
num_timesteps = 10
beta_start = 0.1
beta_end = 0.9

# Create an instance of LinearNoiseScheduler
scheduler = LinearNoiseScheduler(num_timesteps, beta_start, beta_end)

# Add noise over timesteps
noisy_data_list = []
for t in range(num_timesteps):
    noisy_data = scheduler.add_noise(original_data, noise, t)
    noisy_data_list.append(noisy_data)

# Plot original and noisy data
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Data')
plt.imshow(original_data.squeeze().numpy(), cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Noisy Data at timestep {}'.format(num_timesteps))
plt.imshow(noisy_data_list[-1].squeeze().numpy(), cmap='gray')
plt.show()