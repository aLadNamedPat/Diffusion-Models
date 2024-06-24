# STILL A WORK IN PROGRESS

# Diffusion-Based Models

![Original Paper](https://arxiv.org/pdf/2006.11239)

Lots of help from: https://github.com/filipbasara0/simple-diffusion/blob/main/simple_diffusion/model/unet.py
Diffusion-based models operate slightly differently from our VAE model.

The model is no longer attempting to learn how to generate images from a latent space. Instead, the learning process is simplified
by the model learning the random mixing to unmix photos and generate new ones from random samples.

## How does this work?

### Training
We want to get our model to grow good at learning the gaussian distribution by predicting what the select gaussian term actually is in a sample.

Training occurs as a series of steps as described here:

![TrainingSteps](/Diffusion/DescribingDiffusion/training.png)

I wish papers would make explanations less math-dependent (we should expand out of our ecosterism!), but tldr, the model is trying to learn the 
noise that is generated at each timestep. 

I want to point out a key point you're probably wondering, why do we need to sample the timesteps when we are producing noisy images? Remember, we 
want to make sure that our model is not learning dependencies between samples (this is one of the central theories of ML). 
All samples taken should be therefore independent of one another.

### Producing New Images
Then, as we want to produce new image samples, it's as simple as producing a new noisy image that we want to de-noise through a series of timesteps

![Training](/Diffusion/DescribingDiffusion/sampling.png)

Using the series of timesteps we specified in the noising process, we continually apply the noise predictor to redefine the original image. Notice the z
term in the image is continaully sampled and added to the denoised image which serves to model the stochasticity of the noising process to make denoising 
non-deterministic.