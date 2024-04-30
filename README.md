# Diffusion-Model
Here I trained the Denoising Diffusion Model using U-Net architecture and sampled the newly generated images from it. This involves generating images unconditionally and randomly, without any reliance on dataset labels. The dataset used was [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) having 202,599 number of face images. For any other dataset just change the [model_params.yaml](model_params.yaml) file accordingly. 
# Result
![x0_0](https://github.com/AAArpan/Diffusion-Model/assets/108794407/51f3add7-af18-4e4f-958a-fe33e5cc3b4c)

Here [sample.py](sample.py) file will save the generated images at each timestep within a folder.
For training and sampling
```python
! python3 train.py --config /path to model_params file/
! python3 sample.py --config /path to model_params file/
```
Use this [notebook]([TrainDemo.ipynb](https://github.com/AAArpan/Diffusion-Model/blob/main/Trainingdemo.ipynb)) to replicate the results and for direct sampling download the weights from [here](https://drive.google.com/file/d/1HPCfqnGULjmc8NRrveATN5MRmjCLXtZc/view?usp=sharing) 
