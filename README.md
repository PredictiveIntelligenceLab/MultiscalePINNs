## Multi-scale Fourier features for physics-informed neural networks

Code and data (available upon request) accompanying the manuscript titled "On the eigenvector bias of Fourier feature networks: From regression to solving multi-scale PDEs with physics-informed neural networks", authored by Sifan Wang, Hanwen Wang, and Paris Perdikaris.

## Abstract

Physics-informed neural networks (PINNs) are demonstrating remarkable promise in integrating physical models with gappy and noisy observational data, but they still struggle in cases where the target functions to be approximated exhibit high-frequency or multi-scale features. 
In this work we investigate this limitation through the lens of Neural Tangent Kernel (NTK) theory and elucidate how PINNs are biased towards learning functions along the dominant eigen-directions of their limiting NTK. Using this observation, we construct novel architectures that employ spatio-temporal and multi-scale random Fourier features, and justify how such coordinate embedding layers can lead to robust and accurate PINN models. Numerical examples are presented for several challenging cases where conventional PINN models fail,  including wave propagation and reaction-diffusion dynamics, illustrating how the proposed methods can be used to effectively tackle both forward and inverse problems involving partial differential equations with multi-scale behavior. 

## Citation

    @article{wang2021eigenvector,
      title={On the eigenvector bias of fourier feature networks: From regression to solving multi-scale pdes with physics-informed neural networks},
      author={Wang, Sifan and Wang, Hanwen and Perdikaris, Paris},
      journal={Computer Methods in Applied Mechanics and Engineering},
      volume={384},
      pages={113938},
      year={2021},
      publisher={Elsevier}
      }
