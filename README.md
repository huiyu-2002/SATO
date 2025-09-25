# Spatially-Aware Transformer Operator for Real-Time Aerodynamic Evaluations of Arbitrary Three-dimensional Vehicles

Code and data accompanying the manuscript titled "Spatially-Aware Transformer Operator for Real-Time Aerodynamic Evaluations of Arbitrary Three-dimensional Vehicles", authored by Huiyu Yang, Jianghang Gu, Yuntian Chen, Yuanwei Bin, Jianchun Wang and Shiyi Chen.

## Abstract

With the increasing demand for rapid aerodynamic evaluations in the automotive industry, data-driven surrogate models for predicting vehicle surface pressure have emerged as promising alternatives to traditional computational fluid dynamics (CFD) simulations. However, most existing models fall short in capturing the intrinsic spatial correlations of complex 3D vehicle geometries, particularly when represented as unstructured point clouds. 

To address these limitations, we propose the spatially-aware transformer operator (SATO), a novel neural operator framework that unifies global and local spatial correlation modeling through two complementary modules: physics-attention and serialized-attention. SATO introduces a spatial aggregation mechanism to capture large-scale geometric structures while simultaneously employing a serialization technique based on space-filling curves to transform unstructured 3D points into structured sequences, allowing efficient local feature extraction. 

This dual-attention mechanism enables SATO to achieve multi-scale feature fusion with nearly linear computational complexity. Extensive evaluations are conducted on two datasets: the ShapeNet Car dataset, containing a wide variety of simplified vehicle shapes, and the DrivAerNet dataset, comprising industrial-grade, high-fidelity car models. 

Results demonstrate that SATO reduces the relative L₂ error in pressure prediction by 13% and 11%, respectively, compared to state-of-the-art methods, while maintaining real-time inference speed (under one second) for vehicles with over 0.42 million mesh vertices. The complementarity of the two attention mechanisms is further substantiated through a visual analysis of their internal operations. Our study highlights the effectiveness of integrating global and local spatial correlations in transformer-based operator learning and establishes SATO as a robust surrogate model for real-time aerodynamic evaluations of arbitrary vehicle geometries.

## Implementation Framework
We release the code for the PyTorch environment, and the code for the PaddlePaddle environment will come soon.

## Datasets

1. [ShapeNet Car Dataset 🔗](https://github.com/thuml/Transolver/tree/main/Car-Design-ShapeNetCar)
2. [DrivAerNet Dataset 🔗](https://github.com/Mohamedelrefaie/DrivAerNet/tree/main/DrivAerNet_v1)



## References

1. Transolver: [GitHub Repository](https://github.com/thuml/Transolver)
2. PointTransformerV3: [GitHub Repository](https://github.com/Pointcept/PointTransformerV3)
