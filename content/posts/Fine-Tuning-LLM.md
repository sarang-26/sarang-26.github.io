---
title: "Understanding the Anatomy of GPU Utilisation"
date: 2023-10-29T17:55:54+01:00
draft: true
tableofcontent: true
---

Model Training Anatomy


We can begin this blog by answering some very important questions:

How does GPU gets utlised while training the Model ?
- What is GPU RAM/VRAM 
- Components that take up memory
- why we end up using more memory than the size of the model ?
  

Trasformer Model Operations:
1. Tensor Contraction 
2. Statistical Normalisation 
3. Element Wise Operators

(This knowledge can be helpful to know when analyzing performance bottlenecks.)

Components that occupy memory while training the model:
1. Model Weights
2. Optimiser States
3. gradients
4. forward activation saved for gradient computation 
5. temporary buffers
6. functionality specific memory



Understanding TRL, PEFT and trainig arguments and bits and bytes configuration 



- Understanding Bits and bytes (Weight quantisation)
- Understanding Datasets
- Model Tokenizers and model configurations
- 