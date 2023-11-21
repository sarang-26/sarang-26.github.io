---
title: "Anatomy of Memory Utilisation"
date: 2023-10-20T09:44:50+01:00
draft: false
ShowToc: true
---

To understand, how to train the model efficiently, its very important to understand how the memory is behaving in different training stages. In this blog, we will try to understand the anatomy of memory utilisation while training a model.\



Anatomy of Model Memory while training:
   
1. Optimizer states
2. Gradients
3. Forward Activation for gradient computation 
4. Tempory Buffers




### Optimizer States:

AdamW is primarly used for training models which requires more training cycles and has effectively higher trainable parameter. It efficently penalises high weight values, by decoupling the process of weight decay in an additional step. For us to understand, what actually goes on under the hood, we need to understand how AdamW is different from Adam optimisation mathematically.



#### Adam Optimisation Algorithm

1. Compute the gradients of the loss with respect to the parameters:
   $$ g_t = \nabla_{\theta}f_t(\theta_{t-1}) $$

2. Update biased first moment estimate:
   $$ m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t $$
   This is exponetially decaying average of past gradients. It helps SGD in relevant direction and dampens oscillation.

3. Update biased second raw moment estimate:
   $$ v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 $$
   This is the exponentially decaying average of the past squared gradients. It is used to adapt the learning rate of each parameters, scaling down the steps for parameters with large gradients
   (the ones, which initally had a large update) and scaling up the steps for parameter with small gradients.

4. Compute bias-corrected first moment estimate:
   $$ \hat{m}_t = \frac{m_t}{1 - \beta_1^t} $$

5. Compute bias-corrected second raw moment estimate:
   $$ \hat{v}_t = \frac{v_t}{1 - \beta_2^t} $$

6. Update the parameters:
   $$ \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t $$



#### AdanW Optimisation Algorithm

7. Perform a weight decay step:
   $$ \theta_t = \theta_{t-1} - \eta \cdot \lambda \cdot \theta_{t-1} $$

8. Then, proceed with the Adam update:
   $$ \theta_t = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t $$


The key difference here is that the weight decay is applied directly to the parameters before the adaptive learning rate is applied. This small change helps in regularizing and improving the generalization of the model. 

Both m_t and v_t are maintained for each parameter. Which mean for a model with N parameters, it Adam/AdamW will require to store 2N moments. Since, each moment is a floating point number typically stored as a 32 bit(4 bytes), this results in 8N bytes of memory usage while training. Usally deep learning models and increasing becoming popular LLMs have billions of parameter. To put this to more prespective, lets consider Chat-GPT 4, which consists of 175 billion parameters. In order to train all the parameter, we would require a memory of \
                                                      \
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; 175 Billion * 8 Bytes = 1.4e + 9 bytes = 1120 GB of cache memory !

Hence, the 8-bit version of Adam (instead of 32 bits ) reduced memory footprint of the optimiser by storing the optimiser states in lower precision. The lower precision, doesnt significanlty affect the model performance while finetuning.


### Gradients

One of the most noble ideas which makes deep learning unique, is the idea of backpropogation. It's behaves like an examinaiton which the the model undergoes, and verifies its weaknesses and try to perfect itself, by exactly working and targetting on its weak areas of understanding. 

It put this into a more mathematicaly terms, it provides the direction in which the parameters should be adjusted which would lead to net reduction of loss. Gradients are partial derivatives of the loss function with respect to each parameter of the model. They are calculated for every iteration(number of batches to compelete an epoch). Given a loss function, which is typically the combination of Softmax and Cross-Entropy looks like this:

{{< figure src="images/cross_entropy.png" title="" >}}

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Fig 1. Loss Function (Cross Entropy) 



 
Given the loss function L and parameters to be $$ \theta =(\theta_{1}, \theta_{2} ... ,\theta_{n} )$$ we get gradients 


{{< figure src="images/gradients.png" title="" >}}
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Fig 2. Gradients 



During the training, we calculate the gradients for each parameter
{{< figure src="images/one_grad.png" title="" >}}
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Fig 3. Calculating a single gradient


\
and update the parameter value accordingly

{{< figure src="images/update_func.png" title="" >}}

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Fig 4. Update Function


Usually the calculated gradients, are stored in 32 bit precision [(single precision)](https://en.wikipedia.org/wiki/Single-precision_floating-point_format), which means each if there are a 175 billion parameters, and each gradient is 4 bytes in size\

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; 175 Billion * 4 Bytes = 652 GB of memory !!!

This is a huge amount of memory, and is not feasible to store all the gradients in memory. Hence, we store the gradients in lower precision, which is 16 bit precision [(half precision)](https://en.wikipedia.org/wiki/Half-precision_floating-point_format). This reduces the memory footprint by half, and is a good tradeoff between memory and performance.

### Forward Activation for gradient computation

In order to calculate the gradients, we need to compute the forward activation of the model. This is the process of passing the input through the model, and calculating the output. This is done for each batch, and the gradients are calculated for each batch. The forward activation is stored in 32 bit precision, and is stored in the cache memory. This is the reason why, we need to have a large cache memory, to store the forward activation of the model.

### Temporary Buffers

Its very common, to encounter the OOM (Out of Memory) error while training a model, even when we know that the number of parameters in the model are compareltive less. Sometimes, we can see the cache memory is emptied after a few epochs, and the training resumes. This is because, the cache memory is not only used to store the model parameters, but also to store the temporary buffers.
Temporary variables are created during computation. They temporarily require a lot if memory, before they are finally released. For example,

{{< figure src="images/before_activation.png" title="" >}}

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Fig 5. Function before activation

The output of the matrix multiplication requires memory, before its is operated again by the activation function.

{{< figure src="images/activation_function.png" title="" >}}

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Fig 6. Activation Function

The output of the activation function, requires memory before its is operated again by the next layer or being used by backpropogation to calculate the gradients. 




