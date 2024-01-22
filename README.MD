# Bounding Box LP Trainer

## Introduction
I hope to create a program that can take in a input of a 1000x1000 px image and return the center of a target and the targets radius

## Progress
- SGD outperforms Adam
- Custom loss function called PFLoss that separately weights the different outputs and penalizes non-matching standard deviation to address MSELoss diversity issue
- Use an identity block to improve accuracy
- You don't need a complex fully connected layer to get good results
- Experiment with different optimizers that embrace SGD over RMSProp and Adam
- Multi-gpu training has not performed well. Struggling with loading times and synchronisation. Best options lie in training separate networks with different optimizers for a couple epochs with each gpu using a different optimizer. The best performing networks will be picked and redistributed across the gpus and rinse repeat to use optimizers only at the time when they perform best.
- Must step the training through baby steps
  - Identify between left and right x values
  - Identify between top and bottom y values as well
  - then identify between 4 different positions (corners)
  - then choose between 9 different positions
  - Random scatter positions of auto generated targets (squares)
  - Add in noise to the input images
  - Increase the noise until crowded but model can locate target
  - Start training on cleaned LP Training data (helper training with license plate highlighted to make more obvious)
  - Train on final training data set
- Do not use dropout layer, has not improved performance (later?)

## Ideas
- Multi-gpu optimizer switching model
- A uniquely perfect custom loss function is necessary to get perfect results