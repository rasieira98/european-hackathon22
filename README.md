# Schneider Electric European Hackathon
## Challenge Data-Science: Zero Deforestation Mission
- - -
<img src="https://user-images.githubusercontent.com/116558787/201539352-ad0e29cb-501d-4644-9134-33a8b07ad693.PNG" alt="Schneider" width="800"/>


### Group SDG-UR: 
- - -

- Ramón Sieira Martínez (SDG Group, Spain)
- Francisco Javier Martínez de Pisón (University of La Rioja, Spain)
- Jose Divasón (University of La Rioja, Spain)

---

This Jupyter notebook allows one to train and create submissions for the Data-Schneider-European-Hackathon "Zero Deforestation Mission". It also contains the methodology and the general ideas that we have followed.

##### Key ideas of our approach:
- - -

1. Training is performed with a 3-fold cross-validation. The model that obtains the best F1-Macro is saved.
2. The training dataset has been balanced.
3. The final prediction in the submission corresponds to an ensemble of 8 models, each one trained using a 5-fold cross-validation.


##### Detailed methodology:
- - -

###### Step 1: Tuning the learning rate, bath size and epochs

After a first visual inspection of the images contained in the dataset, we have performed several test in order to find an optimal batch size, also adjusting the learning rate and finding an adequate value of epochs needed.

We performed some test with a baseline approach. The goal at this step is to have some idea about the performance of a simple model.

We have also tried with different batch sizes (from 16 to 72), which is very important for us because, in principle, a larger batch size should speed up the calculations, which means we can perform more tests in less time (which is the key in a competition like this). Surprisingly, we have obtained slower and worse results with large bath sizes, even though we also tried with different learning rates (keeping it constant, increasing it linearly with respect to the learning rate, etc.). This has been very surprising and an unexpected result, but due to lack of time we have not studied this problem further. Therefore, we have proceeded with 16 as batch size, 0.00001 as learning rate and 100 epochs.

###### Step 2: Finding augmentation filters

Data augmentation is a fundamental task when training any deep learning model. To try to make the process as fast as possible, we use the Kornia library to perform data augmentation on GPU instead of using CPU.

The first thing we have done is to test various types of data augmentation that are available in the library (separately). For this task, we used a model "tf_efficientnet_b3_ns".

We have tried the following augmentations. Due to the lack of time we have not performed many tests, only a few of them:

- HORIZONTAL_FLIP
- VERTICAL_FLIP
- ERASING_MAX_RATIO
- COLORJITTER_BRIGHTNESS
- COLORJITTER_CONTRAST
- COLORJITTER_SATURATION
- COLORJITTER_HUE
- ROTATION_DEGREES
- SHARPNESS_VALUE
- MOTIONBLUR_KERNEL_SIZE
- MOTIONBLUR_KERNEL_ANGLE
- MOTIONBLUR_KERNEL_DIRECTION

The goal has been to find which augmentations really increase the final score. Some of them decrease the score, so we directly discarded them. We have discovered that some of them do produce a high increase, like saturation, sharpness, rotations and blur. It was not clear if some of the other filters really produce an improvement, so we did not perform further experiments with them to save up time.

###### Step 3: Finding an augmentation range

Those filters with the best results are selected and their maximum and minimum values are determined. We have not performed many test to find an optimal range because here is not enough time for that, so sometimes we have had to estimate it based only on one or two results.

Finally, the following augmentations and ranges have been selected:
```
    saturation_min_max = (0.01, 0.20) 
    rotation_min_max = (0.01, 20.00)
    sharpness_min_max = (0.06, 0.20)  
    blur_motion_min_max = np.array([5, 7]) 
```
We always perform a horizontal flip with probability 0.50, since we have seen it increases the performance of the model.


###### Step 4: Random search with different backbones and augmentations

A random search is performed with different backbones and values of the selected filters and within the ranges defined in step 2. This is done in several GPUs. Some models failed because they did not fit in some of the GPUs (Nvidia 3070), so sometimes we were not able to test each configuration.

###### Step 5: Selection of the best models and use of pseudolabelling

After the random search, we have selected the models with the best scores. Such models are again trained using pseudo-labelling, i.e., using the prediction that has been previously obtained from the test dataset.

###### Step 6: Ensemble

The best models of the previous step are selected and we build a Weighted Blending Ensemble to achieve a final XXXX f1-score. To do it, we tested the combinations of the 2, 3, 4, ..., N best models that have been obtained in the previous step, where N is optimized to get the best result. The weights of the models are obtained by optimization with the validation predictions.


---

#### Further comments:
---

The code can be used directly from the notebook or with its python version from the console. 

The following example allows one to train a "tf_efficientnet_b3_ns" model with a 3-fold cross-validation and several augmentations (horizontal flip, color saturation, rotarion, sharpness and motion blur): 

```
python 01006_FINAL_CODE.py --OUTPUT_DIR results/MODEL_FIRST/ --VERBOSE 0 --BACKBONE tf_efficientnet_b3_ns --GPU_DEVICE cuda:0 --VERSION 01003 --NUM_EPOCHS 130 --BATCH_SIZE 16 --LR 0.00001 --NUM_FOLDS 3 --RUN_FOLDS 1 --AUGMENTATION True --HORIZONTAL_FLIP 0.50 --COLORJITTER_PROB 0.181641967391821 --COLORJITTER_SATURATION 0.0927803557147827 --ROTATION_PROB 0.181641967391821 --ROTATION_DEGREES 19.1576227380687 --SHARPNESS_PROB 0.181641967391821 --SHARPNESS_VALUE 0.140246520091931 --MOTIONBLUR_PROB 0.181641967391821 --MOTIONBLUR_KERNEL_SIZE 7
```

The parameters' description follows:


*Training Parameters*

- OUTPUT_DIR = Directory for results
- TRAIN_MODEL = To train models
- CREATE_SUBMISSION = To create submissions
- VERBOSE = Verbose
- INCLUDE_TEST = Include test in train for pseudo-labeling
- MIN_THR_TEST = Threshold of preds to include in pseudo
- TARGET_DIR_PREDS = Directory with preds for test in pseudo
- PREVIOUS_MODEL = Dir with previous model
- BACKBONE = Backbone
- GPU_DEVICE = GPU device
- SEED = Main Seed
- NUM_FOLDS = Number of folds
- RUN_FOLDS = Number of folds to run (1 to quick validation)
- LR = Min Learning Rate
- NUM_EPOCHS = Max epochs
- BATCH_SIZE = Batch size
- NUM_WORKERS = Numworkers in Dataloader
- SAMPLES_BY_CLASS = Number of row of each class included in balanced train db
- USA_FP16 = To use 16 Float in GPU
- VERSION = Code Version

[Kornia Augmentation Filters](https://kornia.readthedocs.io/en/v0.4.1/augmentation.html#module-api)

- AUGMENTATION = Apply augmentation
- HORIZONTAL_FLIP = Probability for flip
- VERTICAL_FLIP = Probability for flip
- ERASING_MAX_PROB = Probability for erasing
- ERASING_MAX_RATIO = Max ratio box to erase
- COLORJITTER_PROB = Probability for colorjitter
- COLORJITTER_BRIGHTNESS = Value for brightness
- COLORJITTER_CONTRAST = Value for contrast
- COLORJITTER_SATURATION = Value for saturation
- COLORJITTER_HUE = Value for hue
- ROTATION_PROB = Probability for rotation
- ROTATION_DEGREES = Max degrees for rotation
- SHARPNESS_PROB = Probability for sharpness
- SHARPNESS_VALUE = Max value of sharpness
      
- MOTIONBLUR_PROB = Probability for motionblur
- MOTIONBLUR_KERNEL_SIZE = Max size ofkernel motion
- MOTIONBLUR_KERNEL_ANGLE = Max angle of the motion
- MOTIONBLUR_KERNEL_DIRECTION = Direction of the motion
