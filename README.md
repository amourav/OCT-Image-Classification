# OCT Image Classification

## Background

http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

Retinal optical coherence tomography (OCT) is an imaging technique used to capture high-resolution cross sections of the retinas of living patients. It has become a standard imaging modality for diagnosis and treatment management of disease leading to vision loss with nearly 30 million scans acquired annually (Swanson and Fujimoto, 2017). Age related diseases such as age-related macular degeneration (AMD) and diabetic macular edema (DME), choroidal neovascularization (CNV) are the leading causes of blindness worldwide (Varma et al., 2014, Wong et al., 2014), and are likely to become more prevalent with an aging population. Early diagnosis of CNV and DME are crucial, as delayed treatment could lead to irreversible loss of vision.

![Image](https://github.com/amourav/OCT-Image-Classification/blob/master/readMeImgs/oct.jpg)

Figure 1. (Far left) shows choroidal neovascularization (CNV) with neovascular membrane (white arrowheads) and associated subretinal fluid (arrows). (Middle left) Diabetic macular edema (DME) with retinal-thickening-associated intraretinal fluid (arrows). (Middle right) Multiple drusen (arrowheads) present in early AMD. (Far right) Normal retina with preserved foveal contour and absence of any retinal fluid/edema.

source: [Kermany et. al. 2018](http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)


### Dataset

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (NORMAL,CNV,DME,DRUSEN). There are 84,495 OCT images (JPEG) and 4 categories (NORMAL,CNV,DME,DRUSEN).

Images are labeled as (disease)-(randomized patient ID)-(image number by this patient) and split into 4 directories: CNV, DME, DRUSEN, and NORMAL.

Optical coherence tomography (OCT) images (Spectralis OCT, Heidelberg Engineering, Germany) were selected from retrospective cohorts of adult patients from the Shiley Eye Institute of the University of California San Diego, the California Retinal Research Foundation, Medical Center Ophthalmology Associates, the Shanghai First People’s Hospital, and Beijing Tongren Eye Center between July 1, 2013 and March 1, 2017.

Before training, each image went through a tiered grading system consisting of multiple layers of trained graders of increasing expertise for verification and correction of image labels. Each image imported into the database started with a label matching the most recent diagnosis of the patient. The first tier of graders consisted of undergraduate and medical students who had taken and passed an OCT interpretation course review. This first tier of graders conducted initial quality control and excluded OCT images containing severe artifacts or significant image resolution reductions. The second tier of graders consisted of four ophthalmologists who independently graded each image that had passed the first tier. The presence or absence of choroidal neovascularization (active or in the form of subretinal fibrosis), macular edema, drusen, and other pathologies visible on the OCT scan were recorded. Finally, a third tier of two senior independent retinal specialists, each with over 20 years of clinical retina experience, verified the true labels for each image. The dataset selection and stratification process is displayed in a CONSORT-style diagram in Figure 2B. To account for human error in grading, a validation subset of 993 scans was graded separately by two ophthalmologist graders, with disagreement in clinical labels arbitrated by a senior retinal specialist.

Source: http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5


## Project Overview

We constructed an ensemble CNN classifier to diagnose the most common retinal diseases and improve upon the results presented by [Kermany et. al 2018](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5). The reported method consisted of the [InceptionV3](https://arxiv.org/abs/1512.00567) network pretrained on the [ImageNet](http://www.image-net.org/) dataset, then fine-tuned on a set of target OCT images. We extended this architecture by utilizing several classifiers in an ensemble fashion.

Our method consists of four CNNs widely used CNNs in image classification tasks ([VGG16](https://arxiv.org/abs/1409.1556), [ResNet50](https://arxiv.org/abs/1512.03385), [Xception](https://arxiv.org/abs/1610.02357), [InceptionV3](https://arxiv.org/abs/1512.00567)). These networks were pretrained on the [ImageNet](http://www.image-net.org/) dataset, then fine-tuned on 1000 OCT images (250 NORMAL, 250 CNV, 250 DME, 250 DRUSEN) labeled by a team of expert clinicians. An independent test set of 1000 OCT images (250 NORMAL, 250 CNV, 250 DME, 250 DRUSEN) was used to compare each model (Fig. 2). To reflect the potential clinical use of these predictions, labels were also binarized into two groups: URGET (CNV, DME) and NON-URGENT (NORMAL, DRUSEN).

![Image](https://github.com/amourav/OCT-Image-Classification/blob/master/readMeImgs/comparison.png)

Figure 2. Comparison of several ImageNet pretrained CNNs, fine tuned on OCT scans, and evaluated on the test set.

Although we were able to achieve superior performance on the test set with [Xception](https://arxiv.org/abs/1610.02357), even higher performance was obtained by combining the predictions across all the model (Fig. 3).

![Image](https://github.com/amourav/OCT-Image-Classification/blob/master/readMeImgs/meta.png)

Figure 3. Performance comparison of different methods of combining outputs of VGG16, ResNet50, InceptionV3, and Xception.

Our comparison found that averaging probabilities across all models as a simple yet accurate technique which improved above the performance of any individual model as well as the previous published results. This method yielded a 4-class accuracy of 96.4% (in distinguishing CNV, DME,NORMAL, DRUSEN), which is a 3% improvement on what was reported by [Kermany et. al 2018](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5) on the same test data. For the binary classification task of URGENT vs NON-URGENT the ensemble CNN achieved an accuracy of 97.5%, sensitivity of 98.6%, specificity of 96.5%, and ROC AUC of 99.8%. 


## Installation

Download 'python 3.6' and install the packages in `requerements.txt`.

## Usage

### Data

Download the data from (here)[https://data.mendeley.com/datasets/rscbjbr9sj/2].

### Training

From the root project directory run `train.py` with the following arguments:

`-iTrn` Path to the training data directory. Must contain 4 subdirectories (NORMAL,CNV,DME,DRUSEN) corresponding to the image labels. Simply use the path to the downloaded training data

`-iVal` Path to the val data directory (optional). This is used for stopping the model training at the lowest validation loss. If not provided, a portion (10%) of training data will be used for validation.

`-n` Number of sample to use during training (default=1000).


example:
```
python ./train.py -iTrn "./RawData/OCT2017/train" -iVal "./RawData/OCT2017/val"
```


This will automatically save the contents of the model to the `modelOutput` directory.

### Inference

To predict the class of a new OCT image, run `predict.py` with the following arguments

`-i` Path to directory of new oct images. Images must be in the root folder.


`-o` Path to directory where output will be generated.


`-m` Path to trained model directory.

example:
```
python ./predict.py -i "./RawData/testDir" -o "./modelOutput/metaClf_2019-06-22" -m "./modelOutput/metaClf_2019-06-22"
```



## Authors

Andrei Mouraviev

Eric Chou


## Acknowledgements

Data: https://data.mendeley.com/datasets/rscbjbr9sj/2

Citation: http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

**Inspiration**

Automated methods to detect and classify human diseases from medical images.
