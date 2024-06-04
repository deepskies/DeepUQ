---
title: 'DeepUQ: A systematic comparison of uncertainty prediction techniques in deep learning'
tags:
  - Python
  - astronomy
  - physics
  - uncertainty quantification
  - deep learning
  - probability
authors:
  - name: Becky Nevin
    orcid: 0000-0003-1056-8401
    equal-contrib: false
    affiliation: "1"
  - name: Brian Nord
    orcid: 0000-0001-6706-8972
    equal-contrib: true
    affiliation: "1, 2, 3" # (Multiple affiliations must be quoted)
  - name: Aleksandra Ciprijanovic
    orcid: 
    equal-contrib: true
    affiliation: "1, 2"
affiliations:
 - name:  Fermi National Accelerator Laboratory, P.O. Box 500, Batavia, IL 60510
   index: 1
 - name: Department of Astronomy and Astrophysics, University of Chicago, 5801 S Ellis Ave, Chicago, IL 60637
   index: 2
 - name: Kavli Institute for Cosmological Physics, University of Chicago, 5801 S Ellis Ave, Chicago, IL 60637
   index: 3
date: 4 June 2024
bibliography: paper.bib

---


# Summary
We introduce **DeepUQ**, a Python library that assess measurements of aleatoric and epistemic uncertainty from deep learning methods. It generates simple datasets with user-adjustable uncertainty prescriptions for use in testing the predicted uncertainty from the deep learning methods. It then runs the deep learning models (Deep Ensembles and Deep Evidential Regression) with adjustable hyperparameters. Finally, it compares the predicted uncertainties to the known uncertainties.

This software includes methods to curate and store these datasets and model parameters in order to maximize reproducibility.


# Statement of Need
Physically and statistically interpretable uncertainties are critical for measurements, predictions, and decision-making. There is a large and recently populated literature regarding formulations and experiments for uncertainty quantification (UQ) in deep neural networks. However, the development of a coherent and consistent theoretical framework for estimating and calibrating uncertainties remains a significant challenge. The primary issues include matters of statistical interpretability, consistency of models for each approach, and consistency across approaches. 


* A prinicipled method for comparing UQ predictions from different techniques
* A method for comparing true to predicted uncertainties and a framework for our expectations for uncertainty behavior
* A method that is flexible to testing with different types and prescriptions of uncertainty and to adding on new deep learning UQ methods as they emerge

In the nascent field of uncertainty quantification, a principled method for comparing the uncertainty predictions from different deep learning models is lacking. This is important when new uncertainty quantification methods are constantly emerging, necessitating an independent method for direct comparison and documentation. Said method should also be flexible to adding on newly emerging techniques.

Furthermore, a software that is publicly available and that provides an end-to-end framework for comparing methods including data generation, training, and assessment is lacking.


## Related Work




We briefly review the landscape of existing UQ approaches in the literature; they present different methods to obtain aleatoric and epistemic uncertainties.
Deep Evidential Regression (`[@Amini2019DER]`) offers a novel approach to train a neural network to learn a continuous prediction as well as the associated evidence, enabling the practitioner to retrieve both aleatoric and epistemic uncertainties from one model.
Deep Ensembles (`[@Lakshminarayanan2016arXiv161201474L]`) are an ensembling UQ method that predicts aleatoric and epistemic uncertainty estimates by combining multiple mean-variance estimation networks (`[@Nix374138]`). 
Monte Carlo Dropout (MC Dropout; `[@GalMCDropout2015arXiv150602142G]`) utilizes dropout at test time to produce predictive uncertainties.
Bayesian Neural Networks (`[@LV2000,@T2004,@PS2017]`) replace the deterministic network weights with probability distributions and also output a mean and a variance for each of the network outputs. 
This allows BNNs to capture both epistemic and aleatoric uncertainty. At inference time, multiple sets of weights are sampled from the learned weight distributions, allowing the computation of a series of possible network outputs. Because of this, BNNs can be considered a special case of ensemble learning (`[@ZZ2012]`).


Within the landscape of deep learning UQ approaches, there exist taxonomical disagreements or inconsistencies regarding the definitions of aleatoric and epistemic uncertainties in the UQ literature.
For example, `[@Gal2022NatRP...4..573G]` presents a discussion between UQ experts on how different types of uncertainties should be estimated.
`[@brando2022thesis]` also presents a thorough review of the tension between some of these definitions.
Often, aleatoric uncertainty is presented by the deep learning literature as being {\it irreducible} and related only to the data, while epistemic is presented as {\it reducible} and related only to the model.
Unfortunately, these definitions don't account for the gamut of interpretations in the deep learning community or across the domains in which the techniques are applied.


Previous efforts have attempted to benchmark models or standardize the definitions of types of uncertainty `[@Caldeira2020arXiv200410710C, @brando2023standardizing]`.
Other work has focused on developing metrics to assess the quality of the uncertainty estimates `[@Tran2019arXiv191210066T]`.

This work uses the probabilistic conceptualization of deep learning-based UQ from `[@brando2023standardizing]` and examines assumptions in the context of two existing UQ methods---Deep 



# DeepUQ Software 

The **DeepBench** software simulates data for analysis tasks that require precise numerical calculations. First, the simulation models are fundamentally mechanistic -- based on relatively simple analytic mathematical expressions, which are physically meaningful. This means that for each model, the number of input parameters that determine a simulation output is small (<$10$ for most models). These elements make the software fast and the outputs interpretable -- conceptually and mathematically relatable to the inputs. Second, **DeepBench** also includes methods to precisely prescribe noise for inputs, which are propagated to outputs. This permits studies and the development of statistical inference models that require uncertainty quantification, which is a significant challenge in modern machine learning research. Third, the software framework includes features that permit a high degree of reproducibility: e.g., random seeds at every key stage of input, a unique identification tag for each simulation run, tracking and storage of metadata (including input parameters) and the related outputs. Fourth, the primary user interface is a YAML configuration file, which allows the user to specify every aspect of the simulation -- e.g., types of objects, numbers of objects, noise type, and number of classes. This feature -- which is especially useful when building and studying complex models like deep learning neural networks -- permits the user to incrementally decrease or increase the complexity of the simulation with a high level of granularity.


**DeepUQ** has the following features:

* Exact reproducibility
* Control over uncertainty properties
* Control over deep learning technique hyperparameters
* Option to save model diagnostics with epoch
* Analysis module for summarizing predicted uncertainties
* Exensible to new UQ methods


# Primary Modules 

* data: Utilities for saving and loading a dataframe for training a linear regression with user-controlled noise levels. Uses `numpy` [@numpy], `scikit-learn` [@scikitlearn], `pickle`, `PyTorch` [@pytorch], and `h5py` [@h5py].
* models: Architecture for the Deep Ensemble and Deep Evidential Regression models including modified loss functions and the internal calculations for epistemic and aleatoric uncertainties. Uses PyTorch .
* train: Trains and saves the aforementioned models.
* analysis: Utilities for assessing the performance of the models given our uncertainty desiderata. Uses `matplotlib` `[@hunterMatplotlib2DGraphics2007b]`
* utils:

# Example Outputs 

![Example output of the training script **DeepUQ**. The training scripts have an option to save model parameters such as different types of model loss by epoch. Training (dashed) and validation (solid) loss as a function of epoch for DER (left) and DE (right). The MSE loss (top) and NIG and $\beta$-NLL loss (bottom).](figures/all_loss.png)

![Example output of the analysis script from **DeepUQ**. Aleatoric uncertainty as a function of epoch for both the DER (left) and the DE (right) models. The thick lines are the model we use for the DER (left) and the mean DE model (right). The thin lines demonstrate the jitter from five individual runs of each method with a new random seed for the initialization of the weight parameters.](figures/aleatoric_and_jitter.png)

# Acknowledgements

*Becky Nevin*: conceptualization, methodology, software, project administration, writing. *Brian Nord*: conceptualization, methodology, project administration, funding acquisition, supervision, writing. *Alex \'Ciprijanovi\'c* 

We acknowledge contributions from Maggie Voetberg.

Work supported by the Fermi National Accelerator Laboratory, managed and operated by Fermi Research Alliance, LLC under Contract No. DE-AC02-07CH11359 with the U.S. Department of Energy. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a non-exclusive, paid-up, irrevocable, world-wide license to publish or reproduce the published form of this manuscript, or allow others to do so, for U.S. Government purposes.

We acknowledge the Deep Skies Lab as a community of multi-domain experts and collaborators who have facilitated an environment of open discussion, idea-generation, and collaboration. This community was important for the development of this project.


# References