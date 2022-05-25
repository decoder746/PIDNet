# PIDNet: A Real-time Semantic Segmentation Network Inspired from PID Controller
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This is the official repository for our recent work: PIDNet

## Highlights
<p align="center">
  <img src="figs/cityscapes_score.jpg" alt="overview-of-our-method" width="600"/></br>
  <span align="center">Comparison of inference speed and accuracy for real-time models on test set of Cityscapes</span> 
</p>

* **Towards Real-time Apps**: PIDNet could be directly used for the real-time applications, such as autonomous vehicle and medical imaging.
* **Novel Three-branch Network**: Addtional boundary branch is introduced to two-branch network to mimic the PID controller architecture.
* **More Accurate and Faster**: PIDNet-S presents 78.6% mIOU with speed of 93.2 FPS on Cityscapes test set and 81.6% mIOU with speed of 150.6 FPS on CamVid test set. Also, PIDNet-L acheives the highest accuracy (80.6% mIOU) in real-time domain (>30 FPS) for Cityscapes.

## Updates
   - Our paper was submitted to arXiv and paperwithcode for public access. (May/30/2022)
   - The training and testing codes for PIDNet is available here. (May/25/2022)

## Prerequisites
- Pytorch 1.1

