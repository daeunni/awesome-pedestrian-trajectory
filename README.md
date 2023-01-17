# Awesome-pedestrian-trajectory-prediction ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)
> Awesome repository for pedestrian trajectory prediction 

### ✅ Note that this repo contains mainly for pedestrian trajectory prediction‼️

## Contents of subjects 
[1. Graph based](#graph-based)    
[2. Using map information](#using-map-information)     
[3. Generative model based](#generative-model-based)     
[4. Transformer based](#transformer-based)   
[5. Other perspectives](#other-perspectives)  
 


## Graph based

- [Learning Pedestrian Group Representations for Multi-modal Trajectory Prediction](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820263.pdf) (ECCV, 22) [[code](https://github.com/InhwanBae/GPGraph)]
- [GroupNet: Multiscale Hypergraph Neural Networks for Trajectory Prediction with Relational Reasoning](https://arxiv.org/pdf/2204.08770.pdf) (CVPR, 22) `eth/ucy` `SDD` `NBA`
- [Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction](https://arxiv.org/pdf/2002.11927.pdf) (CVPR, 20) [[code](https://github.com/abduallahmohamed/Social-STGCNN/)] `eth/ucy`
- [Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data](https://arxiv.org/pdf/2001.03093.pdf) (ECCV, 20) [[code](https://github.com/StanfordASL/Trajectron-plus-plus)] `eth/ucy` `nuScenes`

## Using map information

- [PreTraM: Self-Supervised Pre-training via Connecting Trajectory and Map](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136990034.pdf) (ECCV, 22) [[code](https://github.com/chenfengxu714/PreTraM)]
- [Graph-based Spatial Transformer with Memory Replay for Multi-future Pedestrian Trajectory Prediction](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Graph-Based_Spatial_Transformer_With_Memory_Replay_for_Multi-Future_Pedestrian_Trajectory_CVPR_2022_paper.pdf)  (CVPR, 22) [[code](https://github.com/Jacobieee/ST-MR)] `VIRAT/ActEV`
- [ScePT: Scene-consistent, Policy-based Trajectory Predictions for Planning](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_ScePT_Scene-Consistent_Policy-Based_Trajectory_Predictions_for_Planning_CVPR_2022_paper.pdf) (CVPR, 22)[[code](https://github.com/nvr-avg/ScePT)] `eth/ucy` `nuScenes`
- [End-to-End Trajectory Distribution Prediction Based on Occupancy Grid Maps](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_End-to-End_Trajectory_Distribution_Prediction_Based_on_Occupancy_Grid_Maps_CVPR_2022_paper.pdf) (CVPR, 22) [[code](https://github.com/Kguo-cs/TDOR)] `SDD` `inD`
- [PECNet: It Is Not the Journey but the Destination: Endpoint Conditioned Trajectory Prediction (ECCV, 20 **Oral**)](https://arxiv.org/abs/2004.02025) [[code](https://github.com/HarshayuGirase/Human-Path-Prediction)] `eth/ucy` `SDD` `inD`
- [Y-Net: From goals, waypoints & paths to long term human trajectory forecasting](https://openaccess.thecvf.com/content/ICCV2021/html/Mangalam_From_Goals_Waypoints__Paths_to_Long_Term_Human_Trajectory_ICCV_2021_paper.html) (ICCV, 21) [[code](https://github.com/HarshayuGirase/Human-Path-Prediction/tree/master/ynet)]  `eth/ucy` `SDD` `inD`
- [Goal-GAN: Multimodal Trajectory Prediction Based on Goal Position Estimation](https://arxiv.org/abs/2010.01114) (ACCV, 20 Oral)[[code](https://github.com/dendorferpatrick/GoalGAN)] `eth/ucy` `SDD`
- [Three Steps to Multimodal Trajectory Prediction: Modality Clustering, Classification and Synthesis](https://arxiv.org/pdf/2103.07854.pdf) (ICCV, 21) [[code](https://github.com/ApeironY/PCCSNet)] `eth/ucy` `SDD` 

## Generative model based

- [MUSE-VAE: Multi-Scale VAE for Environment-Aware Long Term Trajectory Prediction](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_MUSE-VAE_Multi-Scale_VAE_for_Environment-Aware_Long_Term_Trajectory_Prediction_CVPR_2022_paper.pdf) (CVPR, 22) `SDD` `nuScenes`
- [MG-GAN: A Multi-Generator Model Preventing Out-of-Distribution Samples in Pedestrian Trajectory Prediction](https://arxiv.org/pdf/2108.09274.pdf) (ICCV, 21) [[code](https://github.com/selflein/MG-GAN)] `eth/ucy` `SDD`

## Transformer based

- [AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting (ICCV, 21)](https://arxiv.org/abs/2103.14023) [[code](https://github.com/Khrylx/AgentFormer)] `eth/ucy` `nuScenes`
- [Spatio-Temporal Graph Transformer Networks for Pedestrian Trajectory Prediction](https://arxiv.org/abs/2005.08514) (ECCV, 20) [[code](https://github.com/Majiker/STAR)] `eth/ucy`

## Other perspectives 

- [Action-based Contrastive Learning for Trajectory Prediction](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136990140.pdf) (ECCV, 22) `PIE` `JAAD`
- [Adaptive Trajectory Prediction via Transferable GNN](https://arxiv.org/pdf/2203.05046.pdf) (CVPR, 22) `eth/ucy` - Domain Adaptation
- [Human Trajectory Prediction with Momentary Observation](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_Human_Trajectory_Prediction_With_Momentary_Observation_CVPR_2022_paper.pdf) (CVPR, 22) `eth/ucy` `SDD`  - Sudden events for pedestrians (short trajectory)
- [Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion](https://openaccess.thecvf.com/content/CVPR2022/papers/Gu_Stochastic_Trajectory_Prediction_via_Motion_Indeterminacy_Diffusion_CVPR_2022_paper.pdf) (CVPR, 22) [[code](https://github.com/gutianpei/MID)] `eth/ucy` `SDD`
- [How many Observations are Enough? Knowledge Distillation for Trajectory Forecasting](https://openaccess.thecvf.com/content/CVPR2022/papers/Monti_How_Many_Observations_Are_Enough_Knowledge_Distillation_for_Trajectory_Forecasting_CVPR_2022_paper.pdf) (CVPR, 22) `eth/ucy` `SDD` `Lyft`
- [Remember Intentions: Retrospective-Memory-based Trajectory Prediction](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Remember_Intentions_Retrospective-Memory-Based_Trajectory_Prediction_CVPR_2022_paper.pdf) (CVPR, 22) `eth/ucy` `SDD` `NBA`
- [ATPFL: Automatic Trajectory Prediction Model Design under Federated Learning Framework](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_ATPFL_Automatic_Trajectory_Prediction_Model_Design_Under_Federated_Learning_Framework_CVPR_2022_paper.pdf) (CVPR, 22) `eth/ucy`
- [Non-Probability Sampling Network for Stochastic Human Trajectory Prediction](https://openaccess.thecvf.com/content/CVPR2022/papers/Bae_Non-Probability_Sampling_Network_for_Stochastic_Human_Trajectory_Prediction_CVPR_2022_paper.pdf) (CVPR, 22) [[code](https://github.com/InhwanBae/NPSN)] `eth/ucy` `SDD`
- [Social-SSL: Self-Supervised Cross-Sequence Representation Learning Based on Transformers for Multi-Agent Trajectory Prediction](https://basiclab.lab.nycu.edu.tw/assets/Social-SSL.pdf) (ECCV, 22, oral)


