# multimodal-behaviour-cloning-vae
\documentclass{article}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{margin=1in}

\title{Behaviour Cloning for Robotic Manipulation \\ 
\large Multimodal Learning with Self-Supervised Variational Autoencoder}
\author{Your Name}
\date{2024}

\begin{document}

\maketitle

\section{Project Overview}

This project investigates behaviour cloning for robotic manipulation using multimodal observations. The objective is to predict robotic arm actions from RGB visual inputs combined with proprioceptive data (joint states, positional and velocity vectors).

The system integrates self-supervised representation learning through a Variational Autoencoder (VAE) to learn compact latent embeddings from visual observations before downstream action prediction.

\section{Problem Setting}

Given multimodal inputs:
\begin{itemize}
    \item RGB images
    \item Joint positions and velocities
    \item End-effector states
\end{itemize}

The goal is to learn a policy:
\[
\pi(a_t | o_t)
\]
where $o_t$ represents multimodal observations and $a_t$ represents continuous control actions.

\section{Methodology}

\subsection{Self-Supervised Visual Representation Learning}

A Variational Autoencoder (VAE) was trained on RGB images to learn compact latent representations. The objective function is:

\[
\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \beta D_{KL}(q(z|x) \| p(z))
\]

The learned latent embedding was used as input to the downstream policy network.

\subsection{Behaviour Cloning Model}

The behaviour cloning model integrates:
\begin{itemize}
    \item VAE latent embeddings
    \item Robot state vectors
\end{itemize}

The model predicts continuous control actions using supervised regression, optimised via Mean Squared Error (MSE) loss.

\section{Baseline Comparison}

Performance was compared against:
\begin{itemize}
    \item A fully supervised model trained directly on raw image features
    \item A state-only baseline
\end{itemize}

Evaluation metrics included:
\begin{itemize}
    \item Mean Squared Error (MSE)
    \item Action prediction error distribution
\end{itemize}

\section{Training Pipeline}

\begin{itemize}
    \item Implemented in PyTorch
    \item Mixed-precision training (FP16) for efficiency
    \item Experiment tracking via Weights \& Biases
    \item Modular dataset and training loop structure
\end{itemize}

\section{Results and Observations}

\begin{itemize}
    \item Latent dimensionality significantly influenced downstream performance.
    \item Self-supervised pretraining improved generalisation compared to raw image baselines.
    \item Overfitting was mitigated via regularisation and KL balancing.
\end{itemize}

\section{Future Improvements}

\begin{itemize}
    \item Contrastive self-supervised methods (e.g., SimCLR)
    \item Temporal modelling (LSTMs / Transformers)
    \item Domain randomisation for robustness
\end{itemize}

\section{Dataset}

Due to dataset size constraints (approximately 15GB), raw data is not included in this repository. 
Users may adapt the dataset loading module to their own robotic datasets.

\section{Technologies Used}

\begin{itemize}
    \item Python
    \item PyTorch
    \item OpenCV
    \item Weights \& Biases
    \item Docker (for containerisation)
\end{itemize}

\end{document}
