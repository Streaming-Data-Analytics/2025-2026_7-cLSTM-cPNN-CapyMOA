# 2025/2026: (7) Implementing Continuous LSTM and Continuous PNN in CapyMOA

Optional project of the [Streaming Data Analytics](https://emanueledellavalle.org/teaching/streaming-data-analytics-2025-26/) course provided by [Politecnico di Milano](https://www11.ceda.polimi.it/schedaincarico/schedaincarico/controller/scheda_pubblica/SchedaPublic.do?&evn_default=evento&c_classe=837284&__pj0=0&__pj1=36cd41e96fcd065c47b49d18e46e3110).

Student: **[To be assigned]**

This project focuses on interfacing **Streaming Continual Learning (SCL)** architectures (specifically Continuous LSTM (cLSTM) and Continuous Progressive Neural Networks (cPNN)) as native, single-instance classifiers within the [open-source CapyMOA framework](https://capymoa.org/index.html). 
______

# Background: The Rise of Streaming Continual Learning (SCL)
When applying Machine Learning to evolving data streams, traditional paradigms face severe obstacles. Currently, two main research areas tackle these challenges, but with complementary and sometimes opposing objectives:

1. **Streaming Machine Learning (SML):** SML focuses on real-time analytics, monitoring changes, and reacting to *concept drifts* (both virtual and real). SML prioritizes **rapid adaptation** over retaining previous knowledge. Its primary objective is to perform well on the current concept, even at the cost of catastrophic forgetting. 
2. **Continual Learning (CL):** CL, on the other hand, specifically prioritizes the preservation of previously acquired knowledge (stability). It aims to mitigate catastrophic forgetting when learning new tasks. However, CL is traditionally designed for batch or offline scenarios where concepts do not contradict each other (*virtual drifts*), and it often struggles with the strict constraints and real drifts of a true data stream.

To jointly solve these problems, **Streaming Continual Learning (SCL)** has emerged as a unifying paradigm. SCL models must possess the rapid adaptation capabilities of SML, the knowledge consolidation of CL, and the ability to learn continuously under strict computational constraints.

Operating in an SCL environment also often requires overcoming a third major challenge: **Temporal Dependence**. In many real-world scenarios, data points are not independent. Current outcomes are often correlated with past data. Formally, temporal dependence exists if $\exists\tau \; P(a_t|b_{t-\tau})\neq P(a_t)$, meaning that a feature or target at time $t$ is statistically dependent on information from previous time steps. Despite its relevance, temporal dependence is frequently ignored by both SML and CL communities.

The **cPNN** architecture represents a pioneering SCL solution that bridges all three gaps. It uses **cLSTM** to model temporal dependencies via sliding windows. To handle concept drifts and forgetting, cPNN dynamically expands its architecture. For each new concept, a new cLSTM ("column") is added. This new column receives lateral connections from the frozen hidden layers of previous columns, enabling transfer learning to speed up adaptation while completely preserving past knowledge.

cPNN and cLSTM were originally introduced in [**this paper**](https://arxiv.org/pdf/2603.03040). However, this version applies both periodic training and inference. The loss function considers all the sequences in which a data point appears.

A [**second paper**](https://arxiv.org/pdf/2603.08972) proposes the optimized version for anytime inference, which uses a loss function considering only one sequence for each data point (containing from  $X_{t-W+1}, \cdots, X_t)$. It then applies this new version to a network of edge devices. For the purpose of this project, you can ignore this specific application.

The cLSTM and cPNN code is available [here](https://github.com/Sandrodand/MagicNet/tree/dev)
You can focus on the folders `crnn` (for this project, we can focus just on cLSTM) and `cpnn`.

#### **1. Base Learner: Anytime cLSTM**
You will adapt the baseline cLSTM model to fit CapyMOA's standard SML paradigm.
* **Anytime inference**: Inference is produced each time a new feature vector $X_t$ is available, considering the window $X_{t-W+1}, \cdots, X_t$. This is implemented by the method `predict` (it should manage a continuous sliding window with the previous $W-1$ data points).
* **Periodic training:** Data points are accumulated in mini-batches of size $B$. Whenever a mini-batch is full, a sliding window produces sequences on the mini-batch, and the model is trained. The idea is that each time a couple $(X_t, y_t)$ is available, the method `train` shall be called. This method must add the couple to the buffer, when it contains $B$ couples, it must update the model's weights (given a specified number of epochs), and clear the buffer.

## **2. Streaming Continual Learner: cPNN**
You will adapt the baseline cLSTM model to fit CapyMOA's standard SML paradigm. You will extend the `cLSTM` into the full `cPNN` ensemble.
The model initializes with a single cLSTM column.
* **Lateral Connections:** Ensure the feature vector of a given sequence item in column $k$ properly concatenates the original features with the output of the hidden layer from column $k-1$.


## **2. Evalution 
### Datasets
To evaluate the integration, the project will utilize [this data stream](https://polimi365-my.sharepoint.com/:x:/g/personal/10780444_polimi_it/IQDCgaKx14mVR7zozlZM0yStAaR0ONNRc-_GHGV1z75_Jyg?e=mXeuBp). 

It is associated with the weather domain and contains the following features: RH (humidity), T_d (dew point temperature), w_s/w_d (wind speed/direction). Features are already standardized.

8 different binary classification labels are engineered by comparing current values (of the air temperature, which is not given) to previous median or minimum values within a specific temporal window. Each classification function corresponds to a specific concept. For instance, one concept may assign 1 if the current air temperature has increased since the previous timestamp, and 0 otherwise. Another task may assign 1 if the current temperature is greater than the mean of the previous 10, … and so on. 

Concept drifts in this case are **real** and **abrupt**.

The column **task** represents the index of the concept. You can use it to trigger a cPNN expansion (when it changes, it indicates a drift and you must call the method `add_new_column`).
The **Target** containes represents the label associated with each data point. 

### **Prequential Evaluation (SML Focus)**
For the purpose of this project you can focus on the **Test-then-Train** (Prequential) approach (ignoring the forgetting measures):
* The model receives $X_t$ and outputs prediction $\hat{y}_t$.
* The prediction is scored against the true label $y_t$ (method `test`) using a rolling Cohen's Kappa.
* The method `train` is called with $(X_t, y_t)$.

# Deliverable
For this project, you are required to refactor the provided research code and package it as standard CapyMOA classifiers.

1. **Modify and Integrate the Code**
   - Refactor the PyTorch cLSTM implementation to act as a CapyMOA classifier
   - Implement the cPNN wrapper that dynamically handles column addition, weight freezing, and lateral connections.
   
2. **Prepare a Presentation (Notebook)**
   - Demonstrate the correct functioning of your CapyMOA-integrated cPNN on the provided datasets.
   - Plot the prequential evaluation results (e.g., Cohen's Kappa over time) showing the moments of concept drift and the model's rapid adaptation.
   - Provide a comparison highlighting how the cPNN avoids catastrophic forgetting (via CL metrics like BWT) compared to a baseline cLSTM or standard SML model.

## Note for Students

* Clone the created repository offline;
* Add your name and surname into the Readme file;
* Make any changes to your repository, according to the specific assignment;
* Add a `requirements.txt` file for code reproducibility and instructions on how to replicate the results;
* Commit your changes to your local repository;
* Push your changes to your online repository.
