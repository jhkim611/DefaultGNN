# DefaultGNN: A Dual-Perspective GNN Framework for Predicting Corporate Default from Buyer-Seller Transaction Networks

<p align="center">   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>
    <a href="https://uobevents.eventsair.com/cikm2023//" alt="Conference">
        <img src="https://img.shields.io/badge/CIKM'26-green" /></a>
</p>

The official source code for "[DefaultGNN: A Dual-Perspective GNN Framework for Predicting Corporate Default from Buyer-Seller Transaction Networks](https://dsail.kaist.ac.kr/publications_conference/)", accepted at CIKM 2026.

## Overview

Corporate default prediction is a core problem in financial risk management, yet traditional credit models rely heavily on financial statements that are often sparse or unavailable for many firms. Corporate transaction networks offer a complementary view of real economic activity, but how risk propagates through buyer–seller relationships remains underexplored. We conduct a large-scale empirical study using real-world electronic tax-invoice data spanning six years that links transaction histories with default events, revealing that transaction-driven risk is both role-dependent (buyer or seller) and scale-dependent. Based on these findings, we construct multiplex buyer-view and seller-view transaction networks and propose **DefaultGNN, a dual-perspective graph neural network-based framework for corporate default prediction**. DefaultGNN integrates both views to model how risk flows through transactional relationships, achieving strong improvements over both attribute-based and graph-based baselines, especially for firms with limited intrinsic risk signals. We further provide interpretable network-based explanations by visualizing how distressed trading partners contribute to default risk. In collaboration with a licensed credit rating agency, we validate that DefaultGNN's predictions complement existing credit scoring models, improving approval rates by 7-11\%p without increasing default risk among approved firms.

<img width="750" alt="architecture" src="https://github.com/user-attachments/assets/fc4c6ea2-790e-4662-a36e-eefa696a1e97" />

