# Wealth & Well-Being: Global Life Satisfaction Analysis
## Overview
This project explores the relationship between various socioeconomic indicators and global well-being. Using machine learning (Linear Regression), we predict **Life Satisfaction** and **Life Expectancy** for different countries based on their GDP Per Capita, Social Support, Freedom to Make Choices, and Corruption Perception. 
## Technologies Used
* **Python** 
* **Pandas** (Data manipulation & cleaning)
* **Matplotlib** (Data visualization, scatter plots, & heatmaps)
* **Scikit-Learn** (Linear Regression modeling)
## The Math Behind the Model
### 1. Why we use Logarithmic GDP
When modeling the relationship between GDP and Life Satisfaction, we don't use raw GDP data. Instead of a standard straight-line equation (`Satisfaction = θ₀ + θ₁ * GDP`), we use a **log-linear model**:
`Life Satisfaction = θ₀ + θ₁ * log(GDP_Per_Capita)`
**Why?** Because wealth suffers from **diminishing marginal returns**. 
An increase in GDP from \$1,000 to \$10,000 has a massive, transformative impact on a country's quality of life (e.g., access to clean water, basic healthcare). However, an identical \$9,000 increase from \$80,000 to \$89,000 barely moves the needle on overall national life satisfaction. The natural logarithm (`log`) mathematically captures this curve, rising steeply for lower-income brackets and flattening out for higher-income brackets. This transformation yields a much higher R² accuracy score.
### 2. Upgrading to Multiple Linear Regression
To make our model more robust, we expanded from predicting based solely on GDP to a multiple linear regression model utilizing several socioeconomic factors:
`Life Satisfaction = θ₀ + (θ₁ * log(GDP)) + (θ₂ * Social_Support) + (θ₃ * Freedom) + (θ₄ * Corruption)`
This allows us to analyze the "weights" (coefficients) assigned to each individual factor, revealing a fascinating story about what truly drives global happiness beyond just financial wealth.
### 3. How the Model Learns: The Normal Equation
Under the hood, this project utilizes `sklearn.linear_model.LinearRegression()`. 
It is important to note that scikit-learn **does not use Gradient Descent** (neither Batch nor Stochastic) for this specific class. 
Instead, it calculates the exact mathematical optimal values for our parameters (θ) using a method based on the **Normal Equation** (specifically, it solves the Ordinary Least Squares problem using Singular Value Decomposition, or SVD). Because our dataset is relatively small (less than a few million rows), this advanced linear algebra approach is able to calculate the perfectly accurate answer almost instantly, completely eliminating the need to guess-and-check learning rates (alpha) or run through multiple training epochs.
## Features in the Notebook
* Statistical data summaries using Pandas.
* Continent-based color-coded scatter plots.
* Visualized regression trend lines with R² scores.
* Custom country annotations to highlight outliers (e.g., countries that are exceptionally happy despite low GDPs).

## Data Sources
The dataset used in this project is compiled from the following official sources:
1. **Wealth Data (X):** International Monetary Fund (IMF), *World Economic Outlook Database, April 2026 Edition*. (Used for nominal GDP per capita).
2. **Happiness Data (Y):** Oxford Wellbeing Research Centre & Gallup, *World Happiness Report 2026* (Published March 2026).
3. **Longevity Data (Y):** United Nations, *World Population Prospects* (Recent Demographic Estimates).
