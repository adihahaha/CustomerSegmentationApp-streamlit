# CustomerSegmentationApp-streamlit

## This app allows users to  group customers from the given dataset into meaningful segments.
## Access the app at https://customersegmentation-app-task.streamlit.app/

## Purpose of the app
The app demonstrates the employment of an unsupervised machine learning model in segmenting customers based on common characteristics or buisness target. K-Means clustering algorithms (Mini-batch and Elkan) have been used to segment the dataset. Visualizations have been provided to illustrate the distribution and traits of each segment.

## Features of the app
- Choose your desired K-Means variation - either Mini-Batch or Elkan
- Segment the model according to specific goals, like Channel Engagement, Discount Sensitivity, Recent Activity, Purchase Frequency, and many more
- Or even better, customize the segmentation by choosing your own features....whatever is relevant to your business goal
- A **2D plot** visualizing the Principle Component Decompostion of the features
- A **cluster summary table** showing the important characteristics of each segment (or cluster)
- **Bar and box graphs** to show how features differ among clusters
- **Dominant product** for each of the clusters
