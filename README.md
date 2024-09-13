# Water Quality Anomaly Detection Dashboard

This project focuses on building a real-time water quality anomaly detection system using machine learning algorithms. The system monitors water parameters such as **turbidity** and **pH** to detect anomalies that may indicate contamination or equipment malfunctions. The project uses **Local Outlier Factor (LOF)**, **Isolation Forest (IF)**,**Robust Random Cut Forest (RRCF)** and **OneClassSVM** for anomaly detection.

## Project Structure

- **app.py**: Main script to run the Dash appfor  interactive  and realtime  tracking.
- **.ipynb**: jupyter notebook files  used  to train evaluate the  models.
- **.tex**: Latex files documents  can be  compiled with texstudio or  [Overleaf](https://www.overleaf.com/)
-  **`raspberrypico_streamfiles/`**: files  for  the  pico pi to stream data  to the  cloud for  analysis. 
## Features

- **Real-time Monitoring**: Displays real-time data streams for turbidity and pH with dynamic anomaly detection.
- **Anomaly Detection Algorithms**: Comparison of three machine learning algorithms: LOF, IF, and RRCF for identifying anomalies in water quality data.
- **Interactive Dashboard**: Built with **Dash** for real-time visualization of water quality parameters and detected anomalies. Users can switch between 'Home' and 'Model' pages, and interact with a sidebar to control parameters.
  
## Data

The dataset used for this project is a time series data collected from sensors deployed in a water treatment plant, measuring **turbidity** (in NTU) and **pH** values. The dataset is preprocessed to remove null values and to handle anomalies caused by sensor malfunctions or environmental changes.

## Machine Learning Models

- **Local Outlier Factor (LOF)**: Detects local density-based anomalies by comparing the local density of a point to its neighbors.
- **Isolation Forest (IF)**: A tree-based model that isolates anomalies by recursively partitioning the data.
- **Robust Random Cut Forest (RRCF)**: Designed for detecting anomalies in streaming data using recursive binary partitioning.
- **One-Class SVM**: A support vector machine-based algorithm designed for anomaly detection in high-dimensional data. It tries to separate the normal data from outliers by learning a decision boundary around the normal data points.

## Results

- The **Isolation Forest** algorithm outperformed the other methods in detecting significant spikes and anomalies in turbidity and pH data.
- **RRCF** was more efficient in real-time anomaly detection but required parameter tuning to adapt to smaller anomalies.
- **LOF** was sensitive to local variations but struggled with larger spikes in the data.

## Future Work

- Extend the system to monitor additional water quality parameters (e.g., **Total Dissolved Solids, Oxygen Reduction Potential**).
- Implement additional machine learning algorithms for anomaly detection, such as **Deep Learning** methods.
- Explore hybrid models for better anomaly detection performance.

## Acknowledgments

This project was conducted with support from **Nyeri Water and Sanitation Company (NYEWASCO)** and **Dedan Kimathi University of Technology (DeKUT)**.

## License

This project is licensed under the  CC0-1.0 license
