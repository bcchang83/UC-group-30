# Weather-Enhanced Multi-Input LSTM for Trajectory Prediction

## Abstract  

This study builds upon the work of Deo, Nachiket, and Mohan M.Trivedi [1], which introduced convolutional social pooling (CSP) to learn interdependencies between neighboring vehicles for trajectory prediction. CSP is a novel approach that captures spatial
relationships by pooling LSTM states of surrounding vehicles intoa grid-like social tensor, enabling convolutional layers to efficiently
model interactions between vehicles, similar to how convolutional neural networks extract features from images. Accurate vehicle trajectory prediction is critical for improving road safety and advancing autonomous driving systems. While their approach performed
well, it was limited to vehicle-related data. This study enhances the
accuracy of trajectory prediction by incorporating weather data.
Features like visibility, precipitation, wind speed, temperature, and
humidity are included alongside vehicle-related data. Including
weather features strengthens the model, resulting in lower RMSE
and NLL values compared to the original study. These findings underscore the importance of integrating environmental factors into
vehicle trajectory prediction models for improved performance.
 
![model](https://github.com/user-attachments/assets/bee3769f-cf2c-4cac-b80c-f0fffaadb278)

## Data  

This project utilizes two primary datasets: **trajectory data** from the **NGSIM (Next Generation SIMulation) program** and **weather data** from **Visual Crossing** [4].  

### Trajectory Data  
The trajectory data consists of vehicle movement records from freeway segments on **US-101 (Los Angeles, CA)** [2] and **I-80 (San Francisco, CA)** [3]. These datasets were collected by the **Federal Highway Administration (FHWA)** using video cameras and processed into trajectory data at **10 Hz (10 frames per second)**. Key features include **X/Y coordinates, time frames, and lane IDs**.  

- Data is segmented into **mild, moderate, and congested traffic conditions**.  
- Training and testing sets follow prior research setups, using **3 seconds of track history** and a **5-second prediction horizon** (downsampled to 25 frames for efficiency).  

### Weather Data  
The weather data comes from **Visual Crossing**, providing **hourly weather records** for Los Angeles, CA. Key weather features include:  
- **Default set:** Precipitation, wind speed, and visibility.  
- **Extended set:** Adds temperature and humidity to assess further impacts on driving behavior.  

By integrating both **trajectory and weather data**, the study examines how environmental conditions influence vehicle trajectory predictions.  

## Acknowledgment
We would like to express our gratitude to **Professor Mitra Baratchi** for their guidance and support throughout this project. This work was completed as part of the **Urban Computing** course at **Leiden University** in 2024.  

Throughout the course, we explored various data types, including temporal and spatial data, and learned how to visualize and analyze them effectively. The combination of real-world applications and practical tools significantly enhanced our learning experience and motivation. 

## Results
Our experiments compare the baseline CS-LSTM(M) model with two enhanced versions incorporating weather features.  

- **CS-LSTM-Weather(M)(3):** Adds visibility, wind speed, and precipitation, improving RMSE performance.  
- **CS-LSTM-Weather(M)(5):** Further includes temperature and humidity, achieving the best results across a 5-second prediction horizon.  

While the model with basic weather features showed a slight drop in short-term NLL performance, it improved over longer prediction horizons, suggesting weather impacts long-term trajectory predictions more significantly. The most comprehensive weather-enhanced model consistently outperformed both in RMSE and NLL, highlighting the benefits of systematically integrating weather data. Future work may explore meteorological insights for further refinement.  

(See the table below or address the report **Table 1** and **Figure 8** for details.)
![image](https://github.com/user-attachments/assets/27ddfaee-1750-4796-8578-3395b9910722)

## Quick Start
### Requirements 

## References
[1] Nachiket Deo and Mohan M. Trivedi. 2018. **Convolutional Social Pooling for Vehicle Trajectory Prediction**. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops*, pages 1468–1476.  
[2] J. Colyar and J. Halkias. 2007. **US Highway 101 Dataset**. Federal Highway Administration (FHWA). Tech. Rep. (2007).  
[3] J. Colyar and J. Halkias. 2007. **US Highway I-80 Dataset**. Federal Highway Administration (FHWA). Tech. Rep. (2007).  
[4] Visual Crossing Corporation. [n.d.]. **Weather Data & Weather API | Visual Crossing** — [https://www.visualcrossing.com/](https://www.visualcrossing.com/). [Accessed 18-12-2024].  
