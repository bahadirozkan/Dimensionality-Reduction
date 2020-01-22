# Dimensionality Reduction

Case 1: Feed the original dataset without any dimensionality reduction as input to k-NN.

Accuracy, Precision and Recall values for this case was 92% on train set. A comparative table that includes other cases and results for the test set can be found below.

Case 2: Feature extraction: Use PCA to reduce dimensionality to m, followed by k-NN. Try for different values of m corresponding to proportion of variance of 0.80, 0.81, 0.82, ...., 0.99. Plot the data for m=2.

For m=2 PCA plot is as follows:

<img src="https://user-images.githubusercontent.com/20925510/72899986-25cee380-3d38-11ea-9c8c-ff40b9f446a6.png" width="40%" height="40%">

The components are not clearly separable. Accuracy for m=2 is around 85%.

<img src="https://user-images.githubusercontent.com/20925510/72900432-e5bc3080-3d38-11ea-8397-d1b93293e051.png" width="40%" height="40%">

m values that produce a variance equal or greater than 0.80 starts from 24 therefore PCA was applied for n-components 24 to 57.

<img src="https://user-images.githubusercontent.com/20925510/72900571-2f0c8000-3d39-11ea-8297-f680e489ae18.png" width="40%" height="40%"> <img src="https://user-images.githubusercontent.com/20925510/72900623-4ba8b800-3d39-11ea-9161-7b151eddec3c.png" width="40%" height="40%">

In case of spam mails, users in general wouldn’t want to miss important mails therefore decreasing false positives would be a better metric. In other words, precision is crucial for this case. Above graphs show that all metrics have decreased on the test set but precision the most.

Case 3: Feature Selection: Use forward selection to reduce dimensionality to m using k-NN as predictor. Train the model for each m between 1 and 57. Also plot the data for m=2.

<img src="https://user-images.githubusercontent.com/20925510/72900826-a6421400-3d39-11ea-80c4-40ccac6bb1fa.png" width="40%" height="40%">

Components are not separable for this case also. Accuracy is around 82%.

<img src="https://user-images.githubusercontent.com/20925510/72900907-ca9df080-3d39-11ea-94df-a7e563a1b279.png" width="40%" height="40%"> <img src="https://user-images.githubusercontent.com/20925510/72900958-e5706500-3d39-11ea-8562-322a8ae9e43f.png" width="40%" height="40%">

Graphs for the feature selection shows that test data is subject to spikes when number of features vary. However, all three metrics seem correlated. The best m parameter found on the train set gave a good result also provided a good result on the test data.

Performance of each case can be seen in the following tables:

