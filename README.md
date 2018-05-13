# Nightlight_crop_cover_mask

Predict Employment cluster labels and source of lighting cluster labels.

Nightlight data:
There are several type of nightlight data extracted at district level. We tried dividing intensity from 0-63 into various bins but 64 bins work out best.
a. 64 length vector containing no. of pixels with intensity ranging from 0-63.
b. 64 length vector containing no of pixels with intensity ranging from 0-63 when the pixels with crop-cover are masked out.
c. 128 length vector containing no. of pixels with intensity ranging from 0-63 if it is masked or unmasked.

All the districts are split into 80-20 train-test datset. We tried different supervised learning algorithms like SVM, random forest,MLP classifier and unsupervised learning method like KMeans clustering.


Running classifer:

Set the path of data which is to be used for classification

python classifier.py




Plotting box plot:

Set the path of data whose boxplot is to be plotted.

python plot_boxplot.py



Results:

Using Random forest classifer

Accuracy on test data:

Predicting employment clusters:
a. Unmasked Nightlight

Test accuracy:  54.69 %

Train accuracy: 90.23 %


Confusion matrix:


![unnormalized_emp_confusion_a](Results/emp_unnormalized_cm_a.png)

![normalized_emp_confusion_a](Results/emp_normalized_cm_a.png)


b. Masked Nightlight:   

Test accuracy: 62.5 % 

Train accuracy: 94.14% 


Confusion matrix:


![unnormalized_emp_confusion_b](Results/emp_unnormalized_confusion_b.png)

![normalized_emp_confusion_b](Results/emp_normalized_confusion_b.png)


c.Using 128 length feature vector :  

Test accuracy:  60.16% 

Train accuracy: 91.99%


Confusion matrix:


![unnormalized_emp_confusion_matrix](Results/emp_unnormalized_confusion_matrix.png)

![normalized_emp_confusion_matrix](Results/emp_normalized_confusion_matrix.png)






Boxplot for Mean NTL and source of lighting:

![Lighting_mean_light](Results/lighting_MeanLight_Unmasked_2011.png)


Masked mean NTL and Source of lighting:

![Lighting_mean_NTL](Results/lighting_MeanLight_Urban_masked_2011.png)


Mean NTL and Employment clusters:

![Employment_mean_NTL](Results/unmasked_mean_light_emp_2011.png)


Masked mean NTL and employment clusters:

![Employment_masked_mean_NTL](Results/urban_masked_2011_emp_Mean_light.png)

