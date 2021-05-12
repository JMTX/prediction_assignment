---
title: "Prediction_assignment"
author: "JMTX"
date: "25 avril 2021"
output:
  html_document:
    keep_md: yes
---



## Synopsis

In this assignment, the goal was to find a robust method to predict which physical activity the user was doing from different measures detected on a smartphone. We tested several models like a decision tree, logistical regression, a combination of these both models but the method which provides the best accuracy (0.99) was the random forest model.

a few comments: 

- The "out-of-sample accuracy" of the methods were evaluated using a validation set separated from the train set to select the best method. Indeed the "out-of-sample accuracy" seems to me a more suitable way to evaluate methods for a classification task than using "out-of-sample error". 

- All models were fitted using a 4 folds cross validation.


## Cleaning the data
The data for this project come from the Human Activity Recognition (HAR) project: http://groupware.les.inf.puc-rio.br/har.
A training set and a test set can be downloaded on the coursera project page. As I was not sure if I was allowed to share the link, I will not write it here and I will assume that you have downloaded the data in the following section.

Before fitting our models, we need to clean the data because many measures contains NA values or DIV#0 values. This will help us to reduce the number of usefull variables.


```r
rm(list=ls())
#go to the project directory, replace "." by your work directory path in the work_directory variable.
work_directory="."
setwd(work_directory)
getwd()
```

```
## [1] "D:/JM/informatique/datascience_specialization/Course IIIIIIII - Practical Machine Learning/Week4 -  Regularized_Regression_and_Combining_Predictors/prediction_assignment"
```

```r
#Load the training data
training <- read.csv("pml-training.csv")

#remove col which have too many NA values
ts_training<-training[ , colSums(is.na(training)) ==0]

#remove the variables which have weird value like div0 or ""
ts_training2<-ts_training[ , colSums((ts_training =="") | (ts_training =="#DIV/0!")) < 5]

#remove some useless variables for classification 
#(like timestamps or window because we wont use temporal information like 
#we could use in model like sequential neural network) 
ts_training3 <- ts_training2[,-c(1:7)]

dim(ts_training3)
```

```
## [1] 19622    53
```

After cleaning the data, we obtain a dataframe of 19622 observations of 53 variables.

just a comment: As you can see in the code the 7 first variables were removed because they represented some timestamps or windows which represents some temporal information I did not use in the follwing models. Maybe these variable could be interesting if we used some model which take into account this information like sequential neural network.


## Create the training set and validation set
In order to evaluate the different methods it is important to use dataset which were not used during the training part.
That is why I will split the original training data into a training set "train_set" containing 75% of the original training data, and a validation set "validation_set" containing 25% of the original training data. In the following sections the models will be trained using the "train_set" and evaluated using the "validation_set".


```r
#create a training (75%) and validation set (25%) from the originalm training dataset
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
in_train <- createDataPartition(y=ts_training3$classe,
                               p=0.75,list=FALSE)
train_set <- ts_training3[in_train,];validation_set<-ts_training3[-in_train,]

#define the cross validation parameter to divide the train_set in 4 folds for cv
fit_control <- trainControl(method="cv", 4)
```
I also define in the code the cross validation parameter to use during the fit of the models.

## Prediction Models

In this part we will define 4 prediction models and test their performance on the validation set. The 4 models will be:
- a decision tree model
- a logistical regression model
- a combination of both previous models
- a random forest model

# the decision tree model

```r
#Define first model : simple decision tree
model_fit1 <- train(classe ~ ., data=train_set, method="rpart", trControl=fit_control)

#compute the prediction on the train _set just to have an idea of the in-sample accuracy
model_pred_train1 <- predict(model_fit1, train_set)
in_sample_accuracy1 = sum(train_set$classe==model_pred_train1)/length(model_pred_train1)
in_sample_accuracy1
```

```
## [1] 0.496399
```

```r
#compute the prediction of this model on the validation set and compute the confusion matrix
model_pred1 <- predict(model_fit1, validation_set)
out_sample_accuracy1 = sum(validation_set$classe==model_pred1)/length(model_pred1)
confusionMatrix(validation_set$classe, model_pred1)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1264   21  104    0    6
##          B  409  301  239    0    0
##          C  403   24  428    0    0
##          D  377  140  287    0    0
##          E  132  117  227    0  425
## 
## Overall Statistics
##                                          
##                Accuracy : 0.4931         
##                  95% CI : (0.479, 0.5072)
##     No Information Rate : 0.5271         
##     P-Value [Acc > NIR] : 1              
##                                          
##                   Kappa : 0.3368         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.4890  0.49917  0.33307       NA  0.98608
## Specificity            0.9435  0.84934  0.88201   0.8361  0.89358
## Pos Pred Value         0.9061  0.31718  0.50058       NA  0.47170
## Neg Pred Value         0.6235  0.92364  0.78834       NA  0.99850
## Prevalence             0.5271  0.12296  0.26203   0.0000  0.08789
## Detection Rate         0.2577  0.06138  0.08728   0.0000  0.08666
## Detection Prevalence   0.2845  0.19352  0.17435   0.1639  0.18373
## Balanced Accuracy      0.7162  0.67425  0.60754       NA  0.93983
```
This fist model obtain an out-of-sample accuracy of 0.4930669 which is not really good. At least this model, does not overfit the data because the in-sample (0.496399) and out-of-sample (0.4930669) accuracy are pretty similar.

# the logistical regression model

```r
#define second model : multiple logistical regression
model_fit2 <- train(classe ~ ., data=train_set, method="multinom",
                    trControl=fit_control,trace=FALSE)

#compute the prediction on the train _set just to have an idea of the in-sample accuracy
model_pred_train2 <- predict(model_fit2, train_set)
in_sample_accuracy2 = sum(train_set$classe==model_pred_train2)/length(model_pred_train2)
in_sample_accuracy2
```

```
## [1] 0.6724419
```

```r
#compute the prediction of this model on the validation set and compute the confusion matrix
model_pred2 <- predict(model_fit2, validation_set)
out_sample_accuracy2 = sum(validation_set$classe==model_pred2)/length(model_pred2)
confusionMatrix(validation_set$classe, model_pred2)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1163   34   82  107    9
##          B  142  548   73   78  108
##          C  129  102  501   77   46
##          D   92   22   84  570   36
##          E   59  111   55  150  526
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6746          
##                  95% CI : (0.6612, 0.6877)
##     No Information Rate : 0.3232          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.5868          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7338   0.6707   0.6302   0.5804   0.7255
## Specificity            0.9301   0.9019   0.9138   0.9403   0.9103
## Pos Pred Value         0.8337   0.5774   0.5860   0.7090   0.5838
## Neg Pred Value         0.8797   0.9320   0.9274   0.8995   0.9503
## Prevalence             0.3232   0.1666   0.1621   0.2002   0.1478
## Detection Rate         0.2372   0.1117   0.1022   0.1162   0.1073
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.8319   0.7863   0.7720   0.7604   0.8179
```
This second model (logistic regression ) obtain an out-of-sample accuracy of 0.6745514 which is better than the first one but not good enough for our problem. As the first model, the second one does not overfit the training data because the in-sample (0.6724419) and out-of-sample (0.6745514) accuracy are pretty similar.


# Stacking both precedent models

The previous models did not obtain very good performance, thus it could be interesting to combine both ones to see if this could increase the accuracy.


```r
#let s try to combine both first models to see if we can increase accuracy----
#1) compute prediction on training data for both model
model_pred_train1 <- predict(model_fit1, train_set);model_pred_train2 <- predict(model_fit2, train_set);

#2) create a dataframe of prediction values 
pred_data <- data.frame(model_pred_train1,model_pred_train2,classe=train_set$classe)

#3) fit the model
model_stack <- train(classe~.,method="multinom",data=pred_data, trControl=fit_control, trace=FALSE)


#compute the prediction on the train_set just to have an idea of the in-sample accuracy
model_pred_train_stack <- predict(model_stack, pred_data)
in_sample_accuracy_stack = sum(train_set$classe==model_pred_train_stack)/length(model_pred_train_stack)
in_sample_accuracy_stack
```

```
## [1] 0.6926892
```

```r
#4) create a dataframe of prediction on the validation test 
pred_data_val <- data.frame(model_pred_train1= model_pred1,model_pred_train2=model_pred2,classe=validation_set$classe)
pred_stack <- predict(model_stack,pred_data_val)

out_sample_accuracy_stack = sum(validation_set$classe== pred_stack)/length(pred_stack)
confusionMatrix(validation_set$classe, pred_stack)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1149   48   76  107   15
##          B  135  555   73   78  108
##          C  124  107  501   77   46
##          D   92   22   84  570   36
##          E   38   99   19  104  641
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6966          
##                  95% CI : (0.6835, 0.7094)
##     No Information Rate : 0.3136          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6151          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7471   0.6679   0.6653   0.6090   0.7577
## Specificity            0.9269   0.9033   0.9147   0.9410   0.9359
## Pos Pred Value         0.8237   0.5848   0.5860   0.7090   0.7114
## Neg Pred Value         0.8891   0.9302   0.9378   0.9107   0.9488
## Prevalence             0.3136   0.1695   0.1535   0.1909   0.1725
## Detection Rate         0.2343   0.1132   0.1022   0.1162   0.1307
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.8370   0.7856   0.7900   0.7750   0.8468
```
This stacked model (logistic regression + tree decision ) obtain an out-of-sample accuracy of 0.6965742 which is just slitly better than the second one but not nearly as good as we need for our problem. One more time, this model does not overfit the training data because the in-sample (0.6926892) and out-of-sample (0.6965742) accuracy are pretty similar.

```r
summary(cars)
```

```
##      speed           dist       
##  Min.   : 4.0   Min.   :  2.00  
##  1st Qu.:12.0   1st Qu.: 26.00  
##  Median :15.0   Median : 36.00  
##  Mean   :15.4   Mean   : 42.98  
##  3rd Qu.:19.0   3rd Qu.: 56.00  
##  Max.   :25.0   Max.   :120.00
```
# The random forest model

Now, it is time to use a more complex model: the random forest which consists into combining several decision trees and take the class which obtain the majority of the vote of the different trees. Let's see the performance of such a model.


```r
#Define the fourth model and fit it : random forest model
model_fit4 <- train(classe ~ ., data=train_set, method="rf", trControl=fit_control, trace=FALSE)

#compute the prediction on the train_set just to have an idea of the in-sample accuracy
model_pred_train4 <- predict(model_fit4, train_set)
in_sample_accuracy4 = sum(train_set$classe==model_pred_train4)/length(model_pred_train4)
in_sample_accuracy4
```

```
## [1] 1
```

```r
#compute the prediction of this model on the validation set and compute the confusion matrix
model_pred4 <- predict(model_fit4, validation_set)
out_sample_accuracy4 = sum(validation_set$classe==model_pred4)/length(model_pred4)
confusionMatrix(validation_set$classe, model_pred4)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    0    0    0    1
##          B    2  946    1    0    0
##          C    0    2  851    2    0
##          D    0    0   10  793    1
##          E    0    0    1    3  897
## 
## Overall Statistics
##                                         
##                Accuracy : 0.9953        
##                  95% CI : (0.993, 0.997)
##     No Information Rate : 0.2847        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.9941        
##                                         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9986   0.9979   0.9861   0.9937   0.9978
## Specificity            0.9997   0.9992   0.9990   0.9973   0.9990
## Pos Pred Value         0.9993   0.9968   0.9953   0.9863   0.9956
## Neg Pred Value         0.9994   0.9995   0.9970   0.9988   0.9995
## Prevalence             0.2847   0.1933   0.1760   0.1627   0.1833
## Detection Rate         0.2843   0.1929   0.1735   0.1617   0.1829
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9991   0.9986   0.9926   0.9955   0.9984
```

This random forest model obtain an out-of-sample accuracy of 0.99531 which is really great and largely better than all the other models that we've tested in the previous sections. Indeed the confusion matrix seems really nice too, nearly a diagonal matrix. As for the other models, this one does not overfit the training data because the in-sample (1) and out-of-sample (0.99531) accuracy are pretty similar.

## Conclusion

We've tested several models to predict activities from the Human Activity project data and the best model was the random forest obtaining an out of sample precision of (0.99531). Thus we advised to use this model for this problem.
