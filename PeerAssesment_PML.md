

#Practical Machine Learning Peer Assessment - Weight Lifting Exercise Manner Prediction

##Summary 

This analysis was done to predict the manner in which the subjects performed weight lifting exercises. The data is collected from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. The outcome variable has five classes and the total number of predictors are 159.

##Preparation


```r
library(caret)
library(randomForest)
```

## Data Loading


```r
Url1 <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
Url2 <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

 file1 <- "training.csv"
  file2 <- "testing.csv"

  download.file(url=Url1, destfile=file1)
   download.file(url=Url2, destfile=file2)

training <- read.csv("training.csv",row.names=1,na.strings = "")
 testing <- read.csv("testing.csv",row.names=1,na.strings = "NA")
```

##Preprocessing

First we need to get rid of the variables that have close to zero variance in both training and testing data. Then remove the columns with missing values to avoid issues in training models. If the result is not good, we can add back those columns with missing values imputed.


```r
nsv <- nearZeroVar(training,saveMetrics=TRUE)
training <- training[,!nsv$nzv]
testing <- testing[,!nsv$nzv]

# Remove variables with missing values
training_filter_na <- training[,(colSums(is.na(training)) == 0)]
testing_filter_na <- testing[,(colSums(is.na(testing)) == 0)]

# Remove unnecessary columns
colRm_train <- c("user_name","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","num_window")
colRm_test <- c("user_name","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","num_window","problem_id")
training_colRm <- training_filter_na[,!(names(training_filter_na) %in% colRm_train)]
testing_colRm <- testing_filter_na[,!(names(testing_filter_na) %in% colRm_test)]
dim(training_colRm)
```

```
## [1] 19622    53
```

```r
dim(testing_colRm)
```

```
## [1] 20 52
```

Now we split the preprocessed training data into training set and validation set.


```r
inTrain <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
training_clean <- training_colRm[inTrain,]
validation_clean <- training_colRm[-inTrain,]
```

In the new training set and validation set we just created, there are 52 predictors and 1 response. Check the correlations between the predictors and the outcome variable in the new training set. There doesn’t seem to be any predictors strongly correlated with the outcome variable, so linear regression model may not be a good option. Random forest model may be more robust for this data.

```r
cor <- abs(sapply(colnames(training_clean[, -ncol(training)]), function(x) cor(as.numeric(training_clean[, x]), as.numeric(training_clean$classe), method = "spearman")))

#cor
```

##Random Forest Model

We try to fit a random forest model and check the model performance on the validation set.


```r
set.seed(999)
# Fit rf model
rfFit <- train(classe ~ ., method = "rf", data = training_clean, importance = T, trControl = trainControl(method = "cv", number = 4))
validation_pred <- predict(rfFit, newdata=validation_clean)
# Check model performance
confusionMatrix(validation_pred,validation_clean$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671    7    0    0    0
##          B    2 1130    6    0    0
##          C    1    2 1016   15    1
##          D    0    0    4  949    2
##          E    0    0    0    0 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9932          
##                  95% CI : (0.9908, 0.9951)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9914          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9921   0.9903   0.9844   0.9972
## Specificity            0.9983   0.9983   0.9961   0.9988   1.0000
## Pos Pred Value         0.9958   0.9930   0.9816   0.9937   1.0000
## Neg Pred Value         0.9993   0.9981   0.9979   0.9970   0.9994
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2839   0.1920   0.1726   0.1613   0.1833
## Detection Prevalence   0.2851   0.1934   0.1759   0.1623   0.1833
## Balanced Accuracy      0.9983   0.9952   0.9932   0.9916   0.9986
```

##Prediction

The last step is to use the random forest model to predict on the testing set without the outcome variable and save the prediction output.


```r
testing_pred <- predict(rfFit, newdata=testing_colRm)
write_files <- function(x) {
        n <- length(x)
        for (i in 1:n) {
                filename <- paste0("problem_id", i, ".txt")
                write.table(x[i], file=filename, quote=FALSE, row.names=FALSE,col.names=FALSE)
        }
}
write_files(testing_pred)
```


##Results

We used 52 variables to build the random forest model with 4-fold cross validation.

