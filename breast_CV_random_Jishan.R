rm(list=ls())
library(Amelia)
library(dplyr)
library(ggplot2)
library(caTools)
library(stringr)
library(randomForest)
#library(readr) # CSV file I/O, e.g. the read_csv function
library(caTools)
#library(rpart.plot)
library(corrplot)
#library(Hmisc)
#library(rpart.plot)
library(class)
library(e1071)
wbcd <- read.csv("C:/Users/User 1/Desktop/data.csv")
str(wbcd)
head(wbcd)
any(is.na(wbcd))
missmap(wbcd, main='Missing data', col=c("yellow","black"), legend = FALSE)
wbcd$X <- NULL
wbcd$ids <- wbcd$id
wbcd$id <- NULL
wbcd$diagnosis <- as.factor(wbcd$diagnosis)
str(wbcd)
wbcd.cor <- test <- select(wbcd,-ids)
wbcd.cor$diagnosis <- as.numeric(wbcd$diagnosis)

B <- cor(wbcd.cor)
corrplot(B, method="circle")

set.seed(81)
split <- sample.split(wbcd$diagnosis, Split = 0.7)
train <- subset(wbcd, split == T)
test <- subset(wbcd, split == F)
dim(test)

rf.model <- randomForest(diagnosis ~ ., data=train, importance=TRUE)
rf.model
plot(rf.model)

#Prediction on test data
predicted.values <- predict(rf.model, test[2:32])
d <- table(predicted.values, test$diagnosis)
print(d)
library(caret)
confusionMatrix(predicted.values,test$diagnosis)

d.t <-sum(diag(d))/sum(d)
print(d.t)

#Prediction on train data
predicted.values <- predict(rf.model, train[2:32])
d_train <- table(predicted.values, train$diagnosis)
print(d_train)
confusionMatrix(predicted.values,train$diagnosis)

d.train <-sum(diag(d_train))/sum(d_train)
print(d.train)
dim(train)
set.seed(178)



# get the data from wbcd and specify number of folds
nrFolds <- 10
outs <- NULL
# generate array containing fold-number for each sample (row)
folds <- rep_len(1:nrFolds, nrow(wbcd))
folds
dim(folds)
length(folds)
# actual cross validation
for(k in 1:nrFolds) {
  # actual split of the data
  fold <- which(folds == k)
  data.train <- wbcd[-fold,]
  dim(data.train)
  
  data.test <- wbcd[fold,]
  dim(data.test)
  
# Train and test your model with data.train and data.test

  dim(data.test)
  rf_cv <- randomForest(diagnosis ~.,data= data.train, importance=TRUE)
  
# Compute predictions
  
  predicted.values <- predict(rf_cv, data.test[2:32])
  
  
 # Extract results
  
  g_cv <- table(predicted.values, data.test$diagnosis)
  print(g_cv)
  print(dim(g_cv))
  
  # Accuracy (test set)
  
  outs[k] <- sum(diag(g_cv))/sum(g_cv)
  print(outs[k])
  
}


# Average accuracy
mean(outs)



dim(data.test)
dim(data.train)
folds
dim(folds)
length(folds)
