rm(list=ls())
library(Amelia)
library(dplyr)
library(ggplot2)
library(caTools)
library(stringr)
library(caret)
#library(readr) # CSV file I/O, e.g. the read_csv function
library(caTools)
#library(rpart.plot)
library(corrplot)
#library(Hmisc)
#library(rpart.plot)
library(class)
library(e1071)
library(neuralnet)
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
normalize <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}
wbcd1<- as.data.frame(lapply(wbcd[2:31],normalize))
wbcd1$diagnosis <- wbcd$diagnosis
wbcd1$diagnosis <- as.numeric(wbcd1$diagnosis)
binary <- function(dg){
  for(i in 1:length(dg)){
    if(dg[i] == 1){
      dg[i] <- 0
    }else{
      dg <- 1
    }
  }
  return(dg)
}

wbcd1$diagnosis <-sapply(wbcd1$diagnosis,binary)
split <- sample.split(wbcd1$diagnosis, Split = 0.7)
train <- subset(wbcd1, split == T)
test <- subset(wbcd1, split == F)
dim(test)
dim(train)
set.seed(745550)
neuralnn <- neuralnet(diagnosis ~ radius_mean + texture_mean + perimeter_mean + area_mean + 
                  smoothness_mean + compactness_mean + concavity_mean + concave.points_mean + 
                  symmetry_mean + fractal_dimension_mean + radius_se + texture_se + 
                  perimeter_se + area_se + smoothness_se + compactness_se + 
                  concavity_se + concave.points_se + symmetry_se + fractal_dimension_se + 
                  radius_worst + texture_worst + perimeter_worst + area_worst + 
                  smoothness_worst + compactness_worst + concavity_worst + 
                  concave.points_worst + symmetry_worst + fractal_dimension_worst, data=train, hidden = c(29), linear.output = FALSE)
neuralnn
plot(neuralnn)

#Prediction on test data
predicted.nn.values <- compute(neuralnn, test[,1:30])
predictions <- sapply(predicted.nn.values$net.result,round)
g_neural <-table(predictions, test$diagnosis)
g_neural
g.t_neural <- sum(diag(g_neural))/sum(g_neural)
print(g.t_neural)

confusionMatrix(predictions,test$diagnosis)

#Prediction on train data
predicted.nn.values_training <- compute(neuralnn, train[,1:30])
predictions_training <- sapply(predicted.nn.values_training$net.result,round)
g_training_neural <-table(predictions_training, train$diagnosis)
g_training_neural
g.training_neural <- sum(diag(g_training_neural))/sum(g_training_neural)
print(g.training_neural)
dim(train)
set.seed(7770)
# get the data from wbcd and specify number of folds
nrFolds_neural <- 10
outs_neural <- NULL
# generate array containing fold-number for each sample (row)
folds_neural <- rep_len(1:nrFolds_neural, nrow(wbcd))
folds_neural
dim(folds_neural)
length(folds_neural)
# actual cross validation
for(k in 1:nrFolds_neural) {
  # actual split of the data
  fold_neural <- which(folds_neural == k)
  data.train_neural <- wbcd1[-fold_neural,]
  dim(data.train_neural)
  
  data.test_neural <- wbcd1[fold_neural,]
  dim(data.test_neural)
  
  # Train and test your model with data.train and data.test
  
  dim(data.test_neural)
  nn_cv_neural <- neuralnet(diagnosis ~ radius_mean + texture_mean + perimeter_mean + area_mean + 
                       smoothness_mean + compactness_mean + concavity_mean + concave.points_mean + 
                       symmetry_mean + fractal_dimension_mean + radius_se + texture_se + 
                       perimeter_se + area_se + smoothness_se + compactness_se + 
                       concavity_se + concave.points_se + symmetry_se + fractal_dimension_se + 
                       radius_worst + texture_worst + perimeter_worst + area_worst + 
                       smoothness_worst + compactness_worst + concavity_worst + 
                       concave.points_worst + symmetry_worst + fractal_dimension_worst, data=data.train_neural, hidden = c(29), linear.output = FALSE)
  
  
  # Compute predictions
  
  predicted.nn.values_cv_neural <- compute(nn_cv_neural,  data.test_neural[,1:30])
  
  
  
  # Extract results
  
  predictions_cv_neural <- sapply(predicted.nn.values_cv_neural$net.result,round)
  print(predictions_cv_neural)
  print(dim(predictions_cv_neural))
  
  # Accuracy (test set)
  
  
  g_cv_neural <-table(predictions_cv_neural, data.test_neural$diagnosis)
  print(g_cv_neural)
  
  outs_neural[k] <- sum(diag(g_cv_neural))/sum(g_cv_neural)
  print(outs_neural[k])
  
}


# Average accuracy
mean(outs_neural)



dim(data.test_neural)
dim(data.train_neural)
folds_neural
dim(folds_neural)
length(folds_neural)
