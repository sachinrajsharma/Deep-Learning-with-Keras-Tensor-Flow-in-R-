library(tidyverse)
library(naniar)
library(ggplot2)
library(caTools)
library(dplyr)
#install.packages("keras")
library(tensorflow)
library(keras)
#install_keras()

data <- read.csv(file.choose(), header = TRUE)

head(data)
ncol(data)
names(data)

# Change data to Matrix 

data <- as.matrix(data)
str(data)

# Normalize the data from 1:21 column as last column NSP is dependent variable
data[, 1:21]<- normalize(data[, 1:21])
str(data)
data[, 22]<- as.numeric(data[, 22])-1
head(data)
summary(data)
# Lets remove the column names 
dimnames(data)<- NULL


summary(data)

#Doing data partition into training and test set 

split <- sample(2, nrow(data), replace= T, prob = c(0.7, 0.3))

training <- data[ split ==1, 1:21]
test <- data[split ==2, 1:21]

trainingtarget <- data[ split==1,22]
testtarget <- data[split==2, 22]
#One hot encoding 

trainlabels <- to_categorical(trainingtarget)
testlabels <- to_categorical(testtarget)

print(trainlabels)

# Creating Sequential Model 
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 8, activation = 'relu', input_shape = c(21)) %>% 
  layer_dense(units= 3, activation = 'softmax')

summary(model)

# Compiling the moel 

model %>% 
  compile(loss = 'categorical_crossentropy',
          optimizer = 'adam',
          metrics = 'accuracy')

# Fitting the model 

history <- model %>% 
  fit(training,
      trainlabels,
      epoch = 200,
      batch_size = 32,
      validation_split = 0.2)

# To plot static graph of history 

plot(history)

# Now Evaluate the test data 

model %>% 
  evaluate(test, testlabels)

# Prediction and Making Confusion Matrix - with test data 

prob <-model %>% predict_proba(test)

pred <- model %>% 
  predict_classes(test)

print(pred)

print(testtarget)
table(Predicted = pred, Actual = testtarget)


##prediction 
pred <- model %>% predict(test, batch_size = 128)
Y_pred = round(pred)

Y_pred
# Checking accuracy with table 
cbind(Y_pred, testtarget)


