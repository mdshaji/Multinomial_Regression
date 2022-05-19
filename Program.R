### Analyzing the business problem ###

# Output Variable (y) = Type of program (prog) which the student take the most
# Input Variables = x,id,female,ses,schtyp,read,write,math,science,honors

# Importing required packages
require('mlogit')
require('nnet')

### Importing Dataset ###
education <- read.csv(file.choose())
View(education)
attach(education)

head(education) # Shows first 6 rows of the dataset
tail(education) # Showa last 6 rows of the dataset

# Tabular represntation of output variable
table(education$prog)
# Student who opt academic = 105 , general = 45 , vocation = 50

### Data Preprocessing ##

# Checking of NA values
sum(is.na(education)) # No NA Values found

# Removing Uncessary columns
education <- education[ , 3:11]
View(education)

# Renaming the column names
colnames(education) <- c("gender","ses","schtyp","prog","read","write","math","science","honors")
View(education)

# Reorder the variables
education <- education[,c(4,1,2,3,5,6,7,8,9)]
View(education)

# Creating dummy variables

install.packages("dummies")
library(dummies)

str(education)

education$gender <- as.factor(education$gender)
education$gender <- as.numeric(education$gender)

education$ses <- as.factor(education$ses)
education$ses <- as.numeric(education$ses)

education$schtyp <- as.factor(education$schtyp)
education$schtyp <- as.numeric(education$schtyp)

education$honors <- as.factor(education$honors)
education$honors = as.numeric(education$honors)

str(education)


#Exploratory data analysis
summary(education)

install.packages("Hmisc")
library(Hmisc)
describe(education)

install.packages("lattice") # Highly used for data visualization
library("lattice") # dotplot is part of lattice package

# Graphical exploration
dotplot(education$prog, main = "Dot Plot of Type of Program")
dotplot(education$read, main = "Dot Plot of Read")
dotplot(education$write, main = "Dot Plot of Write")
dotplot(education$math, main = "Dot Plot of Math")


#Boxplot Representation

boxplot(education$read, col = "dodgerblue4")
boxplot(education$math, col = "dodgerblue4")
boxplot(education$science, col = "dodgerblue4")
boxplot(education$write, col = "red", horizontal = T)

#Histogram Representation

hist(education$read, col = 'red',main = "Read")
hist(education$write,col = 'red',main = "Write")
hist(education$math,col = 'red',main = "Math")
hist(education$science,col = 'red',main = "Science")


# Normal QQ plot
qqnorm(education$read ,main = "Read")
qqline(education$read ,main = "Read")

qqnorm(education$write,main = "Write")
qqline(education$write,main = "Write")

qqnorm(education$math,main = "Math")
qqline(education$math,main = "Math")

qqnorm(education$science,main = "Science")
qqline(education$science,main = "Science")

#Scatter plot for all pairs of variables
plot(education)


# Data Partitioning
n <- nrow(education)
n1 <- n * 0.8
n2 <- n - n1
train_index <- sample(1:n,n1)
train <- education[train_index,]
test <- education[-train_index,]

## Model Building ##
education.prog <- multinom(prog ~ ., data=train)
summary(education.prog)

#Residual Deviance: 250.0223
#AIC: 286.0223


##### Significance of Regression Coefficients###
z <- summary(education.prog)$coefficients / summary(education.prog)$standard.errors
z

p_value <- (1-pnorm(abs(z),0,1))*2

summary(education.prog)$coefficients
p_value

# odds ratio 
exp(coef(education.prog))

# predict probabilities
prob <- fitted(education.prog)
prob

# Prediction on test data
pred_test <- predict(education.prog, newdata = test , type = "probs")
pred_test

# Find the accuracy of the model
class(pred_test)
pred_test <- data.frame(pred_test)
View(pred_test)
pred_test["prediction"] <- NULL

# Custom function that returns the predicted value based on probability
get_names <- function(i){
  return (names(which.max(i)))
}

predtest_name <- apply(pred_test, 1, get_names)
pred_test$prediction <- predtest_name
View(pred_test)

# Confusion matrix
table(predtest_name, test$prog)

# confusion matrix visualization
barplot(table(predtest_name, test$prog),beside = T,col=c("red","lightgreen","blue","orange"), legend=c("academic","general","Vocation"), main = "Predicted(X-axis) - Legends(Actual)",ylab ="count")
barplot(table(predtest_name, test$prog),beside = T,col=c("red","lightgreen","blue","orange"), main = "Predicted(X-axis) - Legends(Actual)",ylab ="count")

# Accuracy 
mean(predtest_name == test$prog)


# Training Data
pred_train <- predict(education.prog , newdata = train , type = "probs")
pred_train

# # Find the accuracy of the model
class(pred_train)
pred_train <- data.frame(pred_train)
View(pred_train)
pred_train["prediction"] <- NULL

predtrain_name <- apply(pred_train, 1, get_names)
pred_train$prediction <- predtrain_name
View(pred_train)

# Confusion Matrix
table(predtrain_name , train$prog)

# confusion matrix visualization
barplot(table(predtrain_name, train$prog),beside = T,col=c("red","lightgreen","blue","orange"), legend=c("academic","general","Vocational"), main = "Predicted(X-axis) - Legends(Actual)",ylab ="count")
barplot(table(predtrain_name, train$prog),beside = T,col=c("red","lightgreen","blue","orange"), main = "Predicted(X-axis) - Legends(Actual)",ylab ="count")

# Accuracy 
mean(predtrain_name == train$prog)

