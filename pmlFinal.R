#Project...
library(doSNOW)
library(foreach)
library(caret)
library(corrplot)

#cl <- makeCluster(8)
#registerDoSNOW(cl)

set.seed(2014)

training <- read.csv("C:/00_Practical Machine Learning/pml-training.csv")
testing <- read.csv("C:/00_Practical Machine Learning/pml-testing.csv")


NAstrainig <- apply(training,2,function(x) {sum(is.na(x))}) 
validData <- training[,which(NAstrainig == 0)]
rIndex <- grep("num_window|timestamp|X|user_name|new_window|kurtosis|skewness|max_yaw_belt|min_yaw_belt|amplitude_yaw_belt|max_yaw_dumbbell|min_yaw_dumbbell|amplitude_yaw_dumbbell|max_yaw_forearm|min_yaw_forearm|amplitude_yaw_forearm",names(validData))
trainData <- validData[,-rIndex]

nzvRaw <- nearZeroVar(training,saveMetrics=TRUE)
print(nzvRaw)
nzvtrainData <- nearZeroVar(trainData,saveMetrics=TRUE)
print(nzvtrainData)

NAstesting <- apply(testing,2,function(x) {sum(is.na(x))}) 
validDataTesting <- testing[,which(NAstesting == 0)]
rIndex <- grep("num_window|timestamp|X|user_name|new_window|kurtosis|skewness|max_yaw_belt|min_yaw_belt|amplitude_yaw_belt|max_yaw_dumbbell|min_yaw_dumbbell|amplitude_yaw_dumbbell|max_yaw_forearm|min_yaw_forearm|amplitude_yaw_forearm",names(validDataTesting))
testingData <- validDataTesting[,-rIndex]

# Now We Have 19622 x 53 Training Data after removing the NA columns & Near Zero Variables. 

targetCorrelationInspection <- trainData[,-grep("classe",names(trainData))]

correlationMatrix <- cor(targetCorrelationInspection)
#corrplot(corelationMatrix, order = "FPC", type = "upper",method = "pie")
#correlationMatrix[(abs(correlationMatrix) > 0.9) & abs(correlationMatrix) < 1]

corrplot(correlationMatrix, order = "original", type = "upper",method = "pie")
z <- which((abs(correlationMatrix) > 0.95) & abs(correlationMatrix) < 1, arr.ind=T)
a <- data.frame(rownames(z),colnames(correlationMatrix)[z[,2]])

print("Highly Correlated Predictors: ")
print(a)

# %75 Training, %25 Validation ...
SeparationIndexes = createDataPartition(y = trainData$classe, p = 0.75, list = FALSE)
trainingSet <- trainData[SeparationIndexes, ]
validationSet <- trainData[-SeparationIndexes, ]

# PCA Analysis


print("Preproccessing...")
preProccessing <- preProcess(trainingSet[, -grep("classe",names(trainData))], method = "pca", thresh = 0.99)

trainingPC <- predict(preProccessing, trainingSet[, -grep("classe",names(trainData))])
validatingPC <- predict(preProccessing, validationSet[, -grep("classe",names(trainData))])
print("Preproccessing Done!...")

print("Training The Random Forest Model...")
traincontrol <- trainControl(method="cv", number=4)
st <- system.time(modelFit <- train(trainingSet$classe ~ ., method = "rf", data = trainingPC,
                                    trControl = traincontrol,importance = TRUE))
print("Training Done...")

print(st)
print(modelFit, digits=3)
print(modelFit$finalModel,digits=3)

varImpPlot(modelFit$finalModel, sort = TRUE, type = 1, pch = 18, col = 4, 
           main = "Principal Component Importance Levels")

print("Validating...")
validationResults <- predict(modelFit, validatingPC)
confusionM <- confusionMatrix(validationSet$classe, validationResults)
print(confusionM)

mA<- postResample(validationSet$classe, validationResults)
modelAccuracy <- mA[[1]]

print("Model Accuracy: ")
print(modelAccuracy,digits=3)

outOfSampleError <- 1 - modelAccuracy
print("Out Of Sample Error")
print(outOfSampleError,digits=3)


# Find The Predictions...
testingPC <- predict(preProccessing, testingData[, -grep("problem_id",names(testingData))])
predictions <- predict(modelFit,testingPC)
# Prepare For Writing to Disk!
predVector <- as.vector(predictions)
#Print Results
print("Classification Results:")
print(predVector)

source('C:/00_Practical Machine Learning/pml_write_files.R')
pml_write_files(predVector)


#source('C:/00_Practical Machine Learning/pml_write_files.R')
#pml_write_files(predVector)
