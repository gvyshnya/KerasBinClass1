library (readr)
library(plyr)
library(dplyr)
library(ggplot2)

library(Hmisc)
library(corrplot)
library(PerformanceAnalytics)

library(cluster)
library(factoextra)
library(NbClust)
library(clValid)
library(fpc)

library(Boruta)

########################
### Read in the data ###
########################
fname_training <- "input/obtrain.csv"
fname_testing <- "input/obval.csv"

df_training <- read.csv(fname_training, header=FALSE)
df_testing <- read.csv(fname_testing, header=FALSE)

View(df_training)
View(df_testing)

str(df_training)
colnames(df_training)

str(df_testing)
colnames(df_testing)

#Which of the following variables has at least one missing observation?
# Note: there is no missing values in both data frames
sapply(df_training, function(x) sum(is.na(x)))
sapply(df_testing, function(x) sum(is.na(x)))

#######################################
### Do feature selection via Boruta ###
#######################################

# References:
# https://www.analyticsvidhya.com/blog/2016/03/select-important-variables-boruta-package/
# https://www.r-bloggers.com/feature-selection-all-relevant-selection-with-the-boruta-package/

set.seed(123)
boruta.train <- Boruta(V560~., data = df_training, doTrace = 2)
print(boruta.train)

# plotting Boruta importance diagram
plot(boruta.train, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i)
  boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
       at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)

# Now is the time to take decision on tentative attributes.  
# The tentative attributes will be classified as confirmed or rejected by 
# comparing the median Z score of the attributes with the median Z score of the best shadow attribute. 

final.boruta <- TentativeRoughFix(boruta.train)
print(final.boruta)

# list of finally confirmed variables (21): 
# V2, V131, V168, V170, V205, V275, V279, V287, V345, V346, V368,
# V384, V407, V417, V460, V487, V514,V521, V523, V524, V537

