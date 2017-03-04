library(tm)
library(RTextTools)
library(e1071)
library(dplyr)
library(caret)

sen=read.csv("sentiment.csv",stringsAsFactors=FALSE)

sen1=rbind(sen$text,sen$sentiment)

matrix=create_matrix(sen1[,1],language="english",removeStopwords=FALSE,removeNumbers=TRUE,stemWords=FALSE)

sen2$text=as.character(sen2$text)

sen2$sentiment=as.factor(sen2$sentiment)

summary(sen2)

sen3=as.data.frame(sen$sentiment,sen$text)

sen3$text=sen$text

sen3$class=sen$sentiment

corpus <- Corpus(VectorSource(sen$text))

corpus.clean <- corpus %>%

  tm_map(content_transformer(tolower)) %>%

  tm_map(removePunctuation) %>%

  tm_map(removeNumbers) %>%

  tm_map(removeWords, stopwords(kind="en")) %>%

  tm_map(stripWhitespace)

dtm <- DocumentTermMatrix(corpus.clean)

sen3$class=as.factor(sen3$class)
##############division of training and testing data #####################
sen3.train=sen3[1:1500,]

sen3.test=sen3[1501:2000,]

dtm.train <- dtm[1:1500,]

dtm.test <- dtm[1501:2000,]

corpus.clean.train <- corpus.clean[1:1500]

corpus.clean.test <- corpus.clean[1501:2000]

dim(dtm.train)

fivefreq <- findFreqTerms(dtm.train, 5)

dtm.train.nb <- DocumentTermMatrix(corpus.clean.train, control=list(dictionary = fivefreq))

dtm.test.nb <- DocumentTermMatrix(corpus.clean.test, control=list(dictionary = fivefreq))

convert_count <- function(x) {
y <- ifelse(x > 0, 1,0)
y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
y
}
# Apply the count function to get final training and testing dtms
trainNB <- apply(dtm.train.nb, 2, convert_count)

testNB <- apply(dtm.test.nb, 2, convert_count)

system.time( classifier <- naiveBayes(trainNB, df.train$class, laplace = 1) )

system.time( classifier <- naiveBayes(trainNB, sen3.train$class, laplace = 1) )

system.time( pred <- predict(classifier, newdata=testNB) )

table("Predictions"= pred,  "Actual" = sen3.test$class )

conf.mat <- confusionMatrix(pred, sen3.test$class)

conf.mat
#SVM ACCURACY
conf.mat$overall['Accuracy']
