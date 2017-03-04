library(RTextTools)
library(dplyr)
library(caret)

sen=read.csv("sentiment.csv",stringsAsFactors=FALSE)

posr=subset(sen,sen$sentiment=="positive")

negr=subset(sen,sen$sentiment=="negative")

###################testing data for maximum entropy and svm##################

testdat=sen[2001:2500,]

testdat1=subset(testdat,testdat$sentiment=="positive"|testdat$sentiment=="negative")

posrevs=cbind(posr$text,"positive")

negrevs=cbind(negr$text,"negative")

tesrevs=cbind(testdat1$text,testdat1$sentiment)

twts1=rbind(posrevs,negrevs,tesrevs)

revmatrix=create_matrix(twts1[,1],language="english",removeStopwords=FALSE,stemWords=FALSE,toLower=TRUE)

mat3 = as.matrix(revmatrix)
#########################classifying  svm for testing data ######################

container = create_container(mat3, as.numeric(as.factor(twts1[,2])),trainSize=1:1349, testSize=1350:1589,virgin=FALSE)

models = train_models(container, algorithms=c( "SVM"),cost=5)

results = classify_models(container, models)

recall_accuracy(as.numeric(as.factor(twts1[1350:1589, 2])), results[,"SVM_LABEL"])

###################################classifying maximum entropy for testing data############

containermaxent = create_container(mat3, as.numeric(as.factor(twts1[,2])),trainSize=1:1000, testSize=1001:1550,virgin=FALSE)

modelsmaxent = train_models(containermaxent, algorithms=c("MAXENT"))

resultsmaxent = classify_models(containermaxent, modelsmaxent)

cross_validate(containermaxent, N, "MAXENT")
####################plots####################
ggplot(sen, aes(x=sentiment)) +
  geom_bar(aes(y=..count.., fill=sentiment)) +
  scale_fill_brewer(palette="Dark2") +
  labs(x="polarity categories", y="number of tweets")

 freq <- sort(colSums(as.matrix(dtm)), decreasing=TRUE)   
 head(freq, 14)
 wf <- data.frame(word=names(freq), freq=freq)   
 head(wf)

 p <- ggplot(subset(wf, freq>50), aes(word, freq))    
 p <- p + geom_bar(stat="identity")   
 p <- p + theme(axis.text.x=element_text(angle=45, hjust=1))   
 p   
 p <- ggplot(subset(wf, freq>100), aes(word, freq))    
 p <- p + geom_bar(stat="identity")   
 p <- p + theme(axis.text.x=element_text(angle=45, hjust=1))   
 p


 wordcloud(names(freq), freq, min.freq=30, scale=c(4, 1), colors=brewer.pal(6, "Dark2"))

  

