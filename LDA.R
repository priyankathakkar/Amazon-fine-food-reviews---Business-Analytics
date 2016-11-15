library(quanteda) 
library(stm)
library(tm)
library(NLP)
library(openNLP)
library(ggplot2)
library(ggdendro)
library(cluster)
library(fpc)
library(dplyr)
require(magrittr)
library(stringr)
library(lda)
library(LDAvis)
library(servr)

reviews<-read.csv("Reviews.csv",header=TRUE, stringsAsFactors=FALSE)
summary(reviews)
require(quanteda)
newcorpus<- corpus(reviews$Text,
                   docnames=reviews$Id)
corpus<- toLower(newcorpus, keepAcronyms = FALSE) 
cleancorpus <- tokenize(corpus, 
                        removeNumbers=TRUE,  
                        removePunct = TRUE,
                        removeSeparators=TRUE,
                        removeTwitter=FALSE,
                        verbose=TRUE)
swlist = c("br", "one", "can","also","go","much","even","use","make","now","get")
dfm<- dfm(cleancorpus,
          toLower = TRUE, 
          ignoredFeatures = c(swlist, stopwords("english")), 
          verbose=TRUE, 
          stem=TRUE)
topfeatures(dfm, 50)

#Hierc. Clustering
dfm.tm<-convert(dfm, to="tm")
dfm.tm
dtmss <- removeSparseTerms(dfm.tm, 0.77)
dtmss
d.dfm <- dist(t(dtmss), method="euclidian")
fit <- hclust(d=d.dfm, method="average")
hcd <- as.dendrogram(fit)
require(cluster)
k<-5
plot(hcd, ylab = "Distance", horiz = FALSE, 
     main = "Five Cluster Dendrogram", 
     edgePar = list(col = 2:3, lwd = 2:2))
rect.hclust(fit, k=k, border=1:5) # draw dendogram with red borders around the 5 clusters

ggdendrogram(fit, rotate = TRUE, size = 4, theme_dendro = FALSE,  color = "blue") +
  xlab("Features") + 
  ggtitle("Cluster Dendrogram")

require(fpc)   
d <- dist(t(dtmss), method="euclidian")   
kfit <- kmeans(d, 5)   
clusplot(as.matrix(d), kfit$cluster, color=T, shade=T, labels=2, lines=0)

#Advanced Methods for topic modelling
library(dplyr)
require(magrittr)
library(tm)
library(ggplot2)
library(stringr)
library(NLP)
library(openNLP)

precorpus<- read.csv("reviews.csv", 
                     header=TRUE, stringsAsFactors=FALSE)
stop_words <- stopwords("us")
## additional junk words showing up in the data
stop_words <- c(stop_words, "br", "one", "can","also","go","much","even","use","make","now","get","hes")
stop_words <- tolower(stop_words)
#passing Full Text to variable dataset
dataset<-precorpus$Text
dataset

dataset <- gsub("'", "", dataset) # remove apostrophes
dataset <- gsub("[[:punct:]]", " ", dataset)  # replace punctuation with space
dataset <- gsub("[[:cntrl:]]", " ", dataset)  # replace control characters with space
dataset <- gsub("^[[:space:]]+", "", dataset) # remove whitespace at beginning of documents
dataset <- gsub("[[:space:]]+$", "", dataset) # remove whitespace at end of documents
dataset <- gsub("[^a-zA-Z -]", " ", dataset) # allows only letters
dataset <- tolower(dataset)  # force to lowercase

## get rid of blank docs
dataset <- dataset[dataset != ""]

# tokenize on space and output as a list:
doc.list <- strsplit(dataset, "[[:space:]]+")

# compute the table of terms:
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)

del <- names(term.table) %in% stop_words | term.table < 5
term.table <- term.table[!del]
term.table <- term.table[names(term.table) != ""]
vocab <- names(term.table)

# now put the documents into the format required by the lda package:
get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)

#############
# Compute some statistics related to the data set:
D <- length(documents)  # number of documents (1)
W <- length(vocab)  # number of terms in the vocab (1741)
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document [312, 288, 170, 436, 291, ...]
N <- sum(doc.length)  # total number of tokens in the data (56196)
term.frequency <- as.integer(term.table) 

# MCMC and model tuning parameters:
K <- 10
G <- 3000
alpha <- 0.02
eta <- 0.02

# Fit the model:
library(lda)
set.seed(357)
t1 <- Sys.time()
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, 
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)
t2 <- Sys.time()
## display runtime
t2 - t1  
theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))

news_for_LDA <- list(phi = phi,
                     theta = theta,
                     doc.length = doc.length,
                     vocab = vocab,
                     term.frequency = term.frequency)
library(LDAvis)
library(servr)

# create the JSON object to feed the visualization:
json <- createJSON(phi = news_for_LDA$phi, 
                   theta = news_for_LDA$theta, 
                   doc.length = news_for_LDA$doc.length, 
                   vocab = news_for_LDA$vocab, 
                   term.frequency = news_for_LDA$term.frequency)
serVis(json, out.dir = 'vis_2', open.browser = TRUE)
