library(tm)
library(qdap)
library(spacyr)
library(wordcloud)
library(plotrix)
library(FactoMineR)
library(factoextra)
library(dplyr)
library(tibble)
library(ggplot2)
library(proxy)
library(dbscan)
library(RWeka)
library(tidyr)
library(tidytext)
library(tidyverse)
library(igraph)
library(ggraph)
library(lubridate)
library(purrr)

Sys.setlocale(category = "LC_ALL", locale = "Italian") 
spacy_initialize(model = "it_core_news_sm")

replacePunctuation <- function(corpus){
  gsub("[[:punct:]]+"," ",corpus)
}

removeSymbols <- content_transformer(function(x, pattern) gsub("[€”“’‘]"," ",x))

lemmatize <- function(corpus){
  # create a list with all corpus contents
  corpus_txt <- c()
  for(i in 1:length(corpus)){
    corpus_txt[i] <- corpus[[i]]$content
  }
  # parse the list and replace each token with its lemma, except for masculine singular sostantives
  parsed_df <- spacy_parse(corpus_txt, tag = TRUE, entity = FALSE)
  parsed_df$lemma <- ifelse(parsed_df$tag == "S__Gender=Masc|Number=Sing", parsed_df$token, parsed_df$lemma)
  # re-create corpus e apply pre-processing once again
  agg <- aggregate(lemma~doc_id, data = parsed_df, paste0, collapse=" ")
  new_corpus <- Corpus(VectorSource(agg$lemma))
  new_corpus <- dataPreProcessing(new_corpus)
  return(new_corpus)
}

dataPreProcessing <- function(corpus){
  corpus <- tm_map(corpus, tolower)
  corpus <- tm_map(corpus, replacePunctuation)
  corpus <- tm_map(corpus, removeSymbols)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removeWords, stopwords("italian"))
  corpus <- tm_map(corpus, removeWords, stop_commenti)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, trimws)
  return(corpus)
}


reviews1 <- read.csv2("reviews1.csv", sep = ",", header = FALSE, col.names = c("rating","date","user","review"), stringsAsFactors = FALSE)
reviews2 <- read.csv2("reviews2.csv", sep = ",", header = FALSE, col.names = c("rating","date","user","review"), stringsAsFactors = FALSE)
stop_words <- read.delim2("stopwords-it.txt", sep = "\t", header = FALSE, encoding = "UTF-8", col.names = c("word"), stringsAsFactors = FALSE)
stop_commenti <- stop_words$word

#data cleaning and preparation
reviews1$rating <- as.numeric(reviews1$rating/10)
reviews1$date <- as.Date(reviews1$date, format="%d %B %Y")

reviews2$rating <- as.numeric(reviews2$rating/10)
reviews2$date <- as.Date(reviews2$date, format="%d %B %Y")

corpus1 <- Corpus(VectorSource(reviews1$review)) %>% dataPreProcessing() %>% lemmatize()
corpus2 <- Corpus(VectorSource(reviews2$review)) %>% dataPreProcessing() %>% lemmatize()


#### Exploratory Data Analysis ####

# Top 10 words frequency plot for corpus1
frequency1 <- freq_terms(corpus1, top=10, at.least = 2)
plot(frequency1)

## Wordcloud for corpus1 ##

ndocs <-length(corpus1)
# ignore extremely rare words i.e. terms that appear in less then 2% of the documents
minTermFreq <- ndocs * 0.02
# ignore overly common words i.e. terms that appear in more than 80% of the documents
maxTermFreq <- ndocs * .80

corpus1_tdm <- TermDocumentMatrix(corpus1, 
                                  control = list(
                                    wordLengths=c(3, 20),
                                    bounds = list(global = c(minTermFreq, maxTermFreq),
                                                  weighting=weightTfIdf)))

corpus1_tdm <- removeSparseTerms(corpus1_tdm, 0.99)
# Convert TDM to matrix
corpus1_m <- as.matrix(corpus1_tdm)
# Sum rows and frequency data frame and sort in descending order
corpus1_term_freq <- sort(rowSums(corpus1_m), decreasing = TRUE)
# Create a data frame with word and frequency
df1 <- data.frame(word = names(corpus1_term_freq), freq = corpus1_term_freq)

set.seed(123)
# Wordcloud 1
par(mfrow=c(1,1))
wordcloud(words=df1$word, freq=df1$freq, max.words = 100, scale=c(3,.2), 
          random.order = FALSE, rot.per=.5, colors= c("tomato1","tomato2","tomato3","tomato4"))

# Top 10 words frequency plot for corpus2
frequency2 <- freq_terms(corpus2, top=10, at.least = 2)
plot(frequency2)

## Wordcloud for corpus 2 ##

ndocs2 <-length(corpus2)
# ignore extremely rare words i.e. terms that appear in less then 2% of the documents
minTermFreq2 <- ndocs2 * 0.02
# ignore overly common words i.e. terms that appear in more than 80% of the documents
maxTermFreq2 <- ndocs2 * .80
corpus2_tdm <- TermDocumentMatrix(corpus2, 
                                  control = list(
                                    wordLengths=c(3, 20),
                                    bounds = list(global = c(minTermFreq2, maxTermFreq2),
                                                  weighting=weightTfIdf)))

corpus2_tdm <- removeSparseTerms(corpus2_tdm, 0.99)
# Convert TDM to matrix
corpus2_m <- as.matrix(corpus2_tdm)
# Sum rows and frequency data frame and sort in descending order
corpus2_term_freq <- sort(rowSums(corpus2_m), decreasing = TRUE)
# Create a data frame with word and frequency
df2 <- data.frame(word = names(corpus2_term_freq), freq = corpus2_term_freq)

par(mfrow=c(1,1))
wordcloud(words=df2$word, freq=df2$freq, max.words = 100, scale=c(3,.2), 
          random.order = FALSE, rot.per=.5, random.color = FALSE, colors= c("deepskyblue1","deepskyblue2","deepskyblue3","deepskyblue4"))

## Comparison between corpora ##

# Combine both corpora: all reviews
all_rev1 <- paste(reviews1$review, collapse = " ")
all_rev2 <- paste(reviews2$review, collapse = " ")
all_combine <- c(all_rev1, all_rev2)
# Creating corpus for combination
corpus_review_all <- Corpus(VectorSource(all_combine)) %>% dataPreProcessing() %>% lemmatize()

review_tdm_all <- TermDocumentMatrix(corpus_review_all)
all_m <- as.matrix(review_tdm_all)
colnames(all_m)=c("corpus1","corpus2")

#Sum rows and frequency data frame
review_term_freq_all <- rowSums(all_m) %>% data.frame(word=names(review_term_freq_all), freq = review_term_freq_all)

# Create comparison cloud
comparison.cloud(all_m,
                 colors = c("tomato", "deepskyblue"),
                 max.words = 100)

#Make commonality cloud
commonality.cloud(all_m, random.order=FALSE, scale=c(3, .5),
                  colors = "black", max.words=100)


## Common words ##

# Identify terms shared by both documents
common_words <- subset(
  all_m,
  all_m[, 1] > 0 & all_m[, 2] > 0
)

# calc common words and difference
difference <- abs(common_words[, 1] - common_words[, 2])
common_words <- cbind(common_words, difference)
common_words <- common_words[order(common_words[, 3],
                                   decreasing = T), ]

top25_df <- data.frame(x = common_words[1:25, 1],
                       y = common_words[1:25, 2],
                       labels = rownames(common_words[1:25, ]))


# Make pyramid plot
pyramid.plot(top25_df$x, top25_df$y,
             labels = top25_df$labels, 
             main = "Words in Common",
             gap = 80,
             laxlab = NULL,
             raxlab = NULL, 
             unit = NULL,
             top.labels = c("Corpus1",
                            "Words",
                            "Corpus2")
)

#### Correspondence Analysis ####

#corpus1

rev1_rate5 <- reviews1 %>% select(4) %>% filter(reviews1$rating == 5)
rev1_rate4 <- reviews1 %>% select(4) %>% filter(reviews1$rating == 4)
rev1_rate3 <- reviews1 %>% select(4) %>% filter(reviews1$rating == 3)
rev1_rate2 <- reviews1 %>% select(4) %>% filter(reviews1$rating == 2)
rev1_rate1 <- reviews1 %>% select(4) %>% filter(reviews1$rating == 1)

all_rev1_rate5 <- paste(rev1_rate5, collapse = " ")
all_rev1_rate4 <- paste(rev1_rate4, collapse = " ")
all_rev1_rate3 <- paste(rev1_rate3, collapse = " ")
all_rev1_rate2 <- paste(rev1_rate2, collapse = " ")
all_rev1_rate1 <- paste(rev1_rate1, collapse = " ")

all_rate_rev1_combine <- c(all_rev1_rate5, all_rev1_rate4, all_rev1_rate3, all_rev1_rate2, all_rev1_rate1)
# Creating corpus for combination
corpus_rate_rev1_all <- Corpus(VectorSource(all_rate_rev1_combine)) %>% 
  dataPreProcessing() %>% 
  lemmatize()

corpus_rate_rev1_tdm <- TermDocumentMatrix(corpus_rate_rev1_all) %>% removeSparseTerms(0.99)
rate_rev1_m <- as.matrix(corpus_rate_rev1_tdm)
colnames(rate_rev1_m) <- c("cinque","quattro", "tre", "due", "uno")
dfrev1 <- data.frame(rate_rev1_m) 

# for better visualization drop the row less than 10
dfr1 <- dfrev1 %>%
  rownames_to_column('gene') %>%
  filter(dfrev1$cinque>20 | dfrev1$quattro>20 | dfrev1$tre>20 | dfrev1$due>20 | dfrev1$uno>20) %>%
  column_to_rownames('gene') %>% as.matrix()

rev1_ca_fit <- CA(dfr1)
print(rev1_ca_fit) # basic results
summary(rev1_ca_fit) # extended results
fviz_ca_biplot(rev1_ca_fit, select.row = list(contrib = 50)) # symmetric map
fviz_ca_biplot(rev1_ca_fit, map ="rowprincipal",
               arrow = c(FALSE, TRUE), select.row = list(contrib = 50)) # asymmetric map


# corpus2

rev2_rate5 <- reviews2 %>% select(4) %>% filter(reviews2$rating == 5)
rev2_rate4 <- reviews2 %>% select(4) %>% filter(reviews2$rating == 4)
rev2_rate3 <- reviews2 %>% select(4) %>% filter(reviews2$rating == 3)
rev2_rate2 <- reviews2 %>% select(4) %>% filter(reviews2$rating == 2)
rev2_rate1 <- reviews2 %>% select(4) %>% filter(reviews2$rating == 1)

all_rev2_rate5 <- paste(rev2_rate5, collapse = " ")
all_rev2_rate4 <- paste(rev2_rate4, collapse = " ")
all_rev2_rate3 <- paste(rev2_rate3, collapse = " ")
all_rev2_rate2 <- paste(rev2_rate2, collapse = " ")
all_rev2_rate1 <- paste(rev2_rate1, collapse = " ")

all_rate_rev2_combine <- c(all_rev2_rate5, all_rev2_rate4, all_rev2_rate3, all_rev2_rate2, all_rev2_rate1)
# Creating corpus for combination
corpus_rate_rev2_all <- Corpus(VectorSource(all_rate_rev2_combine)) %>% 
  dataPreProcessing() %>% 
  lemmatize()

corpus_rate_rev2_tdm <- TermDocumentMatrix(corpus_rate_rev2_all) %>% removeSparseTerms(0.99)
rate_rev2_m <- as.matrix(corpus_rate_rev2_tdm)
colnames(rate_rev2_m) <- c("cinque","quattro", "tre", "due", "uno")
dfrev2 <- data.frame(rate_rev2_m) 
dfr2 <- dfrev2 %>%
  rownames_to_column('gene') %>%
  filter(dfrev2$cinque>20 | dfrev2$quattro>20 | dfrev2$tre>20 | dfrev2$due>20 | dfrev2$uno>20) %>%
  column_to_rownames('gene') %>% as.matrix()

rev2_ca_fit <- CA(dfr2)
print(rev2_ca_fit) # basic results
summary(rev2_ca_fit) # extended results
fviz_ca_biplot(rev2_ca_fit, select.row = list(contrib = 50)) # symmetric map
fviz_ca_biplot(rev2_ca_fit, map ="rowprincipal",
               arrow = c(FALSE, TRUE), select.row = list(contrib = 50)) # asymmetric map


#### Clustering ####

all_rev <- unique(rbind(reviews1, reviews2)) 
all_rev_corpus <- Corpus(VectorSource(all_rev$review)) %>% 
  dataPreProcessing() %>% lemmatize()

ndocs3 <-length(all_rev_corpus)
# ignore extremely rare words i.e. terms that appear in less then 1% of the documents
minTermFreq3 <- ndocs3 * 0.01
# ignore overly common words i.e. terms that appear in more than 90% of the documents
maxTermFreq3 <- ndocs3 * .90
all_rev_corpus_tdm <- TermDocumentMatrix(all_rev_corpus, 
                                         control = list(
                                           wordLengths=c(3, 20),
                                           bounds = list(global = c(minTermFreq3, maxTermFreq3),
                                                         weighting=weightTfIdf)))

# corpus2_tdm <- removeSparseTerms(corpus2_tdm, 0.999)
all_rev_corpus_m <- as.matrix(all_rev_corpus_tdm)
all_freq_m <- sort(rowSums(all_rev_corpus_m), decreasing = TRUE)
df_all <- data.frame(word = names(all_freq_m), freq = all_freq_m)

# Cosine distance matrix 
distMatrix = dist(all_rev_corpus_m, method = "cosine")


## k-means ##
set.seed(123)

#kmeans – determine the optimum number of clusters (elbow method)
#look for “elbow” in plot of summed intra-cluster distances (withinss) as fn of k

#Elbow Method for finding the optimal number of clusters

# Compute and plot wss for k = 2 to k = 8.
k.max <- 8
wss <- sapply(1:k.max, 
              function(k){kmeans(all_rev_corpus_m, k, nstart=50,iter.max = 15 )$tot.withinss})

plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")
abline(v=5, col="blue", lty=2)


clustering.kmeans <- eclust(all_rev_corpus_m, "kmeans", k = 5, nstart = 50)

## hierarchical ##

clustering.hierarchical <- hcut(distMatrix, k = 5, stand = TRUE, hc_method="ward.D2")

fviz_dend(clustering.hierarchical, rect = TRUE, cex = 0.5,
          k_colors = c("#00AFBB", "#EB8C21", "#2E9FDF", "#E7B800", "#FC4E07"))

## DBscan ##

dbscan::kNNdistplot(distMatrix, k =  5)
abline(h = 0.8, lty = 2)
clustering.dbscan <- dbscan::dbscan(distMatrix, eps = 0.8, minPts = 5)
fviz_cluster(clustering.dbscan, data = all_rev_corpus_m, stand = FALSE,
             ellipse = FALSE, show.clust.cent = FALSE,
             geom = "point",palette = "jco", ggtheme = theme_classic())


# kmeans vs hierarchical
kmeans_m <- as.matrix(clustering.kmeans$cluster)
df.kmeans <- data.frame(kmeans_m)
colnames(df.kmeans) <- c("cluster")
km_cluster <- rbind(head(subset(df.kmeans, cluster==1), 10),head(subset(df.kmeans, cluster==2), 10),head(subset(df.kmeans, cluster==3), 10),
                    head(subset(df.kmeans, cluster==4), 10),head(subset(df.kmeans, cluster==5), 10))
km_cluster

hier_m <- as.matrix(clustering.hierarchical$cluster)
df.hier <- data.frame(hier_m)
colnames(df.hier) <- c("cluster")
hc_cluster <- rbind(head(subset(df.hier, cluster==1), 10),head(subset(df.hier, cluster==2), 10),head(subset(df.hier, cluster==3), 10),
                    head(subset(df.hier, cluster==4), 10), head(subset(df.hier, cluster==5), 10))
hc_cluster

#### Other ####

## Bigram and Tigram ##

ngram_corpus <- Corpus(VectorSource(all_rev$review)) %>% dataPreProcessing()

ngram_txt <- c()
for(i in 1:length(ngram_corpus)){
  ngram_txt[i] <- ngram_corpus[[i]]$content
}

ngram_corpus2 <- VCorpus(VectorSource(ngram_txt))


BigramTokenizer <- function(x)
  NGramTokenizer(x, Weka_control(min = 2, max = 2))

tdm.bigram = TermDocumentMatrix(ngram_corpus2,
                                control = list(tokenize = BigramTokenizer)) %>% removeSparseTerms(0.99)

freq = sort(rowSums(as.matrix(tdm.bigram)),decreasing = TRUE)
freq.df = data.frame(word=names(freq), freq=freq, stringsAsFactors = FALSE)
head(freq.df, 20)

# Bigram wordcloud
set.seed(123)
wordcloud(words=freq.df$word, freq=freq.df$freq, max.words = 20, scale=c(3,.2), 
          random.order = FALSE, rot.per=.5, random.color = FALSE, colors= c("deepskyblue1","deepskyblue2","deepskyblue3","deepskyblue4"))


TrigramTokenizer <- function(x)
  NGramTokenizer(x, Weka_control(min = 3, max = 3))

tdm.trigram = TermDocumentMatrix(ngram_corpus2,
                                 control = list(tokenize = TrigramTokenizer)) %>% removeSparseTerms(0.998)

freq2 = sort(rowSums(as.matrix(tdm.trigram)),decreasing = TRUE)
freq2.df = data.frame(word=names(freq2), freq=freq2, stringsAsFactors = FALSE)
head(freq2.df, 20)

# Trigram wordcloud
wordcloud(words=freq2.df$word, freq=freq2.df$freq, max.words = 20, scale=c(2,.2), 
          random.order = FALSE, rot.per=.5, random.color = FALSE, colors= c("deepskyblue1","deepskyblue2","deepskyblue3","deepskyblue4"))

# wordmap bigram
b <- tibble(txt = ngram_txt)
trip_bigrams <- b %>%
  unnest_tokens(bigram, txt, token = "ngrams", n = 2) %>%
  count(bigram, sort = T) %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(n>8)

visualize_bigrams <- function(bigrams) {
  set.seed(2016)
  a <- grid::arrow(type = "closed", length = unit(.15, "inches"))
  
  bigrams %>%
    graph_from_data_frame() %>%
    ggraph(layout = "fr") +
    geom_edge_link(aes(edge_alpha = n), show.legend = FALSE, arrow = a) +
    geom_node_point(color = "cyan3", size = 5) +
    geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
    theme_void() +
    ggtitle('Word network in TripAdvisor reviews')
}

visualize_bigrams(trip_bigrams)

# wordmap trigram
t <- tibble(txt = ngram_txt)
trip_trigrams <- t %>%
  unnest_tokens(trigram, txt, token = "ngrams", n = 3) %>%
  count(trigram, sort = T) %>%
  separate(trigram, c("word1", "word2", "word3"), sep = " ") %>%

visualize_trigrams <- function(trigrams) {
  set.seed(123)
  a <- grid::arrow(type = "closed", length = unit(.15, "inches"))
  
  trigrams %>%
    graph_from_data_frame() %>%
    ggraph(layout = "fr") +
    geom_edge_link(aes(edge_alpha = n), show.legend = FALSE, arrow = a) +
    geom_node_point(color = "cyan3", size = 5) +
    geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
    theme_void() +
    ggtitle('Word network in TripAdvisor reviews')
}

visualize_trigrams(trip_trigrams)

spacy_finalize()

## Trends in words ##

library(readr)
library(stringr)
library(scales)
library(broom)
library(purrr)
library(widyr)
library(reshape2)

min(all_rev$date); max(all_rev$date)

all_rev %>%
  count(date = round_date(all_rev$date, "week")) %>%
  ggplot(aes(date, n)) +
  geom_line(color="blue") + 
  ggtitle('Andamento del Numero di Recensioni da Maggio 2013 a Febbraio 2020')

all_rev <- tibble::rowid_to_column(all_rev, "ID")
all_rev <- all_rev %>%
  mutate(review_date = as.POSIXct(all_rev$date, origin = "1970-01-01"), month = round_date(all_rev$date, "month"))
 
all_rev$review <- gsub("[[:punct:]]+"," ", all_rev$review)
all_rev$review <- gsub("[€”“’‘]","", all_rev$review)
all_rev$review <- gsub("[0-9]","", all_rev$review)
all_rev$review <- gsub("[ \t]{2,}"," ", all_rev$review)
all_rev$review <- gsub("^\\s+|\\s+$"," ", all_rev$review)
all_rev$review <- tolower(all_rev$review)


review_words <- all_rev %>%
  distinct(review, .keep_all = TRUE) %>%
  unnest_tokens(word, review, drop = FALSE) %>%
  distinct(ID, word, .keep_all = TRUE) %>%
  anti_join(stop_words, by = "word") %>%
  filter(str_detect(word, "[^\\d]")) %>%
  group_by(word) %>%
  mutate(word_total = n()) %>%
  ungroup()

reviews_per_month <- all_rev %>%
  group_by(month) %>%
  summarize(month_total = n())

word_month_counts <- review_words %>%
  filter(word_total >= 200) %>%
  count(word, month) %>%
  complete(word, month, fill = list(n = 0)) %>%
  inner_join(reviews_per_month, by = "month") %>%
  mutate(percent = n / month_total) %>%
  mutate(year = year(month) + yday(month) / 365)

library(broom)

mod <- ~ glm(cbind(n, month_total - n) ~ year, ., family = "binomial")
slopes <- word_month_counts %>%
  nest(-word) %>%
  mutate(model = map(data, mod)) %>%
  unnest(map(model, tidy)) %>%
  filter(term == "year") %>%
  arrange(desc(estimate))

# growing trend
slopes %>%
  head(9) %>%
  inner_join(word_month_counts, by = "word") %>%
  mutate(word = reorder(word, -estimate)) %>%
  ggplot(aes(month, n / month_total, color = word)) +
  geom_line(show.legend = FALSE) +
  scale_y_continuous(labels = percent_format()) +
  facet_wrap(~ word, scales = "free_y") +
  expand_limits(y = 0) +
  labs(x = "Year",
       y = "% di reviews che contengono questo termine",
       title = "9 parole con trend di crescita veloce nelle recensioni",
       subtitle = "Calcolato su un tasso di crescita di 7 anni")

# shrinking trend	   
slopes %>%
  tail(9) %>%
  inner_join(word_month_counts, by = "word") %>%
  mutate(word = reorder(word, estimate)) %>%
  ggplot(aes(month, n / month_total, color = word)) +
  geom_line(show.legend = FALSE) +
  scale_y_continuous(labels = percent_format()) +
  facet_wrap(~ word, scales = "free_y") +
  expand_limits(y = 0) +
  labs(x = "Year",
       y = "% di reviews che contengono questo termine",
       title = "9 parole con trend di crescita lento nelle recensioni",
       subtitle = "Calcolato su un tasso di crescita di 7 anni")

# Comparison between two words' trends	   
word_month_counts %>%
  filter(word %in% c("antipasti", "pesce")) %>%
  ggplot(aes(month, n / month_total, color = word)) +
  geom_line(size = 1, alpha = .8) +
  scale_y_continuous(labels = percent_format()) +
  expand_limits(y = 0) +
  labs(x = "Year",
       y = "% di reviews che contengono questo termine", title = "Antipasti vs Pesce in termini di interesse dei recensori")