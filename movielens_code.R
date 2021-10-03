#title: "HarvardX Data Science Capstone Project _ MovieLens"
#author: "Muditha Hapudeniya"
#date: "01/10/2021"

# Heading numbers are corresponding with the heading numbers in the report
###=====================================================###
#### 2.1.1. Installing Packpage

if(!require(tidyverse)) install.packages("tidyverse",
                                         repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret",
                                     repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table",
                                          repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

###=====================================================###
#### 2.1.2. Downloading the data set and Creating Train Set and Validation Set


dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

###=====================================================###
### 2.2 Exploring the Training Dataset

#### 2.2.1 Structure and basic information


nrow<-dim(edx)[1]
ncol<-dim(edx)[2]
nmovie<-n_distinct(edx$movieId)
nuser<-n_distinct(edx$userId)

tibble(Summary = c("Number of Row in Dataset",
                   "Number of Column in Dataset",
                   "Number of Movies",
                   "Number of Users"),
       Count = c(nrow,ncol,nmovie,nuser))%>% knitr::kable(caption = "Summary")

###=====================================================###
# The data set has the following column headings.


colnames(edx)


###=====================================================###
## Missing values in the data set


sum(is.na(edx))

###=====================================================###
#### 2.2.2. Top 10 user rated the movies


edx %>% group_by(userId) %>% summarise(Number_of_ratings = n()) %>%
  arrange(desc(Number_of_ratings)) %>%
  head(10) %>% knitr::kable(caption = "Top 10 users who had given most ratings")

###=====================================================###
#### 2.2.3. Top rated movies


edx %>% group_by(movieId) %>% summarise(Number_of_ratings = n()) %>%
  arrange(desc(Number_of_ratings)) %>%
  head(10)%>% knitr::kable(caption = "Top 10 movies which had most ratings")

###=====================================================###
#### 2.2.4. Distribution of rating

edx %>%
	group_by(rating) %>%
	summarize(count = n()) %>%
	ggplot(aes(x = rating, y = count)) +
	geom_line() +
  ggtitle("Figure1: Distribution of Counting Per Rating") +
  xlab("Rating") +
  ylab("Count") +
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE))


###=====================================================###
#### 2.2.5. Distribution of average rating per movie


edx %>% group_by(movieId) %>% summarise(avg_rating = mean(rating)) %>%
  ggplot(aes(avg_rating)) +
  geom_histogram(aes(y=..density..), colour="black", fill="white")  +
  geom_density(alpha=.2, fill="#FF6666") +
  geom_vline(aes(xintercept=mean(avg_rating)),
            color="blue", linetype="dashed", size=1)
  ggtitle("Figure 1: Distribution of average ratings for movies")


###=====================================================###
#### 2.2.6. Distribution of average rating for users

edx %>% group_by(userId) %>% summarise(avg_rating = mean(rating)) %>%
  ggplot(aes(avg_rating)) +
  geom_histogram(aes(y=..density..), colour="black", fill="white")  +
  geom_density(alpha=.2, fill="#a4c496") +
  geom_vline(aes(xintercept=mean(avg_rating)),
            color="blue", linetype="dashed", size=1)
  ggtitle("Figure 2: Distribution of average ratings for users")


###=====================================================###
  #### 2.3.1. Root Mean Squared Error


  RMSE <- function(true_rating, predicted_rating){
    sqrt(mean((true_rating - predicted_rating)^2))}

###=====================================================###
#### 2.3.2. Model by Considering average only


# calculate the mean value for all the movie ratings
mu_hat <- mean(edx$rating)
mu_hat

naive_rmse <- RMSE(validation$rating, mu_hat)

# save this value for future comparisons
rmse_results <- data.frame("Predictive Method" = "Using the Global Average",
                           RMSE = naive_rmse)
rmse_results %>%
  knitr::kable(caption = "RMSE for models ")

###=====================================================###

#### 2.3.3. Modeling Movie Effects



movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_m = mean(rating - mu_hat))

predicted_rating <- mu_hat + validation %>%
  left_join(movie_avgs, by='movieId') %>%
  .$b_m

model_1_rmse <- RMSE(predicted_rating, validation$rating)

# save RMSE for future reference

rmse_results <- bind_rows(rmse_results,
                          data.frame("Predictive Method"="Adding Movie Effect to the Model",
                                     RMSE = model_1_rmse ))
rmse_results %>% knitr::kable(caption="RMSE for models" )

###=====================================================###

#### 2.3.4. Modeling by adding the User Effects


user_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_m))

predicted_rating <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu_hat + b_m + b_u) %>%
  .$pred

model_2_rmse <- RMSE(predicted_rating, validation$rating)

# save the result
rmse_results <- bind_rows(rmse_results,
                          data.frame("Predictive Method"="Add Movie + User bias to the Model",
                                     RMSE = model_2_rmse ))

rmse_results %>% knitr::kable(caption="RMSE for models")


###=====================================================###

#### 2.3.5. Regularization for movie and User Effect in the model


# define lambda
lambdas <- seq(0, 10, 0.25)

# calculate RMSE for each lambda
rmses <- sapply(lambdas, function(l){
# calculate the global mean
  mu_hat <- mean(edx$rating)

  # calculate the movie bias
  b_m <- edx %>%
    group_by(movieId) %>%
    summarize(b_m = sum(rating - mu_hat)/(n()+l))

  # calculate the user bias
  b_u <- edx %>%
    left_join(b_m, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_m - mu_hat)/(n()+l))

  # predict the rating by adding the movie and user bias
  predicted_rating <-
    validation %>%
    left_join(b_m, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu_hat + b_m + b_u) %>%
    .$pred

    return(RMSE(predicted_rating, validation$rating))
})

# Generate the plot to visualize the RMSE for each lambda value
df <- data.frame(lambdas,rmses)
ggplot(df,aes(lambdas,rmses)) +
  geom_point(color="blue") +
  ggtitle("Figure 3: RMSE by lambda")

lambdas[which.min(rmses)]


###=====================================================###

## Calculate the final RMSE by the lambda value (5.25)
lambda <- 5.25

movie_reg_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_m2 = sum(rating - mu_hat)/(n()+lambda), n_i = n())

user_reg_avgs <- edx %>%
  left_join(movie_reg_avgs, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u2 = sum(rating - mu_hat-b_m2)/(n()+lambda), n_i = n())

predicted_rating <- validation %>%
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(user_reg_avgs, by='userId') %>%
  mutate(pred = mu_hat + b_m2 + b_u2) %>%
  .$pred

model_3_rmse <- RMSE(predicted_rating, validation$rating)

rmse_results <- bind_rows(rmse_results,
data.frame("Predictive Method"="Add Regularized Movie +
                                     Regularized User Effects to the Model",
                                     RMSE = model_3_rmse ))

rmse_results %>% knitr::kable(caption="RMSE for models")


###=====================================================###
## 3. Results


data_frame("Final Result" ="Regularized Movie +
           Regularized User Effects Model",RMSE = model_3_rmse ) %>%
  knitr::kable(caption="Final RMSE value achived")

