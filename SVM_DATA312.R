#CLEAR
rm(list=ls())
gc()
# Load required libraries
library(tidyverse)
library(kernlab)
library(e1071)
library(Ecdat)

# 1. Load and prepare the dataset
df <- Ecdat::UnempDur
df1 <- subset(df, select = -ui)

# 2. Split the dataset into training, query, and test samples
raw_data_samplingframe <- df1 %>%
  mutate(snum = sample.int(n(), n()) / n())

# Create training set (60% of data)
training <- raw_data_samplingframe %>%
  filter(snum < 0.6) %>%
  select(-snum)

# Create query set (20% of data)
query <- raw_data_samplingframe %>%
  filter(snum >= 0.6, snum < 0.8) %>%
  select(-snum)

# Create test set (20% of data)
test <- raw_data_samplingframe %>%
  filter(snum >= 0.8) %>%
  select(-snum)

# Save datasets to CSV files
training %>% write_csv('4991024_trainingf3.csv')
query %>% write_csv('4991024_queryf3.csv')
test %>% write_csv('4991024_testf3.csv')

# 3. Train SVMs on the training data
# Explanatory variables: logwage, tenure, disrate, reprate
# Response variable: censor1

# Prepare input and truth data for each set
training_input <- training %>% select(-censor1)
training_truth <- training$censor1 %>% as.factor()

query_input <- query %>% select(-censor1)
query_truth <- query$censor1 %>% as.factor()

test_input <- test %>% select(-censor1)
test_truth <- test$censor1 %>% as.factor()

# Perform PCA on training data
spam_pca <- training_input %>% prcomp()

# Visualize PCA results
spam_pca$x %>%
  as_tibble() %>%
  mutate(type = training_truth) %>%
  ggplot(aes(PC1, PC2, color = type)) +
  geom_point()

# Perform k-means clustering
spam_kmeans <- training_input %>% kmeans(2)

# Analyze k-means results
kmeans_results <- tibble(training_truth, km = spam_kmeans$cluster)
kmeans_results %>%
  count(training_truth, km) %>%
  pivot_wider(names_from = km, values_from = n)

# Perform chi-square test on k-means results
ct <- kmeans_results %>%
  count(training_truth, km) %>%
  pivot_wider(names_from = km, values_from = n) %>%
  column_to_rownames('training_truth')
chisq.test(ct)

# Visualize k-means clustering results
spam_pca$x %>%
  as_tibble() %>%
  ggplot(aes(PC1, PC2, color = factor(spam_kmeans$cluster))) +
  geom_point()

# Train SVM models with different kernels
svm_linear <- svm(x = training_input, y = training_truth, kernel = 'linear')
svm_polynomial <- svm(x = training_input, y = training_truth, kernel = 'polynomial')
svm_radial <- svm(x = training_input, y = training_truth, kernel = 'radial')
svm_sigmoid <- svm(x = training_input, y = training_truth, kernel = 'sigmoid')

# Train SVM on PCA results
training_pca <- spam_pca$x %>% as_tibble() %>% mutate(type = training_truth)
sl <- svm(type ~ ., data = training_pca, kernel = 'linear')

# Visualize SVM decision boundary
plot(sl, training_pca, PC1 ~ PC2)

# 4. Evaluate SVM performance using the query set
query_results <- tibble(
  query_truth,
  linear = predict(svm_linear, query_input),
  poly = predict(svm_polynomial, query_input),
  radial = predict(svm_radial, query_input),
  sigmoid = predict(svm_sigmoid, query_input)
)

# Calculate performance metrics for each model
query_results1 <- query_results %>% pivot_longer(cols = !query_truth)
query_results2 <- query_results1 %>%
  mutate(
    tp = (query_truth == 0 & value == 0),
    tn = (query_truth == 1 & value == 1),
    fp = (query_truth == 0 & value == 1),
    fn = (query_truth == 1 & value == 0)
  )

query_results3 <- query_results2 %>%
  group_by(name) %>%
  summarize(
    tp = sum(tp),
    tn = sum(tn),
    fp = sum(fp),
    fn = sum(fn)
  )

# Calculate and display performance metrics
query_results3 %>%
  mutate(
    accuracy = (tp + tn) / (tp + tn + fp + fn),
    sensitivity = tp / (tp + fn),
    specificity = tn / (tn + fp),
    ppv = tp / (tp + fp),
    npv = fn / (tn + fn),
    f1 = 2 * tp / (2 * tp + fp + fn)
  )

# 5. Evaluate the best performing model (linear SVM) on the test set
test_results <- tibble(test_truth, linear = predict(svm_linear, test_input))

# Calculate performance metrics for the test set
test_results1 <- test_results %>% pivot_longer(cols = !test_truth)
test_results2 <- test_results1 %>%
  mutate(
    tp = (test_truth == 0 & value == 0),
    tn = (test_truth == 1 & value == 1),
    fp = (test_truth == 0 & value == 1),
    fn = (test_truth == 1 & value == 0)
  )

test_results3 <- test_results2 %>%
  group_by(name) %>%
  summarize(
    tp = sum(tp),
    tn = sum(tn),
    fp = sum(fp),
    fn = sum(fn)
  )

# Calculate and display performance metrics for the test set
test_results3 %>%
  mutate(
    accuracy = (tp + tn) / (tp + tn + fp + fn),
    sensitivity = tp / (tp + fn),
    specificity = tn / (tn + fp),
    ppv = tp / (tp + fp),
    npv = fn / (tn + fn),
    f1 = 2 * tp / (2 * tp + fp + fn)
  )
