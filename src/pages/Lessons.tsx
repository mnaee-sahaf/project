import React, { useState } from 'react';
import { Search, BookOpen } from 'lucide-react';
import { LessonCard } from '../components/LessonCard';
import { LessonContent } from '../components/LessonContent';
import { Quiz } from '../components/Quiz';
import type { Lesson } from '../types';

const allLessons: Lesson[] = [
  {
    id: '1',
    title: 'Introduction to Machine Learning',
    description: 'Learn the fundamental concepts of machine learning and its applications.',
    category: 'basics',
    difficulty: 'beginner',
    completed: false,
    content: `
      <h2>What is Machine Learning?</h2>
      <p>Machine Learning is a subset of artificial intelligence that focuses on developing systems that can learn and improve from experience without being explicitly programmed.</p>
      
      <h2>Key Concepts</h2>
      <ul>
        <li><strong>Supervised Learning:</strong> The algorithm learns from labeled training data</li>
        <li><strong>Unsupervised Learning:</strong> The algorithm finds patterns in unlabeled data</li>
        <li><strong>Reinforcement Learning:</strong> The algorithm learns through trial and error</li>
      </ul>

      <h2>Applications</h2>
      <p>Machine learning is used in various fields:</p>
      <ul>
        <li>Image and Speech Recognition</li>
        <li>Natural Language Processing</li>
        <li>Recommendation Systems</li>
        <li>Autonomous Vehicles</li>
        <li>Medical Diagnosis</li>
      </ul>
    `,
    quiz: {
      id: 'q1',
      questions: [
        {
          id: 'q1',
          text: 'What is the primary goal of supervised learning?',
          options: [
            'To discover hidden patterns in unlabeled data',
            'To predict outcomes using labeled training data',
            'To reduce computational complexity',
            'To optimize resource allocation'
          ],
          correctAnswer: 1,
          explanation: 'Supervised learning uses labeled data to train models to predict outcomes for unseen data.'
        },
        {
          id: 'q2',
          text: 'Which algorithm is commonly used for regression tasks?',
          options: [
            'K-Means',
            'Decision Tree',
            'Linear Regression',
            'Support Vector Machine (SVM)'
          ],
          correctAnswer: 2,
          explanation: 'Linear Regression is a foundational algorithm for predicting continuous numerical outputs.'
        },
        {
          id: 'q3',
          text: 'What does the term "overfitting" mean?',
          options: [
            'The model performs well on training data but poorly on unseen data',
            'The model is too simple to capture patterns',
            'The model has high bias',
            'The model is optimized for speed'
          ],
          correctAnswer: 0,
          explanation: 'Overfitting occurs when a model memorizes training data noise, leading to poor generalization.'
        },
        {
          id: 'q4',
          text: 'Which Python library is primarily used for machine learning?',
          options: [
            'Matplotlib',
            'Pandas',
            'Scikit-learn',
            'TensorFlow'
          ],
          correctAnswer: 2,
          explanation: 'Scikit-learn provides tools for preprocessing, model training, and evaluation.'
        },
        {
          id: 'q5',
          text: 'What is the purpose of a training-test split?',
          options: [
            'To reduce dataset size',
            'To evaluate model performance on unseen data',
            'To speed up training',
            'To balance class labels'
          ],
          correctAnswer: 1,
          explanation: 'Splitting data ensures the model is tested on data it hasn’t seen during training.'
        },
        {
          id: 'q6',
          text: 'Which evaluation metric is used for classification tasks?',
          options: [
            'Mean Absolute Error (MAE)',
            'R-squared',
            'Accuracy',
            'Euclidean Distance'
          ],
          correctAnswer: 2,
          explanation: 'Accuracy measures the percentage of correct predictions in classification.'
        },
        {
          id: 'q7',
          text: 'What does KNN stand for?',
          options: [
            'Kernel Neural Network',
            'K-Nearest Neighbors',
            'K-Means Network',
            'Key Normalization Node'
          ],
          correctAnswer: 1,
          explanation: 'K-Nearest Neighbors classifies data points based on the majority class of their nearest neighbors.'
        },
        {
          id: 'q8',
          text: 'Which activation function is commonly used in neural network output layers for binary classification?',
          options: [
            'ReLU',
            'Sigmoid',
            'Tanh',
            'Linear'
          ],
          correctAnswer: 1,
          explanation: 'Sigmoid outputs probabilities between 0 and 1, ideal for binary classification.'
        },
        {
          id: 'q9',
          text: 'What is the role of a loss function?',
          options: [
            'To measure model performance during training',
            'To preprocess data',
            'To visualize decision boundaries',
            'To select hyperparameters'
          ],
          correctAnswer: 0,
          explanation: 'Loss functions quantify the error between predictions and actual values.'
        },
        {
          id: 'q10',
          text: 'Which code snippet correctly splits data into training and test sets?',
          options: [
            'train, test = split(data, test_size=0.2)',
            'X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)',
            'train_test_split(X, y, shuffle=True)',
            'split(data, ratio=0.7)'
          ],
          correctAnswer: 1,
          explanation: 'scikit-learn’s `train_test_split` returns four arrays: features and labels for training/testing.'
        },
        {
          id: 'q11',
          text: 'What is regularization used for?',
          options: [
            'To increase model complexity',
            'To reduce overfitting by penalizing large weights',
            'To speed up training',
            'To handle missing data'
          ],
          correctAnswer: 1,
          explanation: 'Regularization (e.g., L1/L2) discourages overly complex models by adding penalty terms to the loss.'
        },
        {
          id: 'q12',
          text: 'Which algorithm creates splits based on information gain?',
          options: [
            'Linear Regression',
            'Decision Tree',
            'K-Means',
            'Logistic Regression'
          ],
          correctAnswer: 1,
          explanation: 'Decision Trees use metrics like information gain or Gini impurity to split nodes.'
        },
        {
          id: 'q13',
          text: 'What does the following code do?\n`from sklearn.preprocessing import StandardScaler`',
          options: [
            'Imports a decision tree classifier',
            'Imports a tool to normalize data',
            'Imports a hyperparameter tuning library',
            'Imports a visualization library'
          ],
          correctAnswer: 1,
          explanation: 'StandardScaler standardizes features by removing the mean and scaling to unit variance.'
        },
        {
          id: 'q14',
          text: 'What is a "feature" in machine learning?',
          options: [
            'A performance metric',
            'An input variable used for making predictions',
            'A type of neural network layer',
            'A data visualization technique'
          ],
          correctAnswer: 1,
          explanation: 'Features are individual measurable properties of the data (e.g., age, height).'
        },
        {
          id: 'q15',
          text: 'Which of the following is a clustering algorithm?',
          options: [
            'Random Forest',
            'K-Means',
            'Linear Regression',
            'SVM'
          ],
          correctAnswer: 1,
          explanation: 'K-Means groups data points into clusters based on similarity.'
        },
        {
          id: 'q16',
          text: 'What is the output layer size for a multiclass classification problem with 5 classes?',
          options: [
            '1 neuron',
            '5 neurons',
            '10 neurons',
            'Equal to the number of features'
          ],
          correctAnswer: 1,
          explanation: 'In multiclass classification, the output layer typically has one neuron per class.'
        },
        {
          id: 'q17',
          text: 'Which code trains a logistic regression model?',
          options: [
            'model.fit(X_train, y_train)',
            'model = LogisticRegression().train(X, y)',
            'model = LogisticRegression().fit(X_train, y_train)',
            'model.predict(X_test)'
          ],
          correctAnswer: 2,
          explanation: 'scikit-learn uses the `fit()` method to train models.'
        },
        {
          id: 'q18',
          text: 'What is the purpose of a confusion matrix?',
          options: [
            'To visualize model architecture',
            'To evaluate classification performance',
            'To preprocess text data',
            'To reduce dimensionality'
          ],
          correctAnswer: 1,
          explanation: 'A confusion matrix shows true vs. predicted labels (TP, TN, FP, FN).'
        },
        {
          id: 'q19',
          text: 'What is the default value of `k` in KNN?',
          options: [
            '1',
            '3',
            '5',
            'Defined by the user'
          ],
          correctAnswer: 3,
          explanation: 'In scikit-learn, `n_neighbors` defaults to 5, but it must be explicitly set by the user.'
        },
        {
          id: 'q20',
          text: 'What is cross-validation used for?',
          options: [
            'To increase training speed',
            'To reduce model bias',
            'To evaluate model generalizability',
            'To preprocess images'
          ],
          correctAnswer: 2,
          explanation: 'Cross-validation splits data into multiple folds to assess performance across different subsets.'
        },
        {
          id: 'q21',
          text: 'What does the following code compute?\n`np.mean(np.abs(y_true - y_pred))`',
          options: [
            'Mean Squared Error (MSE)',
            'R-squared',
            'Mean Absolute Error (MAE)',
            'Accuracy'
          ],
          correctAnswer: 2,
          explanation: 'MAE calculates the average absolute difference between predictions and true values.'
        },
        {
          id: 'q22',
          text: 'Which hyperparameter controls tree depth in a Decision Tree?',
          options: [
            'min_samples_split',
            'max_depth',
            'n_estimators',
            'learning_rate'
          ],
          correctAnswer: 1,
          explanation: '`max_depth` limits how deep the tree can grow to prevent overfitting.'
        },
        {
          id: 'q23',
          text: 'What is the output of this code?\n`print(confusion_matrix([1,0,1], [1,1,0]))`',
          options: [
            '[[1 1], [1 0]]',
            '[[1 0], [1 1]]',
            '[[1 1], [0 1]]',
            '[[1 0], [0 1]]'
          ],
          correctAnswer: 0,
          explanation: 'Row 1: True Positives (1) and False Positives (1). Row 2: False Negatives (1) and True Negatives (0).'
        },
        {
          id: 'q24',
          text: 'Which gradient descent update rule is correct?',
          options: [
            'weights = weights - learning_rate * gradient',
            'weights = weights + learning_rate * gradient',
            'weights = gradient - learning_rate * weights',
            'weights = learning_rate * gradient'
          ],
          correctAnswer: 0,
          explanation: 'Weights are updated by subtracting the gradient scaled by the learning rate.'
        },
        {
          id: 'q25',
          text: 'What is the primary use of PCA?',
          options: [
            'Classification',
            'Dimensionality reduction',
            'Clustering',
            'Regularization'
          ],
          correctAnswer: 1,
          explanation: 'Principal Component Analysis (PCA) reduces feature space while preserving variance.'
        }
      ]
    }
  },
  {
    id: '2',
    title: 'Linear Regression Fundamentals',
    description: 'Understand how linear regression works and its practical applications.',
    category: 'regression',
    difficulty: 'beginner',
    completed: false,
    content: `
      <h2>Understanding Linear Regression</h2>
      <p>Linear regression is a fundamental algorithm in machine learning used to predict a continuous output value based on one or more input features.</p>
      <p>Structure & Difficulty:

      Foundational: Questions 1 to 10 cover theory (e.g., bias term, residuals, assumptions).

      Intermediate: Questions 11 to 20 involve code snippets, metrics (R²), and interpretation.

      Advanced: Questions 21 to 25 focus on regularization, multicollinearity, and real-world applications.

      Key Concepts Tested:

     Assumptions (linearity, homoscedasticity).

     Cost functions (MSE), gradient descent.

     Regularization (Ridge/Lasso), polynomial features.

     Code implementation (scikit-learn syntax).

     Interpretation of coefficients, p-values, and metrics.</p>

      <h2>Key Components</h2>
      <ul>
        <li><strong>Features (X):</strong> Input variables used for prediction</li>
        <li><strong>Target (y):</strong> The output variable we want to predict</li>
        <li><strong>Weights (w):</strong> Parameters that determine the line's slope</li>
        <li><strong>Bias (b):</strong> The y-intercept of the line</li>
      </ul>

      <h2>The Linear Equation</h2>
      <p>y = wx + b</p>
      <p>Where:</p>
      <ul>
        <li>y is the predicted output</li>
        <li>w is the weight (slope)</li>
        <li>x is the input feature</li>
        <li>b is the bias (y-intercept)</li>
      </ul>
    `,
    quiz: {
      id: 'q2',
      questions: [
        {
          id: 'q2_1',
          text: 'What is the purpose of the bias term in linear regression?',
          options: [
            'To make the model more complex',
            'To allow the line to intersect the y-axis at any point',
            'To reduce overfitting',
            'To increase the learning rate'
          ],
          correctAnswer: 1,
          explanation: 'The bias term (intercept) shifts the regression line vertically, enabling it to fit data not passing through the origin.'
        },
        {
          id: 'q2_2',
          text: 'Which of the following is true about linear regression?',
          options: [
            'It can only handle one input feature',
            'It always produces perfect predictions',
            'It assumes a linear relationship between features and target',
            'It can only be used for classification tasks'
          ],
          correctAnswer: 2,
          explanation: 'Linear regression assumes a linear relationship between independent variables and the dependent variable.'
        },
        {
          id: 'q2_3',
          text: 'What does the term "residual" refer to in linear regression?',
          options: [
            'The difference between predicted and actual values',
            'The slope of the regression line',
            'The bias term',
            'The learning rate'
          ],
          correctAnswer: 0,
          explanation: 'Residuals are the errors between the model’s predictions and the true target values.'
        },
        {
          id: 'q2_4',
          text: 'Which cost function is used in linear regression?',
          options: [
            'Cross-Entropy Loss',
            'Mean Absolute Error (MAE)',
            'Mean Squared Error (MSE)',
            'Hinge Loss'
          ],
          correctAnswer: 2,
          explanation: 'MSE (average of squared residuals) is the default loss function for linear regression.'
        },
        {
          id: 'q2_5',
          text: 'What is the role of gradient descent in linear regression?',
          options: [
            'To normalize features',
            'To minimize the cost function by updating weights',
            'To calculate p-values',
            'To visualize residuals'
          ],
          correctAnswer: 1,
          explanation: 'Gradient descent iteratively adjusts model parameters (weights) to minimize the loss function.'
        },
        {
          id: 'q2_6',
          text: 'Which code snippet correctly implements linear regression in scikit-learn?',
          options: [
            'model = LinearRegression().train(X, y)',
            'model = LinearRegression().predict(X)',
            'model = LinearRegression().fit(X_train, y_train)',
            'model = LinearRegression().score(X, y)'
          ],
          correctAnswer: 2,
          explanation: 'The `fit()` method trains the model on the training data.'
        },
        {
          id: 'q2_7',
          text: 'What does R-squared (R²) measure?',
          options: [
            'The magnitude of residuals',
            'The proportion of variance explained by the model',
            'The learning rate',
            'The number of features'
          ],
          correctAnswer: 1,
          explanation: 'R² quantifies how well the model explains the variability of the target variable.'
        },
        {
          id: 'q2_8',
          text: 'What is multicollinearity?',
          options: [
            'High correlation between features',
            'Low correlation between features and target',
            'A type of regularization',
            'Non-linear relationships in data'
          ],
          correctAnswer: 0,
          explanation: 'Multicollinearity occurs when features are highly correlated, destabilizing coefficient estimates.'
        },
        {
          id: 'q2_9',
          text: 'Which assumption is violated if residuals form a funnel shape?',
          options: [
            'Linearity',
            'Homoscedasticity',
            'Independence',
            'Normality'
          ],
          correctAnswer: 1,
          explanation: 'Heteroscedasticity (non-constant residual variance) invalidates homoscedasticity assumptions.'
        },
        {
          id: 'q2_10',
          text: 'What does this code compute?\n`np.sum((y_true - y_pred) ** 2)`',
          options: [
            'Mean Absolute Error (MAE)',
            'R-squared',
            'Total Sum of Squares (TSS)',
            'Sum of Squared Residuals (SSR)'
          ],
          correctAnswer: 3,
          explanation: 'SSR is the sum of squared differences between true and predicted values.'
        },
        {
          id: 'q2_11',
          text: 'What is Ridge Regression?',
          options: [
            'A classification algorithm',
            'Linear regression with L1 regularization',
            'Linear regression with L2 regularization',
            'A non-parametric method'
          ],
          correctAnswer: 2,
          explanation: 'Ridge adds an L2 penalty term to the loss function to prevent overfitting.'
        },
        {
          id: 'q2_12',
          text: 'Which code standardizes features before training?',
          options: [
            'StandardScaler().transform(X)',
            'MinMaxScaler().fit(X)',
            'Normalizer().fit_transform(X)',
            'RobustScaler().predict(X)'
          ],
          correctAnswer: 0,
          explanation: 'StandardScaler standardizes features by removing the mean and scaling to unit variance.'
        },
        {
          id: 'q2_13',
          text: 'What is the normal equation for linear regression?',
          options: [
            'θ = (XᵀX)⁻¹Xᵀy',
            'θ = Xᵀy',
            'θ = X⁻¹y',
            'θ = gradient_descent(X, y)'
          ],
          correctAnswer: 0,
          explanation: 'The normal equation is a closed-form solution for finding optimal weights without iteration.'
        },
        {
          id: 'q2_14',
          text: 'What is the effect of increasing the learning rate in gradient descent?',
          options: [
            'Slower convergence',
            'Faster convergence but risk of overshooting',
            'No effect',
            'Reduced computational cost'
          ],
          correctAnswer: 1,
          explanation: 'A high learning rate may cause the algorithm to diverge instead of converging.'
        },
        {
          id: 'q2_15',
          text: 'Which of the following is a limitation of linear regression?',
          options: [
            'Cannot handle categorical features',
            'Assumes linear relationships',
            'Requires GPU acceleration',
            'Only works with small datasets'
          ],
          correctAnswer: 1,
          explanation: 'Linear regression fails if relationships between features and target are non-linear.'
        },
        {
          id: 'q2_16',
          text: 'How do you interpret a coefficient β₁ = 2.5 in a linear regression model?',
          options: [
            'A 1-unit increase in X₁ decreases y by 2.5 units',
            'A 1-unit increase in X₁ increases y by 2.5 units',
            'X₁ has no effect on y',
            'The model’s R² is 2.5'
          ],
          correctAnswer: 1,
          explanation: 'Coefficients represent the change in y per unit change in the corresponding feature.'
        },
        {
          id: 'q2_17',
          text: 'Which code calculates R-squared?',
          options: [
            'r2_score(y_true, y_pred)',
            'mean_squared_error(y_true, y_pred)',
            'accuracy_score(y_true, y_pred)',
            'confusion_matrix(y_true, y_pred)'
          ],
          correctAnswer: 0,
          explanation: 'scikit-learn’s `r2_score` function computes the R² metric.'
        },
        {
          id: 'q2_18',
          text: 'What does a p-value > 0.05 for a coefficient imply?',
          options: [
            'The feature is statistically insignificant',
            'The feature is highly significant',
            'The model is overfit',
            'The data is homoscedastic'
          ],
          correctAnswer: 0,
          explanation: 'A high p-value suggests the feature’s effect may be due to chance.'
        },
        {
          id: 'q2_19',
          text: 'What is the dummy variable trap?',
          options: [
            'Perfect multicollinearity caused by one-hot encoding all categories',
            'Using too many features',
            'Ignoring outliers',
            'Including interaction terms'
          ],
          correctAnswer: 0,
          explanation: 'Including all dummy variables leads to redundancy (e.g., n categories encoded as n columns).'
        },
        {
          id: 'q2_20',
          text: 'Which code adds polynomial features to a linear regression model?',
          options: [
            'PolynomialFeatures(degree=2).fit_transform(X)',
            'StandardScaler().transform(X)',
            'PCA(n_components=2).fit(X)',
            'KMeans(n_clusters=2).fit(X)'
          ],
          correctAnswer: 0,
          explanation: 'PolynomialFeatures generates interaction and higher-degree terms for non-linear relationships.'
        },
        {
          id: 'q2_21',
          text: 'What is the purpose of the `fit_intercept` parameter in scikit-learn’s LinearRegression?',
          options: [
            'To enable regularization',
            'To include/exclude the bias term',
            'To set the learning rate',
            'To shuffle training data'
          ],
          correctAnswer: 1,
          explanation: 'Setting `fit_intercept=False` forces the regression line through the origin.'
        },
        {
          id: 'q2_22',
          text: 'What does VIF (Variance Inflation Factor) measure?',
          options: [
            'Model accuracy',
            'Severity of multicollinearity',
            'Residual distribution',
            'Feature importance'
          ],
          correctAnswer: 1,
          explanation: 'VIF quantifies how much a feature’s variance is inflated due to multicollinearity.'
        },
        {
          id: 'q2_23',
          text: 'Which code snippet performs Lasso regression?',
          options: [
            'Lasso(alpha=0.1).fit(X, y)',
            'Ridge(alpha=0.1).fit(X, y)',
            'LinearRegression().fit(X, y)',
            'ElasticNet().predict(X)'
          ],
          correctAnswer: 0,
          explanation: 'Lasso (L1 regularization) is implemented via `Lasso()` in scikit-learn.'
        },
        {
          id: 'q2_24',
          text: 'What is Adjusted R-squared?',
          options: [
            'R-squared adjusted for the number of features',
            'A metric for classification tasks',
            'A regularization technique',
            'A feature scaling method'
          ],
          correctAnswer: 0,
          explanation: 'Adjusted R² penalizes adding unnecessary features to avoid overfitting.'
        },
        {
          id: 'q2_25',
          text: 'Which scenario is best suited for linear regression?',
          options: [
            'Predicting house prices based on size and location',
            'Classifying emails as spam or not spam',
            'Clustering customers into segments',
            'Generating images with GANs'
          ],
          correctAnswer: 0,
          explanation: 'Linear regression is ideal for predicting continuous numerical outcomes (e.g., prices).'
        }
      ]
    }
  },
  {
    id: '3',
    title: 'K-Nearest Neighbors (KNN)',
    description: 'Explore the KNN algorithm and its use in classification problems.',
    category: 'classification',
    difficulty: 'intermediate',
    completed: false,
    content: `
      <h2>Understanding KNN</h2>
      <p>K-Nearest Neighbors is a simple but powerful algorithm used for both classification and regression tasks.</p>
  
      <h2>How KNN Works</h2>
      <ol>
        <li>Calculate the distance between the new point and all training points</li>
        <li>Select the K nearest points</li>
        <li>For classification: take a majority vote</li>
        <li>For regression: take the average value</li>
      </ol>
  
      <h2>Choosing K</h2>
      <p>The value of K is crucial:</p>
      <ul>
        <li>Small K: More sensitive to noise</li>
        <li>Large K: Smoother decision boundaries</li>
        <li>K should be odd for binary classification</li>
      </ul>
  
      <h2>Distance Metrics</h2>
      <p>KNN relies on distance metrics to find nearest neighbors:</p>
      <ul>
        <li>Euclidean Distance: Most common, works well for continuous data</li>
        <li>Manhattan Distance: Useful for high-dimensional data</li>
        <li>Minkowski Distance: Generalization of Euclidean and Manhattan</li>
      </ul>
  
      <h2>Feature Scaling</h2>
      <p>KNN is sensitive to feature scales. Always normalize or standardize features before applying KNN.</p>
  
      <h2>Pros and Cons</h2>
      <ul>
        <li>Pros: Simple, no training phase, works well for small datasets</li>
        <li>Cons: Computationally expensive for large datasets, sensitive to irrelevant features</li>
      </ul>
    `,
    quiz: {
      id: 'q3',
      questions: [
        {
          id: 'q3_1',
          text: 'Why should K be an odd number in binary classification?',
          options: [
            'It makes the algorithm faster',
            'To avoid ties in voting',
            'It reduces the memory usage',
            'It improves accuracy in all cases'
          ],
          correctAnswer: 1,
          explanation: 'Using an odd number for K in binary classification helps avoid tie situations in the majority voting process.'
        },
        {
          id: 'q3_2',
          text: 'What happens when K is too small?',
          options: [
            'The model becomes more robust',
            'The model becomes more sensitive to noise',
            'The model always underfits',
            'The computation becomes slower'
          ],
          correctAnswer: 1,
          explanation: 'When K is too small, the model becomes more sensitive to noise in the training data, potentially leading to overfitting.'
        },
        {
          id: 'q3_3',
          text: 'What is the default distance metric used in scikit-learn’s KNN implementation?',
          options: [
            'Manhattan Distance',
            'Euclidean Distance',
            'Minkowski Distance',
            'Cosine Similarity'
          ],
          correctAnswer: 1,
          explanation: 'The default distance metric in scikit-learn’s KNN is Euclidean Distance.'
        },
        {
          id: 'q3_4',
          text: 'Which of the following is a disadvantage of KNN?',
          options: [
            'It requires a training phase',
            'It is computationally expensive for large datasets',
            'It cannot handle categorical features',
            'It always underfits the data'
          ],
          correctAnswer: 1,
          explanation: 'KNN requires calculating distances for all training points, making it slow for large datasets.'
        },
        {
          id: 'q3_5',
          text: 'What is the effect of increasing K in KNN?',
          options: [
            'Decision boundaries become more complex',
            'Decision boundaries become smoother',
            'The model becomes more sensitive to noise',
            'The model becomes faster'
          ],
          correctAnswer: 1,
          explanation: 'Increasing K leads to smoother decision boundaries, reducing overfitting but potentially increasing bias.'
        },
        {
          id: 'q3_6',
          text: 'Which preprocessing step is critical for KNN?',
          options: [
            'Feature scaling',
            'Feature selection',
            'One-hot encoding',
            'Dimensionality reduction'
          ],
          correctAnswer: 0,
          explanation: 'KNN is sensitive to feature scales, so normalization or standardization is essential.'
        },
        {
          id: 'q3_7',
          text: 'What does the following code do?\n`KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)`',
          options: [
            'Trains a KNN model with K=5',
            'Predicts labels for the test set',
            'Calculates distances between points',
            'Normalizes the training data'
          ],
          correctAnswer: 0,
          explanation: 'The `fit()` method trains the KNN model using the training data.'
        },
        {
          id: 'q3_8',
          text: 'Which distance metric is best suited for high-dimensional data?',
          options: [
            'Euclidean Distance',
            'Manhattan Distance',
            'Cosine Similarity',
            'Minkowski Distance'
          ],
          correctAnswer: 2,
          explanation: 'Cosine Similarity is often preferred for high-dimensional data as it focuses on the angle between vectors rather than magnitude.'
        },
        {
          id: 'q3_9',
          text: 'What is the purpose of the `weights` parameter in KNN?',
          options: [
            'To assign higher importance to closer neighbors',
            'To normalize the features',
            'To reduce the number of neighbors',
            'To speed up computation'
          ],
          correctAnswer: 0,
          explanation: 'The `weights` parameter can assign higher importance to closer neighbors (e.g., using `distance` instead of `uniform`).'
        },
        {
          id: 'q3_10',
          text: 'Which code snippet correctly predicts labels using a trained KNN model?',
          options: [
            'model.predict(X_test)',
            'model.fit(X_test, y_test)',
            'model.score(X_test, y_test)',
            'model.transform(X_test)'
          ],
          correctAnswer: 0,
          explanation: 'The `predict()` method is used to generate predictions for new data.'
        },
        {
          id: 'q3_11',
          text: 'What is the time complexity of KNN during prediction?',
          options: [
            'O(1)',
            'O(n)',
            'O(n log n)',
            'O(n²)'
          ],
          correctAnswer: 3,
          explanation: 'KNN requires calculating distances to all training points, resulting in O(n²) time complexity.'
        },
        {
          id: 'q3_12',
          text: 'Which of the following is NOT a valid way to choose K?',
          options: [
            'Using cross-validation',
            'Setting K=1 for maximum accuracy',
            'Using domain knowledge',
            'Randomly selecting K'
          ],
          correctAnswer: 3,
          explanation: 'Randomly selecting K is not a valid approach; K should be chosen using cross-validation or domain knowledge.'
        },
        {
          id: 'q3_13',
          text: 'What does the following code compute?\n`np.mean(np.abs(y_true - y_pred))`',
          options: [
            'Mean Absolute Error (MAE)',
            'Mean Squared Error (MSE)',
            'R-squared',
            'Accuracy'
          ],
          correctAnswer: 0,
          explanation: 'This code computes the Mean Absolute Error (MAE) between true and predicted values.'
        },
        {
          id: 'q3_14',
          text: 'Which of the following is a hyperparameter of KNN?',
          options: [
            'Number of features',
            'Number of neighbors (K)',
            'Number of training samples',
            'Number of classes'
          ],
          correctAnswer: 1,
          explanation: 'The number of neighbors (K) is a hyperparameter that must be tuned.'
        },
        {
          id: 'q3_15',
          text: 'What is the effect of using a large K on decision boundaries?',
          options: [
            'They become more complex',
            'They become smoother',
            'They become more sensitive to noise',
            'They become non-linear'
          ],
          correctAnswer: 1,
          explanation: 'A large K results in smoother decision boundaries, reducing overfitting.'
        },
        {
          id: 'q3_16',
          text: 'Which code snippet standardizes features before applying KNN?',
          options: [
            'StandardScaler().fit_transform(X)',
            'MinMaxScaler().predict(X)',
            'Normalizer().fit(X)',
            'RobustScaler().score(X)'
          ],
          correctAnswer: 0,
          explanation: 'StandardScaler standardizes features by removing the mean and scaling to unit variance.'
        },
        {
          id: 'q3_17',
          text: 'What is the purpose of cross-validation in KNN?',
          options: [
            'To reduce the number of neighbors',
            'To select the optimal value of K',
            'To normalize the data',
            'To speed up computation'
          ],
          correctAnswer: 1,
          explanation: 'Cross-validation helps determine the best value of K by evaluating model performance on different subsets.'
        },
        {
          id: 'q3_18',
          text: 'Which of the following is a limitation of KNN for large datasets?',
          options: [
            'It cannot handle categorical features',
            'It is computationally expensive',
            'It always underfits the data',
            'It requires feature selection'
          ],
          correctAnswer: 1,
          explanation: 'KNN’s computational cost increases with dataset size due to distance calculations.'
        },
        {
          id: 'q3_19',
          text: 'What does the following code do?\n`KNeighborsClassifier(n_neighbors=3, weights="distance").fit(X_train, y_train)`',
          options: [
            'Trains a KNN model with K=3 and uniform weights',
            'Trains a KNN model with K=3 and distance-based weights',
            'Predicts labels using K=3',
            'Normalizes the data before training'
          ],
          correctAnswer: 1,
          explanation: 'The `weights="distance"` parameter assigns higher importance to closer neighbors.'
        },
        {
          id: 'q3_20',
          text: 'Which of the following is true about KNN?',
          options: [
            'It requires a training phase',
            'It is a parametric algorithm',
            'It is a lazy learner',
            'It works well with imbalanced datasets'
          ],
          correctAnswer: 2,
          explanation: 'KNN is a lazy learner because it does not learn a model during training; it stores the entire dataset.'
        },
        {
          id: 'q3_21',
          text: 'What is the effect of irrelevant features on KNN?',
          options: [
            'It improves accuracy',
            'It has no effect',
            'It degrades performance',
            'It reduces computation time'
          ],
          correctAnswer: 2,
          explanation: 'Irrelevant features can distort distance calculations, reducing KNN’s performance.'
        },
        {
          id: 'q3_22',
          text: 'Which code snippet calculates the accuracy of a KNN model?',
          options: [
            'accuracy_score(y_true, y_pred)',
            'mean_squared_error(y_true, y_pred)',
            'r2_score(y_true, y_pred)',
            'confusion_matrix(y_true, y_pred)'
          ],
          correctAnswer: 0,
          explanation: 'scikit-learn’s `accuracy_score` function computes classification accuracy.'
        },
        {
          id: 'q3_23',
          text: 'What is the purpose of the `algorithm` parameter in KNN?',
          options: [
            'To specify the distance metric',
            'To choose the optimization algorithm',
            'To speed up neighbor search',
            'To normalize the data'
          ],
          correctAnswer: 2,
          explanation: 'The `algorithm` parameter (e.g., `auto`, `kd_tree`, `ball_tree`) optimizes neighbor search for efficiency.'
        },
        {
          id: 'q3_24',
          text: 'Which of the following is a valid way to handle imbalanced data in KNN?',
          options: [
            'Increase K',
            'Use distance-based weights',
            'Oversample minority classes',
            'All of the above'
          ],
          correctAnswer: 3,
          explanation: 'All these techniques can help mitigate the impact of imbalanced data in KNN.'
        },
        {
          id: 'q3_25',
          text: 'What does the following code do?\n`KNeighborsClassifier(n_neighbors=7, metric="manhattan").fit(X_train, y_train)`',
          options: [
            'Trains a KNN model with K=7 and Euclidean distance',
            'Trains a KNN model with K=7 and Manhattan distance',
            'Predicts labels using K=7',
            'Normalizes the data before training'
          ],
          correctAnswer: 1,
          explanation: 'The `metric="manhattan"` parameter specifies the use of Manhattan Distance.'
        }
      ]
    }
  },
  {
    id: '4',
    title: 'K-Nearest Neighbors (KNN)',
    description: 'Explore the KNN algorithm and its use in classification problems.',
    category: 'classification',
    difficulty: 'intermediate',
    completed: false,
    content: `
      <h2>Understanding KNN</h2>
      <p>K-Nearest Neighbors is a simple but powerful algorithm used for both classification and regression tasks.</p>

      <h2>How KNN Works</h2>
      <ol>
        <li>Calculate the distance between the new point and all training points</li>
        <li>Select the K nearest points</li>
        <li>For classification: take a majority vote</li>
        <li>For regression: take the average value</li>
      </ol>

      <h2>Choosing K</h2>
      <p>The value of K is crucial:</p>
      <ul>
        <li>Small K: More sensitive to noise</li>
        <li>Large K: Smoother decision boundaries</li>
        <li>K should be odd for binary classification</li>
      </ul>
    `,
    quiz: {
      id: 'q3',
      questions: [
        {
          id: 'q3_1',
          text: 'Why should K be an odd number in binary classification?',
          options: [
            'It makes the algorithm faster',
            'To avoid ties in voting',
            'It reduces the memory usage',
            'It improves accuracy in all cases'
          ],
          correctAnswer: 1,
          explanation: 'Using an odd number for K in binary classification helps avoid tie situations in the majority voting process.'
        },
        {
          id: 'q3_2',
          text: 'What happens when K is too small?',
          options: [
            'The model becomes more robust',
            'The model becomes more sensitive to noise',
            'The model always underfits',
            'The computation becomes slower'
          ],
          correctAnswer: 1,
          explanation: 'When K is too small, the model becomes more sensitive to noise in the training data, potentially leading to overfitting.'
        }
      ]
    }
  }
];

export function Lessons() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [selectedDifficulty, setSelectedDifficulty] = useState<string | null>(null);
  const [selectedLesson, setSelectedLesson] = useState<Lesson | null>(null);
  const [showQuiz, setShowQuiz] = useState(false);

  const filteredLessons = allLessons.filter(lesson => {
    const matchesSearch = lesson.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         lesson.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = !selectedCategory || lesson.category === selectedCategory;
    const matchesDifficulty = !selectedDifficulty || lesson.difficulty === selectedDifficulty;
    return matchesSearch && matchesCategory && matchesDifficulty;
  });

  const handleQuizComplete = (score: number) => {
    // Here you would typically save the score to the database
    console.log(`Quiz completed with score: ${score}`);
    // Reset the view after a delay
    setTimeout(() => {
      setShowQuiz(false);
      setSelectedLesson(null);
    }, 3000);
  };

  if (selectedLesson) {
    if (showQuiz && selectedLesson.quiz) {
      return (
        <div className="container mx-auto px-4 py-8">
          <Quiz quiz={selectedLesson.quiz} onComplete={handleQuizComplete} />
        </div>
      );
    }

    return (
      <div className="container mx-auto px-4 py-8">
        <LessonContent
          lesson={selectedLesson}
          onBack={() => setSelectedLesson(null)}
          onStartQuiz={() => setShowQuiz(true)}
        />
      </div>
    );
  }

  return (
    <>
      <header className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <BookOpen className="text-indigo-600" size={32} />
          <h1 className="text-3xl font-bold text-gray-900">Lessons</h1>
        </div>
        <p className="text-gray-600">Explore our comprehensive curriculum of machine learning lessons.</p>
      </header>

      <div className="mb-8 grid gap-4 md:grid-cols-[1fr,auto]">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
          <input
            type="text"
            placeholder="Search lessons..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 rounded-lg border border-gray-200 focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
          />
        </div>

        <div className="flex gap-4">
          <select
            value={selectedCategory || ''}
            onChange={(e) => setSelectedCategory(e.target.value || null)}
            className="rounded-lg border border-gray-200 px-4 py-2 focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
          >
            <option value="">All Categories</option>
            <option value="basics">Basics</option>
            <option value="regression">Regression</option>
            <option value="classification">Classification</option>
            <option value="clustering">Clustering</option>
          </select>

          <select
            value={selectedDifficulty || ''}
            onChange={(e) => setSelectedDifficulty(e.target.value || null)}
            className="rounded-lg border border-gray-200 px-4 py-2 focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
          >
            <option value="">All Levels</option>
            <option value="beginner">Beginner</option>
            <option value="intermediate">Intermediate</option>
            <option value="advanced">Advanced</option>
          </select>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {filteredLessons.map(lesson => (
          <LessonCard
            key={lesson.id}
            lesson={lesson}
            onClick={() => setSelectedLesson(lesson)}
          />
        ))}
      </div>

      {filteredLessons.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-500">No lessons found matching your criteria.</p>
        </div>
      )}
    </>
  );
}