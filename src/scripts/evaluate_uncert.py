import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

class ConformalPredictor:
    def __init__(self, model):
        """
        Initialize the ConformalPredictor with a machine learning model.

        Parameters:
        - model: The machine learning model (e.g., a regression model) to be used for conformal prediction.
        """
        self.model = model

    def conformal_predict(self, X_train, y_train, X_test):
        """
        Perform conformal prediction using the given training data and test data.

        Parameters:
        - X_train: Training features.
        - y_train: Training labels.
        - X_test: Test features for which predictions are to be made.

        Returns:
        - predicted_p_values: Predicted p-values for each test instance.
        - prediction_intervals: Prediction intervals for each test instance.
        """
        # Initialize the ICPRegressor with the provided model
        icp = ICPRegressor(self.model)

        # Fit the ICPRegressor using the training data
        icp.fit(X_train, y_train)

        # Perform conformal prediction on the test data
        predicted_p_values, prediction_intervals = icp.predict(X_test)

        return predicted_p_values, prediction_intervals
    
    def conformal_intervals(self,
                            Xs_true,
                            y_true,
                            y_pred,
                            y_pred_error,
                            invcov):
        labels, labels_pred, upper, lower = y_true, y_pred, y_pred + y_pred_error, y_pred - y_pred_error
        assert np.shape(labels) == np.shape(upper) == np.shape(lower), "not the same shapes"
        # Problem setup
        n=50
        alpha = invcov # 1 - alpha is desired coverage, here we're going for 1 sigma or 68%
        print(f'Guaranteeing {round(1-alpha,2)*100}% coverage')
        # split the softmax scores into calibration and validation sets (save the shuffling)
        idx = np.array([1] * n + [0] * (labels.shape[0]-n)) > 0
        np.random.shuffle(idx)
        cal_labels, val_labels = labels[idx], labels[~idx]
        cal_labels_pred, val_labels_pred = labels_pred[idx], labels_pred[~idx]
        cal_upper, val_upper = upper[idx], upper[~idx]
        cal_lower, val_lower = lower[idx], lower[~idx]
        cal_X, val_X = Xs_true[idx], Xs_true[~idx]
        # Get scores. cal_upper.shape[0] == cal_lower.shape[0] == n
        cal_scores = np.maximum(cal_labels-cal_upper, cal_lower-cal_labels)
        # Get the score quantile
        qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')
        # Deploy (output=lower and upper adjusted quantiles)
        prediction_sets = [val_lower - qhat, val_upper + qhat]
        # let's visualize what is happening on the validation set
        plt.scatter(val_X[:,0], val_labels, label = 'true value', color = '#334E58')
        plt.plot(val_X[:,0], val_labels_pred, label = 'predicted value', color = '#334E58')
        plt.fill_between(val_X[:,0], val_lower, val_upper, label = r'Meinert+2022 $u_{al}$', alpha = 0.5, color = '#D33F49')
        plt.fill_between(val_X[:,0], prediction_sets[0], prediction_sets[1], label = r'conformal correction, 1$\sigma$', alpha = 0.5, color = '#FCBFB7')
        plt.legend()
        plt.show()
        return prediction_sets
    
class SpearmanRankCalculator:
    @staticmethod
    def calculate_spearman_rank(y_true, y_pred):
        """
        Calculate the Spearman rank correlation coefficient between true and predicted values.

        Parameters:
        - y_true: True labels.
        - y_pred: Predicted labels.

        Returns:
        - spearman_rank: Spearman rank correlation coefficient.
        """
        spearman_rank, _ = spearmanr(y_true, y_pred)
        return spearman_rank

    def visualize_spearman_rank(y_true, y_pred):
        plt.hist(y_true)
        plt.hist(y_pred)
        plt.legend()
        plt.show()
