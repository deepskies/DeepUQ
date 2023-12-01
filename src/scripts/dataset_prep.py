# Contains modules used to prepare a dataset
# with varying noise properties

import pandas as pd
import numpy as np


class DatasetPreparation:
     """
    A class for loading, preprocessing, and simulating datasets.

    Parameters:
    - file_path (str): The path to the dataset file.

    Methods:
    - load_data(): Load data from the specified file path.
    - preprocess_data(): Preprocess the loaded data.
    - simulate_data(simulation_name, num_samples=1000): Simulate data based on the specified simulation.
    - save_data(output_file='output_data.csv'): Save the current dataset to a CSV file.
    - get_data(): Retrieve the current dataset.

    Example Usage:
    ```
    dataset_manager = DatasetPreparation('your_dataset.csv')
    dataset_manager.load_data()
    dataset_manager.preprocess_data()
    dataset_manager.simulate_data('linear')
    dataset_manager.save_data('simulated_data.csv')
    simulated_data = dataset_manager.get_data()
    ```

    Note: Replace 'your_dataset.csv' with the actual dataset file path.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            print("Data loaded successfully.")
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")

    def preprocess_data(self):
        if self.data is not None:
            # Example: Dropping missing values for simplicity
            self.data = self.data.dropna()
            print("Data preprocessed successfully.")
        else:
            print("Error: No data loaded. Please use load_data() first.")

    def simulate_data(self, x, parameters, simulation_name):
        if simulation_name == 'linear':
            # Example linear simulation
            m, b, sigma = parameters
            #x = np.linspace(0, 100, 101)
            rs = np.random.RandomState()#2147483648)# 
            ε = rs.normal(loc=0, scale=sigma, size = len(x)) 
            y =  m * x + b + ε
            #x = np.linspace(0, 10, num_samples)
            #y = 2 * x + 1 + np.random.normal(0, 1, num_samples)
            simulated_data = pd.DataFrame({'Feature': x, 'Target': y})
            print("Linear simulation data generated.")
        elif simulation_name == 'quadratic':
            # Example quadratic simulation
            x = np.linspace(0, 10, num_samples)
            y = 3 * x**2 + 2 * x + 1 + np.random.normal(0, 1, num_samples)
            simulated_data = pd.DataFrame({'Feature': x, 'Target': y})
            print("Quadratic simulation data generated.")
        else:
            print(f"Error: Unknown simulation name '{simulation_name}'. No data generated.")
            return

        self.data = simulated_data

    def save_data(self, output_file='output_data.csv'):
        if self.data is not None:
            self.data.to_csv(output_file, index=False)
            print(f"Data saved to {output_file} successfully.")
        else:
            print("Error: No data available to save. Please load, preprocess, or simulate data first.")

    def get_data(self):
        return self.data


class ParameterSampler:
    """
    A class for randomly generating and saving parameter values.

    Methods:
    - random_parameters(num_samples=5): Generate random parameter values.
    - save_parameters(output_file='parameter_values.csv'): Save generated parameter values to a CSV file.

    Example Usage:
    ```
    param_sampler = ParameterSampler()
    param_sampler.random_parameters(num_samples=10)
    param_sampler.save_parameters('random_parameters.csv')
    ```

    Note: Adjust the parameter generation logic in the `random_parameters` method based on specific requirements.
    """
    def __init__(self):
        self.parameter_values = None

    def random_parameters(self, num_samples=1):
        # Example: Randomly generate parameter values
        parameter_values = {
            'param1': np.random.uniform(0, 1, num_samples),
            'param2': np.random.normal(0, 1, num_samples),
            'param3': np.random.choice(['A', 'B', 'C'], size=num_samples)
        }
        self.parameter_values = pd.DataFrame(parameter_values)
        print(f"Random parameter values generated for {num_samples} samples.")

    def save_parameters(self, output_file='parameter_values.csv'):
        if self.parameter_values is not None:
            self.parameter_values.to_csv(output_file, index=False)
            print(f"Parameter values saved to {output_file} successfully.")
        else:
            print("Error: No parameter values available to save. Please generate random parameters first.")

# Example usage:
if __name__ == "__main__":
    # Replace 'your_dataset.csv' with your actual dataset file path
    dataset_manager = DatasetPreparation('your_dataset.csv')
    dataset_manager.load_data()
    dataset_manager.preprocess_data()

    # Simulate linear data
    dataset_manager.simulate_data('linear')

    # Access the simulated data
    simulated_data = dataset_manager.get_data()
    print(simulated_data.head())
