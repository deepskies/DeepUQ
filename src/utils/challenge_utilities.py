
import numpy as np
import matplotlib.pyplot as plt
from deepbench.astro_object import StarObject, GalaxyObject 
import math 
import tensorflow as tf 
from sklearn.metrics import roc_curve, confusion_matrix


class SkyGeneratorTrue(tf.keras.utils.Sequence): 
    def __init__(self, n_samples, image_size=28, pre_processing=None, train=True, shuffle=False, batch_size=64):
        self.n_samples = n_samples
        self.train = train
        self.pre_processing = pre_processing

        self.shuffle = shuffle

        self.image_size = image_size
        self.noise_level = 0.05

        self.rng = np.random.default_rng(seed=42) # Seed for the main notebook
        self.batch_size=batch_size
        self.labels = self.decide_labels()

    def decide_labels(self): 
        n_stars = self.rng.integers(low=int(.45*self.n_samples), high=int(.65*self.n_samples))
        n_galaxies = self.n_samples-n_stars
        labels = [0 for _ in range(n_stars)] + [1 for _ in range(n_galaxies)]

        if self.shuffle: 
            self.rng.shuffle(labels)

        return np.asarray(labels)
 
    def generate_image(self, label): 
        radius = self.rng.integers(low=1, high=self.image_size/2)
        center_x = self.rng.integers(low=1, high=self.image_size)
        center_y = self.rng.integers(low=1, high=self.image_size)

        if label == 0: 
            image = StarObject(
                image_dimensions=(self.image_size, self.image_size), 
                noise_level=self.noise_level,
                radius=radius
                    ).create_object(
                        center_x=center_x, center_y=center_y
                        )
 
        else: 
            image = GalaxyObject(
                image_dimensions=(self.image_size, self.image_size), 
                noise_level=self.noise_level,
                radius=radius
                    ).create_object(
                        center_x=center_x, center_y=center_y
                        )

        if self.pre_processing is not None: 
            image = self.pre_processing(image)

        return image

    def __len__(self):
        return math.ceil(self.n_samples / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.labels))
        batch_y = self.labels[low:high]
        batch_x = np.zeros((len(batch_y), self.image_size, self.image_size))
        for index, label in enumerate(batch_y): 
            batch_x[index] = self.generate_image(label)
        return batch_x, batch_y


class SkyGenerator01(SkyGeneratorTrue):
    def __init__(self, n_samples, pre_processing=None, train=True, shuffle=False, batch_size=64):
        super().__init__(n_samples, pre_processing, train, shuffle, batch_size)


class SkyGenerator02(SkyGeneratorTrue): 
    def __init__(self, n_samples, pre_processing=None, train=True, shuffle=False, batch_size=64):
        super().__init__(n_samples, pre_processing, train, shuffle, batch_size)

    def decide_labels(self): 
        n_stars = self.rng.integers(low=int(.85*self.n_samples), high=int(.95*self.n_samples))
        n_galaxies = self.n_samples-n_stars
        labels = [0 for _ in range(n_stars)] + [1 for _ in range(n_galaxies)]

        if self.shuffle: 
            self.rng.shuffle(labels)

        return np.asarray(labels)
    
class SkyGenerator03(SkyGeneratorTrue): 
    def __init__(self, n_samples, pre_processing=None, train=True, shuffle=False, batch_size=64):
        super().__init__(n_samples, pre_processing, train, shuffle, batch_size)
        self.noise_level = 0.6
        self.image_size = 64

class SkyGenerator04(SkyGeneratorTrue): 
    def __init__(self, n_samples, pre_processing=None, train=True, shuffle=False, batch_size=64):
        super().__init__(n_samples, pre_processing, train, shuffle, batch_size)
        
    def decide_labels(self): 
        n_stars = self.rng.integers(low=int(.45*self.n_samples), high=int(.65*self.n_samples))
        n_galaxies = self.rng.integers(low=int(.6*(self.n_samples-n_stars)), high=int(.9*(self.n_samples-n_stars)))
        wild_card = self.n_samples - (n_stars+n_galaxies)
        labels = [0 for _ in range(n_stars)] + [1 for _ in range(n_galaxies)] + [2 for _ in range(wild_card)]

        if self.shuffle: 
            self.rng.shuffle(labels)

        return np.asarray(labels)
    
class SkyGenerator05(SkyGeneratorTrue): 
    def __init__(self, n_samples, pre_processing=None, train=True, shuffle=False, batch_size=64):
        super().__init__(n_samples, pre_processing, train, shuffle, batch_size)


class Eval: 
    @staticmethod
    def plot_loss_history(history): 
        loss = history['loss']
        epochs = range(len(loss))

        val_loss = history['val_loss']

        plt.plot(epochs, loss, label="Train", marker='o')
        plt.plot(epochs, val_loss, label='Validation', marker='x')

        plt.title("Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def ROC_curve(prediction_classes, labels): 
        score_fpr, score_tpr, _ = roc_curve(labels, prediction_classes)
        plt.plot(score_fpr, score_tpr, label='Your classifier')
        plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1.5, color='black', label='Random Classifier')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC AUC Curve")
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def confusion_matrix(prediction_classes, labels): 
        confusion = confusion_matrix(labels.ravel(), prediction_classes.ravel())
        plt.imshow(confusion)

        for true in range(confusion.shape[0]):
            for predicted in range(confusion.shape[1]):
                plt.text(predicted, true, confusion[true, predicted],
                            ha="center", va="center", fontdict={
                                "color":"white", 
                                "backgroundcolor":"black", 
                                "size": 5})

        plt.xticks([0, 1], labels=["Star", "Galaxy"])
        plt.yticks([0, 1], labels=["Star", "Galaxy"])  
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()