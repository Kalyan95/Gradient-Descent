import numpy as np
from dataclasses import dataclass, field
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt


@dataclass
class LogisticRegressionGD:
    lr: float = 0.01
    epochs: int = 50
    W: np.ndarray | None = field(init=False, default=None)
    losses: list[float] = field(init=False, default_factory=list)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.c_[X, np.ones(X.shape[0])]
        y = y.reshape(-1, 1)
        self.W = np.random.randn(X.shape[1], 1)
        self.losses.clear()
        for epoch in range(self.epochs):
            scores = X.dot(self.W)
            preds = self._sigmoid(scores)
            error = preds - y
            loss = np.sum(error**2)
            self.losses.append(loss)
            gradient = X.T.dot(error)
            self.W -= self.lr * gradient
            if epoch == 0 or (epoch + 1) % 5 == 0:
                print(f"[INFO] epoch={epoch + 1}, loss={loss:.7f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise ValueError("Model has not been trained yet.")
        X = np.c_[X, np.ones(X.shape[0])]
        scores = X.dot(self.W)
        preds = self._sigmoid(scores)
        return (preds >= 0.5).astype(int)


def demo() -> None:
    X, y = make_moons(n_samples=1000, noise=0.15)
    trainX, testX, trainY, testY = train_test_split(
        X, y, test_size=0.5, random_state=12
    )

    model = LogisticRegressionGD(lr=0.01, epochs=50)
    print("[INFO] training...")
    model.fit(trainX, trainY)
    print("[INFO] evaluating...")
    preds = model.predict(testX)
    print(classification_report(testY, preds))
    print("Accuracy:", accuracy_score(testY, preds))

    plt.style.use("ggplot")
    plt.figure()
    plt.title("Data")
    plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY, s=30)

    plt.figure()
    plt.plot(range(len(model.losses)), model.losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    demo()
