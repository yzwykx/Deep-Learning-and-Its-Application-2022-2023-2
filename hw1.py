import jittor as jt
from jittor import nn, Module
import numpy as np
import matplotlib.pyplot as plt

class MLP(Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        
    def execute(self, x):
        x = self.layer1(x)
        x = nn.relu(x)
        x = self.layer2(x)
        return x

def generate_data(num_samples):
    data = jt.array(np.random.rand(num_samples, 2))
    labels = data[:, 0] * data[:, 1]
    return data, labels

input_size = 2
hidden_size = 16
output_size = 1
learning_rate = 0.01
num_epochs = 1000

model = MLP(input_size, hidden_size, output_size)
optimizer = nn.SGD(model.parameters(), learning_rate, momentum=0.9)

train_data, train_labels = generate_data(4000)
test_data, test_labels = generate_data(1000)

print(train_labels)

loss_func = nn.MSELoss()

for epoch in range(num_epochs):
    train_preds = model(train_data)
    train_preds = train_preds.reshape(-1)
    train_loss = loss_func.execute(train_preds, train_labels)
    optimizer.zero_grad()
    optimizer.backward(train_loss)
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item()}")

with jt.no_grad():
    x = np.linspace(1,1000,1000,endpoint=False)
    test_preds = model(test_data)
    test_preds = test_preds.reshape(-1)
    plt.scatter(x, test_preds, s = 3)
    plt.scatter(x, test_labels, s = 3)
    plt.savefig('./scatter.jpg')
    plt.show()
    test_loss = loss_func.execute(test_preds, test_labels)
    print(abs(test_preds-test_labels).mean())
    print(f"Test Loss: {test_loss.item()}")
