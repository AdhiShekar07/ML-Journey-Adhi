import torch
import torch.nn as nn
import numpy as np

# Sample data: [test_score, attendance] for 10 days
student_data = [
    [80, 1], [82, 1], [79, 0], [75, 1], [78, 1],
    [74, 0], [70, 1], [72, 1], [73, 1], [71, 0]
]

# Normalize data
student_data = np.array(student_data, dtype=np.float32)
student_data /= np.max(student_data, axis=0)

# Reshape for LSTM: (seq_len, batch_size, input_size)
input_seq = torch.tensor(student_data).unsqueeze(1)

# LSTM Model
class StudentLSTM(nn.Module):
    def __init__(self):
        super(StudentLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=16, num_layers=1)
        self.fc = nn.Linear(16, 1)  # Predict next test score

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[-1])  # Use the last time step output
        return out

model = StudentLSTM()

# Dummy target: Next day score (normalized)
target = torch.tensor([[0.75]])

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    output = model(input_seq)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Predict next score
model.eval()
predicted = model(input_seq).item() * 100
print(f"\n🎯 Predicted next test score: {predicted:.2f}")
