from micrograd.nn import MLP

# 1. Create a small dataset (Binary Classification)
# Inputs: 4 examples, each has 3 features
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
# Targets: The desired output for each example (1.0 or -1.0)
ys = [1.0, -1.0, -1.0, 1.0] 

# 2. Initialize the Neural Network
# 3 inputs -> Layer of 4 -> Layer of 4 -> 1 output
model = MLP(3, [6, 6, 1])

# 3. The Training Loop
print(f"Initial Loss: ???")

for k in range(500): # Run 20 training steps
    
    # A. Forward Pass
    ypred = [model(x) for x in xs] # Predict inputs
    
    # B. Calculate Loss (Mean Squared Error)
    # loss = sum((prediction - target)^2)
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    # C. Zero Gradients (Reset the "backpacks")
    model.zero_grad() # Not relevant for the first run and only after that
    
    # D. Backward Pass (Calculate new gradients)
    loss.backward()
    
    # E. Update (Gradient Descent)
    # Move every weight slightly opposite to the gradient
    if k < 100:
        learning_rate = 0.01
    else:
        learning_rate = 0.002
    for p in model.parameters():
        p.data += -learning_rate * p.grad
        
    print(f"Step {k}: Loss = {loss.data:.4f}") # .data just returns the value of the Value object

print("\nFinal Predictions:")
print([f"{y.data:.2f}" for y in ypred])
