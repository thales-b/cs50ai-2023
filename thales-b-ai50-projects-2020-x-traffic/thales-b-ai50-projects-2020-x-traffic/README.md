# Experimentation Process 
- For the base model, I used a simple architecture with one convolutional layer, one max pooling layer, and one dense layer. The results were the following:
```python
loss: 0.9133 - accuracy: 0.9035
```
- Then, I added additional convolutional and pooling layers to capture finer details. The results were noticeably more accurate:
```python
loss: 0.3246 - accuracy: 0.9408
```
- Next, I increased the number of units. Although the results were slightly more accurate, the runtime was longer.
```python
loss: 0.2465 - accuracy: 0.9531
```
- I also added two dropout layers after each convolutional layer with a 0.25 rate and a final one before the dense layer with a rate of 0.5. But surprisingly, it only slightly reduced loss while causing a setback in accuracy:
```python
loss: 0.2293 - accuracy: 0.9390
```
- So, after tinkering with the dropout layers, I found that keeping only two of them (one for a convenlutional layer and one for the dense layer) was the best configuration.
```python
loss: 0.1127 - accuracy: 0.9730
```
# Conclusion
Adding layers on top of others without obeying some kind of logic won't do much to enhance the quality of our model. Otherwise we would sacrifice much-needed computational power for meager gains, or worse, for a model that performs worse.