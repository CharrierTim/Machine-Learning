# Plot the data

def plot_data():
    plt.figure(figsize=(10, 10))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_train[i], cmap='gray')
        plt.title("Label = " + str(Y_train[i]))
        plt.axis('off')
    plt.show()
    
# Main

plot_data()