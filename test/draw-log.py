import matplotlib.pyplot as plt

# Initialize a list to hold the running average probabilities.
running_averages = []
sum_probabilities = 0

# Open and read the file.
filename = '../log/20240723_233519.txt'
with open(filename, 'r') as file:
    for i, line in enumerate(file):
        items, total = map(int, line.split())
        probability = items / total if total != 0 else 0
        sum_probabilities += probability
        current_average = sum_probabilities / (i + 1)
        running_averages.append(current_average)

# Now, running_averages list contains the average probability up to each line.

# Plotting the graph.
plt.plot(range(1, len(running_averages) + 1), running_averages, marker='o', linestyle='-', color='b')
#plt.title('Running Average Probability Over Lines')
plt.xlabel('Program Number')
plt.ylabel('Average Probability')
plt.grid(True)

if __name__ == '__main__':
    plt.show()
