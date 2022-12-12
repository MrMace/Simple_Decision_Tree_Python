# Training data
X_train = [
    [3, "red"],  # Small red fruit
    [2, "yellow"],  # Small yellow fruit
    [5, "red"],  # Large red fruit
    [4, "yellow"],  # Large yellow fruit
]

# Training labels
y_train = [
    "apple",  # Small red fruit is an apple
    "banana",  # Small yellow fruit is a banana
    "apple",  # Large red fruit is an apple
    "banana",  # Large yellow fruit is a banana
]

# Test data
X_test = [
    [1, "red"],  # Small red fruit
    [6, "yellow"],  # Large yellow fruit
]

# Define a class to represent a decision tree
class DecisionTree:
    def __init__(self, feature, value, left, right):
        self.feature = feature  # The feature to split on
        self.value = value  # The value of the feature to split on
        self.left = left  # The left subtree
        self.right = right  # The right subtree

    # Function to make predictions using the decision tree
    def predict(self, x):
        # If the current node is a leaf node, return the predicted label
        if self.left is None and self.right is None:
            return self.value

        # If the current node is not a leaf node,
        # split the data on the feature and value of the current node
        if x[self.feature] < self.value:
            return self.left.predict(x)
        else:
            return self.right.predict(x)


# Function to calculate the Gini index for a given split
def gini_index(y_left, y_right):
    # Calculate the Gini index for the left split
    p = float(y_left.count(1)) / len(y_left)
    gini_left = 1 - p**2 - (1 - p) ** 2

    # Calculate the Gini index for the right split
    p = float(y_right.count(1)) / len(y_right)
    gini_right = 1 - p**2 - (1 - p) ** 2

    # Calculate the weighted average of the Gini indices for the left and right splits
    return (len(y_left) / (len(y_left) + len(y_right))) * gini_left + (
        len(y_right) / (len(y_left) + len(y_right))
    ) * gini_right


# Function to train a decision tree
def train(X, y):
    # If all the labels are the same, return a leaf node with the label
    if len(set(y)) == 1:
        return DecisionTree(None, y[0], None, None)

    # If there are no more features to split on, return a leaf node with the most common label
    if len(X[0]) == 0:
        return DecisionTree(None, max(set(y), key=y.count), None, None)

    # Find the best feature and value to split on
    feature, value = find_best_split(X, y)

    # Split the data on the best feature and value
    X_left, y_left, X_right, y_right = split_data(X, y, feature, value)

    # Create a decision tree node with the best feature and value
    tree = DecisionTree(feature, value, None, None)

    # Recursively build the left and right subtrees
    tree.left = train(X_left, y_left)
    tree.right = train(X_right, y_right)

    return tree


# Function to find the best feature and value to split on
def find_best_split(X, y):
    best_feature = None
    best_value = None
    best_score = float("inf")  # Initialize the best score to infinity

    # Iterate over all features and values
    for feature in range(len(X[0])):
        for value in set([x[feature] for x in X]):
            # Split the data on the current feature and value
            X_left, y_left, X_right, y_right = split_data(X, y, feature, value)

            # If one of the splits is empty, skip this split
            if len(X_left) == 0 or len(X_right) == 0:
                continue

            # Calculate the Gini index for the current split
            gini = gini_index(y_left, y_right)

            # If the Gini index for the current split is lower than the best score,
            # update the best feature, value, and score
            if gini < best_score:
                best_feature = feature
                best_value = value
                best_score = gini

    return best_feature, best_value


# Function to split the data on a given feature and value
def split_data(X, y, feature, value):
    X_left = []  # List for the left split
    y_left = []  # List for the left labels
    X_right = []  # List for the right split
    y_right = []  # List for the right labels

    # Iterate over the data
    for i in range(len(X)):
        # If the value of the current feature is less than the split value,
        # add the data point and label to the left split
        if X[i][feature] < value:
            X_left.append(X[i])
            y_left.append(y[i])
        # Otherwise, add the data point and label to the right split
        else:
            X_right.append(X[i])
            y_right.append(y[i])

    return X_left, y_left, X_right, y_right


# Train a decision tree
tree = train(X_train, y_train)

# Use the trained decision tree to make predictions on the test data
predictions = [tree.predict(x) for x in X_test]

# Print the predictions
print(predictions)  # ["apple", "banana"]

