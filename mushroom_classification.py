from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Fetch dataset
mushroom = fetch_ucirepo(id=73)
X = mushroom.data.features
y = mushroom.data.targets

# Preprocessing
encoder = LabelEncoder()
X_encoded = X.apply(encoder.fit_transform)
y_encoded = encoder.fit_transform(y).ravel()

# Train-test split (normal shuffling, no stratification)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, train_size=100, shuffle=True, random_state=42)

# higher weight for poisonous mushrooms
class_weights = {0: 1, 1: 2}

# Train Random Forest with class weights
rf_clf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5, class_weight=class_weights)
rf_clf.fit(X_train, y_train)

# Prediction on training data
y_train_pred = rf_clf.predict(X_train)

# Prediction on full test data
y_test_pred = rf_clf.predict(X_test)

# Evaluate accuracy on training data
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

# Evaluate accuracy on full test data
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Print classification report for test data
print(classification_report(y_test, y_test_pred))
