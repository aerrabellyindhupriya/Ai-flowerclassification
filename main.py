from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os

# ==================================================
# FLOWER INTELLIGENCE SYSTEM
# ==================================================

flower_info = {
    "setosa": {
        "color": "Purple / Violet",
        "fragrance": "Mild Floral",
        "lifespan": "Perennial",
        "description": "A small and elegant iris flower with beautiful violet petals."
    },

    "versicolor": {
        "color": "Blue-Violet",
        "fragrance": "Light Sweet Aroma",
        "lifespan": "Perennial",
        "description": "A medium-sized iris flower known for its attractive colors."
    },

    "virginica": {
        "color": "Deep Violet / Blue",
        "fragrance": "Rich Floral Fragrance",
        "lifespan": "Perennial",
        "description": "A large iris flower with vibrant petals and strong growth."
    }
}

# ==================================================
# LOAD DATASET
# ==================================================

iris = load_iris()

X = iris.data
y = iris.target

target_names = iris.target_names

# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==================================================
# TRAIN MODEL
# ==================================================

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

# ==================================================
# ACCURACY
# ==================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

# ==================================================
# MAIN MENU
# ==================================================

while True:

    print("\n" + "=" * 60)
    print("🌸 FLOWER INTELLIGENCE SYSTEM 🌸")
    print("=" * 60)

    print(f"Model Accuracy : {accuracy * 100:.2f}%")

    print("\nEnter Flower Measurements")

    try:

        sepal_length = float(input("Sepal Length (cm): "))
        sepal_width = float(input("Sepal Width  (cm): "))
        petal_length = float(input("Petal Length (cm): "))
        petal_width = float(input("Petal Width  (cm): "))

        user_data = [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]]

        prediction = model.predict(user_data)

        probability = model.predict_proba(user_data)

        confidence = max(probability[0]) * 100

        flower_name = target_names[prediction[0]]

        details = flower_info[flower_name]

        print("\n" + "=" * 60)
        print("🌺 FLOWER ANALYSIS REPORT")
        print("=" * 60)

        print(f"Flower Name      : {flower_name.title()}")
        print(f"Flower Color     : {details['color']}")
        print(f"Fragrance        : {details['fragrance']}")
        print(f"Life Span        : {details['lifespan']}")
        print(f"Confidence Score : {confidence:.2f}%")

        print("\nDescription:")
        print(details['description'])

        print("\n" + "=" * 60)

        choice = input(
            "\nWould you like to predict another flower? (yes/no): "
        ).lower()

        if choice != "yes":
            print("\nThank you for using Flower Intelligence System 🌸")
            break

    except ValueError:
        print("\n❌ Please enter valid numeric values.")