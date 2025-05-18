#Nazif ÇELİK
import pandas as pd #data manipulation library
from multiprocessing import Process, Queue #multithread library
from sklearn.model_selection import train_test_split #Splitting the dataset into train and test sections
from sklearn.feature_extraction.text import CountVectorizer #for vectorization
from sklearn.ensemble import RandomForestClassifier #for random forest classification algorithm
from sklearn.metrics import accuracy_score, classification_report #for classification report 

# Function to vectorize and train the model
def vectorize_and_train(X_train, X_test, y_train, queue):
    # Vectorize the training data
    vectorizer = CountVectorizer()
    X_train_vect = vectorizer.fit_transform(X_train)
    
    # Vectorize the test data
    X_test_vect = vectorizer.transform(X_test)
    
    # Train the Random Forest classifier
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X_train_vect, y_train)
    
    # Make predictions on the test data
    y_pred = rfc.predict(X_test_vect)
    
    # Put the predictions in the queue for retrieval
    queue.put(y_pred)

# Load the dataset
df = pd.read_csv('WELFake_Dataset.csv')
df.dropna(inplace=True)
print(f"Number of samples in the dataset: {len(df)}")
print(df.head())

# Split the data into train and test sets
title = df['title']
typr = df['label']
X_train, X_test, y_train, y_test = train_test_split(title, typr, test_size=0.25, random_state=42, shuffle=True)

# Create a queue to store the predictions
queue = Queue()

# Create a process for vectorization and training
p = Process(target=vectorize_and_train, args=(X_train, X_test, y_train, queue))

# Start the process
p.start()

# Wait for the process to finish and retrieve the predictions
y_pred = queue.get()

# Join the process
p.join()

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

# Generate classification report
report = classification_report(y_test, y_pred)
print(report)
