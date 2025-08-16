import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

df=pd.read_csv('ecommerce.csv')

df.columns = ["Category", "Description"]


df['Category_num']=df['Category'].map({
    'Household':0,
    'Books':1,
    'Electronics':2,
    'Clothing & Accessories':3,
})
df=df.dropna(subset=['Description', 'Category_num','Category'])
X_train, X_test, y_train, y_test = train_test_split(
    df['Description'], df['Category_num'],
    test_size=0.2,
    random_state=42,
    stratify=df['Category_num']
)


pipe=Pipeline(
    [
        ('tfidf', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ]
)
pipe.fit(X_train, y_train)  
y_pred=pipe.predict(X_test)
print(classification_report(y_test, y_pred))