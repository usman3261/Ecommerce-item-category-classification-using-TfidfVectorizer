import pandas as pd

df=pd.read_csv('ecommerce.csv')

df.columns = ["Category", "Description"]
print(df.Category.value_counts())

df['Category_num']=df['Category'].map({
    'Household':0,
    'Books':1,
    'Electronics':2,
    'Clothing & Accessories':3,
})

print(df[df['Category_num'] == 1].head(10))