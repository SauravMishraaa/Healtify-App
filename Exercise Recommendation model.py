!pip install scikit-surprise
import surprise
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
df = pd.read_csv('exercise_data2.csv')
print(df.head(10))
reader = Reader(rating_scale=(0, 1000))
data = Dataset.load_from_df(df[['user_id', 'exercise_id', 'calories']], reader)
trainset, testset = train_test_split(data, test_size=0.2)
model = SVD()
model.fit(trainset)
predictions = model.test(testset)
accuracy = surprise.accuracy.rmse(predictions)
print('RMSE:', accuracy)

# Make recommendations for a new user
new_user_id = 6
new_user_data = [(new_user_id, exercise_id, 0) for exercise_id in df['exercise_id'].unique() if exercise_id not in df[df['user_id']==new_user_id]['exercise_id'].unique()]
new_user_predictions = model.test(new_user_data)
new_user_recommendations = sorted(new_user_predictions, key=lambda x: x.est, reverse=True)[:5]

print('Recommended exercises for user', new_user_id)
for recommendation in new_user_recommendations:
    print('Exercise ID:', recommendation.iid, '| Predicted Calories:', round(recommendation.est))
