# DisasterRecoveryMLPipeline
For this project data has been prepared using and Extract, Transform, Load (ETL Pipeline). The cleaned dataframe was saved to an sql database.

After cleaning, further data processing was undertaken to normalize and tokenize the data in preparation to be used in the machine learning model.

Scikit-Learn has then been used to build a machine learning model to categorise twitter response messages to natural disasters. Countvectoriser and TF-IDF transformations were carried out before train, test split was applied. 

Required libraries:
- Numpy
- Scikit-Learn
- Pandas
- sqlalchemy
- sqlite3
- nltk
- re
- pickle
