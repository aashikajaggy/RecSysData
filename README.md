# RecSysData
Data processing to create an RL-powered ethical recommendation system (Kullback-leibler divergences, profile distributions, item/user minority indices, rating data filtering)

The movie lens dataset was used for experimentation, which consists of user demographics, user ids, rankings, movie ids, and time stamps.

To identify user preferences, I use item response theory, which can predict a user's ratings to new movies based on previous rankings. This was used to simulate the recommendation process.

The files in this folder contain the following:
- Kullback-leibler divergence, which calculated the statistical difference between the user's predicted responses and initial rankings to simulate the shift in user views over time.
- User minority indices, which reflected a user's position as underrepresented or a part of a marginalized community (this was calculated through the user demographic information)
- Item minority index, which demonstrated recommended items that were least preferred or interacted with ("not popular")

The objective of the ethical recommendation system is to include underrepresented items/users, as well as to diversify recommended items and prevent the echo chamber effect.


