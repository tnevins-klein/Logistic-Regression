import numpy
import pandas
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

penguins = pandas.read_csv('penguins.csv').dropna()
adelie = penguins[penguins['species'] == "Adelie"]

normalized_sex = OrdinalEncoder().fit_transform(
    adelie.sex.values.reshape(-1, 1)
)

body_train, body_test, sex_train, sex_test = train_test_split(
    adelie.body_mass_g.values.reshape(-1, 1),
    normalized_sex,
    random_state=2001,
    train_size=0.7
)

regression = LogisticRegression()
regression.fit(body_train, sex_train)

print("Regression ", regression.coef_)

print("Finished training with score: ", regression.score(sex_test, body_test))

plt.title("Body Mass vs Sex")
plt.ylabel("Sex (probability of being male)")
plt.xlabel("Body Mass (g)")

plt.scatter(adelie.body_mass_g.values, normalized_sex)

logistic_x = numpy.linspace(plt.xlim()[0], plt.xlim()[1])
logistic_y = regression.predict_proba(logistic_x.reshape(-1, 1))

plt.plot(logistic_x, logistic_y[:, 1])

plt.savefig("figure.svg")
