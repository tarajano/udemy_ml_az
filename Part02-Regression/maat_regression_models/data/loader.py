import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

salaries_dtf = pd.read_csv('data/SalariesSalariesForProfessors.csv')
# print(salaries_dtf.na())


## Viz
# salaries_male_dtf = salaries_dtf[salaries_dtf.sex == 'Male']
# salaries_female_dtf = salaries_dtf[salaries_dtf.sex == 'Female']
# plt.scatter(salaries_male_dtf['yrs.service'], salaries_male_dtf['salary'], label='Male')
# plt.scatter(salaries_female_dtf['yrs.service'], salaries_female_dtf['salary'], label='Female')
# plt.xlabel('Years in Service')
# plt.ylabel('Salary')
# plt.legend()
# plt.show()


