{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Simple Linear Regression\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>YearsExperience</th>\n",
       "      <th>Salary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.1</td>\n",
       "      <td>39343.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1.3</td>\n",
       "      <td>46205.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.5</td>\n",
       "      <td>37731.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2.0</td>\n",
       "      <td>43525.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2.2</td>\n",
       "      <td>39891.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   YearsExperience   Salary\n",
       "0              1.1  39343.0\n",
       "1              1.3  46205.0\n",
       "2              1.5  37731.0\n",
       "3              2.0  43525.0\n",
       "4              2.2  39891.0"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv('Salary_Data.csv')\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = data.iloc[:, :-1].values\n",
    "y = data.iloc[:, -1].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "regressor = LinearRegression()\n",
    "regressor.fit(x_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_predict = regressor.predict(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 40835.10590871, 123079.39940819,  65134.55626083,  63265.36777221,\n",
       "       115602.64545369, 108125.8914992 , 116537.23969801,  64199.96201652,\n",
       "        76349.68719258, 100649.1375447 ])"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_predict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(x_train, y_train, color='red')\n",
    "plt.plot(x_train, regressor.predict(x_train), color='blue')\n",
    "plt.title('Salary vs Experience(Training Set)')\n",
    "plt.xlabel('Years of Experience')\n",
    "plt.ylabel('Salary')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(x_test, y_test, color='red')\n",
    "plt.plot(x_test, y_predict, color='blue')\n",
    "plt.title('Salary vs Experience(Test Set)')\n",
    "plt.xlabel('Years of Experience')\n",
    "plt.ylabel('Salary')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
