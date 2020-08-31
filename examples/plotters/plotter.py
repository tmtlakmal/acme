import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_result_from_csv(lower, upper, file_name='../RL_Agent/Vehicle/Episode_data.csv'):
    df = pd.read_csv(file_name)

    def func(df, lower, upper):
        data = df[(df['episode number'] > lower) & (df['episode number'] < upper)]
        return data


    df = func(df, lower ,upper)

    annot = df.pivot('episode number', 'step', 'speed')
    df = df.pivot('episode number', 'step', 'action')

    print(df)

    sns.heatmap(df, annot=annot,
                vmin=-1, vmax=3
                )
    plt.show()

file_name='/home/pgunarathna/PycharmProjects/acme/examples/gym/Episode_data.csv'
if __name__ == '__main__':
    load_result_from_csv(11000,11050,
                         file_name=file_name
                         )