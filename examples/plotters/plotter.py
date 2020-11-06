import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_result_from_csv(lower, num_items, file_name='../RL_Agent/Vehicle/Episode_data.csv'):
    upper = lower+num_items
    df = pd.read_csv(file_name)

    def func(df, lower, upper):
        data = df[(df['episode number'] >= lower) & (df['episode number'] < upper)]
        return data


    df = func(df, lower ,upper)
    df['new'] = df['distance'] - df['gap']
    df['new'] = df['new'].apply(lambda x : 396 - x)

    annot = df.pivot('episode number', 'step', 'gap')
    annot = annot.astype('int32')
    df = df.pivot('episode number', 'step', 'action')

    sns.heatmap(df, annot=annot, fmt='d',
                vmin=-1, vmax=3
                )
    plt.show()

#file_name='../gym/backup_episode_data/episode_data_2.csv'
file_name='../gym/episode_data_10.csv'
#file_name='../../external_interface/episode_data_2_.csv'
if __name__ == '__main__':
    load_result_from_csv(4400, 50,
                         file_name=file_name
                         )