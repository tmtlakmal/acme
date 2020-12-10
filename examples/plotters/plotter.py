import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_result_from_csv(lower, num_items, file_name='../RL_Agent/Vehicle/Episode_data.csv'):
    upper = lower+num_items
    df = pd.read_csv(file_name)

    def func(df, lower, upper, string='episode number'):
        data = df[(df[string] >= lower) & (df[string] < upper)]
        return data

    df = func(df, lower ,upper)
    #df = func(df, 140, 180, 'step')

    def get_headway(x,y):
        if not (x == -1 or x == 99):
            return x/max(0.1, y)
        else:
            return x

    def filter_range(x):
        if (x < 15 and x > 6):
            return x
        else:
            return 99

    #df['new'] = df.apply(lambda x : get_headway(x['gap'], x['speed']), axis=1)
    #df['new'] = df['gap'] / df['speed']

    #df['new'] = df['distance'] - df['gap']
    #df['new'] = df['new'].apply(lambda x : round(396 - x))

    df['new'] = df['gap'].apply(lambda x : filter_range(x))
    decimals = pd.Series([0, 1], index=['distance', 'step'])

    df = df.round(decimals)

    print(max(df['episode number']))
    annot = df.pivot('episode number', 'step', 'distance')
    #annot = annot.round(decimals)
    annot = annot.astype('Int64')
    df = df.pivot('episode number', 'step', 'speed')

    sns.heatmap(df,
                #annot=annot,
                #fmt='d',
                vmin=0, vmax=22
                )
    plt.show()

#file_name='../gym/backup_episode_data/episode_data_2.csv'
#file_name='../gym/episode_data_9.200000000000001.csv'
file_name='../gym/episode_data_x1x1x0.9.csv'
#file_name='../../external_interface/episode_data_2_x0.9x1x1.csv'
if __name__ == '__main__':
    print(file_name)
    load_result_from_csv(1330, 20,
                         file_name=file_name
                         )