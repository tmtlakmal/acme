import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def func(df, lower, upper, string='episode number'):
    data = df[(df[string] >= lower) & (df[string] < upper)]
    return data

def smooth_data(file_name, column, weight):
    df = pd.read_csv(file_name)

    df = func(df, 0, 400000, 'Step')

    df['Smoothed Value'] = df[column].rolling(100, min_periods=1).mean()
    df.to_csv(file_name)


def compare_trajectories(file_name1, file_name2, lower, num_items, output_file='lp_comparison.csv'):
    upper = lower+num_items
    df_original_1 = pd.read_csv(file_name1)
    df_original_2 = pd.read_csv(file_name2)

    l = []
    l1 = []
    l2 = []
    l3 = []
    for i in range(1,num_items):

        len = (21+i)*5-1

        df1 = func(df_original_1, i-1, i)
        df2 = func(df_original_2, i-1, i)

        result = df1['distance']-df2['distance']
        avg = (result[0:len].abs().sum())/len
        print(avg, len, df1['distance'][0:len].sum()/400, df2['distance'][0:len].sum()/400)
        l.append(len)
        dist1 = df1[(df1['distance'] != 99) | (df1['distance'] != -1)]['distance']
        dist2 = df2[(df2['distance'] != 99) | (df2['distance'] != -1)]['distance']
        l1.append(dist1.sum()/dist1.size)
        l2.append(dist2.sum()/dist2.size)
        l3.append(dist1.sum()/dist1.size-dist2.sum()/dist2.size)

    df = pd.DataFrame(list(zip(l, l1, l2, l3)), columns=['index', 'DQN', 'Gurobi', 'diff'])
    df.to_csv(output_file)

def load_result_from_csv(lower, num_items, annotations, file_name='../RL_Agent/Vehicle/Episode_data.csv'):
    upper = lower+num_items
    df = pd.read_csv(file_name)

    df = func(df, lower ,upper)
    #df = func(df, 130, 160, 'step')

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
    #decimals = pd.Series([0, 1], index=['distance', 'step'])

    #df = df.round(decimals)
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    #print(df['speed'])
    #print(max(df['episode number']))
    annot = df.pivot('episode number', 'step', 'gap')
    annot = annot.round()
    #annot = annot.astype('Int64')
    df = df.pivot('episode number', 'step', 'speed')

    sns.heatmap(df,
                annot=annot if annotations else None,
                fmt='d',
                vmin=-1, vmax=30
                )
    plt.show()

#file_name='../gym/backup_episode_data/episode_data_2.csv'
#file_name='../gym/episode_data_9.200000000000001.csv'
file_name='../gym/episode_data_x0.99x0.99x0.99.csv'
file_name1='../gym/episode_data_x1x1x1.csv'
file_name2='../../logs/DQN_46/episode_data_single.csv'
#file_name='/home/student.unimelb.edu.au/pgunarathna/Downloads/run-DQN_11-tag-environment loop_episode_reward_2.csv'
#file_name='../../external_interface/episode_data_2_x0.9x1x1.csv'

if __name__ == '__main__':

    #file_name = "/home/student.unimelb.edu.au/pgunarathna/Downloads/reward_2_12.csv"
    #smooth_data(file_name, 'Value', 0.9)

    file_name1 = '../../logs/DQN_19/episode_data.csv'
    #file_name2 = '../../logs/DQN_76/episode_data.csv'
    load_result_from_csv(1400, 100, annotations=False, file_name=file_name1)
    #compare_trajectories(file_name1, file_name2, 0 , num_items=20, output_file='lp_comparison_heuristic.csv')
