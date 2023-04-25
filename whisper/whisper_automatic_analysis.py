import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Levenshtein import distance as lev_distance
import pandas as pd
import os


class WhisperAutoAnalysis():

    def __init__(self):
        self.only_test_classes = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'] # only test classes
        self.only_number_classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'] # only number classes
        self.only_other_classes = ['bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow'] # only other classes
        self.figsize = (7, 3.5)
        self.labels_fontsize = 12
        self.conf_matrix_size = (10, 8)
    def clean_text(self, text):
        text = '' if pd.isnull(text) else text
        text = text.lower().replace(',', '').replace('.', '').replace('?', '').replace('!', '').replace('(', '').replace(')', '')\
                            .replace(';', '').replace(':', '').replace('"', '').replace("'", '').replace('  ', ' ').strip()
        
        text = text.replace('0', 'zero').replace('1', 'one').replace('2', 'two').replace('3', 'three').replace('4', 'four')\
                    .replace('5', 'five').replace('6', 'six').replace('7', 'seven').replace('8', 'eight').replace('9', 'nine')
        
        if len(text) == 0:
            text = '.'
        if len(text) + 1  == len(text.replace(' ', '')) * 2:
            text = text.replace(' ', '')
        return text
    
    def plot_lev(self, df, model_name_title_addition, plots_root):

        filtered_df_lev = df[df['levenshtein']>0]

        fig, ax = plt.subplots(1, 2, figsize=self.figsize, gridspec_kw={'width_ratios': [3, 1]})
        sns.histplot(filtered_df_lev['levenshtein'], kde=False, binwidth=1, ax=ax[0], stat='percent')
        ax[0].set_title(f'Distribution (without correct transcriptions)', loc='left')
        ax[0].set_xlabel(f'Levenshtein distance, quantile 95%: {np.quantile(filtered_df_lev["levenshtein"], 0.95):.1f}', fontsize=self.labels_fontsize)
        ax[0].set_ylabel('Percent', fontsize=self.labels_fontsize)
        ax[0].set_xlim(0, 10)
        ax[0].set_yticks([y for y in ax[0].get_yticks() if y % 5 == 0])
        ax[0].set_yticklabels([f'{int(y)}%' for y in ax[0].get_yticks()])

        sns.boxplot(data=pd.DataFrame(filtered_df_lev['levenshtein']), ax=ax[1], width=0.5)
        ax[1].set_title(f'Boxplot', loc='left')
        ax[1].set_xlabel('')
        ax[1].set_xticks([])
        ax[1].set_ylabel('Levenshtein distance', fontsize=self.labels_fontsize)
        ax[1].set_ylim(0, 10)

        fig.suptitle(f'Whisper {model_name_title_addition}: Levenshtein distance for {len(filtered_df_lev.reference.unique())} classes', y=0.95)
        fig.tight_layout()
        fig.savefig(plots_root + f'levenshtein_distance_{len(df["reference"].unique())}_classes.png', dpi=300)
        plt.show()


    def plot_per_class_acc(self, df, model_name_title_addition, accuracy, plots_root, filter_name):

        accuracies_per_class = df.groupby('reference').apply(lambda x: x[x['hypothesis'] == x['reference']].shape[0] / x.shape[0])

        fig, ax = plt.subplots(figsize=self.figsize)

        sns.barplot(x=accuracies_per_class.index, y=accuracies_per_class.values, ax=ax, color='C0')
        ax.set_title(f'Whisper {model_name_title_addition}: Accuracy per class, average {accuracy:.0%}', loc='left')
        ax.set_xlabel('Class', fontsize=self.labels_fontsize)
        ax.set_ylabel('Accuracy', fontsize=self.labels_fontsize)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([y for y in ax.get_yticks()])
        ax.set_yticklabels([f'{int(y * 100)}%' for y in ax.get_yticks()], fontsize=self.labels_fontsize)
        ax.set_ylim(0, 1.05)

        if filter_name == 'all':
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            # # add vertical line after 10 and 20 words
            ax.axvline(x=9.5, color='k', linestyle='-', linewidth=0.5)
            ax.axvline(x=19.5, color='k', linestyle='-', linewidth=0.5)

            # # add texts at x=5 and x=15 and y=0.8 and x=25
            ax.text(x=4.5, y=0.9, s='Test words', ha='center', va='center')
            ax.text(x=15.5, y=0.9, s='Numbers', ha='center', va='center')
            ax.text(x=25.5, y=0.9, s='Other words', ha='center', va='center')

            for p in ax.patches:
                ax.annotate(f'{int(p.get_height()*100)}', (p.get_x() + p.get_width() / 2., p.get_height()-0.03), ha='center', va='center', xytext=(0, 10), textcoords='offset points', rotation=0)
        else: 
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=self.labels_fontsize)
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height()*100)}%', (p.get_x() + p.get_width() / 2., p.get_height()-0.03), ha='center', va='center', xytext=(0, 10), textcoords='offset points', rotation=0)

        fig.tight_layout()
        fig.savefig(plots_root + f'accuracy_per_class_{len(df["reference"].unique())}_classes.png', dpi=300)
        plt.show()

    def plot_per_class_mistakes(self, df, model_name_title_addition, accuracy, plots_root, filter_name, classes_to_leave):
        percentage_minimum = 0.03 if filter_name != 'all' else 0.08

        df_counts = df.groupby('reference').apply(lambda x: x['hypothesis'].value_counts()).reset_index()
        df_counts.columns = ['reference', 'hypothesis', 'count']
        df_counts['percentage'] = df_counts['count'] / df_counts.groupby('reference')['count'].transform('sum')
        df_counts = df_counts[df_counts['percentage'] >= percentage_minimum]
        df_counts = df_counts[df_counts['hypothesis'] != df_counts['reference']]
        for class_ in classes_to_leave:
            if class_ not in df_counts['reference'].values:
                emtpy_df = pd.DataFrame({'reference': [class_], 'hypothesis': [''], 'count': [0], 'percentage': [0]})
                df_counts = pd.concat([df_counts, emtpy_df])
        df_counts['reference'] = pd.Categorical(df_counts['reference'], categories=classes_to_leave, ordered=True)
        df_counts.sort_values(by=['reference', 'count'], ascending=[True, False], inplace=True)


        weight_counts = {}

        for h in df_counts['hypothesis'].unique():
            weight_counts[h] = np.array([0.0]*len(classes_to_leave))

        for i in range(df_counts.shape[0]):
            id = classes_to_leave.index(df_counts['reference'].values[i])
            weight_counts[df_counts['hypothesis'].values[i]][id] = df_counts['percentage'].values[i]

        width = 0.7

        fig, ax = plt.subplots(figsize=self.figsize)
        bottom = np.zeros(len(classes_to_leave))

        rotation = 90 if filter_name == 'all' else 0

        for label, weight_count in weight_counts.items():
            p = ax.bar(classes_to_leave, weight_count, width, label=label, bottom=bottom)
            for i, r in enumerate(p):
                if r.get_height() > percentage_minimum:
                    ax.text(r.get_x() + r.get_width() / 2., r.get_height() / 2. + bottom[i], f'{label}', ha='center', va='center', color='white', rotation=rotation)
            bottom += weight_count

        for i, r in enumerate(p):
            t = max(bottom[i], r.get_height())
            formatting = '{:.0%}' if t > 0 else '{:.0f}'
            ax.text(r.get_x() + r.get_width() / 2., 0.025 + bottom[i], formatting.format(t), ha='center', va='center', color='black')

        ax.set_title(f'Whisper {model_name_title_addition}: Most common mistakes (each above >{percentage_minimum}%), accuracy: {accuracy:.0%}', loc='right')
        ax.set_xlabel('Reference class', fontsize=self.labels_fontsize)
        ax.set_ylabel('Percentage of predictions', fontsize=self.labels_fontsize)
        ax.set_xticks([x for x in ax.get_xticks()])
        ax.set_ylim(0, 0.65)
        ax.set_yticks([y for y in ax.get_yticks()])
        ax.set_yticklabels([f'{int(y * 100)}%' for y in ax.get_yticks()], fontsize=self.labels_fontsize)
        ax.set_ylim(0, 0.65)

        if filter_name == 'all':

            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            # add vertical line after 10 and 20 words
            ax.axvline(x=9.5, color='k', linestyle='-', linewidth=0.5)
            ax.axvline(x=19.5, color='k', linestyle='-', linewidth=0.5)

            # add texts at x=5 and x=15 and y=0.8 and x=25
            ax.text(x=4.5, y=0.6, s='Test words', ha='center', va='center')
            ax.text(x=15.5, y=0.6, s='Numbers', ha='center', va='center')
            ax.text(x=25.5, y=0.6, s='Other words', ha='center', va='center')
        else:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=self.labels_fontsize)

        fig.tight_layout()
        fig.savefig(plots_root + f'most_common_mistakes_{len(df["reference"].unique())}_classes.png', dpi=300)

        plt.show()

    def plot_confusion_matrix(self, df, model_name_title_addition, accuracy, plots_root, filter_name, classes_to_leave):
        df_cf = df.loc[:, ['reference', 'hypothesis']]
        df_cf['hypothesis'] = df_cf.apply(lambda x: x['hypothesis'] if x['hypothesis'] in classes_to_leave else 'mistake', axis=1)
        df_confusion = pd.crosstab(df_cf['reference'], df_cf['hypothesis'])
        df_confusion = df_confusion.div(df_confusion.sum(axis=1), axis=0)
        df_confusion = df_confusion[classes_to_leave + ['mistake']]
        df_confusion = df_confusion.reindex(classes_to_leave)

        fig, ax = plt.subplots(figsize=self.conf_matrix_size) 
        if filter_name == 'all':
            sns.heatmap(df_confusion, linewidths=1, annot=False, cmap="flare", ax=ax)
        else:
            sns.heatmap(df_confusion, linewidths=1, annot=True, cmap="flare", ax=ax, fmt='.0%', annot_kws={"size": 16})
        plt.title(f"Whisper {model_name_title_addition}: Confusion matrix, accuracy {accuracy:.0%}", size=20, loc='left')
        plt.xlabel('Predicted class', size=16)
        plt.ylabel('Reference class', size=16)
        
        if filter_name == 'all':
            # add vertical line after 10 and 20 words
            ax.axvline(x=10, color='k', linestyle='-', linewidth=2)
            ax.axvline(x=20, color='k', linestyle='-', linewidth=2)
            # add horizontal line after 10 and 20 words
            ax.axhline(y=10, color='k', linestyle='-', linewidth=2)
            ax.axhline(y=20, color='k', linestyle='-', linewidth=2)

        ax.tick_params(axis='y', which='major', labelsize=16, rotation=0)
        ax.tick_params(axis='x', which='major', labelsize=16, rotation=90)

        fig.tight_layout()
        fig.savefig(plots_root + f'confusion_matrix_{len(df["reference"].unique())}_classes.png', dpi=300)
        plt.show()
        
    def analyze(self, model_size_name, filter_name):
        root_dir = f'whisper_{model_size_name}_raw/'
        overall_results_root = f'whisper_plots_{model_size_name}/'

        raw_csvs = [f for f in os.listdir(root_dir) if f.endswith('raw.csv')]
        raw_csvs_dict = {f.split('_')[5]: pd.read_csv(root_dir + f, index_col=0) for f in raw_csvs}
        df = pd.concat(raw_csvs_dict.values()).reset_index(drop=True)

        if filter_name == 'test classes':
            classes_to_leave = self.only_test_classes
            plots_root = overall_results_root + 'test_classes_'
        elif filter_name == 'number classes':
            classes_to_leave = self.only_number_classes
            plots_root = overall_results_root + 'numbers_only_'
        elif filter_name == 'other':
            classes_to_leave = self.only_other_classes
            plots_root = overall_results_root + 'others_'
        elif filter_name == 'all':
            classes_to_leave = [c for c in self.only_test_classes + self.only_number_classes + self.only_other_classes if c in df['reference'].unique()] # all present classes
            plots_root = overall_results_root + 'all_'

        if model_size_name == 'large':
            model_name_title_addition = 'large-v2'
        elif model_size_name == 'medium':
            model_name_title_addition = 'medium.en'

        df = df[df['reference'].isin(classes_to_leave)]

        df['hypothesis'] = df.apply(lambda x: self.clean_text(x['hypothesis']), axis=1)
        accuracy = df[df['hypothesis'] == df['reference']].shape[0] / df.shape[0]

        df['levenshtein'] = df.apply(lambda x: lev_distance(x['reference'],x['hypothesis']), axis=1)

        df['reference'] = pd.Categorical(df['reference'], categories=classes_to_leave, ordered=True)
        df.sort_values(by=['reference'], inplace=True)


        self.plot_lev(df, model_name_title_addition, plots_root)

        self.plot_per_class_acc(df, model_name_title_addition, accuracy, plots_root, filter_name)

        self.plot_per_class_mistakes(df, model_name_title_addition, accuracy, plots_root, filter_name, classes_to_leave)

        self.plot_confusion_matrix(df, model_name_title_addition, accuracy, plots_root, filter_name, classes_to_leave)