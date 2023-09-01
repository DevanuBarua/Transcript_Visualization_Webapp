import streamlit as st
import pandas as pd
from pydeseq2.ds import DeseqStats
from pydeseq2.dds import DeseqDataSet, deseq2_norm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from functools import reduce
import requests
import io


###Main Page

def home_page():

    st.write('''
    # MSc Bioinformatics Thesis Project 2695664B
    
    # Transcriptomics Data Visualization Webapp
    ''')

    # Master_counts upload
    countdata = st.file_uploader('Upload Master counts file')
    st.write('OR')
    # Merge_counts function
    if st.button('Create Counts file'):
        st.session_state['page'] = 'Counts'

    if countdata is None:
        st.write('#### Please upload a Counts file')
        st.write('''
            ##### Description: A tab-separated master file for all counts with the sample names as rows and the genes with their counts as columns
            ''')
        eg_data = {
            'GeneA': [1.2, 2.4, 2.5, 4.8],
            'GeneB': [1.9, 2.9, 5.6, 1.0]
        }
        eg_rows = ['Sample1_rep1', 'Sample1_rep2', 'Sample2_rep1', 'Sample2_rep2']
        eg_df = pd.DataFrame(eg_data, index=eg_rows)
        st.write('Example dataset;', eg_df)
    else:
        # Read master_counts
        counts_df = pd.read_csv(countdata, sep='\t', index_col=0)
        st.write('Number of Genes found:', counts_df.shape[1])  # Number of columns (genes)
        # Clinical information
        clinicaldata = st.file_uploader('Upload clinical information file')
        if clinicaldata is None:
            st.write('#### Please upload clinical file')
            st.write('''
                    ##### Description: A tab-separated file with the two columns; 'samples' and 'condition'. 'samples' should contain
                    ##### sample specific replicate names, 'condition' should contain the sample names the replicates belong to
                    ##### Please ensure the columns are named exactly as specified
                    ''')
            clin_data = {
                'sample': ['Male_rep1', 'Male_rep2', 'Female_rep1', 'Female_rep2'],
                'condition': ['Male', 'Male', 'Female', 'Female']
            }
            clin_df = pd.DataFrame(clin_data)
            st.write(clin_df)
        else:
            # Read clinical info
            clinical_df = pd.read_csv(clinicaldata, sep='\t', index_col=0)
            # Pull out unique clinical conditions
            st.write('Number of Conditions:', len(clinical_df['condition'].unique()))
            st.write('Condtions:', clinical_df['condition'].unique())

            # Remove genes that are NA or have total expression < 10 across all conditions
            samples_to_keep = ~clinical_df.condition.isna()
            counts_df = counts_df.loc[samples_to_keep]
            clinical_df = clinical_df.loc[samples_to_keep]
            genes_to_keep = counts_df.columns[counts_df.sum(axis=0) >= 10]
            counts_df = counts_df[genes_to_keep]

            # Count genes kept after filtration
            st.write('Genes kept after initial filtering:', counts_df.shape[1])

            # Pass count and clinical dfs into session states for other pages
            st.session_state['counts_df'] = counts_df
            st.session_state['clinical_df'] = clinical_df


###Counts page


def counts_page():
    st.write('Counts Page')
    # Upload csv file of unique sample names
    samples_tsv = st.file_uploader('Upload Sample Names')
    # make an empty list to contain individual count dfs to be merged
    df_list = []
    if samples_tsv is None:
        st.write('Please upload Sample Names')
        st.write('''
                ##### Description: A tab-delimited file for the names of all samples in dataset
                ##### Example; Female{Tab}Male{Tab}Asexual
                ''')
    else:
        # read unique sample names
        samples_file = pd.read_csv(samples_tsv, header=None, sep='\t')
        # input replicates for each sample
        reps = st.number_input('Insert Replicates for each sample', min_value=1, value=1)
        st.write('''
        ##### Input number of replicates for each sample, the samples must have same number of replicates
        ''')
        # provide filepath for the sample folder (before sample folders)
        filepath = st.text_input('Input Samples Folder Path')
        st.write('''
        ##### Input the absolute path for the folder which contains the separate folders for the counts of each replicate
        ##### without the '/' at the end of the path
        ##### The replicate folders must be of the form '{sample}_rep{replicate_no}', eg. Male_rep3
        ''')
        # provide name for counts file for each sample (has to be the same)
        countfilename = st.text_input('Input Raw Counts Filename')
        st.write('''
        ##### Input the name for the raw counts file inside the replicate folders in the above filepath. Must be the same for all replicates
        ''')
        if st.button('Confirm Samples'):
            # for each sample name in samples_csv
            for sample in samples_file.iloc[0]:
                for rep in range(1, reps+1):
                    # open each counts file for particular sample and replicate
                    sample_df = pd.read_csv(f'{filepath}/{sample}_rep{rep}/{countfilename}', delimiter='\t',
                                             header=None, skipfooter=5, names=['ID',f"{sample}_rep{rep}"],
                                             engine='python', index_col=0) # skipfooter since last 5 lines not needed
                    # append each individual sample df to df_list for merging
                    df_list.append(sample_df)
            # make master_counts using reduce and outer join on ID for each df in df_list
            master_counts_df = reduce(lambda left, right: pd.merge(left, right, on='ID', how='outer'), df_list)
            # Transpose to have gene names as columns and samples as row names
            master_counts_df = master_counts_df.T
            st.write(master_counts_df)
            # save master_counts to same folder as samples as tsv
            # tsv should be used in home page for analysis
            @st.cache_data
            def convert_df(df):
                return df.to_csv(sep='\t').encode('utf-8')

            st.download_button(
                label='Download Master Counts file',
                data=convert_df(master_counts_df),
                file_name='master_counts.tsv',
                mime='text/csv',
            )


###PCA page
def pca_page():
    # use counts and clinical file from session state passed  on from home_page
    counts_df = st.session_state['counts_df']
    clinical_df = st.session_state['clinical_df']

    # normalize counts file using ratio of means
    norm_counts_df = deseq2_norm(counts=counts_df)

    # perform PCA on normalized counts
    pca = PCA(n_components=2)
    counts_pca = pca.fit_transform(norm_counts_df[0])
    pca_df = pd.DataFrame(counts_pca, columns=['PC1', 'PC2'])

    # create 'condition' column using clinical file for clustering
    pca_df['condition'] = clinical_df['condition'].values
    pca_ratios = pca.explained_variance_ratio_

    # cluster pca using 'conditions' column on clinical file
    kmeans = KMeans(n_clusters=len(clinical_df['condition'].unique()))
    clusters = kmeans.fit_predict(pca_df[['PC1', 'PC2']])
    clusters_df = pd.DataFrame(clusters, columns=['clusters'])
    clusters_df['condition'] = pca_df['condition']

    # create colormap to colour the clusters
    cmap = matplotlib.colormaps.get_cmap('tab20')

    # generate scatterplot and colour the datapoints using the clusters
    fig, ax = plt.subplots(figsize=(10, 10))
    for label in clinical_df['condition'].unique():
        pca_subset = pca_df.loc[pca_df['condition'] == label]
        cluster_subset = clusters_df.loc[clusters_df['condition'] == label]
        cluster_index = cluster_subset['clusters'].values[0]
        color = cmap(cluster_index)
        colors = [color] * len(pca_subset)
        ax.scatter(pca_subset['PC1'], pca_subset['PC2'], c=colors, label=label)
    plt.legend()
    plt.xlabel(f'PC1:{pca_ratios[0] * 100:.2f}%')
    plt.ylabel(f'PC2:{pca_ratios[1] * 100:.2f}%')
    plt.title('PCA with Clustered Samples')
    plt.figure(dpi=10)

    # show PCA plot
    st.write('''
    ### PCA plot of first two principal components for groups in clinical dataframe\n
    ##### If any groups cluster together, New groups can be made on DESeq2 to consider them as the same group\n\n 
    ''')
    st.pyplot(fig)

    # pass on normalized counts into session state
    st.session_state['norm_counts_df'] = norm_counts_df


###Deseq Page

def deseq_page():
    # use counts and clinical files from session state
    counts_df = st.session_state['counts_df']
    clinical_df = st.session_state['clinical_df']

    # create empty dictionary for each Deseq2 result
    dds_dict = {}

    ####PyDeseq2

    st.header('Select New Groups')
    # Show unique samples to be used for Deseq2 taken from clinical file
    unique_samples = clinical_df['condition'].unique()
    st.write('Unique Groups found:')
    sample_count = 1
    for i in unique_samples:
        st.write(f'{sample_count} {i}')
        sample_count += 1

    # select new groups if needed from groups in clinical file
    if 'user_groups' not in st.session_state:
        # empty dict for group_count function
        st.session_state['user_groups'] = {}

    # define function that counts number of groups in clinical file and 'user_groups' dict
    def group_count(old_group, new_group):
        old_group_count = len(old_group)
        new_group_count = 0
        for item in new_group.keys():
            new_group_count += 1

        return old_group_count == new_group_count

    # define function for selecting new groups from groups provided in clinical file
    # provide old group list and empty dictionary to store new groups
    def select_new_groups(old_group, new_group):
        # select groups for new group, then name the new group
        selected_samples = st.multiselect('Select Samples', old_group)
        st.write('''
        ##### Select One or more samples to take as a new group
        ##### Remove the samples after new  group is made to add another or to add column to clinical dataframe
        ''')
        group_name = st.text_input(f'Enter group name for {selected_samples}')
        st.write('''
        ##### Enter the new group name for the selected samples, This will be added into the clinical dataframe and used in the DESeq2 calculation of FoldChange
        ''')

        # add groups to empty dict
        if st.button('Create New Group'):
            for sample_name in selected_samples:
                new_group.update({sample_name: group_name})

    # if all groups in old groups doesnt match the count of groups used to make new ones, run both functions
    # use 'user_groups' as empty dict for saving new groups
    if group_count(unique_samples, st.session_state['user_groups']) is False:
        select_new_groups(unique_samples, st.session_state['user_groups'])
    else:
        st.write('New Groups Selected, they can be now added to the clinical dataframe')
        # add user defined groups to clinical_df for further processing
        if st.button('Add Column to Clinical_Dataframe'):
            clinical_df['user_condition'] = clinical_df['condition'].map(st.session_state['user_groups'])
    # show new groups
    st.write(st.session_state['user_groups'])

    # show modified clinical_df
    st.write(clinical_df)
    st.write('''
    The clinical dataframe that will be used as the condition for DESeq2, if satisfactory, 'Perform DESeq2' then go to 'Analysis' page
    ''')

    # if user defined new conditions, use them for Deseq2
    if st.button('Perform DESeq2'):
        if 'user_condition' in clinical_df.columns:
            user_unique_samples = clinical_df['user_condition'].unique()
            # perform deseq for each sample in user defined conditions
            for sample in user_unique_samples:
                dds = DeseqDataSet(
                    counts=counts_df,
                    clinical=clinical_df,
                    design_factors='user_condition',
                    ref_level=['user_condition', sample],
                    refit_cooks=True,
                    n_cpus=8)
                dds.deseq2()
                # make list to store each comparison for current sample
                dds_list = []
                for ref_sample in user_unique_samples:
                    # use only samples that are not current sample for comparison against current sample
                    if ref_sample != sample:
                        dds2_res = DeseqStats(dds, contrast=['user_condition', ref_sample, sample], alpha=0.05,
                                              cooks_filter=True, independent_filter=True)
                        dds2_res.summary()
                        # store each comparison in dds_list
                        dds_list.append(dds2_res)
                # store all comparisons for a sample in dict under sample name
                dds_dict.update({sample: dds_list})

        # perform deseq on conditions given in clinical file if user hasnt defined new groups
        elif 'user_condition' not in clinical_df.columns:
            for sample in unique_samples:
                # perform deseq on unique each sample in clinical file
                dds = DeseqDataSet(
                    counts=counts_df,
                    clinical=clinical_df,
                    design_factors='condition',
                    ref_level=['condition', sample],
                    refit_cooks=True,
                    n_cpus=8)
                dds.deseq2()
                # make an empty list to store comparison results for each sample
                dds_list = []
                for ref_sample in unique_samples:
                    # perform comparison on all samples that aren't current sample
                    if ref_sample != sample:
                        dds_res = DeseqStats(dds, contrast=['condition', ref_sample, sample], alpha=0.05,
                                             cooks_filter=True, independent_filter=True)
                        dds_res.summary()
                        # store comparison results to dds_list
                        dds_list.append(dds_res)
                # store all comparison results for each sample into dict under sample name
                dds_dict.update({sample: dds_list})

        st.write('Deseq Done!')
        # pass deseq results and cliniical file into session state for use in other pages
        st.session_state['dds_dict'] = dds_dict
        st.session_state['clinical_df'] = clinical_df


### Feature Page
def features_page():
    # pull in all files to be used for page from session state
    dds_dict = st.session_state['dds_dict']
    clinical_df = st.session_state['clinical_df']
    counts_df = st.session_state['counts_df']
    # use normalized counts generated from pca_page or perform normalization
    if 'norm_counts_df' not in st.session_state:
        norm_counts_df = deseq2_norm(counts=counts_df)
    else:
        norm_counts_df = st.session_state['norm_counts_df']
    norm_counts_df = pd.DataFrame(norm_counts_df[0]).T
    # extract unqiue samples based on whether user has defined or not
    if 'user_condition' in clinical_df:
        samples = clinical_df['user_condition'].unique()
    else:
        samples = clinical_df['condition'].unique()
    # create empty df that will store all foldChange values for each comparison based on samples
    exp_df = pd.DataFrame()
    for key, val in dds_dict.items():
        for i in val:
            # store all foldChange and adjusted p-values for each comparison in exp_df
            fc_column_name = (f'{i.contrast[1]}v{i.contrast[2]}')
            p_value = (f'{i.contrast[1]}v{i.contrast[2]}_pval')
            exp_data = (i.results_df['log2FoldChange'])
            exp_p = i.results_df['padj']
            exp_df[fc_column_name] = exp_data
            exp_df[p_value] = exp_p
    # if user has not defined groups
    # store average expressions from normalized counts
    if 'user_condition' not in clinical_df:
        pc_fc_df = pd.DataFrame()
        for sample in samples:
            # filter norm_counts for only the current sample
            filtered_df = norm_counts_df.filter(like=sample, axis=1)
            # calculate average expression across all replicates for current sample
            filtered_df.loc[:, 'avg_exp'] = filtered_df[filtered_df.columns[0:]].mean(numeric_only=True, axis=1)
            # store average expression for each gene in current sample in new df
            # then rank the expression through pct, calculate percentile for each gene then store as new column
            pc_fc_df[f'{sample}_avg_exp'] = filtered_df['avg_exp']
            pc_fc_df[f'{sample}_pc'] = pc_fc_df[f'{sample}_avg_exp'].rank(pct=True)
            pc_fc_df[f'{sample}_pc'] = pc_fc_df[f'{sample}_pc'] * 100

    # if user has defined groups
    else:
        pc_fc_df = pd.DataFrame()
        for sample in samples:
            # pull out all samples that are part of new groups
            relevant_samples_df = clinical_df[(clinical_df['user_condition'] == sample)]
            relevant_samples = relevant_samples_df['condition'].unique()
            # make a new df that will store average expressions for each sample that is a part of new group
            user_exp_df = pd.DataFrame()
            for osamples in relevant_samples:
                # filter norm_counts for each sample that is part of new group
                filtered_df = norm_counts_df.filter(like=osamples, axis=1)
                # calculate average expression across replicates
                filtered_df.loc[:, 'avg_exp'] = filtered_df[filtered_df.columns[0:]].mean(numeric_only=True, axis=1)
                # add expressions to user_exp_df for further calculation
                # each new group will average expressions for all samples that are part of new group
                user_exp_df[f'{osamples}_exp'] = filtered_df['avg_exp']
            # calculate average for the new groups using average expressions for each sample part of the new group
            pc_fc_df[f'{sample}_avg_exp'] = user_exp_df[user_exp_df.columns[0:]].mean(numeric_only=True, axis=1)
            # rank with pct and add percentile column
            pc_fc_df[f'{sample}_pc'] = pc_fc_df[f'{sample}_avg_exp'].rank(pct=True)
            pc_fc_df[f'{sample}_pc'] = pc_fc_df[f'{sample}_pc'] * 100

    # show each plot function that can be made with the data
    with st.sidebar:
        # make radio buttons for each feature
        feature = st.radio('Choose Feature', ('DE', 'FoldChange', 'Percentile'))
    if feature == 'DE':
        st.write('### DE_Comparison')
        # provide inputs for comparison plot
        ref_fc = st.selectbox('Select Reference Sample', samples)
        com_fc = st.selectbox('Select Comparison Sample', samples)
        fc_filter = st.number_input('Minimum log2(FoldChange)')
        p_filter = st.number_input('Minimum adjusted p-value')
        # generate volcano plot for selected inputs
        if st.button('Plot Volcano'):
            # create new df that will be used for plot and results
            de_df = pd.DataFrame()
            # extract fc and padj for selected inputs
            if ref_fc == com_fc:
                st.write('#### Error: Please ensure Reference and Comparison samples are different')
            else:
                foldChange = f'{com_fc}v{ref_fc}'
                padj = f'{com_fc}v{ref_fc}_pval'
                de_df['log2(foldChange)'] = exp_df[foldChange]
                de_df['padj'] = exp_df[padj]

                # define function to check for significance according to filters
                def significant_check(row):
                    if row['log2(foldChange)'] >= fc_filter and row['padj'] < p_filter:
                        return 'Upregulated'
                    elif row['log2(foldChange)'] <= -fc_filter and row['padj'] < p_filter:
                        return 'Downregulated'
                    else:
                        return 'Not_Significant'

                # create new 'significance' column that will use above function to return significance values
                de_df['significance'] = de_df.apply(significant_check, axis=1)
                # reindex df according to absolute values  of foldchange
                de_df = de_df.reindex(de_df['log2(foldChange)'].abs().sort_values(ascending=False).index)

                # create colormap for datapoint based on significance
                cmap = {'Upregulated': 'red', 'Downregulated': 'blue', 'Not_Significant': 'gray'}

                # generate scatterplot using foldchange and log10(padj) of de_df
                # color the datapoints based on significance column and colormap
                plt.scatter(de_df['log2(foldChange)'], -np.log10(de_df['padj']), c=de_df['significance'].map(cmap))
                # create legends for each group
                legend_handles = [
                    plt.Line2D([0], [0], marker='o', color='red', label='Upregulated'),
                    plt.Line2D([0], [0], marker='o', color='blue', label='Downregulated'),
                    plt.Line2D([0], [0], marker='o', color='gray', label='Not Significant')
                ]
                plt.legend(handles=legend_handles, title='Significance')
                # create lines on the significance thresholds
                plt.axvline(x=fc_filter, color='black', linestyle='--')
                plt.axvline(x=-fc_filter, color='black', linestyle='--')
                plt.axhline(y=-np.log10(p_filter), color='black', linestyle='--')
                plt.title('log2(FoldChange) vs Adj-pval')
                plt.xlabel('log2(FoldChange)')
                plt.ylabel('-log10(adjusted p-value)')
                # show plot
                st.pyplot(plt)

                # show upto 6 decimal points for padj column
                de_df['padj'] = de_df['padj'].apply('{:.6e}'.format)
                # display final df that only contains significant genes
                de_df_final = de_df[(de_df['significance'] == 'Upregulated') | (de_df['significance'] == 'Downregulated')]
                st.write(de_df_final)
                # show different stats for genes; Significant, Non_significant etc.
                st.write('Non-Significant Genes =', len(de_df[de_df['significance'] == 'Not_Significant']))
                st.write('Significant Genes =', len(de_df_final))
                st.write('Upregulated Genes =', len(de_df_final[de_df_final['significance'] == 'Upregulated']))
                st.write('Downregulated Genes =', len(de_df_final[de_df_final['significance'] == 'Downregulated']))

                st.download_button(
                    label='Download Significant Genes',
                    data=de_df_final.to_csv(sep='\t').encode('utf-8'),
                    file_name=f'{com_fc}v{ref_fc}_DE.tsv',
                    mime='text/csv'
                )

                st.session_state['GeneList'] = de_df_final

    elif feature == 'FoldChange':
        st.write('### FoldChange Comparison')
        # provide inputs for plot and results
        ref_sample = st.selectbox('Reference Sample', samples)
        com_sample = st.selectbox('Comparison Sample', samples)
        fc_filter = st.number_input('Minimum log2(FoldChange)')
        if st.button('Scatter Plot'):
            # make dataframe to store results and make plot
            fc_df_pre = pd.DataFrame()
            # pull out 'foldchange' column for comparison
            if ref_sample == com_sample:
                st.write('#### Error: Please ensure Reference and Comparison samples are different')
            else:
                fc_df_pre['log2(foldChange)'] = exp_df[f'{com_sample}v{ref_sample}']
                # calculate average for expression values for both samples in comparison
                fc_df_pre['log2(Counts)'] = (pc_fc_df[f'{ref_sample}_avg_exp'] + pc_fc_df[f'{com_sample}_avg_exp']) / 2
                # remove genes that have no expression across both samples
                fc_df = fc_df_pre.copy()
                fc_df['log2(Counts)'] = fc_df['log2(Counts)'].replace(0, np.nan)
                fc_df.dropna(inplace=True)
                # reindex df according to descending absolute foldchange values
                fc_df = fc_df.reindex(fc_df['log2(foldChange)'].abs().sort_values(ascending=False).index)
                # perform log transformation of expressions
                fc_df['log2(Counts)'] = np.log2(fc_df['log2(Counts)'])

                # define significant function based on filters
                def significant_check(row):
                    if abs(row['log2(foldChange)']) >= fc_filter:
                        return 'Significant'
                    else:
                        return 'Non_Significant'

                # generate new column for significance using above function
                fc_df['significance'] = fc_df.apply(significant_check, axis=1)

                # create colormap based on significance
                cmap = {'Significant': 'red', 'Non_Significant': 'gray'}
                # generate scatterplot on fc vs exp and colour them based on significance
                plt.scatter(fc_df['log2(Counts)'], fc_df['log2(foldChange)'], c=fc_df['significance'].map(cmap))
                # create legend for significance
                legend_handles = [
                    plt.Line2D([0], [0], marker='o', color='red', label='Significant'),
                    plt.Line2D([0], [0], marker='o', color='gray', label='Not Significant')
                ]
                plt.legend(handles=legend_handles)
                # create lines to show filter cutoffs
                plt.axhline(y=fc_filter, color='black', linestyle='--')
                plt.axhline(y=-fc_filter, color='black', linestyle='--')
                plt.title('log2(FoldChange) vs log2(Normalized Counts)')
                plt.xlabel('log2(Normalized Counts)')
                plt.ylabel('log2(FoldChange)')
                # show MA plot
                st.pyplot(plt)

                # create final df that contains only significant genes
                fc_df_final = fc_df[fc_df['significance'] == 'Significant']
                # display final df
                st.write(fc_df_final)
                # show stats for genes
                st.write('Total Genes:', len(fc_df_pre))
                st.write('Genes after Preprocessing:', len(fc_df))
                st.write('Non-Significant Genes:', len(fc_df[fc_df['significance'] == 'Non_Significant']))
                st.write('Significant Genes:', len(fc_df_final))

                st.download_button(
                    label='Download Significant Genes',
                    data=fc_df_final.to_csv(sep='\t').encode('utf-8'),
                    file_name=f'{com_sample}v{ref_sample}_FCvsExp.tsv',
                    mime='text/csv'
                )

                st.session_state['GeneList'] = fc_df_final

    elif feature == 'Percentile':
        st.write('### Percentile Comparison')
        # provide inputs for histogram
        pc_sample = st.selectbox('Select Sample', samples)
        min_pc = st.number_input('Minimum Expression Percentile', value=80.0)
        max_pc = st.number_input('Maximum Expression Percentile', value=100.0)

        if st.button('Plot Histogram'):
            # make empty dataframe for plot and results
            pc_df = pd.DataFrame()
            # extract relevant columns from avg exp df based on given inputs
            pc_df['Counts'] = pc_fc_df[f'{pc_sample}_avg_exp']
            pc_df['Percentile'] = pc_fc_df[f'{pc_sample}_pc']
            # sort df (descending) on 'percentile' column
            pc_df = pc_df.sort_values('Percentile', ascending=False)
            # make final df to store results
            pc_df_final = pc_df.copy()
            # remove genes with zero expression in the sample to perform log2 change
            pc_df_final['Counts'] = pc_df_final['Counts'].replace(0, np.nan)
            pc_df_final.dropna(inplace=True)
            # perform log transformation for expression values
            pc_df_final['log2(Counts)'] = np.log2(pc_df_final['Counts'])
            # filter df based on percentile thresholds
            pc_filtered_df = pc_df_final[(pc_df_final['Percentile'] >= min_pc) & (pc_df_final['Percentile'] <= max_pc)]
            # find exact gene that are above and below the thresholds (to be used for threshold lines)
            min_pc_threshold = pc_filtered_df.iloc[0]['log2(Counts)']
            max_pc_threshold = pc_filtered_df.iloc[len(pc_filtered_df) - 1]['log2(Counts)']
            # generate histogram on log2 expression values
            plt.hist(pc_df_final['log2(Counts)'])
            # make threshold lines for percentile thresholds
            plt.axvline(x=min_pc_threshold, color='black', linestyle='--')
            plt.axvline(x=max_pc_threshold, color='black', linestyle='--')
            plt.title(f'Expression Frequency {min_pc}-{max_pc} Percentile')
            plt.xlabel('log2(Normalized Counts)')
            plt.ylabel('Frequency of Genes')
            # show plot
            st.pyplot(plt)

            # show stats on genes
            st.write(pc_filtered_df)
            st.write('Initial Genes:', len(pc_df))
            st.write('Genes After Preprocessing:', len(pc_df_final))
            st.write('Genes in Percentile Threshold:', len(pc_filtered_df))

            st.download_button(
                label='Download Significant Genes',
                data=pc_filtered_df.to_csv(sep='\t').encode('utf-8'),
                file_name=f'{pc_sample}_percentile.tsv',
                mime='text/csv'
            )

            st.session_state['GeneList'] = pc_df_final

def products_page():
    genelist = st.session_state['GeneList']
    st.write(
        'If dataset is present in the VEuPathDB project databases and can be used to get product descriptions;')
    websites = ['None', 'amoebadb', 'plasmodb', 'cryptodb']
    website = st.selectbox('Select VEuPathDB website', websites)
    website2 = website.rstrip('db')
    organism = st.text_input('Organism Reference Name separated by spaces')
    if website != 'None':
        organism_url = organism.replace(' ', '%20')
        url = f"https://{website}.org/{website2}/service/record-types/transcript/searches/GenesByTaxon/reports/attributesTabular?organism=%5B%22{organism_url}%22%5D&reportConfig=%7B%22attributes%22%3A%5B%22primary_key%22%2C%22source_id%22%2C%22organism%22%2C%22gene_location_text%22%2C%22gene_product%22%5D%2C%22includeHeader%22%3Atrue%2C%22attachmentType%22%3A%22csv%22%7D"
        if st.button('Fetch Data'):
            response = requests.get(url)
            if response.status_code == 200:
                # Convert the response content to a DataFrame
                try:
                    products_df = pd.read_csv(io.BytesIO(response.content))
                    products_df.set_index('Gene ID', inplace=True)

                    # Display the DataFrame
                    st.write(f'Data fetched from the website: {website}')
                    # Make a common index from the genelist df
                    common_index = products_df.index.intersection(genelist.index)
                    products_df_filtered = products_df.loc[common_index]
                    # Remove duplicated rows in products_df
                    products_df_filtered = products_df_filtered[~products_df_filtered.index.duplicated(keep='last')]
                    # Extract relevant column
                    genelist['Product_Descriptions'] = products_df_filtered['Product Description']
                    st.write(genelist)

                    st.download_button(
                        label='Download Gene List',
                        data=genelist.to_csv(sep='\t').encode('utf-8'),
                        file_name='Genelist_Products.tsv',
                        mime='text/csv'
                    )

                except Exception as e:
                    st.write("Error converting response to DataFrame:", e)

            else:
                st.write(f"Error getting data, please check if name of organism and website has been correctly inputted")

### Page_layout

if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

if st.session_state['page'] == 'Home':
    home_page()
    if st.button('PCA'):
        st.session_state['page'] = 'PCA'
    elif st.button('Deseq'):
        st.session_state['page'] = 'Deseq'
elif st.session_state['page'] == 'Deseq':
    deseq_page()
    if st.button('Analysis'):
        st.session_state['page'] = 'Features'
    elif st.button('Home'):
        st.session_state['page'] = 'Home'
elif st.session_state['page'] == 'PCA':
    pca_page()
    if st.button('Home'):
        st.session_state['page'] = 'Home'
    elif st.button('Deseq'):
        st.session_state['page'] = 'Deseq'
elif st.session_state['page'] == 'Features':
    features_page()
    if st.button('Add Product Descriptions'):
        st.session_state['page'] = 'Products'
    elif st.button('Home'):
        st.session_state['page'] = 'Home'
elif st.session_state['page'] == 'Counts':
    counts_page()
    if st.button('Home'):
        st.session_state['page'] = 'Home'
elif st.session_state['page'] == 'Products':
    products_page()
    if st.button('Back to Analysis'):
        st.session_state['page'] = 'Features'
    elif st.button('Home'):
        st.session_state['page'] = 'Home'

