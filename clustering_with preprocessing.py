
#%%
import numpy as np
import pandas as pd
from Bio import Phylo
from Bio.Phylo import NexusIO
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import homogeneity_score

#%%
# reading file
data_df = pd.read_csv('/Users/gabby/Documents/GitHub/DataMiningProject/data copy/species_data.csv')
tree = Phylo.read('/Users/gabby/Documents/GitHub/DataMiningProject/data copy/terrestrial_vertebrate_tree.nexus.trees', "nexus")

#%%

#list of terminal descentants sorted
leaf_names = [leaf.name for leaf in tree.get_terminals()]
leaf_names = pd.Series(leaf_names)
leaf_names = leaf_names.sort_values()

#data already sorted and making list 
data_names = data_df.iloc[:, 0].values

not_in_tree = []
not_in_data = []

for leaf in leaf_names:
    if leaf not in data_names:
        not_in_data.append(leaf)
        
for name in data_names:
    if name not in leaf_names.values:
        not_in_tree.append(name)

# found the leaf nodes that are in tree but not in data_df

#removing the leaf nodes from tree than are not in data_df
for clade in not_in_data:
    tree.prune(clade)

#new sorted list with all terminal leaf names that are consistent with data_df    
new_leaf_names = [leaf.name for leaf in tree.get_terminals()]
new_leaf_names = pd.Series(new_leaf_names)
new_leaf_names = new_leaf_names.sort_values()

#new tree file
with open("new_tree", "w") as f:
    NexusIO.write([tree], f)
    


#%%
#Retrospectivley, this code is not really necessary
#Looking at tree we should do all four classes first (aves, amphibans, mamamals, repitiles seperately and then try to go smaller)    

# Amphibians
clade_amp1 = next(tree.find_clades("Typhlonectes_natans"))
clade_amp2 = next(tree.find_clades("Anaxyrus_baxteri"))
amp_com_anc_node = tree.common_ancestor(clade_amp1, clade_amp2)
amp_list = []
for tc in amp_com_anc_node.find_clades(terminal=True):
    amp_list.append(tc.name)
    
#mammals
clade_mam1 = next(tree.find_clades("Didelphis_marsupialis"))
clade_mam2 = next(tree.find_clades("Mustela_putorius"))
mam_com_anc_node = tree.common_ancestor(clade_mam1, clade_mam2)
mam_list = []
for tc in mam_com_anc_node.find_clades(terminal=True):
    mam_list.append(tc.name)

#Reptiles
clade_rep1 = next(tree.find_clades("Eublepharis_macularius"))
clade_rep2 = next(tree.find_clades("Crotalus_horridus"))
rep_com_anc_node = tree.common_ancestor(clade_rep1, clade_rep2)
rep_list = []
for tc in rep_com_anc_node.find_clades(terminal=True):
    rep_list.append(tc.name)

#Birds
clade_ava1 = next(tree.find_clades("Eudromia_elegans"))
clade_ava2 = next(tree.find_clades("Uraeginthus_bengalus"))
ava_com_anc_node = tree.common_ancestor(clade_ava1, clade_ava2)
ava_list = []
for tc in ava_com_anc_node.find_clades(terminal=True):
    ava_list.append(tc.name)

#making binary list for each class
def indicator(clade_list):
    bin_list = []
    indicator_ind = np.where(data_df['Species'].isin(clade_list))[0]
    for i in range(0, len(data_df)):
        if i in indicator_ind:
            bin_list.append(1)
        else:
            bin_list.append(0)
    return bin_list

#indicator lists for each class
rep_bin = indicator(rep_list)
ava_bin = indicator(ava_list)
mam_bin = indicator(mam_list)
amp_bin = indicator(amp_list)

data_df['isMam'] = mam_bin
data_df['isRep'] = rep_bin
data_df['isAmp'] = amp_bin
data_df['isAva'] = ava_bin

amps_df = data_df[data_df["Species"].isin(amp_list)]

#%%
#Binning on Neo
#Created boxplot to bin neoplasia counts
bp = plt.boxplot(data_df["neo"])

#found mins, boxplot  medians and maxs, and overall max
neo_min = min(data_df["neo"]) # 0
neo_boxmin = [line.get_ydata()[1] for line in bp['whiskers'][::2]] # 0
neo_q1 = [line.get_ydata()[1] for line in bp['boxes']] # 0
neo_med = [line.get_ydata()[0] for line in bp['medians']] # 2
neo_q3 = [line.get_ydata()[2] for line in bp['boxes']] # 5
neo_boxmax = [line.get_ydata()[1] for line in bp['whiskers'][1::2]] # 12
neo_max = max(data_df["neo"]) # 122

#design bins based of neo counts
newbins = []
for i in range(len(data_df)):
    if (data_df.iloc[i]["neo"] <= 2):
        newbins.append("low-med")
    elif (data_df.iloc[i]["neo"] <= 5):
        newbins.append("high")
    elif (data_df.iloc[i]["neo"] <= 12):
        newbins.append("very-high")
    elif (data_df.iloc[i]["neo"] > 12):
        newbins.append("outlying-high")
#add bins to new data frame
new_data_df = data_df.assign(bin_neo = newbins)

#add whether endo- or ectotherm
nclass, ntherm = [], []
for i in range(len(new_data_df)):
    if (new_data_df.iloc[i].therm == "ectothermic"):
        ntherm.append(0)
    else:
        ntherm.append(1)
    
    if (new_data_df.iloc[i].Class == "Amphibia"):
        nclass.append(0)
    elif (new_data_df.iloc[i].Class == "Aves"):
        nclass.append(1)
    elif (new_data_df.iloc[i].Class == "Mammalia"):
        nclass.append(2)
    else:
        nclass.append(3)

#find ratios of cancer types           
new_data_df['ben_ratio'] = (new_data_df['neo'] - new_data_df['mal'])/new_data_df['rec']

new_data_df['mal_ratio'] = (new_data_df['neo'] - new_data_df['mal'])/new_data_df['rec']

new_data_df['neo_ratio'] = new_data_df['neo']/new_data_df['rec']

#box plot of neoplasia rates to bin
bp = plt.boxplot(new_data_df["neo_ratio"])

#
neo_min = min(new_data_df["neo_ratio"]) # 0
neo_boxmin = [line.get_ydata()[1] for line in bp['whiskers'][::2]] # 0
neo_q1 = [line.get_ydata()[1] for line in bp['boxes']] # 0
neo_med = [line.get_ydata()[0] for line in bp['medians']] # 0.05
neo_q3 = [line.get_ydata()[2] for line in bp['boxes']] # 0.14285714285714285
neo_boxmax = [line.get_ydata()[1] for line in bp['whiskers'][1::2]] # 0.3541666666666667
neo_max = max(new_data_df["neo_ratio"]) 


neo_ratio_bins = []
for i in range(len(data_df)):
    if (new_data_df.iloc[i]["neo_ratio"] <= 0.05):
        neo_ratio_bins.append("low-med")
    elif (new_data_df.iloc[i]["neo_ratio"] <= 0.1429):
        neo_ratio_bins.append("high")
    elif (new_data_df.iloc[i]["neo_ratio"] <= 0.3542):
        neo_ratio_bins.append("very-high")
    elif (new_data_df.iloc[i]["neo_ratio"] > 0.3542):
        neo_ratio_bins.append("outlying-high")

new_data_df = new_data_df.assign(bin_neo_ratio = neo_ratio_bins)
new_data_df = new_data_df.assign(nClass = nclass)
new_data_df = new_data_df.assign(nTherm = ntherm)

data_feat = ["log_bm", "log_pl", "nClass", "nTherm"]
X = new_data_df[data_feat]
y = new_data_df.bin_neo


#%%

#data set conatining clade lables
clade_labels_df = pd.read_excel('/Users/gabby/Downloads/41559_2020_1321_MOESM2_ESM.xlsx', header=2)

#merging new_data_df with clade labels
df_for_cluster = pd.merge(new_data_df, clade_labels_df[['Binomial', 'Subclade']], left_on='Species',  right_on='Binomial', how='left'
)
#%%
#running ward linkage on cancer rates in mammals 
X_mam = new_data_df[new_data_df['isMam'] == 1]['neo_ratio'].to_numpy()

Z_neo = linkage(np.reshape(X_mam, (len(X_mam), 1)), 'ward')

#plotting dendrogram clustered on cancer rates label points with subclades
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z_neo,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,
    labels = list(df_for_cluster[df_for_cluster['isMam'] == 1]['Subclade'])
      # font size for the x axis labels
)
plt.show()


#%%    
# calculate covariance phlyogenetic matrix
def calculate_covar(tree, new_data_df):
    
    #all of the species 
    leaf_names_2 = list(new_data_df['Species'])
    n_leaf = len(leaf_names_2)
    cov_mat = np.zeros((n_leaf, n_leaf))
    for i in range(n_leaf):
        for j in range(i, n_leaf):
            if i == j:
                #for cell_ij where i = j this is the variance or the distance from the terminal leaf to root
                cell_cov = tree.distance(leaf_names_2[i], tree.root)
            else:
                #for cell_ij where i not = j this is the covariance or the distance from the common ancestor to the root of the tree
                anc = tree.common_ancestor(leaf_names_2[i], leaf_names_2[j])
                cell_cov = tree.distance(tree.root, anc)
            # the matrix is symmetric 
            cov_mat[i,j] = cell_cov
            cov_mat[j,i] = cell_cov
    cov_mat = pd.DataFrame(cov_mat, index = leaf_names_2, columns = leaf_names_2)
    return cov_mat

cov_matrix = calculate_covar(tree, new_data_df)

#calculating the first PCA component of the covariance matrix to put into ward
pca = PCA(n_components=1)
X_1D = pca.fit_transform(cov_matrix)

X_1D = X_1D.flatten()
            
#%% 

#with covariance
clade_dist = pd.DataFrame({"PC1": X_1D})

# un-log normalized to run through normalization such that mean = 0 and variance = 1
bm = pd.DataFrame({"bm": np.exp(new_data_df['log_bm'])})
pl = pd.DataFrame({"pl": np.exp(new_data_df['log_pl'])})

#reset indexs b/c covariance matrix index species rather than numbers - all are stored alphabetically 
X_cluster_pheno = pd.concat([
    bm.reset_index(drop=True), 
    pl.reset_index(drop=True), 
    clade_dist.reset_index(drop=True) 
], axis=1)

#normalize 
scaler = StandardScaler()
X_cluster_pheno = scaler.fit_transform(X_cluster_pheno)

#ward linkage
Z_link_pheno = linkage(X_cluster_pheno, 'ward')

#plot dendrogram
plt.title('Hierarchical Clustering Dendrogram with Covariance Matrix')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z_link_pheno,
    leaf_rotation=90.,
    leaf_font_size=8.,
    labels = list(new_data_df['bin_neo_ratio'])
)
plt.show()



#%%

# without covariance
clade_dist = pd.DataFrame({"PC1": X_1D})

# un-log normalized to run through normalization such that mean = 0 and variance = 1
bm = pd.DataFrame({"bm": np.exp(new_data_df['log_bm'])})
pl = pd.DataFrame({"pl": np.exp(new_data_df['log_pl'])})

#reset indexs to keep consistent with above
X_cluster_no = pd.concat([
    bm.reset_index(drop=True), 
    pl.reset_index(drop=True) 
], axis=1)

#normalize 
scaler = StandardScaler()
X_cluster_no = scaler.fit_transform(X_cluster_no)

#ward linkage
Z_link_no = linkage(X_cluster_no, 'ward')

#plot dendrogram
plt.title('Hierarchical Clustering Dendrogram without Covariance Matrix')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z_link_no,
    leaf_rotation=90., 
    leaf_font_size=8.,
    labels = list(new_data_df['bin_neo_ratio'])
      
)
plt.show()



#%%

#defining clusters and ploting violin plot for method without covariance
num_clusters_no = 5
cluster_labels = fcluster(Z_link_no, num_clusters_no, criterion='maxclust')

data_no = new_data_df.copy()

data_no['Clus'] = cluster_labels 

data_no['Clus'] = data_no['Clus'].astype('category') 


sns.violinplot(data=data_no,x='Clus', y='neo_ratio', palette="dark:#5A9_r", inner='point')
plt.title('Clustering without Covariance Matrix Against Cancer Rate')
plt.xlabel('Clusters')
plt.ylabel('Cancer Rate')
plt.show()

#%%

#defining clusters and ploting violin plot for method with covariance
data_phen = new_data_df.copy()

num_clusters_phen = 5
clus_labels = fcluster(Z_link_pheno, num_clusters_phen, criterion='maxclust')

data_phen['Clus'] = clus_labels 

data_phen['Clus'] = data_phen['Clus'].astype('category') 

sns.violinplot(data=data_phen, x='Clus', y='neo_ratio', palette="dark:#5A9_r", inner='point')
plt.title('Clustering with Covariance Matrix Against Cancer Rate')
plt.xlabel('Clusters')
plt.ylabel('Cancer Rate')
plt.show()

#%%
#plotting scatterplot of clustering method with covariance matrix
sns.scatterplot(x='log_bm', y='neo_ratio', hue='Clus', data=data_phen, palette='pastel', legend='full'
)
plt.title('Clustering with Covariance Matrix Against Cancer Rate')
plt.xlabel('Log Body Mass')
plt.ylabel('Cancer Rate')

#%%
#plotting scatterplot of clustering method without covariance martix
sns.scatterplot(x='log_bm', y='neo_ratio', hue='Clus', data=data_no, palette='pastel', legend='full'
)
plt.title('Clustering without Covariance Matrix Against Cancer Rate')
plt.xlabel('Log Body Mass')
plt.ylabel('Cancer Rate')

#%%

#Generate shilhouette scores
silhouette_no = silhouette_score(X_cluster_no, cluster_labels)

silhouette_phen = silhouette_score(X_cluster_pheno, clus_labels)

#%%

#dbi scores
dbs_no = davies_bouldin_score(X_cluster_no, cluster_labels)

dbs_pheno = davies_bouldin_score(X_cluster_pheno, clus_labels)

#%%
#homogenity scores
labels_true_phen = data_phen['bin_neo_ratio']
labels_cluster_phen = data_phen['Clus']

homo_phen = homogeneity_score(labels_true_phen, labels_cluster_phen)

labels_true_no = data_no['bin_neo_ratio']
labels_cluster_no = data_no['Clus']


homo_no = homogeneity_score(labels_true_no, labels_cluster_no)



