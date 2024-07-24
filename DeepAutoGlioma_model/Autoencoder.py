#3. Data Integration by autoencoder
import numpy as np
import pandas as pd
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras.utils import plot_model
from keras.layers import Input, Dense
from keras.layers.merge import concatenate

#Importing the Libraries
from sklearn.datasets import make_classification
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model
import os


import matplotlib as mpl
from keras.layers import Input, Dense, Dropout
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

GE_data = pd.read_csv('Xtrain_70percent_data_GE.csv')
# define dataset
print(GE_data['sample'].value_counts(normalize=True))


y = GE_data['sample']

GE_data.drop([ 'sample'], axis=1, inplace=True)

# summarize the dataset
#print(X.shape, y.shape)
from sklearn.preprocessing import MinMaxScaler
# split into train test sets
#X_train, X_test, y_train, y_test = train_test_split(GE_data, y, test_size=0.33, random_state=1)
# scale data
t = MinMaxScaler()
t.fit(GE_data)
X_train = t.transform(GE_data)
#X_test = t.transform(X_test)

DNA_methyl_new= pd.read_csv('Xtrain_70percent_data_DNAmethyl.csv')
# define dataset
print(DNA_methyl_new['sample'].value_counts(normalize=True))

y_D = DNA_methyl_new['sample']

DNA_methyl_new.drop([ 'sample'], axis=1, inplace=True)
X_train_D=DNA_methyl_new
X_train_D.dtypes

Y_scRNAseq = y

Y_scProteomics = y_D
print("\n")


# LOG-TRANSFORM DATA
X_scRNAseq = X_train

X_scProteomics = X_train_D

################################################## AUTOENCODER ##################################################

# Input Layer
ncol_scRNAseq = X_scRNAseq.shape[1]
input_dim_scRNAseq = Input(shape = (ncol_scRNAseq, ), name = "scRNAseq")
ncol_scProteomics = X_scProteomics.shape[1]
input_dim_scProteomics = Input(shape = (ncol_scProteomics, ), name = "scProteomics")

# Dimensions of Encoder for each OMIC
encoding_dim_scRNAseq = 1110
encoding_dim_scProteomics = 3204

# Encoder layer for each OMIC
encoded_scRNAseq = Dense(encoding_dim_scRNAseq, activation = 'relu', name = "Encoder_scRNAseq")(input_dim_scRNAseq)
encoded_scProteomics = Dense(encoding_dim_scProteomics, activation = 'relu', name = "Encoder_scProteomics")(input_dim_scProteomics)

# Merging Encoder layers from different OMICs
merge = concatenate([encoded_scRNAseq, encoded_scProteomics])

# Bottleneck compression
bottleneck = Dense(400, kernel_initializer = 'uniform', activation = 'linear', name = "Bottleneck")(merge)

#Inverse merging
merge_inverse = Dense(encoding_dim_scRNAseq + encoding_dim_scProteomics, activation = 'relu', name = "Concatenate_Inverse")(bottleneck)

# Decoder layer for each OMIC
decoded_scRNAseq = Dense(ncol_scRNAseq, activation = 'relu', name = "Decoder_scRNAseq")(merge_inverse)
decoded_scProteomics = Dense(ncol_scProteomics, activation = 'relu', name = "Decoder_scProteomics")(merge_inverse)

# Combining Encoder and Decoder into an Autoencoder model
autoencoder = Model(inputs = [input_dim_scRNAseq, input_dim_scProteomics], outputs = [decoded_scRNAseq, decoded_scProteomics])

# Compile Autoencoder
autoencoder.compile(optimizer = 'adam', loss={'Decoder_scRNAseq': 'mean_squared_error', 'Decoder_scProteomics': 'mean_squared_error'})
autoencoder.summary()

# Autoencoder graph
plot_model(autoencoder, to_file='autoencoder_graph.png')

# Autoencoder training
estimator = autoencoder.fit([X_scRNAseq, X_scProteomics], [X_scRNAseq, X_scProteomics], epochs = 200, batch_size = 16, validation_split = 0.3, shuffle = True, verbose = 1)
print("Training Loss: ",estimator.history['loss'][-1])
print("Validation Loss: ",estimator.history['val_loss'][-1])
#plt.figure(figsize=(20, 15))
plt.plot(estimator.history['loss'])
plt.plot(estimator.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc = 'upper right')
plt.show()


# Encoder model
encoder = Model(inputs = [input_dim_scRNAseq, input_dim_scProteomics], outputs = bottleneck)
bottleneck_representation = encoder.predict([X_scRNAseq, X_scProteomics])
print(pd.DataFrame(bottleneck_representation).shape)
print(pd.DataFrame(bottleneck_representation).iloc[0:5,0:5])

# Dimensionality reduction plot
#plt.figure(figsize=(20, 15))
plt.scatter(bottleneck_representation[:, 0], bottleneck_representation[:, 1], c = Y_scRNAseq, cmap = 'tab20', s = 10)
plt.title('Autoencoder Data Integration')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
#plt.colorbar()
plt.show()

# tSNE on Autoencoder bottleneck representation
model_tsne_auto = TSNE(learning_rate = 200, n_components = 2, random_state = 123, perplexity = 10, n_iter = 1000, verbose = 1)
tsne_auto = model_tsne_auto.fit_transform(bottleneck_representation)
plt.scatter(tsne_auto[:, 0], tsne_auto[:, 1], c = Y_scRNAseq, cmap = 'tab20', s = 10)
plt.title('tSNE on Autoencoder: Data Integration, CITEseq')
plt.xlabel("tSNE1")
plt.ylabel("tSNE2")
plt.legend()
plt.show()

##having label in this file
zx= pd.read_csv('Xtrain_70percent_data_GE_F1.csv')

y_D = zx['label']
tsne_df=pd.DataFrame(tsne_auto)
tsne_df = pd.concat([tsne_df,pd.Series(y_D)], axis=1)
col_name = list(["tSNE Dimension1", "tSNE Dimension2", "Label"])
tsne_df.columns = col_name
tsne_df

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

d = {'color': ["#FF0707","#b82e8a","#00008B"]}
fig = plt.figure(figsize=(20, 20), dpi=300)
#plt.figure(figsize = (10,10))
sns.set(font_scale=3, style='white')
fig1=sns.FacetGrid(tsne_df, hue="Label", size=15, hue_kws=d).map(plt.scatter,"tSNE Dimension1", "tSNE Dimension2",s=80)
plt.legend()
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.legend(loc='center right', bbox_to_anchor=(1.50, 0.5), ncol=1)
fig1.savefig('tsne_LGG_Integrated_data_AE.tif', dpi=300)

bottleneck_representation.shape
bottledPD=pd.DataFrame(bottleneck_representation)
bottledPD_df = pd.concat([bottledPD,pd.Series(y)], axis=1)
#col_name = list(['latentDimension1','latentDimension2','latentDimension3','latentDimension4','latentDimension5','latentDimension6','latentDimension7','latentDimension8','latentDimension9','latentDimension10','latentDimension11','latentDimension12','latentDimension13','latentDimension14','latentDimension15','latentDimension16','latentDimension17','latentDimension18','latentDimension19','latentDimension20','latentDimension21','latentDimension22','latentDimension23','latentDimension24','latentDimension25','latentDimension26','latentDimension27','latentDimension28','latentDimension29','latentDimension30','latentDimension31','latentDimension32','latentDimension33','latentDimension34','latentDimension35','latentDimension36','latentDimension37','latentDimension38','latentDimension39','latentDimension40', 'y'])
#col_name = list(['latentDimension1','latentDimension2','latentDimension3','latentDimension4','latentDimension5','latentDimension6','latentDimension7','latentDimension8','latentDimension9','latentDimension10','latentDimension11','latentDimension12','latentDimension13','latentDimension14','latentDimension15','latentDimension16','latentDimension17','latentDimension18','latentDimension19','latentDimension20','latentDimension21','latentDimension22','latentDimension23','latentDimension24','latentDimension25','latentDimension26','latentDimension27','latentDimension28','latentDimension29','latentDimension30','latentDimension31','latentDimension32','latentDimension33','latentDimension34','latentDimension35','latentDimension36','latentDimension37','latentDimension38','latentDimension39','latentDimension40','latentDimension41','latentDimension42','latentDimension43','latentDimension44','latentDimension45','latentDimension46','latentDimension47','latentDimension48','latentDimension49','latentDimension50','latentDimension51','latentDimension52','latentDimension53','latentDimension54','latentDimension55','latentDimension56','latentDimension57','latentDimension58','latentDimension59','latentDimension60', 'y'])
#col_name = list(['latentDimension1','latentDimension2','latentDimension3','latentDimension4','latentDimension5','latentDimension6','latentDimension7','latentDimension8','latentDimension9','latentDimension10','latentDimension11','latentDimension12','latentDimension13','latentDimension14','latentDimension15','latentDimension16','latentDimension17','latentDimension18','latentDimension19','latentDimension20','latentDimension21','latentDimension22','latentDimension23','latentDimension24','latentDimension25','latentDimension26','latentDimension27','latentDimension28','latentDimension29','latentDimension30','latentDimension31','latentDimension32','latentDimension33','latentDimension34','latentDimension35','latentDimension36','latentDimension37','latentDimension38','latentDimension39','latentDimension40','latentDimension41','latentDimension42','latentDimension43','latentDimension44','latentDimension45','latentDimension46','latentDimension47','latentDimension48','latentDimension49','latentDimension50', 'y'])
col_name = list(['latentDimension1','latentDimension2','latentDimension3','latentDimension4','latentDimension5','latentDimension6','latentDimension7','latentDimension8','latentDimension9','latentDimension10','latentDimension11','latentDimension12','latentDimension13','latentDimension14','latentDimension15','latentDimension16','latentDimension17','latentDimension18','latentDimension19','latentDimension20','latentDimension21','latentDimension22','latentDimension23','latentDimension24','latentDimension25','latentDimension26','latentDimension27','latentDimension28','latentDimension29','latentDimension30','latentDimension31','latentDimension32','latentDimension33','latentDimension34','latentDimension35','latentDimension36','latentDimension37','latentDimension38','latentDimension39','latentDimension40','latentDimension41','latentDimension42','latentDimension43','latentDimension44','latentDimension45','latentDimension46','latentDimension47','latentDimension48','latentDimension49','latentDimension50','latentDimension51','latentDimension52','latentDimension53','latentDimension54','latentDimension55','latentDimension56','latentDimension57','latentDimension58','latentDimension59','latentDimension60','latentDimension61','latentDimension62','latentDimension63','latentDimension64','latentDimension65','latentDimension66','latentDimension67','latentDimension68','latentDimension69','latentDimension70','latentDimension71','latentDimension72','latentDimension73','latentDimension74','latentDimension75','latentDimension76','latentDimension77','latentDimension78','latentDimension79','latentDimension80','latentDimension81','latentDimension82','latentDimension83','latentDimension84','latentDimension85','latentDimension86','latentDimension87','latentDimension88','latentDimension89','latentDimension90','latentDimension91','latentDimension92','latentDimension93','latentDimension94','latentDimension95','latentDimension96','latentDimension97','latentDimension98','latentDimension99','latentDimension100','latentDimension101','latentDimension102','latentDimension103','latentDimension104','latentDimension105','latentDimension106','latentDimension107','latentDimension108','latentDimension109','latentDimension110','latentDimension111','latentDimension112','latentDimension113','latentDimension114','latentDimension115','latentDimension116','latentDimension117','latentDimension118','latentDimension119','latentDimension120','latentDimension121','latentDimension122','latentDimension123','latentDimension124','latentDimension125','latentDimension126','latentDimension127','latentDimension128','latentDimension129','latentDimension130','latentDimension131','latentDimension132','latentDimension133','latentDimension134','latentDimension135','latentDimension136','latentDimension137','latentDimension138','latentDimension139','latentDimension140','latentDimension141','latentDimension142','latentDimension143','latentDimension144','latentDimension145','latentDimension146','latentDimension147','latentDimension148','latentDimension149','latentDimension150','latentDimension151','latentDimension152','latentDimension153','latentDimension154','latentDimension155','latentDimension156','latentDimension157','latentDimension158','latentDimension159','latentDimension160','latentDimension161','latentDimension162','latentDimension163','latentDimension164','latentDimension165','latentDimension166','latentDimension167','latentDimension168','latentDimension169','latentDimension170','latentDimension171','latentDimension172','latentDimension173','latentDimension174','latentDimension175','latentDimension176','latentDimension177','latentDimension178','latentDimension179','latentDimension180','latentDimension181','latentDimension182','latentDimension183','latentDimension184','latentDimension185','latentDimension186','latentDimension187','latentDimension188','latentDimension189','latentDimension190','latentDimension191','latentDimension192','latentDimension193','latentDimension194','latentDimension195','latentDimension196','latentDimension197','latentDimension198','latentDimension199','latentDimension200','latentDimension201','latentDimension202','latentDimension203','latentDimension204','latentDimension205','latentDimension206','latentDimension207','latentDimension208','latentDimension209','latentDimension210','latentDimension211','latentDimension212','latentDimension213','latentDimension214','latentDimension215','latentDimension216','latentDimension217','latentDimension218','latentDimension219','latentDimension220','latentDimension221','latentDimension222','latentDimension223','latentDimension224','latentDimension225','latentDimension226','latentDimension227','latentDimension228','latentDimension229','latentDimension230','latentDimension231','latentDimension232','latentDimension233','latentDimension234','latentDimension235','latentDimension236','latentDimension237','latentDimension238','latentDimension239','latentDimension240','latentDimension241','latentDimension242','latentDimension243','latentDimension244','latentDimension245','latentDimension246','latentDimension247','latentDimension248','latentDimension249','latentDimension250','latentDimension251','latentDimension252','latentDimension253','latentDimension254','latentDimension255','latentDimension256','latentDimension257','latentDimension258','latentDimension259','latentDimension260','latentDimension261','latentDimension262','latentDimension263','latentDimension264','latentDimension265','latentDimension266','latentDimension267','latentDimension268','latentDimension269','latentDimension270','latentDimension271','latentDimension272','latentDimension273','latentDimension274','latentDimension275','latentDimension276','latentDimension277','latentDimension278','latentDimension279','latentDimension280','latentDimension281','latentDimension282','latentDimension283','latentDimension284','latentDimension285','latentDimension286','latentDimension287','latentDimension288','latentDimension289','latentDimension290','latentDimension291','latentDimension292','latentDimension293','latentDimension294','latentDimension295','latentDimension296','latentDimension297','latentDimension298','latentDimension299','latentDimension300','latentDimension301','latentDimension302','latentDimension303','latentDimension304','latentDimension305','latentDimension306','latentDimension307','latentDimension308','latentDimension309','latentDimension310','latentDimension311','latentDimension312','latentDimension313','latentDimension314','latentDimension315','latentDimension316','latentDimension317','latentDimension318','latentDimension319','latentDimension320','latentDimension321','latentDimension322','latentDimension323','latentDimension324','latentDimension325','latentDimension326','latentDimension327','latentDimension328','latentDimension329','latentDimension330','latentDimension331','latentDimension332','latentDimension333','latentDimension334','latentDimension335','latentDimension336','latentDimension337','latentDimension338','latentDimension339','latentDimension340','latentDimension341','latentDimension342','latentDimension343','latentDimension344','latentDimension345','latentDimension346','latentDimension347','latentDimension348','latentDimension349','latentDimension350','latentDimension351','latentDimension352','latentDimension353','latentDimension354','latentDimension355','latentDimension356','latentDimension357','latentDimension358','latentDimension359','latentDimension360','latentDimension361','latentDimension362','latentDimension363','latentDimension364','latentDimension365','latentDimension366','latentDimension367','latentDimension368','latentDimension369','latentDimension370','latentDimension371','latentDimension372','latentDimension373','latentDimension374','latentDimension375','latentDimension376','latentDimension377','latentDimension378','latentDimension379','latentDimension380','latentDimension381','latentDimension382','latentDimension383','latentDimension384','latentDimension385','latentDimension386','latentDimension387','latentDimension388','latentDimension389','latentDimension390','latentDimension391','latentDimension392','latentDimension393','latentDimension394','latentDimension395','latentDimension396','latentDimension397','latentDimension398','latentDimension399','latentDimension400','y'])

bottledPD_df.columns = col_name
bottledPD_df

#latent variables
bottledPD_df.to_csv("bottledPD_df_400_features.csv")


