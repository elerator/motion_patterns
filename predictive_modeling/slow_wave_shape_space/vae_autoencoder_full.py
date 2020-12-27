#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
import pickle
import numpy as np
import sys
sys.path.append("../..")
from utils.diverse import *
from utils.data_transformations import *
from utils.visualization_tools import *
from sklearn.utils import resample

# In[2]:


set_random_state(42)


# # Load dataset

# In[3]:


with open("../../../../main_experiment/sparse_data/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)
slow_wave_ids = list(dataset["sws"].keys())
print(len(slow_wave_ids))


# In[4]:


dataset["sws"][slow_wave_ids[0]].keys()


# # Load data for features, select slow waves

# In[5]:


features = ["iso", "start", "stop", "width", "height", "gcamp_interpolated", "gcamp_mean",
            "hemo_interpolated", "left_too_high", "correlation", "mean_sources", "mean_sinks"]
iso, start, stop, width, height, gcamp, gcamp_mean, hemo, left_too_high, corr, mean_sources, mean_sinks = slow_wave_features(dataset, features)

#Filter waves and get all data for filtered indices (where)
where = corr < .3
where[height/width < 0.1] = False
where[left_too_high] = False
#where[height/width < 0.05] = False

print(np.sum(where))

iso, start, stop, width, height, gcamp, gcamp_mean, hemo, left_too_high, corr, mean_sources, mean_sinks = slow_wave_features(dataset, features, where)


# In[6]:


sources_sinks = [np.hstack([a, b]) for a,b in zip(mean_sources, mean_sinks)]


# In[7]:


fig, ax = plt.subplots(1, dpi = 100)
_ = ax.hist(np.sort(height), bins = 100)
ax.set_xlabel("Amplitude [%]")
ax.set_ylabel("Absolute frequency")


# In[8]:


components = [dataset["sws"][id]["flow_components"]["per_wave"]["left_hemisphere"] for id in np.array(slow_wave_ids)[where]]
components = np.array(components)
relative_components = np.array([normalize(c) for c in components])
vertical_greater_horizontal = np.array([normalize(c) for c in components[:,1:3]])


# In[9]:


abs_up_flow = np.abs(components[:,2])
abs_down_flow = np.abs(components[:,3])
abs_left_flow = np.abs(components[:,0])
abs_right_flow = np.abs(components[:,1])

ud = abs_up_flow + abs_down_flow
lr = abs_right_flow + abs_left_flow
up_flow = abs_up_flow/ud#upwards flow as fraction of total upwards/downwards flow
left_flow = abs_left_flow/lr#upwards flow as fraction of total upwards/downwards flow

flow_per_auc = (ud+lr)/[np.nansum(x) for x in gcamp_mean]

ratio_vertical_horizontal = ud/(ud+lr)


# In[10]:


#abs_up_flow - abs_down_flow


# In[11]:


#for uf in [up_flow[iso == i] for i in list(set(isos))]:
#    plt.scatter(np.arange(len(uf)), uf)
#    plt.plot([0,60], [np.mean(uf), np.mean(uf)])


# ## Prepare data

# Use train test split to split the dataset.
#
# Append features that we aim to plot later such that they are split and shuffled in the same way.

# In[12]:


from sklearn.model_selection import train_test_split
def prepare_data(gcamp, width, height, additional_features, additional_arrays, random_state = 42, test_size=.25, batch_size = 100):
    x = [gcamp.T, height.T, width.T]
    x.extend([f.T for f in additional_features])
    x = np.vstack(x).T
    x[:,128:128+2] = normalize_nan(x[:,128:128+2])#Normalize heights & widths

    x_train, x_test = train_test_split(x, test_size=test_size, random_state = random_state)
    arrays_train, arrays_test = train_test_split(additional_arrays, test_size=test_size, random_state = random_state)

    #Save additional features seperately
    additional_features_train = []
    additional_features_test = []
    for i in range(len(additional_features)):
        feature_train = x_train[:,130+i]
        feature_test = x_test[:,130+i]
        additional_features_train.append(feature_train)
        additional_features_test.append(feature_test)

    #x_train and x_test are the gcamp signal and the width/ height (aka amplitude, phase of it)
    x_train = x_train[:,:130]
    x_test = x_test[:,:130]
    return x_train, x_test, additional_features_train, additional_features_test, arrays_train, arrays_test



# In[13]:


additional_features = [iso, corr, width, height, ratio_vertical_horizontal, up_flow, left_flow, flow_per_auc]


# In[14]:


x_train, x_test, add_train, add_test, sources_sinks_train, sources_sinks_test = prepare_data(gcamp, width, height, additional_features, sources_sinks)


# Save additional features as variables.

# In[15]:


iso_train, corr_train, width_train, height_train, ratio_vertical_horizontal_train, up_flow_train, left_flow_train, flow_per_auc_train = add_train
iso_test, corr_tests, width_test, height_test, ratio_vertical_horizontal_test, up_flow_test, left_flow_test, flow_per_auc_test = add_test


# ## Remove class imbalance
#
# Large slow waves are present less frequently. Hence optimization will be effected less by these samples (Note that training samples are normalized hence large amplitude waves do not cause greater loss). To avoid slective underfitting of large waves we balance the sampels with respect to height classes.

# In[16]:


height_classes = np.round(height_train*2, -1)
height_classes[height_classes > 50] = 50


# In[17]:


upsampled_x_train = resample(x_train, stratify=height_classes, n_samples = 4000, random_state = 42)


# In[18]:


print("There are " + str(len(iso_test)) + " test samples")


# In[19]:


print("The maximal amplitude of the percentage change in time is "
              + str(np.max(height_train).round()))


# # Variational Autoenconder

# In[20]:


## network parameters
from predictive_modeling.models.vae_cnn_v0 import *
from keras import Model

batch_size = 100
epochs = 200


# In[21]:


[image_input, value_input], [z_mean, z_log_var, z] = encoder(n_values = 130)
latent_inputs, [image_decoded, values_decoded] = decoder(n_values = 130)
encoder_model = Model(inputs = [image_input, value_input], outputs = [z_mean, z_log_var, z])
decoder_model = Model(inputs = latent_inputs, outputs = [image_decoded, values_decoded], name='decoder')
image_decoded, values_decoded = decoder_model(encoder_model([image_input, value_input])[2])
vae = Model(inputs = [image_input, value_input], outputs = [image_decoded, values_decoded], name='vae_mlp')


# In[22]:


import keras
vae.add_loss(get_vae_loss(image_input, value_input, image_decoded, values_decoded, z_mean, z_log_var, impact_reconstruction_loss = 100))
#vae.compile(optimizer= keras.optimizers.Adam(learning_rate=0.01))
vae.compile(optimizer = "adam")


# In[23]:


sources_sinks_train1 = np.expand_dims(sources_sinks_train,-1)
x_train_combined = [np.array(sources_sinks_train1), x_train]

sources_sinks_test1 = np.expand_dims(sources_sinks_test,-1)
x_test_combined = [np.array(sources_sinks_test1), x_test]


# In[24]:


# train the autoencoder
history = vae.fit(x_train_combined, validation_data = (x_train_combined, None), epochs= 1000, batch_size=batch_size, verbose=1)


# In[155]:


plt.plot(vae.history.history["val_loss"])


# # Visualize predictions

# In[156]:


res = vae.predict([np.array(sources_sinks_train1), x_train])


# In[157]:


val_predictions = res[1]


# In[158]:


from scipy.ndimage import median_filter


# In[159]:


i = 1
fig, ax = plt.subplots(3, figsize= (5,8))
ax[0].imshow(sources_sinks_train[i])
ax[1].imshow(res[0][i][:,:,0])

ax[2].plot(median_filter(res[1][i][:128], 2))
ax[2].plot(x_train[i][:128])


# In[160]:


from plots import manifold
from plots import manifold_of_images


# In[161]:


#Predictions for z[0] and z[1] neurons i.e. x and y for the train and test datasets
x_pred_train, y_pred_train = encoder_model.predict([np.array(sources_sinks_train1), x_train])[2].T


# In[168]:


#%%capture
x_min = np.mean(x_pred_train)-2*np.std(x_pred_train)
y_min = np.mean(y_pred_train)-2*np.std(y_pred_train)
x_max = np.mean(x_pred_train)+2*np.std(x_pred_train)
y_max = np.mean(y_pred_train)+2*np.std(y_pred_train)
x_range = [x_min, x_max]
y_range = [y_min, y_max]

man_mean_signal = manifold(decoder_model, x_range, y_range, n = 15, dpi = 300, scale="sqrt", pos_in_multi_output = 1, medfilt = 2)


# In[163]:


import random
def manifold_of_vector_components(figsize=(5,5), dpi = 200, n = 15,
                                  up = None, down = None, left = None, right = None, debug = True):
    fig, ax = plt.subplots(n, n, figsize= figsize, dpi = dpi)
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

    for y in range(n):
        for x in range(n):
            if debug:
                ax[y,x].imshow(render_arrow_components(random.uniform(0,1) ,random.uniform(0,1) ,
                                                random.uniform(0,1) ,random.uniform(0,1) ))
            else:
                ax[y,x].imshow(render_arrow_components(up, down, left, right))
            ax[y,x].axis("off")
    return fig2rgb_array(fig)


# In[164]:


get_ipython().run_cell_magic('capture', '', 'manifold_vector_components = manifold_of_vector_components()')


# In[165]:


sources_manifold = manifold_of_images(decoder_model, x_range, y_range, sources_sinks = "sources")
sinks_manifold = manifold_of_images(decoder_model, x_range, y_range, sources_sinks = "sinks")


# In[166]:


z_train = encoder_model.predict([np.array(sources_sinks_train1), x_train])[2].T


# In[167]:


fig = plt.figure(constrained_layout = False, figsize = (8,8), dpi = 300)
gs1 = fig.add_gridspec(nrows = 6, ncols = 4, left=0.05, right=0.48, wspace=0.05)
ax0 = fig.add_subplot(gs1[0:2, 0:2])
ax1 = fig.add_subplot(gs1[0:2, 2:4])
ax2 = fig.add_subplot(gs1[2:4, 0:2])
ax3 = fig.add_subplot(gs1[2:4, 2:4])
ax4 = fig.add_subplot(gs1[4:6, 0:2])
ax5 = fig.add_subplot(gs1[4:6, 2:4])

ax0.imshow(man_mean_signal)
ax1.imshow(manifold_vector_components)
ax2.imshow(man_mean_signal)
ax3.imshow(man_mean_signal)
ax4.imshow(sources_manifold)
ax5.imshow(sinks_manifold)


# In[101]:


iso_plots = []
for x in range(5):
    iso_plots.append(fig.add_subplot(gs1[2,x]))

isos = [1.8, 2.0, 2.2, 2.4, 2.6]
def add_feature_plot(ax, iso):
    ax.scatter(*z_train, c= "lightgray", s= .01)
    ax.scatter(*(z_train.T[iso_train == iso]).T, c = "darkblue", s = .5)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

for iso, a in zip(isos, iso_plots):
    add_feature_plot(a, iso)


# In[115]:


fig, ax = plt.subplots(1,5, figsize= (10,5))
isos = [1.8, 2.0, 2.2, 2.4, 2.6]

def add_feature_plot(ax, iso):
    ax.scatter(*z_train, c= "lightgray")
    ax.scatter(*(z_train.T[iso_train == iso]).T, c = "darkblue")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

for iso, a in zip(isos,ax):
    add_feature_plot(a, iso)


# In[65]:


x_train.shape


# # Visualization

# In[36]:


vae.save("full_vae")


# In[41]:


vae = keras.models.load_model("full_vae", custom_objects={'tf': tf}, compile = False)
