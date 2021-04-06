import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

####################
# Load data
####################

@st.cache
def load_data(dataset='beijing'):

    if dataset == 'beijing':
        data_loc = 'data/beijing_taxi_data_30k.csv'

        colnames=['latitude', 'longitude']
        data = pd.read_csv(data_loc, sep=',', names=colnames, header=None)
        data = data[data['latitude']>116.18]
        data = data[data['latitude']<116.65]
        data = data[data['longitude']>39.6]
        data = data[data['longitude']<40.2]
        x, y = data['longitude'].values, data['latitude'].values

    elif dataset == 'gowalla':
        data_loc = 'data/gowalla_1-8.txt'
        colnames=['user', 'check-in time', 'latitude', 'longitude', 'location id']
        data = pd.read_csv(data_loc, sep='\t', names=colnames, header=None)
        data = data[data['latitude']<180] # remove the 29 invalid outlier points with at latitude>180
        x, y = data['longitude'].values, data['latitude'].values

    else:
        raise ValueError(dataset+' is not a valid data set! Choose a valid one.')
    return x, y

####################
# Main algorithms
####################

def simple_tree(x, y, lam=2, theta=100, h=10, domain_margin=1e-2, plot=False, seed=7):
    # simple tree parameters
    #x, y = data['longitude'].values, data['latitude'].values
    #lam = laplace noise parameter
    #theta = 50 #min count per domain
    #h = 10 # max tree depth
    np.random.seed(seed)

    #initialise counters and holders
    domains = []
    unvisited_domains = []
    counts = []
    noisy_counts = []
    tree_depth = 0
    
    # data limits
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # root domain
    v0 = [x_min, y_min, y_max+domain_margin, x_max+domain_margin] # +margin around borders to include all points
    #domains.append(v0)
    unvisited_domains.append(v0)
    #tree_depth += 1
    #print('tree root initialised.')

    # create subdomains where necessary
    while not not unvisited_domains: # while unvisited_domains is not empty
        for unvisited in unvisited_domains:
            # calculate count and noisy count
            count = count_in_domain(x, y, unvisited)
            noisy_count = count + laplace_noise(lam)
            if (noisy_count > theta) and (tree_depth < h): #split if conditions are met
                v1, v2, v3, v4 = get_domain_subdomains(unvisited)
                # mark new domains as unvisited
                unvisited_domains.append(v1)
                unvisited_domains.append(v2)
                unvisited_domains.append(v3)
                unvisited_domains.append(v4)
                # remove domain that was just visited and split
                unvisited_domains.remove(unvisited)
                # add to tree depth
                tree_depth += 1
                #print('*** domain split ***')
                #print('\ttree depth: {:d}'.format(tree_depth))
            else:
                # remove domain that was just visited
                unvisited_domains.remove(unvisited)
                # record count and noisy count
                counts.append(count)
                # add domain to final domains
                noisy_counts.append(noisy_count)
                domains.append(unvisited)
                #print('domain visited but not split.')

    if plot:
        # plot location points:
        fig, ax = plot_locations_xy(x, y)

        # plot all domains
        for domain in domains:
            plot_rect(ax, domain[0], domain[1], domain[2], domain[3])

        # adjust plot limits to fit everything
        v0_h = v0[2] - v0[1]
        v0_w = v0[3] - v0[0]
        ax.set_xlim(v0[0]-0.05*v0_w, v0[3]+0.05*v0_w)
        ax.set_ylim(v0[1]-0.05*v0_h, v0[2]+0.05*v0_h);

    else:
        fig = None
        ax = None
        
    return domains, noisy_counts, counts, tree_depth, fig, ax


def priv_tree(x, y, lam=2, theta=100, delta=10, domain_margin=1e-2, plot=False, seed=7):
    # simple tree parameters
    #x, y = data['longitude'].values, data['latitude'].values
    #lam = laplace noise parameter
    #theta = 50 #min count per domain
    #h = 10 # max tree depth
    np.random.seed(seed)

    #initialise counters and holders
    domains = []
    unvisited_domains = []
    counts = []
    noisy_counts = []
    tree_depth = 0
    
    # data limits
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # root domain
    v0 = [x_min, y_min, y_max+domain_margin, x_max+domain_margin] # +margin around borders to include all points
    #domains.append(v0)
    unvisited_domains.append(v0)
    #tree_depth += 1
    #print('tree root initialised.')

    # create subdomains where necessary
    while not not unvisited_domains: # while unvisited_domains is not empty
        for unvisited in unvisited_domains:
            # calculate count and noisy count
            count = count_in_domain(x, y, unvisited)
            b = count - (delta*tree_depth)
            b = max(b, (theta - delta))
            noisy_b = b + laplace_noise(lam)
            if (noisy_b > theta): #split if condition is met
                v1, v2, v3, v4 = get_domain_subdomains(unvisited)
                # mark new domains as unvisited
                unvisited_domains.append(v1)
                unvisited_domains.append(v2)
                unvisited_domains.append(v3)
                unvisited_domains.append(v4)
                # remove domain that was just visited and split
                unvisited_domains.remove(unvisited)
                # add to tree depth
                tree_depth += 1
                #print('*** domain split ***')
                #print('\ttree depth: {:d}'.format(tree_depth))
            else:
                # remove domain that was just visited
                unvisited_domains.remove(unvisited)
                # record count and noisy count
                counts.append(count)
                # add domain to final domains
                noisy_counts.append(noisy_b)
                domains.append(unvisited)
                #print('domain visited but not split.')

    if plot:
        # plot location points:
        fig, ax = plot_locations_xy(x, y)

        # plot all domains
        for domain in domains:
            plot_rect(ax, domain[0], domain[1], domain[2], domain[3])

        # adjust plot limits to fit everything
        v0_h = v0[2] - v0[1]
        v0_w = v0[3] - v0[0]
        ax.set_xlim(v0[0]-0.05*v0_w, v0[3]+0.05*v0_w)
        ax.set_ylim(v0[1]-0.05*v0_h, v0[2]+0.05*v0_h);
    else:
        fig = None
        ax = None
        
    return domains, noisy_counts, counts, tree_depth, fig, ax


def uniform_grid(x, y, m=5, lam=2, domain_margin=1e-2, plot=False, seed=7):
    # uniform grid parameters
    #x, y = data['longitude'].values, data['latitude'].values
    #lam = laplace noise parameter

    np.random.seed(seed)

    #initialise counters and holders
    domains = []
    counts = []
    noisy_counts = []
    
    # data limits
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # root domain
    # domain format = [left, bottom, top, right]
    v0 = [x_min, y_min, y_max+domain_margin, x_max+domain_margin] # +margin around borders to include all points
    w = (v0[3] - v0[0])/m # width of uniform grid cells
    h = (v0[2] - v0[1])/m # height of uniform grid cells
    
    # divide root domain into mxm subdomains
    for row in range(m):
        for col in range(m):
            new_v = [v0[0]+(col*w), v0[1]+(row*h) , v0[1]+(row*h)+h , v0[0]+(col*w)+w]
            domains.append(new_v)
            
    for dom in domains:
        count = count_in_domain(x, y, dom)
        noisy_count = count + laplace_noise(lam)
        counts.append(count)
        noisy_counts.append(noisy_count)
        
    if plot:
        # plot location points:
        fig, ax = plot_locations_xy(x, y)

        # plot all domains
        for domain in domains:
            plot_rect(ax, domain[0], domain[1], domain[2], domain[3])

        # adjust plot limits to fit everything
        v0_h = v0[2] - v0[1]
        v0_w = v0[3] - v0[0]
        ax.set_xlim(v0[0]-0.05*v0_w, v0[3]+0.05*v0_w)
        ax.set_ylim(v0[1]-0.05*v0_h, v0[2]+0.05*v0_h);
    else:
        fig = None
        ax = None

    return domains, noisy_counts, counts, fig, ax


####################
# Helper functions
####################

def plot_locations_xy(x, y, alpha=0.3, marker='o', marker_size=5, figsize=(16,8), save_plot_dir=None, dpi=250):
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.scatter(x, y,alpha=alpha, marker=marker, s=marker_size)
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    ax.grid(alpha=0.8,linewidth=0.5);
    
    if not (save_plot_dir == None):
        plt.savefig(save_plot_dir,dpi=dpi)
        
    return fig, ax


def plot_rect(ax, left, bottom, top, right, fill=False, color='r', edgecolor='r', alpha=1, hatch=None):
    p = plt.Rectangle((left, bottom), right-left, top-bottom, fill=fill, color=color, edgecolor=edgecolor, alpha=alpha, hatch=hatch)
    ax.add_patch(p)


def plot_q_grid(ax, x_min, y_min, y_max, x_max, fill=False, edgecolor='r', alpha=1, hatch=None):
    # -----------
    # | q1 | q2 |
    # -----------
    # | q3 | q4 |
    # -----------
    
    width = x_max - x_min
    height = y_max - y_min
    
    # q1
    plot_rect(ax, x_min, y_min+height/2, y_max, x_min+width/2, fill=fill, edgecolor=edgecolor, alpha=alpha, hatch=hatch)
    # q2
    plot_rect(ax, x_min+width/2, y_min+height/2, y_max, x_max, fill=fill, edgecolor=edgecolor, alpha=alpha, hatch=hatch)
    # q3
    plot_rect(ax, x_min, y_min, y_min+height/2,x_max-width/2, fill=fill, edgecolor=edgecolor, alpha=alpha, hatch=hatch)
    # q4
    plot_rect(ax, x_min+width/2, y_min, y_min+height/2, x_max, fill=fill, edgecolor=edgecolor, alpha=alpha, hatch=hatch)


def is_in_domain(x, y, left, bottom, top, right): #includes left and bottom border
        if (x >= left) and (x < right) and (y < top) and (y >= bottom):
            return True
        else:
            return False


def is_domain_partially_in_domain(v1, v0): 
    # checks if domain v1 is partially inside domain v0
    # domain v1 corners (TL = top left, TR = top right, BL = bottom left, BR = botom right)
    is_partially_in = False
    
    v0_L, v0_B, v0_T, v0_R = v0[0], v0[1], v0[2], v0[3]
    v1_L, v1_B, v1_T, v1_R = v1[0], v1[1], v1[2], v1[3]
    
    TL = [v1_L , v1_T]
    TR = [v1_R , v1_T]
    BL = [v1_L , v1_B]
    BR = [v1_R , v1_B]
    
    v1_corners = [TL, TR, BL, BR]
    for corner in v1_corners:
        if is_in_domain(corner[0], corner[1], v0_L, v0_B, v0_T, v0_R):
            is_partially_in = True
    return is_partially_in


def get_domain_subdomains(domain):
    
    # domain = [left, bottom, top, right]
    # left = domain[0]
    # bottom = domain[1]
    # top = domain[2]
    # right = domain[3]
    
    # -----------
    # | q1 | q2 |
    # -----------
    # | q3 | q4 |
    # -----------
    
    dom_h = domain[2] - domain[1]
    dom_w = domain[3] - domain[0]
    
    q1 = [domain[0], domain[1] + dom_h/2, domain[2] , domain[3] - dom_w/2]
    q2 = [domain[0] + dom_w/2, domain[1] + dom_h/2, domain[2] , domain[3]]
    q3 = [domain[0], domain[1], domain[1] + dom_h/2, domain[3] - dom_w/2]
    q4 = [domain[0] + dom_w/2, domain[1], domain[1] + dom_h/2 , domain[3]]
    
    return q1, q2, q3, q4


def count_in_domain(xs, ys, domain):
    count = 0
    
    for i in range(xs.shape[0]):
        if is_in_domain(xs[i], ys[i], domain[0], domain[1], domain[2], domain[3]):
            count += 1
    
    return count


def laplace_noise(Lambda, seed=7): # using inverse transform sampling
    # for numbers between -N and N
    N = Lambda*10
    x = np.arange(-N,N+1,N/20000)

    # pdf P
    P = 1.0 / (2*Lambda) * np.exp(-np.abs(x) / Lambda)
    P = P / np.sum(P)
    
    # cdf C
    C = P.copy()
    for i in np.arange(1, P.shape[0]):
        C[i] = C[i-1] + P[i]
    
    # get sample from laplace distribution wiht uniform random number
    u = np.random.rand()
    sample = x[np.argmin(np.abs(C-u))]
    
    return sample


def get_range_count(q_dom, data_x, data_y, tree_out_doms, tree_noisy_counts, tree_counts):
    true_range_count = count_in_domain(data_x, data_y, q_dom)
    noisy_range_count = 0
    
    # if query domain intersects any tree domain, add its count
    for i in range(len(tree_out_doms)):
        if is_domain_partially_in_domain(q_dom, tree_out_doms[i]):
            noisy_range_count += tree_noisy_counts[i]
    
    return noisy_range_count, true_range_count


####################
# App setup
####################

## page config
st.set_page_config(page_title='Private Range Queries', 
					page_icon=':round_pushpin:', # :pushpin:  :earth_africa:   :world_map:
					layout='wide', # "centered" or "wide"
					initial_sidebar_state='expanded') # "auto" or "expanded" or "collapsed"

## page logo
logo = Image.open('figures/logo3.png')
st.image(logo, use_column_width=True)

## display link to paper
st.markdown('*Read full paper [here](https://github.com/ruankie/differentially-private-range-queries/blob/main/paper.pdf)*')
st.markdown('***')

## sidebar
st.sidebar.markdown('# Parameters')

# data/map
data_map = st.sidebar.selectbox('Select Data (Map)', ['Beijing Taxi Data', 'Gowalla Social Media'])
if data_map == 'Beijing Taxi Data':
    x, y = load_data(dataset='beijing')
elif data_map == 'Gowalla Social Media':
    x, y = load_data(dataset='gowalla')
else:
    raise ValueError(dataset+' is not a valid data set! Choose a valid one.')

# calculations for query
# for default query
x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)
x_range = x_max - x_min
y_range = y_max -y_min

# default query
q_default = np.round([x_min + 0.25*x_range, 
                    y_min + 0.25*y_range, 
                    (y_min + 0.25*y_range) + 0.5*y_range, 
                    (x_min + 0.25*x_range) + 0.5*x_range],2)

q_default_str = str(q_default[0])+', '+str(q_default[1])+', '+str(q_default[2])+', '+str(q_default[3])
left_x = q_default[0]
bottom_y = q_default[1]
top_y = q_default[2]
right_x =q_default[3]

# query algorithm
algo = st.sidebar.selectbox('Select Algorithm', ['PrivTree','Simple QuadTree','Uniform Grid'])

# algorithm parameters
if algo == 'Uniform Grid':
    st.sidebar.markdown(r'#### Privacy Budget ($\epsilon$)')
    epsilon = st.sidebar.slider('', min_value=0.0, max_value=2.0, value=0.1, step=0.01)

    st.sidebar.markdown(r'#### Grid Parameter ($m$)')
    st.sidebar.text('(default value is optimal)')
    m = st.sidebar.slider('', min_value=0, max_value=100, value=int(( (len(x) * epsilon) / 10.0 )**0.5), step=10)

    st.sidebar.markdown(r'#### Laplace Noise Parameter ($\lambda$)')
    lam = st.sidebar.slider('', min_value=(1/epsilon), max_value=2.0*(1/epsilon), value=(1/epsilon), step=(2.0/10.0)*(1/epsilon))

    
elif algo == 'Simple QuadTree':
    st.sidebar.markdown(r'#### Privacy Budget ($\epsilon$)')
    epsilon = st.sidebar.slider('', min_value=0.0, max_value=2.0, value=0.1, step=0.01)

    st.sidebar.markdown(r'#### Count Threshold ($\theta$)')
    theta = st.sidebar.slider('', min_value=100, max_value=5000, value=1000, step=100)

    st.sidebar.markdown(r'#### Max Tree Depth ($h$)')
    h = st.sidebar.slider('', min_value=10, max_value=500, value=150, step=10)

    st.sidebar.markdown(r'#### Laplace Noise Parameter ($\lambda$)')
    st.sidebar.text('(differential pivacy guaranteed\nfor entire given range)')
    lam = st.sidebar.slider('', min_value=(h/epsilon), max_value=2.0*(h/epsilon), value=(h/epsilon), step=(2.0/10.0)*(h/epsilon))

elif algo == 'PrivTree':
    st.sidebar.markdown(r'#### Privacy Budget ($\epsilon$)')
    epsilon = st.sidebar.slider('', min_value=0.0, max_value=2.0, value=0.1, step=0.01)

    st.sidebar.markdown(r'#### Count Threshold ($\theta$)')
    theta = st.sidebar.slider('', min_value=0, max_value=5000, value=0, step=100)

    st.sidebar.markdown(r'#### Laplace Noise Parameter ($\lambda$)')
    st.sidebar.text('(differential pivacy guaranteed\nfor entire given range)')
    lam_def = 7.0/(3.0 * epsilon)
    lam = st.sidebar.slider('', min_value=lam_def, max_value=2.0*lam_def, value=lam_def, step=(2.0/10.0)*lam_def)

    st.sidebar.markdown(r'#### Scaling Parameter ($\delta$)')
    delta_def = lam * 1.3862944 #np.log(4)
    delta = st.sidebar.slider('', min_value=0.1*delta_def, max_value=2.0*delta_def, value=delta_def, step=0.01)


## input range query
st.markdown('## Range Query Input')    

# check boxes
decompose = st.checkbox('Show underlying spatial decomposition (takes time to compute)', value=True)
show_q = st.checkbox('Show query area', value=True)

# query input
q_input = st.text_input('Query Coordinates (left x, bottom y, top y, right x):', q_default_str)
left_x, bottom_y, top_y, right_x = q_input.split(', ')
left_x = float(left_x)
bottom_y = float(bottom_y)
top_y = float(top_y)
right_x = float(right_x)
q = np.array([left_x, bottom_y, top_y, right_x])

if not decompose:
    if algo == 'Uniform Grid':
        with st.spinner('Performing spatial decomposition. Please wait...'):
            domains, noisy_counts, counts, fig, ax = uniform_grid(x, y, m=m, lam=lam, domain_margin=1e-2, plot=True, seed=7)

    elif algo == 'Simple QuadTree':
        with st.spinner('Performing spatial decomposition. Please wait...'):
            domains, noisy_counts, counts, tree_depth, fig, ax = simple_tree(x, y, lam=lam, theta=theta, h=h, domain_margin=1e-2, plot=True, seed=7)

    elif algo == 'PrivTree':
        with st.spinner('Performing spatial decomposition. Please wait...'):
            domains, noisy_counts, counts, tree_depth, fig, ax = priv_tree(x, y, lam=lam, theta=theta, delta=delta, domain_margin=1e-2, plot=True, seed=7)


## Plot map data
st.markdown('## Data Plot')

if decompose:
    if algo == 'Uniform Grid':
        with st.spinner('Performing spatial decomposition. Please wait...'):
            domains, noisy_counts, counts, fig, ax = uniform_grid(x, y, m=m, lam=lam, domain_margin=1e-2, plot=True, seed=7)
        if show_q:
            plot_rect(ax, left_x, bottom_y, top_y, right_x, fill=True, color='g', edgecolor='g', alpha=0.3, hatch=None)
        st.pyplot(fig)

    elif algo == 'Simple QuadTree':
        with st.spinner('Performing spatial decomposition. Please wait...'):
            domains, noisy_counts, counts, tree_depth, fig, ax = simple_tree(x, y, lam=lam, theta=theta, h=h, domain_margin=1e-2, plot=True, seed=7)
        if show_q:
            plot_rect(ax, left_x, bottom_y, top_y, right_x, fill=True, color='g', edgecolor='g', alpha=0.3, hatch=None)
        st.pyplot(fig)

    elif algo == 'PrivTree':
        with st.spinner('Performing spatial decomposition. Please wait...'):
            domains, noisy_counts, counts, tree_depth, fig, ax = priv_tree(x, y, lam=lam, theta=theta, delta=delta, domain_margin=1e-2, plot=True, seed=7)
        if show_q:
            plot_rect(ax, left_x, bottom_y, top_y, right_x, fill=True, color='g', edgecolor='g', alpha=0.3, hatch=None)
        st.pyplot(fig)

else:
    fig, ax = plot_locations_xy(x, y,marker_size=1.5)
    if show_q:
        plot_rect(ax, left_x, bottom_y, top_y, right_x, fill=True, color='g', edgecolor='g', alpha=0.3, hatch=None)
    st.pyplot(fig)


## display range query results
st.markdown('## Range Query Results')
noisy_range_count, true_range_count = get_range_count(q, x, y, domains, noisy_counts, counts)
st.write('True Count:', int(true_range_count))
st.write('Noisy Count:', int(noisy_range_count))


## display algorithm pseudo code
st.markdown('***')
st.markdown('## Algorithm Pseudo Code')
show_pseudo = st.checkbox('Show algorithm pseudo code', value=True)
if show_pseudo:
    if algo == 'Uniform Grid':
        alg_img = Image.open('figures/ug_alg.PNG')
        st.image(alg_img)

    elif algo == 'Simple QuadTree':
        alg_img = Image.open('figures/simple_tree_alg.PNG')
        st.image(alg_img)

    elif algo == 'PrivTree':
        alg_img = Image.open('figures/privtree_alg.PNG')
        st.image(alg_img)


## display author details
st.markdown('***')
st.markdown('*By Ruan Pretorius*')
st.markdown('*[LinkedIn](https://www.linkedin.com/in/ruan-pretorius)*')

## references
st.markdown('***')
st.markdown('## References')
st.markdown('''
    * *Data Sets:* 
        * [Beijing Taxi Data Set](http://snap.stanford.edu/data/loc-gowalla.html)
        * [Gowalla Data Set](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/?from=http%3A%2F%2Fresearch.microsoft.com%2Fapps%2Fpubs%2F%3Fid%3D152883)
    * *Algorithms:*
        * J. Zhang, X. Xiaokui, and X. Xing, ``Privtree: A differentially private algorithm for hierarchical decompositions,''
        In Proceedings of the 2016 International Conference on Management of Data, 2016, pp. 155-170.
        * W. Qardaji, W. Yang and N. Li, ``Differentially private grids for geospatial data,'' 
        2013 IEEE 29th International Conference on Data Engineering (ICDE), Brisbane, QLD, 2013, pp. 757-768, doi: 10.1109/ICDE.2013.6544872.
        * G. Cormode, C. Procopiuc, D. Srivastava, E. Shen and T. Yu, ``Differentially Private Spatial Decompositions,'' 
        2012 IEEE 28th International Conference on Data Engineering, Washington, DC, 2012, pp. 20-31, doi: 10.1109/ICDE.2012.16.''')

