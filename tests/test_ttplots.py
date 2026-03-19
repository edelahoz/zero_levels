#%%
#!/usr/bin/env python3

import os
import sys

import numpy as np
import healpy as hp

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from zero_levels import TTplots
#%%

def generate_map(nside, monopole_value, dipole_amplitude, dipole_direction):

    npix = 12 * nside ** 2
    m = monopole_value * np.ones(npix)
    vecs = hp.pix2vec(nside, np.arange(npix))
    dipole = dipole_amplitude * np.dot(dipole_direction, vecs)
    m += dipole

    return m


nside = 256

np.random.seed(100)
mono1 = 1.0  
d_amp1 = 0.02 
d_dir1 = np.array([0, 0, 1] )
sigma1 = 0
 
mono2 = 0 
d_amp2 = 0.07 
d_dir2 = np.array([1, 0, 1] )
sigma2 = 0.0002

mono3 = -0.8  
d_amp3 = 0.2
d_dir3 = np.array([1/4, -2, 1] )
sigma3 = 0

m1 = generate_map(nside, mono1, d_amp1, d_dir1) + sigma1 * np.random.randn(12 * nside ** 2)
m2 = generate_map(nside, mono2, d_amp2, d_dir2) + sigma2 * np.random.randn(12 * nside ** 2)
m3 = generate_map(nside, mono3, d_amp3, d_dir3) + sigma3 * np.random.randn(12 * nside ** 2)

maps = np.vstack([m1, m2, m3])

fixed_pars = {1: "mono"}
pars = [(mono1, np.sqrt(((d_amp1 * d_dir1) ** 2).sum())), 
        (mono2, np.sqrt(((d_amp2 * d_dir2) ** 2).sum())), 
        (mono3, np.sqrt(((d_amp3 * d_dir3) ** 2).sum()))
        ]

nside_clusters = 16
ttplot = TTplots(nside, nside_clusters)


#%%
monodip = ttplot.calculate_mono_dipole(maps, fixed_pars=fixed_pars)

aux_idx = 0
for i, (monopolo, dipolo) in enumerate(pars):
    
    if (fixed_pars is not None) and (i in fixed_pars.keys()):
        par = fixed_pars[i]
        if par == "mono":
            print(f"Delta d_amp{i + 1} = {np.sqrt((monodip[ i * 4 - aux_idx: (i + 1) * 4 - 1 - aux_idx] ** 2).sum()) - dipolo}")
            aux_idx += 1
        elif par == "dip":
            print(f"Delta mono{i + 1} = {monodip[i * 4 - aux_idx] - monopolo}")
            aux_idx += 3
    else:
        print(f"Delta mono{i + 1} = {monodip[i * 4 - aux_idx] - monopolo}")
        print(f"Delta d_amp{i + 1} = {np.sqrt((monodip[ i * 4 + 1 - aux_idx: (i + 1) * 4 - aux_idx] ** 2).sum()) - dipolo}")

    

# %%

total_monodip, monodip_iter = ttplot.calculate_mono_dipole_iter(maps, fixed_pars=fixed_pars)

#%%
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

n_iters = np.arange(len(monodip_iter))

fig, axes = plt.subplots(nrows=4, ncols=len(maps), sharex=True, )

list_axes = axes.flatten('F')
true_vals = np.concatenate([
    np.array([mono1]), d_amp1 * d_dir1, 
    np.array([mono2]), d_amp2 * d_dir2, 
    np.array([mono3]), d_amp3 * d_dir3])
    
aux_idx = 0
for i, (ax, truth) in enumerate(zip(list_axes, true_vals)):
    if i // 4 in fixed_pars.keys():
        par = fixed_pars[i // 4]
        if par == "mono" and (i % 4 == 0): 
            ax.axis('off')
            aux_idx += 1
        elif par == "dip" and (i % 4 != 0): 
            ax.axis('off')
            aux_idx += 1
        else:
            ax.scatter(n_iters, np.cumsum(monodip_iter.T[i - aux_idx]))
            ax.axhline(y=truth, color='r', linestyle='-')
    else:

        ax.scatter(n_iters, np.cumsum(monodip_iter.T[i - aux_idx]))
        ax.axhline(y=truth, color='r', linestyle='-')
    
    if i % 4 == 3: ax.set_xlabel(r'$N_{\rm iter}$')

list_axes[0].set_ylabel(r"$m$")
list_axes[1].set_ylabel(r"$d_x$")
list_axes[2].set_ylabel(r"$d_y$")
list_axes[3].set_ylabel(r"$d_z$")
#ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#ax.ticklabel_format(style='plain', axis='y')
#ax.yaxis.get_offset_text().set_visible(False)  # Hide the offset text
#ax.ticklabel_format(style='plain', axis='y', useOffset=False)
fig.subplots_adjust(wspace=0.4, hspace=0.25)
fig.savefig("tt_plot_sims.png")
#fig.tight_layout()
# %%

aux_idx = 0
for i, (monopolo, dipolo) in enumerate(pars):
    
    if (fixed_pars is not None) and (i in fixed_pars.keys()):
        par = fixed_pars[i]
        if par == "mono":
            print(f"Delta d_amp{i} = {np.sqrt((total_monodip[ i * 4 - aux_idx: (i + 1) * 4 - 1 - aux_idx] ** 2).sum()) - dipolo}")
            aux_idx += 1
        elif par == "dip":
            print(f"Delta mono{i} = {total_monodip[i * 4 - aux_idx] - monopolo}")
            aux_idx += 3
    else:
        print(f"Delta mono{i} = {total_monodip[i * 4 - aux_idx] - monopolo}")
        print(f"Delta d_amp{i} = {np.sqrt((total_monodip[ i * 4 + 1 - aux_idx: (i + 1) * 4 - aux_idx] ** 2).sum()) - dipolo}")

    
# %%
