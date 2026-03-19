#!/usr/bin/env python3

import numpy as np
import healpy as hp

from typing import List, Union, Dict
from numpy.typing import NDArray


class MonoDip():

    def __init__(self, nside: int, mask: NDArray[np.float64] = None) -> None:

        self.nside = nside
        self.T_array = None
        self.mask = mask

    def get_templates(self):
        if self.T_array is None:
            if self.mask is not None:
                pixels = np.argwhere(self.mask != 0).flatten()
            else:
                pixels = np.arange(12 * self.nside ** 2)

            theta, phi = hp.pix2ang(
                self.nside, 
                pixels, 
                lonlat=False
            )

            T_array = np.vstack([
                np.ones(pixels.size), 
                np.cos(phi) * np.sin(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(theta)]).T
            
            self.T_array = T_array
        return self.T_array

    def remove_mono_dipole(self, maps: NDArray[np.float64], 
                           mono_dipole: NDArray[np.float64],
                           **kwargs) -> NDArray[np.float64]:
        
        corrected_maps = np.zeros(maps.shape)
        vecs = np.array(hp.pix2vec(self.nside, np.arange(12 * self.nside ** 2)))

        for i, (m, md) in enumerate(zip(maps, mono_dipole)):
            monopole = md[0]
                
            dipole_amp = np.sqrt(np.sum(md[1: 4] ** 2))
            dipole_direction = md[1: 4] / dipole_amp
            dipole = dipole_amp * np.dot(dipole_direction, vecs)
                
            corrected_maps[i] = m - monopole - dipole

        return corrected_maps


    def dep_remove_mono_dipole(self, maps: NDArray[np.float64], 
                           mono_dipole: NDArray[np.float64],
                           fixed_pars: Dict = None,) -> NDArray[np.float64]:
        
        corrected_maps = np.zeros(maps.shape)
        vecs = hp.pix2vec(self.nside, np.arange(12 * self.nside ** 2))

        aux_idx = 0
        for i, m in enumerate(maps):
            if (fixed_pars is not None) and (i in fixed_pars.keys()):
                par = fixed_pars[i]
                if par == "mono":
                    corrected_maps[i] = self.remove_dipoles(
                        m, mono_dipole[i * 4 - aux_idx : (i + 1) * 4 - 1 - aux_idx]
                    )
                    aux_idx += 1
                elif par == "dip":
                    corrected_maps[i] = self.remove_monopoles(
                        m, mono_dipole[i * 4 - aux_idx]
                    )
                    aux_idx += 3
            else:
                monopole = mono_dipole[i * 4 - aux_idx]
                
                dipole_amp = np.sqrt(np.sum(mono_dipole[i * 4 + 1 - aux_idx: (i + 1) * 4 - aux_idx] ** 2 ))
                dipole_direction = mono_dipole[i * 4 + 1 - aux_idx : (i + 1) * 4 - aux_idx] / dipole_amp
                dipole = dipole_amp * np.dot(dipole_direction, vecs)
                
                corrected_maps[i] = m - monopole - dipole

        return corrected_maps
    
    def remove_monopoles(self, maps: NDArray[np.float64], 
                        monopoles: Union[float, NDArray[np.float64]]
                        ) -> NDArray[np.float64]:
        
        if isinstance(monopoles, (list, tuple, np.ndarray)):
            assert monopoles.size == maps.shape[0], (
                "The number of monopoles needs to equal the number of maps"
                )
            monopoles = monopoles[..., np.newaxis]
        return maps - monopoles
    
    def remove_dipoles(self, maps: NDArray[np.float64], 
                       dipoles: NDArray[np.float64]) -> NDArray[np.float64]:
        
        assert dipoles.shape[-1] == 3, (
            "Dipoles should be in (dx, dy, dz) format"
        )

        vecs = hp.pix2vec(self.nside, np.arange(12 * self.nside ** 2))
        corrected_maps = maps.copy()

        if dipoles.ndim == 2:
            assert dipoles.shape[0] == maps.shape[0], (
                "The number of dipoles needs to equal the number of maps"
                )
            
            for i, (m, dipole) in enumerate(zip(maps, dipoles)):
                dipole_amp = np.sqrt(np.sum(dipole ** 2 ))
                dipole_direction = dipole / dipole_amp
                dipole_map = dipole_amp * np.dot(dipole_direction, vecs)

                corrected_maps[i] = m - dipole_map
        else:
            dipole_amp = np.sqrt(np.sum(dipoles ** 2 ))
            dipole_direction = dipoles / dipole_amp
            dipole_map = dipole_amp * np.dot(dipole_direction, vecs)

            corrected_maps = maps - dipole_map
        
        return corrected_maps


class TTplots(MonoDip):


    def __init__(self, nside: int, nside_cluster: int = None,
                 clusters: List[NDArray[np.int32]] = None, 
                 mask: NDArray[bool] = None) -> None:

        super().__init__(nside)

        if clusters is None:
            if nside_cluster is None:
                raise ValueError("Either nside_cluster or clusters has to be provided")
            else:
                clusters = self.get_HEALPix_super_clusters(nside, nside_cluster, mask=mask)

        clusters = [cluster for cluster in clusters if cluster.size > 1]
        self.n_clusters = len(clusters)
        self.clusters = clusters
        self.T_array_clusters = None
        

    @staticmethod
    def get_children_pixels(pix: int, nside_in: int, nside_out: int, 
                            in_nest: bool = False, out_nest: bool = False
                            ) -> NDArray[np.int32]:
        """
        Function to evaluate the pixel indices of the pixels at nside_out 
        within the pixel pix at nside_in. The pixel indices returned can
        be at either ring or nest scheme. The default scheme for the pix
        index is ring.
        """
        def get_pixel_tree(ind_pix, nside):
            if nside == 1:
                return [ind_pix]
            if ind_pix == 0: 
                return np.zeros(int(np.log2(nside) + 1))
            
            remainder = ind_pix
            pixel_tree = []
            Npix = nside ** 2
            if remainder == 0: pixel_tree = np.zeros(np.log2(nside)+1)
            while Npix >= 1:
                pixel_tree.append(remainder // Npix)
                remainder = remainder % Npix
                Npix *= 0.25
            return pixel_tree
        
        def get_pixel_indices(pixel_tree, nside):
            lowerlimit = np.sum([el_tree * nside ** 2 / 4 ** k for k, el_tree in enumerate(pixel_tree)])
            upperlimit = (pixel_tree[0] + 1) * nside ** 2
            upperlimit += -np.sum([(3 - el_tree) * nside ** 2 / 4 ** (k + 1) for k, el_tree in enumerate(pixel_tree[1:])])        
            return np.arange(int(lowerlimit), int(upperlimit))
        
        if not np.log2(nside_in).is_integer() or not np.log2(nside_out).is_integer():
            if nside_in == 0 and pix == 0:
                return np.arange(12 * nside_out ** 2)
            else:
                raise ValueError('nside must be a power of 2')
        
        if not in_nest: pix = hp.ring2nest(nside_in,pix)
        
        pixel_tree = get_pixel_tree(pix, nside_in)
        ind_pixels = get_pixel_indices(pixel_tree, nside_out)
        
        return ind_pixels if out_nest else hp.nest2ring(nside_out, ind_pixels)

    def get_HEALPix_super_clusters(self, nside: int, super_nside: int, 
                                   mask: NDArray[bool] = None) -> List[NDArray[np.int32]]:
        clusters = []
        idx_mask = np.argwhere(mask).flatten() if mask is not None else None

        if super_nside == 0:
            if mask is not None:
                clusters.append(idx_mask)
            else:
                clusters.append(np.arange(12 * nside * nside))
            return clusters
        
        for ipix in np.arange(12 * super_nside * super_nside):
            idx_cluster = self.get_children_pixels(
                ipix, nside_in=super_nside, nside_out=nside,
            )
            if mask is not None:
                idx_cluster = idx_cluster[np.in1d(idx_cluster, idx_mask)]

            if idx_cluster.size !=0: clusters.append(idx_cluster)

        return clusters
    
    @staticmethod
    def simple_calculate_slope_intercept(x, y):

        m = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean())**2)
        b = y.mean() - m * x.mean()
        return m, b

    def calculate_slopes_intercepts(self, maps: NDArray[np.float64]):
        Nmaps, Npix = maps.shape

        slopes = np.zeros(Nmaps - 1)
        intercepts = np.zeros(Nmaps - 1)

        if Npix > 15000:
            print("Using simple linear regression for slope and intercept calculation. Should be checked.")
            # Use simple linear regression because of memory demands
            for idx in np.arange(Nmaps -1):
                m, b = self.simple_calculate_slope_intercept(
                    maps[idx], maps[idx+1]
                )
                slopes[idx] = m
                intercepts[idx] = b
        else:
            aux_idx = np.triu_indices(Npix, k=1)
                
            pairwise_diff1 = np.subtract.outer(
                    maps[0], maps[0]
                    )[aux_idx]
            
            for idx in np.arange(Nmaps -1):
                pairwise_diff2 = np.subtract.outer(
                    maps[idx + 1], maps[idx + 1]
                    )[aux_idx]
                
                slopes[idx] = np.median(pairwise_diff2[pairwise_diff1 != 0] / pairwise_diff1[pairwise_diff1 != 0])
                intercepts[idx] = np.median(maps[idx + 1] - slopes[idx] * maps[idx])

                pairwise_diff1 = pairwise_diff2.copy()
            
        return slopes, intercepts
    

    def get_clusters_templates(self):
        if self.T_array_clusters is None: 
            T_array_clusters = np.zeros((self.n_clusters, 4))
            
            if self.T_array is not None:
                for i, idx_cluster in enumerate(self.clusters):
                    T_array_clusters[i] = np.mean(self.T_array[idx_cluster], axis=0)

            else:
                for i, idx_cluster in enumerate(self.clusters):
                    theta, phi = hp.pix2ang(self.nside, idx_cluster, lonlat=False)

                    T_array_clusters[i] = np.array([
                        1, np.mean(np.cos(phi) * np.sin(theta)),
                        np.mean(np.sin(phi) * np.sin(theta)), np.mean(np.cos(theta))
                    ])

            self.T_array_clusters = T_array_clusters

        return self.T_array_clusters
    


    def calculate_mono_dipole(self, maps: NDArray[np.float64],
                              fixed_pars: Dict = None) -> NDArray[np.float64]:
        
        N_maps = len(maps)

        # Monopole and Dipole Templates
        T_array = self.get_clusters_templates()
        n_temp = T_array.shape[-1]

        assert self.n_clusters >= n_temp * N_maps, "Number of clusters must be larger than number of parameters to fit"
        
        # Calculate slopes (a) and intercepts (b)
        a = np.zeros((N_maps - 1, self.n_clusters))
        b = np.zeros((N_maps - 1, self.n_clusters))

        for i, idx_cluster in enumerate(self.clusters):
            s_cluster, i_cluster = self.calculate_slopes_intercepts(
                maps[..., idx_cluster]
            )

            a[:, i] = s_cluster
            b[:, i] = i_cluster

        


        # Linear system A x = b
        A = np.zeros(((N_maps - 1) * self.n_clusters, n_temp * N_maps))
        
        for i, a_m in enumerate(a):
            A[i * self.n_clusters: (i + 1) * self.n_clusters, 
                i * n_temp: (i + 2) * n_temp] = np.hstack([
                    (- a_m * T_array.T).T, T_array])
            
        if fixed_pars is not None:
            for idx, par in fixed_pars.items():
                if par == "mono":
                    A = np.delete(A, idx * n_temp, axis=1)
                elif par == "dip":
                    A = np.delete(A, np.arange(idx * n_temp + 1, (idx + 1) * n_temp), axis=1)
                else:
                    raise ValueError(
                        f'Either mono or dip is fixed for map[{idx}] not {par}'
                        )
        
        b = np.ravel(b)[np.newaxis]

        x = np.linalg.inv(A.T @ A) @ A.T @ b.T
        return x[:, 0]
    
        
    

    def calculate_mono_dipole_iter(self, maps: NDArray[np.float64], fixed_pars: Dict = None,
                                   tolerance: float = 0.01) -> NDArray[np.float64]:
        
        N_maps = len(maps)

        n_fixed_pars = 0
        if fixed_pars is not None:
            for idx, par in fixed_pars.items():
                if par == "mono":
                    n_fixed_pars += 1
                elif par == "dip":
                    n_fixed_pars += 3
                else:
                    raise ValueError(
                        f'Either mono or dip is fixed for map[{idx}] not {par}'
                        )
        
        total_mono_dipole = np.zeros(4 * N_maps - n_fixed_pars)
        mono_dipole = [np.zeros(4 * N_maps - n_fixed_pars)]
        criterion = 1

        iter = 0
        while criterion > tolerance:
            print(f"{iter = }: {criterion = }")
            iter_mono_dipole = self.calculate_mono_dipole(maps, fixed_pars=fixed_pars)
            mono_dipole.append(iter_mono_dipole)


            total_mono_dipole += iter_mono_dipole

            maps = self.dep_remove_mono_dipole(maps, iter_mono_dipole, fixed_pars=fixed_pars)

            criterion = np.sum(
                np.abs(mono_dipole[-1] - mono_dipole[-2]).sum() 
                / np.max([np.abs(total_mono_dipole).sum(), 1e-7])
            )
            iter += 1
        return total_mono_dipole, np.array(mono_dipole[1:])
    


class TemplateFitting(MonoDip):

    def __init__(self, nside: int, mask: NDArray[np.float64] = None) -> None:

        super().__init__(nside, mask=mask)


    def template_fitting(self, m: NDArray[np.float64], sigma: NDArray[np.float64],
                         template_maps: NDArray[np.float64],
                        ) -> NDArray[np.float64]:
        
        T_monodip = self.get_templates()
        T_array = np.hstack([T_monodip, template_maps.T])

        z = np.linalg.inv((T_array.T / (sigma ** 2)) @ T_array) @ T_array.T @ ( m / (sigma ** 2)) [..., np.newaxis]
        return z