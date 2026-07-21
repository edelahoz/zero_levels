#!/usr/bin/env python3

import numpy as np
import healpy as hp
import logging

from scipy.optimize import minimize
from typing import List, Union, Dict, Optional
from numpy.typing import NDArray

class MonoDip():

    def __init__(
            self, nside: int, mask: NDArray[np.float64] = None, 
            calculate_dipole: bool =True) -> None:

        self.nside = nside
        self.T_array = None
        self.mask = mask
        self.calculate_dipole = calculate_dipole

    def get_templates(self):
        if self.T_array is None:
            if self.mask is not None:
                pixels = np.argwhere(self.mask != 0).flatten()
            else:
                pixels = np.arange(12 * self.nside ** 2)

            if self.calculate_dipole:
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
            else:
                T_array = np.ones((pixels.size, 1))
            
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
                 mask: NDArray = None, 
                 calculate_dipole: bool = True,
                 min_pix_per_cluster: float = 2,
                 log_file: Optional[str] = None,
                 mask_name: Optional[str] = None,
        ) -> None:

        super().__init__(
            nside, mask=mask, calculate_dipole=calculate_dipole
        )

        if clusters is None:
            if nside_cluster is None:
                raise ValueError("Either nside_cluster or clusters has to be provided")
            else:
                clusters = self.get_HEALPix_super_clusters(nside, nside_cluster, mask=mask)

        clusters = [cluster for cluster in clusters if cluster.size >= min_pix_per_cluster]
        self.n_clusters = len(clusters)
        self.clusters = clusters
        self.T_array_clusters = None

        # --- Set up Logging ---
        self.logger = logging.getLogger(f"{__name__}.TTplots")
        self.logger.setLevel(logging.INFO)
         
         # Prevent adding multiple handlers if the class is instantiated multiple times
        if not self.logger.handlers:
            if log_file:
                # Log to the specified file
                handler = logging.FileHandler(log_file, mode='w')
            else:
                # Log to console by default
                handler = logging.StreamHandler()
                
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)

        if mask_name is not None:
            mask_str = f"mask: {mask_name}"
        else:
            mask_str
        self.logger.info(f"Initialized: \n"
                         f"\t Nside={nside} \n"
                         f"\t Nside_cluster={nside_cluster} \n" 
                         f"\t {mask_str} \n"
                         "\t Monopole + Dipole \n" if calculate_dipole else "\t Monopole \n"
                         f"\t Min pix per cluster {min_pix_per_cluster} \n"
                         )
        

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
                                   mask: NDArray = None) -> List[NDArray[np.int32]]:
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

            if idx_cluster.size != 0: clusters.append(idx_cluster)

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
            for idx in np.arange(Nmaps - 1):
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
            
            for idx in np.arange(Nmaps - 1):
                pairwise_diff2 = np.subtract.outer(
                    maps[idx + 1], maps[idx + 1]
                    )[aux_idx]
                
                slopes[idx] = np.median(pairwise_diff2[pairwise_diff1 != 0] / pairwise_diff1[pairwise_diff1 != 0])
                intercepts[idx] = np.median(maps[idx + 1] - slopes[idx] * maps[idx])

                pairwise_diff1 = pairwise_diff2.copy()
            
        return slopes, intercepts
    

    def get_clusters_templates(self):
        if self.T_array_clusters is None: 

            if self.calculate_dipole:
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
                
            else:
                T_array_clusters = np.ones((self.n_clusters, 1))

            self.T_array_clusters = T_array_clusters
        return self.T_array_clusters

    def _find_coldest_pixels(
            self,
            maps: NDArray[np.float64],
            n_coldest: int | None = None,
            min_separation_deg: float = 10.0,
    ) -> List[NDArray[np.int32]]:
        """
        For each map, find the coldest pixels separated by at least
        min_separation_deg degrees on the sky. Only pixels inside the
        mask (if any) are considered.

        Parameters
        ----------
        maps : (N_maps, N_pix)
        n_coldest : int
            Maximum number of cold pixels to collect per map.
        min_separation_deg : float
            Minimum angular separation between selected cold pixels [degrees].

        Returns
        -------
        List of length N_maps, each element is an array of pixel indices.
        """
        min_sep_rad = np.radians(min_separation_deg)
        N_maps = maps.shape[0]

        if self.mask is not None:
            valid_pixels = np.argwhere(self.mask != 0).flatten()
        else:
            valid_pixels = np.arange(hp.nside2npix(self.nside))

        coldest_per_map = []

        for i in range(N_maps):
            sorted_idx = np.argsort(maps[i, valid_pixels])
            sorted_pixels = valid_pixels[sorted_idx]

            selected = []
            vecs_selected = []

            for pix in sorted_pixels:
                if (n_coldest is not None) and (len(selected) >= n_coldest):
                    break

                vec = np.array(hp.pix2vec(self.nside, int(pix)))

                too_close = False
                for v in vecs_selected:
                    cos_angle = np.clip(np.dot(vec, v), -1.0, 1.0)
                    if cos_angle > np.cos(min_sep_rad):
                        too_close = True
                        break

                if not too_close:
                    selected.append(pix)
                    vecs_selected.append(vec)

            coldest_per_map.append(np.array(selected, dtype=np.int32))

        return coldest_per_map


    def _build_positivity_constraints(
            self,
            maps: NDArray[np.float64],
            coldest_pixels_per_map: List[NDArray[np.int32]],
            n_temp: int,
            n_params_reduced: int,
            full_to_reduced: Dict[int, int],
            sigma_maps: Optional[NDArray[np.float64]] = None,
            n_sigma: float = 4.0,
    ) -> List[Dict]:
        """
        Build scipy-style inequality constraints enforcing that the
        corrected map is non-negative at the coldest pixels:

            map_i(p) - T_i(p)·x_i ≥ -N·σ_i(p)

        Rearranged for scipy 'ineq' convention (f(x) ≥ 0):

            f(x) = map_i(p) + N·σ_i(p) - T_i(p)·x_i ≥ 0

        Parameters
        ----------
        maps : (N_maps, N_pix)
        coldest_pixels_per_map : list of pixel index arrays, one per map
        n_temp : number of templates per map (1 or 4)
        n_params_reduced : total number of unknowns after fixed_pars removal
        full_to_reduced : mapping from full column index to reduced column index
        sigma_maps : (N_maps, N_pix) noise rms per pixel, optional.
                     If None, constraints are hard (no relaxation).
        n_sigma : relaxation threshold in units of sigma (default 4)

        Returns
        -------
        List of dicts suitable for scipy.optimize.minimize(constraints=...)
        """
        constraints = []

        for i, cold_pixels in enumerate(coldest_pixels_per_map):
            col_start = i * n_temp

            for pix in cold_pixels:
                map_value = maps[i, pix]
                noise_relax = 0.0
                if sigma_maps is not None:
                    noise_relax = n_sigma * sigma_maps[i, pix]

                rhs = map_value + noise_relax

                # Evaluate template at this exact pixel
                theta, phi = hp.pix2ang(self.nside, int(pix), lonlat=False)
                if self.calculate_dipole:
                    t_row = np.array([
                        1.0,
                        np.cos(phi) * np.sin(theta),
                        np.sin(phi) * np.sin(theta),
                        np.cos(theta),
                    ])
                else:
                    t_row = np.array([1.0])

                # Map template row into reduced parameter space
                c_vec = np.zeros(n_params_reduced)
                for k in range(n_temp):
                    full_col = col_start + k
                    if full_col in full_to_reduced:
                        c_vec[full_to_reduced[full_col]] = t_row[k]

                # Closure with default args to avoid late-binding bug
                def make_constraint(cv, r):
                    return {
                        "type": "ineq",
                        "fun": lambda x, cv=cv, r=r: r - cv @ x,
                        "jac": lambda x, cv=cv: -cv,
                    }

                constraints.append(make_constraint(c_vec, rhs))

        return constraints


    def _build_full_to_reduced_mapping(
            self,
            N_maps: int,
            n_temp: int,
            fixed_pars: Optional[Dict] = None,
    ) -> tuple[Dict[int, int], int]:
        """
        Build a mapping from full parameter column indices to reduced
        column indices after removing fixed parameters.

        Returns
        -------
        full_to_reduced : dict mapping full index → reduced index
        n_params_reduced : total number of free parameters
        """
        total_params = n_temp * N_maps

        removed_cols = []
        if fixed_pars is not None and self.calculate_dipole:
            for idx, par in fixed_pars.items():
                if par == "mono":
                    removed_cols.append(idx * n_temp)
                elif par == "dip":
                    removed_cols.extend(
                        list(range(idx * n_temp + 1, (idx + 1) * n_temp))
                    )

        kept_cols = [c for c in range(total_params) if c not in removed_cols]
        n_params_reduced = len(kept_cols)
        full_to_reduced = {full: red for red, full in enumerate(kept_cols)}

        return full_to_reduced, n_params_reduced
    
    @staticmethod
    def _parse_slope_limits(limits, default_val, N_pairs):
        if limits is None:
            return np.full(N_pairs, default_val)
        if np.isscalar(limits):
            return np.full(N_pairs, limits)
        if len(limits) == N_pairs:
            # Convert list to array, replacing any inner None with default_val
            return np.array([val if val is not None else default_val for val in limits])
        raise ValueError(f"Slope limit must be None, a scalar, or have exactly {N_pairs} elements.")


    def calculate_mono_dipole(
            self,
            maps: NDArray[np.float64],
            fixed_pars: Optional[Dict] = None,
            use_prior: bool = False,
            sigma_maps: Optional[NDArray[np.float64]] = None,
            n_sigma: float = 4.0,
            n_coldest: int = 50,
            ftol: float = 1e-6,
            min_separation_deg: float = 10.0,
            iter: int = None,
            ext: str = "",
            path_si: str = None,
            coldest_pixels: Optional[List] = None,
            min_slope: Optional[float] = None,
            max_slope: Optional[float] = None,
    ) -> NDArray[np.float64]:
        """
        Calculate monopole (and optionally dipole) zero levels using the
        TT-plot linear system. Optionally imposes a positivity prior on
        the corrected maps at the coldest sky pixels.

        Parameters
        ----------
        maps : (N_maps, N_pix)
        fixed_pars : dict mapping map index → 'mono' or 'dip', optional.
                     Fixes either the monopole or dipole of a given map.
        use_prior : bool
            If True, impose positivity constraints at the coldest pixels.
            Solved via SLSQP. If False, solve the unconstrained normal
            equations directly. Default False.
        sigma_maps : (N_maps, N_pix) noise rms per pixel, optional.
            Used to relax the positivity constraints by N·σ. Only used
            when use_prior=True. If None, constraints are hard.
        n_sigma : float
            Constraint relaxation threshold in units of sigma. Only used
            when use_prior=True and sigma_maps is not None. Default 4.
        n_coldest : int
            Maximum number of cold pixels per map to use as constraints.
            Only used when use_prior=True and coldest_pixels is None. Default 50.
        min_separation_deg : float
            Minimum angular separation between cold pixels [degrees].
            Only used when use_prior=True. Default 10.
        iter : int, optional
            Iteration number, used for saving slopes/intercepts to disk.
        ext : str
            Filename extension suffix for saving slopes/intercepts.
        path_si : str, optional
            Directory path for saving slopes/intercepts. If None, nothing
            is saved.
        coldest_pixels: List, optional
            Coldest pixels for map combination.
        min_slope : float, list, or None, optional
            Minimum allowed slope(s) to retain a cluster. 
            Can be None (no limits), a scalar, or list of length N_maps - 1.
        max_slope : float, list, or None, optional
            Maximum allowed slope(s) to retain a cluster. 
            Can be None (no limits), a scalar, or list of length N_maps - 1.
        Returns
        -------
        x : (n_params,) array of zero-level coefficients
        """
        N_maps = len(maps)
        T_array = self.get_clusters_templates()
        n_temp = T_array.shape[-1]

        assert self.n_clusters >= n_temp * N_maps, (
            "Number of clusters must be larger than number of parameters to fit"
        )

        # ── Step 1: slopes and intercepts per cluster ────────────────────
        a = np.zeros((N_maps - 1, self.n_clusters))
        b = np.zeros((N_maps - 1, self.n_clusters))

        for i, idx_cluster in enumerate(self.clusters):
            s_cluster, i_cluster = self.calculate_slopes_intercepts(
                maps[..., idx_cluster]
            )
            a[:, i] = s_cluster
            b[:, i] = i_cluster
        
        # ── Filter clusters by allowed slope ranges  ───────────────
        min_slopes_arr = self._parse_slope_limits(min_slope, -np.inf, N_maps - 1)
        max_slopes_arr = self._parse_slope_limits(max_slope, np.inf, N_maps - 1)

        valid_cluster_mask = (a >= min_slopes_arr[:, None]) & (a <= max_slopes_arr[:, None])
        valid_cluster_mask = np.all(valid_cluster_mask, axis=0)
        
        # Logging removed clusters
        num_removed = np.sum(~valid_cluster_mask)
        if num_removed > 0:
            removed_indices = np.where(~valid_cluster_mask)[0]
            
            self.logger.info("\t" + f"[INFO] Filtered out {num_removed} clusters due to slope range limits.")
            self.logger.info(f"Removed cluster indices: {removed_indices.tolist()}")
        else:
            self.logger.info("\t" + f"[INFO] No clusters removed by slope limits.")

        # Apply the mask to keep only valid clusters
        a = a[:, valid_cluster_mask]
        b = b[:, valid_cluster_mask]
        
        valid_clusters_list = [self.clusters[idx] for idx in range(self.n_clusters) if valid_cluster_mask[idx]]
        self.clusters = valid_clusters_list
        self.n_clusters = len(self.clusters)
        
        T_array = T_array[valid_cluster_mask]
        if self.T_array_clusters is not None:
            self.T_array_clusters = self.T_array_clusters[valid_cluster_mask]


        if path_si is not None:
            np.save(f"{path_si}/slopes_iter{ext}_n{iter}.npy", a)
            np.save(f"{path_si}/intercepts_iter{ext}_n{iter}.npy", b)

        # ── Step 2: build linear system A x = b_vec ──────────────────────
        A = np.zeros(((N_maps - 1) * self.n_clusters, n_temp * N_maps))
        for i, a_m in enumerate(a):
            A[i * self.n_clusters: (i + 1) * self.n_clusters,
              i * n_temp: (i + 2) * n_temp] = np.hstack([
                (-a_m * T_array.T).T, T_array
            ])

        if self.calculate_dipole and fixed_pars is not None:
            for idx, par in fixed_pars.items():
                if par == "mono":
                    A = np.delete(A, idx * n_temp, axis=1)
                elif par == "dip":
                    A = np.delete(
                        A, np.arange(idx * n_temp + 1, (idx + 1) * n_temp), axis=1
                    )
                else:
                    raise ValueError(
                        f'Either mono or dip is fixed for map[{idx}] not {par}'
                    )

        b_vec = np.ravel(b)

        # ── Step 3: unconstrained solve ───────────────────────────────────
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            x0 = (np.linalg.inv(A.T @ A) @ A.T @ b_vec[:, np.newaxis])[:, 0]

        if not use_prior:
            return x0

        # ── Step 4: build positivity constraints ──────────────────────────
        full_to_reduced, n_params_reduced = self._build_full_to_reduced_mapping(
            N_maps, n_temp, fixed_pars
        )

        if coldest_pixels is None:
            coldest_pixels = self._find_coldest_pixels(
                maps, n_coldest=n_coldest, min_separation_deg=min_separation_deg
            )

        constraints = self._build_positivity_constraints(
            maps=maps,
            coldest_pixels_per_map=coldest_pixels,
            n_temp=n_temp,
            n_params_reduced=n_params_reduced,
            full_to_reduced=full_to_reduced,
            sigma_maps=sigma_maps,
            n_sigma=n_sigma,
        )


        # Check how many constraints x0 already violates
        n_violated = 0
        for con in constraints:
            if con["fun"](x0) < 0:
                n_violated += 1
        
        if n_violated == 0:
            self.logger.info("\t" + f"[INFO] Unconstrained solution satisfies all \n"
                             "\t" + f"positivity constraints, skipping constrained solve.")
            return x0
        self.logger.info("\t" + f"[INFO] {n_violated}/{len(constraints)} constraints "
                         f"violated by unconstrained solution")


        # ── Step 5: constrained QP solve via SLSQP ────────────────────────
        with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
            AtA = A.T @ A
            Atb = A.T @ b_vec

        def objective(x):
            with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
                r = A @ x - b_vec
                val = float(r @ r)
            return val

        def gradient(x):
            return 2.0 * (AtA @ x - Atb)

        result = minimize(
            fun=objective,
            x0=x0,
            jac=gradient,
            method="trust-constr",
            constraints=constraints,
            options={"gtol": ftol, "maxiter": 1000, "disp": False,},
        )

        # Check how many constraints are violated by the result
        n_violated = 0
        violated_info = []
        for j, con in enumerate(constraints):
            val = con["fun"](result.x)
            if val < 0:
                n_violated += 1
                violated_info.append((j, val))

        if not result.success:
            delta = np.max(np.abs(result.x - x0))

            if n_violated == 0:
                self.logger.info(
                    "\t" + f"[INFO] Constrained solver did not fully converge: "
                    f"{result.message}. "
                    f"All constraints satisfied "
                    f"Max change from unconstrained solution: {delta:.3e} "
                )
            else:
                self.logger.warning(
                    "\t" + f"[WARNING] Constrained solver did not fully converge: "
                    f"{result.message}. "
                    f"{n_violated}/{len(constraints)} constraints violated. "
                    f"Max change from unconstrained solution: {delta:.3e}."
                )
                self.logger.warning("\t" + f"Violated constraints (index, violation value):")
                for j, val in violated_info:
                    self.logger.warning(f"    constraint[{j}]: {val:.3e}")
        else:
            self.logger.info("\t" + '[INFO] Constrained solver converged.')
            self.logger.info("\t" + f"[INFO] {n_violated}/{len(constraints)} constraints "
                  f"violated by constrained solution.")

        return result.x

    
    def remove_zero_levels(self, maps: NDArray[np.float64], 
                           zero_levels: NDArray[np.float64],
                           fixed_pars: Dict = None,) -> NDArray[np.float64]:
        
        corrected_maps = np.zeros(maps.shape)
        vecs = hp.pix2vec(self.nside, np.arange(12 * self.nside ** 2))
        if self.calculate_dipole:
            aux_idx = 0
            for i, m in enumerate(maps):
                if (fixed_pars is not None) and (i in fixed_pars.keys()):
                    par = fixed_pars[i]
                    if par == "mono":
                        corrected_maps[i] = self.remove_dipoles(
                            m, zero_levels[i * 4 - aux_idx : (i + 1) * 4 - 1 - aux_idx]
                        )
                        aux_idx += 1
                    elif par == "dip":
                        corrected_maps[i] = self.remove_monopoles(
                            m, zero_levels[i * 4 - aux_idx]
                        )
                        aux_idx += 3
                else:
                    monopole = zero_levels[i * 4 - aux_idx]
                    
                    dipole_amp = np.sqrt(np.sum(zero_levels[i * 4 + 1 - aux_idx: (i + 1) * 4 - aux_idx] ** 2 ))
                    dipole_direction = zero_levels[i * 4 + 1 - aux_idx : (i + 1) * 4 - aux_idx] / dipole_amp
                    dipole = dipole_amp * np.dot(dipole_direction, vecs)
                    
                    corrected_maps[i] = m - monopole - dipole
        else:
            for i, m in enumerate(maps):
                monopole = zero_levels[i]
                        
                corrected_maps[i] = m - monopole

        return corrected_maps
    

    def calculate_zero_levels_iter(
            self,
            maps: NDArray[np.float64],
            fixed_pars: Optional[Dict] = None,
            tolerance: float = 0.01,
            use_prior: bool = False,
            sigma_maps: Optional[NDArray[np.float64]] = None,
            n_sigma: float = 4.0,
            n_coldest: int = 50,
            min_separation_deg: float = 10.0,
            ext: str = "",
            path_si: str = None,
            N_max_iter: int = 50,
            coldest_pixels: Optional[List] = None,
            case_name: Optional[str] = None,
            min_slope: Optional[float] = None,
            max_slope: Optional[float] = None,
    ) -> NDArray[np.float64]:
        """
        Iteratively calculate zero levels (monopoles and optionally dipoles)
        until convergence.

        Parameters
        ----------
        maps : (N_maps, N_pix)
        fixed_pars : dict mapping map index → 'mono' or 'dip', optional.
        tolerance : float
            Convergence criterion on the relative change in zero levels.
            Default 0.01.
        use_prior : bool
            If True, impose the positivity prior at every iteration.
            Default False.
        sigma_maps : (N_maps, N_pix) noise rms per pixel, optional.
            Used to relax positivity constraints by N·σ per pixel.
            Only used when use_prior=True. If None, constraints are hard.
        n_sigma : float
            Constraint relaxation threshold in units of sigma. Default 4.
        n_coldest : int
            Maximum number of cold pixels per map used as constraints.
            Only used when use_prior=True. Default 50.
        min_separation_deg : float
            Minimum angular separation between cold pixels [degrees].
            Only used when use_prior=True. Default 10.
        ext : str
            Filename suffix for saving slopes/intercepts.
        path_si : str, optional
            Directory for saving slopes/intercepts. If None, nothing saved.
        min_slope : float, list, or None, optional
            Minimum allowed slope(s) to retain a cluster. 
            Can be None (no limits), a scalar, or list of length N_maps - 1.
        max_slope : float, list, or None, optional
            Maximum allowed slope(s) to retain a cluster. 
            Can be None (no limits), a scalar, or list of length N_maps - 1.

        Returns
        -------
        total_zero_levels : (n_params,) cumulative zero levels
        zero_levels_list : (n_iter, n_params) zero levels at each iteration
        """
        if case_name is not None:
            self.logger.info(60 * "-")
            self.logger.info(f"Running: {case_name}...")
            self.logger.info(60 * "-")
        N_maps = len(maps)

        if self.calculate_dipole:
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
            total_zero_levels = np.zeros(4 * N_maps - n_fixed_pars)
            zero_levels_list = [np.zeros(4 * N_maps - n_fixed_pars)]
        else:
            assert fixed_pars is None, (
                "Fixing dipole is not available if calculate_dipole is False"
            )
            total_zero_levels = np.zeros(N_maps)
            zero_levels_list = [np.zeros(N_maps)]

        criterion = 1
        iter = 0
        while criterion > tolerance and iter <= N_max_iter:
            self.logger.info("\t" + 10 * '*' + f"   ITER: {iter}   " + 10 * '*')
            iter_mono_dipole = self.calculate_mono_dipole(
                maps,
                fixed_pars=fixed_pars,
                use_prior=use_prior,
                sigma_maps=sigma_maps,
                n_sigma=n_sigma,
                n_coldest=n_coldest,
                min_separation_deg=min_separation_deg,
                iter=iter,
                ext=ext,
                path_si=path_si,
                coldest_pixels=coldest_pixels,
                min_slope=min_slope,
                max_slope=max_slope,
            )
            zero_levels_list.append(iter_mono_dipole)
            total_zero_levels += iter_mono_dipole

            maps = self.remove_zero_levels(maps, iter_mono_dipole, fixed_pars=fixed_pars)

            criterion = np.sum(
                np.abs(zero_levels_list[-1] - zero_levels_list[-2]).sum()
                / np.max([np.abs(total_zero_levels).sum(), 1e-7])
            )
            
            self.logger.info("\t" + f"[INFO] {criterion = }")
            iter += 1
        if iter > N_max_iter:
            self.logger.warning("\t" + f"[WARNING] Reached maximum iterations ({N_max_iter}) without \n"
                  f"convergence (criterion={criterion:.3e} > tolerance={tolerance:.3e}).")
        self.logger.info("\n")
        return total_zero_levels, np.array(zero_levels_list[1:])


class TemplateFitting(MonoDip):

    def __init__(self, nside: int, mask: NDArray[np.float64] = None,
                 calculate_dipole: bool = True) -> None:

        super().__init__(
            nside, mask=mask, calculate_dipole=calculate_dipole
        )

    def template_fitting(self, m: NDArray[np.float64], sigma: NDArray[np.float64],
                         template_maps: NDArray[np.float64],
                        ) -> NDArray[np.float64]:
        
        T_monodip = self.get_templates()
        T_array = np.hstack([T_monodip, template_maps.T])

        z = np.linalg.inv((T_array.T / (sigma ** 2)) @ T_array) @ T_array.T @ ( m / (sigma ** 2)) [..., np.newaxis]
        return z