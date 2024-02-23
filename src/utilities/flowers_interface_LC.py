#%% minimally edit version of Michael's implementation of FLOWERS
# (I was worried my previous version was too edited and i could have introduced some bugs!)

import numpy as np
import pandas as pd

class FlowersInterface():
    """
    Flowers is a high-level user interface to the FLOWERS AEP model.

    Args:
        wind_rose (pandas.DataFrame): A dataframe for the wind rose in the FLORIS
            format containing the following information:
                - 'ws' (float): wind speeds [m/s]
                - 'wd' (float): wind directions [deg]
                - 'freq_val' (float): frequency for each wind speed and direction
        layout_x (numpy.array(float)): x-positions of each turbine [m]
        layout_y (numpy.array(float)): y-positions of each turbine [m]
        num_terms (int, optional): number of Fourier modes
        k (float, optional): wake expansion rate
        turbine (str, optional): turbine type:
                - 'nrel_5MW' (default)

    """

    ###########################################################################
    # Initialization tools
    ###########################################################################

    def __init__(self, U_i,P_i,theta_i, layout, turb,num_terms=0, k=0.05):

        self.U_i = np.copy(U_i)
        self.P_i = np.copy(P_i)
        self.theta_i = np.copy(theta_i)

        self.layout_x = layout[:,0]
        self.layout_y = layout[:,1]
        self.k = k

        self.turb = turb
        self.D = turb.D
        self.U = turb.U
        
        self._fourier_coefficients(num_terms=num_terms)
    
    def reinitialize(self, wind_rose=None, layout_x=None, layout_y=None, num_terms=None, k=None):

        if wind_rose is not None:
            self.wind_rose = wind_rose
            self._fourier_coefficients(num_terms=num_terms)
        
        if num_terms is not None:
            self._fourier_coefficients(num_terms=num_terms)
        
        if layout_x is not None:
            self.layout_x = layout_x
        
        if layout_y is not None:
            self.layout_y = layout_y
        
        if k is not None:
            self.k = k
    
    ###########################################################################
    # User functions
    ###########################################################################

    def get_layout(self):
        return self.layout_x, self.layout_y
    
    def get_wind_rose(self):
        return self.wind_rose
    
    def get_num_modes(self):
        return len(self.fs)
    
    def calculate_aep(self, gradient=False):
        """
        Compute farm AEP (and Cartesian gradients) for the given layout and wind rose.
        
        Returns:
            aep (float): farm AEP [Wh]
            gradient (numpy.array(float)): (dAEP/dx, dAEP/dy) for each turbine [Wh/m]
        """
        
        # Power component from freestream
        u0 = self.fs.c[0]

        # Normalize and reshape relative positions into symmetric 2D array
        xx = (self.layout_x - np.reshape(self.layout_x,(-1,1)))
        yy = (self.layout_y - np.reshape(self.layout_y,(-1,1)))

        # Convert to normalized polar coordinates
        R = np.sqrt(xx**2 + yy**2)
        THETA = np.arctan2(yy,xx) / (2 * np.pi)

        # Set up mask for rotor swept area
        mask_area = np.array(R <= 0.5, dtype=int)
        mask_val = self.fs.c[0]

        # Critical polar angle of wake edge (as a function of distance from turbine)
        theta_c = np.arctan(
            (1 / (2*R) + self.k * np.sqrt(1 + self.k**2 - (2*R)**(-2)))
            / (-self.k / (2*R) + np.sqrt(1 + self.k**2 - (2*R)**(-2)))
            ) / (2 * np.pi)
        theta_c = np.nan_to_num(theta_c)
        
        # Contribution from zero-frequency Fourier mode
        du = self.fs.a[0] * theta_c / (2 * self.k * R + 1)**2 * (
            1 + (8 * np.pi**2 * theta_c**2 * self.k * R) / (3 * (2 * self.k * R + 1)))

        # Reshape variables for vectorized calculations
        m = np.arange(1, len(self.fs.b))
        a = np.swapaxes(np.tile(np.expand_dims(self.fs.a[1:], axis=(1,2)),np.shape(R.T)),0,2)
        b = np.swapaxes(np.tile(np.expand_dims(self.fs.b[1:], axis=(1,2)),np.shape(R.T)),0,2)
        R = np.tile(np.expand_dims(R, axis=2),len(m))
        THETA = np.tile(np.expand_dims(THETA, axis=2),len(m))
        theta_c = np.tile(np.expand_dims(theta_c, axis=2),len(m))

        # Vectorized contribution of higher Fourier modes
        du += np.sum((1 / (np.pi * m * (2 * self.k * R + 1)**2) * (
            a * np.cos(2 * np.pi * m * THETA) + b * np.sin(2 * np.pi * m * THETA)) * (
                np.sin(2 * np.pi * m * theta_c) + 2 * self.k * R / (m**2 * (2 * self.k * R + 1)) * (
                    ((2 * np.pi * theta_c * m)**2 - 2) * np.sin(2 * np.pi * m * theta_c) + 4*np.pi*m*theta_c*np.cos(2 * np.pi * m * theta_c)))), axis=2)

        # Apply mask for points within rotor radius
        du = du * (1 - mask_area) + mask_val * mask_area
        np.fill_diagonal(du, 0.)
        
        # Sum power for each turbine
        du = np.sum(du, axis=1)
        U_wav = (u0 - du)*self.U #un-normalise
        
        alpha = self.turb.Cp_f(U_wav)*U_wav**3 
        pow_j = (0.5*self.turb.A*1.225*alpha)/(1*10**6)

        return pow_j

    ###########################################################################
    # Private functions
    ###########################################################################

    def _fourier_coefficients(self, num_terms=0):
        """
        Compute the Fourier series expansion coefficients from the wind rose.
        Modifies the Flowers interface in place to add a Fourier coefficients
        dataframe:
            fs (pandas:dataframe): Fourier coefficients used to expand the wind rose:
                - 'a_free': real coefficients of freestream component
                - 'a_wake': real coefficients of wake component
                - 'b_wake': imaginary coefficients of wake component

        Args:
            num_terms (int, optional): the number of Fourier modes to save in the range
                [1, floor(num_wind_directions/2)]
        
        """

        # Transform wind direction to polar angle 
        self.theta_i = 270 - self.theta_i #previously 450
        self.theta_i = np.remainder(self.theta_i, 360)
        # 450 is 360 + 90 so effectively 90 - theta
        # Get the indices that would sort theta_i
        sorted_indices = np.argsort(self.theta_i)

        # Reorder all arrays based on theta_i values
        self.theta_i[:] = self.theta_i[sorted_indices]
        self.U_i[:] = self.U_i[sorted_indices]
        self.P_i[:] = self.P_i[sorted_indices]

        self.U_i = self.U_i / self.U #normalise by cut out

        # Look up thrust and power coefficients for each wind direction bin
        ct = self.turb.Ct_f(self.U_i*self.U)

        # Average freestream term
        c = np.sum(self.U_i * self.P_i)

        # Fourier expansion of wake deficit term
        c1 = (1 - np.sqrt(1 - ct)) * self.U_i* self.P_i
        c1ft = 2 * np.fft.rfft(c1)
        a =  c1ft.real
        b = -c1ft.imag

        # Truncate Fourier series to specified number of modes
        if num_terms > 0 and num_terms <= len(a):
            a = a[0:num_terms]
            b = b[0:num_terms]

        # Compile Fourier coefficients
        self.fs = pd.DataFrame({'a': a, 'b': b, 'c': c})