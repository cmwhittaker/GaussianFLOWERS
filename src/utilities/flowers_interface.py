
"""
this is modified from: https://github.com/locascio-m/flowers 
written by Michael LoCascio
# initialised using (average speed binned) wind rose directly rather than using the wind rose dataframe.
"""

import numpy as np
import pandas as pd
import flowers.tools as tl

class FlowersInterface():
    """
    Flowers is a high-level user interface to the FLOWERS AEP model.

    Args:
        U_i (bins,) : (Averaged) wind speed of the bin
        P_i (bins,) : frequency of that direction bin
        theta_i (bins,) : bin center angle in degrees (North / bearing)
        layout (turbines,2) : (x,y) (layout[:,0],layout[:,1])  posistions of the turbines

        num_terms (int, optional): number of Fourier modes
        k (float, optional): wake expansion rate
        turbine : this is a "custom" turbine object, it has attributes turb.name = 'iea_10mw' etc. and turb.Cp_f(U) for power coefficient etc.

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
    
    def calculate_aep(self):
        """
        Compute farm AEP (and Cartesian gradients) for the given layout and wind rose.
        
        Returns:
            aep (float): farm AEP [Wh]
            gradient (numpy.array(float)): (dAEP/dx, dAEP/dy) for each turbine [Wh/m]
        """
        
        # (removed normalisation here) reshape relative positions into symmetric 2D array
        from .helpers import find_relative_coords
        layout = np.column_stack((self.layout_x,self.layout_y))
        R,THETA = find_relative_coords(layout,layout)

        # Set up mask for rotor swept area
        mask_area = np.where(R<=0.5,1,0) #true within radius

        # Critical polar angle of wake edge (as a function of distance from turbine)
        theta_c = np.arctan(
            (1 / (2*R) + self.k * np.sqrt(1 + self.k**2 - (2*R)**(-2)))
            / (-self.k / (2*R) + np.sqrt(1 + self.k**2 - (2*R)**(-2)))
            ) 
        theta_c = np.nan_to_num(theta_c)
        
        # Contribution from zero-frequency Fourier mode
        du = self.a_0 * theta_c / (2 * self.k * R + 1)**2 * (
            1 + (2*(theta_c)**2 * self.k * R) / (3 * (2 * self.k * R + 1)))
        
        # Reshape variables for vectorized calculations
        m = np.arange(1, len(self.fs.b))
        a = self.fs.a[None, None,1:] 
        b = self.fs.b[None, None,1:] 
        R = R[:, :, None]
        THETA = THETA[:, :, None] 
        theta_c = theta_c[:, :, None] 

        # Vectorized contribution of higher Fourier modes
        du += np.sum(
            (2*(a * np.cos(m*THETA) + b * np.sin(m*THETA)) / (m * (2 * self.k * R + 1))**3 *
            (
            np.sin(m*theta_c)*(m**2*(2*self.k*R*(theta_c**2+1)+1)-4*self.k*R)+ 4*self.k*R*theta_c*m *np.cos(theta_c * m))
            ), axis=2)

        # Apply mask for points within rotor radius
        du = np.where(mask_area,self.a_0,du)
        np.fill_diagonal(du, 0.) #stop self-produced wakes
        # Sum power for each turbine
        du = np.sum(du, axis=1) #superposistion sum
        wav = (self.c_0*np.pi - du)
        alpha = self.turb.Cp_f(wav)*wav**3 #turbine sum #np.sum((u0 - du)**3) 
        aep = (0.5*self.turb.A*1.225*alpha)/(1*10**6)
    
        return aep

    ###########################################################################
    # Private functions
    ###########################################################################

    def _fourier_coefficients(self, num_terms=36):
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

        # Assuming self.turb.Ct_f and self.turb.Cp_f are functions that take wind speeds (self.U_i) and return thrust and power coefficients
        ct = self.turb.Ct_f(self.U_i)
        cp = self.turb.Cp_f(self.U_i)

        # Average freestream term
        c = np.sum(self.U_i * self.P_i)
        self.c_0 = 2*np.sum(self.U_i * self.P_i)/(2*np.pi)
        # Fourier expansion of wake deficit term
        c1 = ((1 - np.sqrt(1 - ct)) * self.U_i* self.P_i)/(2*np.pi)
        print("len(c1): {}".format(len(c1)))
        c1ft = 2 * np.fft.rfft(c1)
        a =  c1ft.real
        b = -c1ft.imag
        print("len(a): {}".format(len(a)))
        print("num_terms: {}".format(num_terms))

        self.a_0 = a[0]

        # Truncate Fourier series to specified number of modes
        if num_terms > 0 and num_terms <= len(a):
            a = a[0:num_terms+1] #dc is the 0 term
            print("len(a): {}".format(len(a)))
            b = b[0:num_terms+1]
        else:
            if num_terms > 0 :
                raise ValueError(f"number of terms is {num_terms}")
            else:
                raise ValueError(f"num_terms ({num_terms}) > max length Fourier ({len(a)})")

        # Compile Fourier coefficients
        self.fs = pd.DataFrame({'a': a, 'b': b, 'c': c})
