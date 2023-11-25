
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
    
    def calculate_aep(self, gradient=False):
        """
        Compute farm AEP (and Cartesian gradients) for the given layout and wind rose.
        
        Returns:
            aep (float): farm AEP [Wh]
            gradient (numpy.array(float)): (dAEP/dx, dAEP/dy) for each turbine [Wh/m]
        """
        
        # Power component from freestream
        u0 = self.fs.c[0]

        # (removed normalisation here) reshape relative positions into symmetric 2D array
        xx = (self.layout_x - np.reshape(self.layout_x,(-1,1)))
        yy = (self.layout_y - np.reshape(self.layout_y,(-1,1)))

        # Convert to normalized polar coordinates
        R = np.sqrt(xx**2 + yy**2)
        THETA = np.arctan2(yy,xx) / (2 * np.pi)

        mask_area = np.array(R <= 0.5, dtype=int)
        mask_val = u0

        # Critical polar angle of wake edge (as a function of distance from turbine)
        theta_c = np.arctan(
            (1 / (2*R) + self.k * np.sqrt(1 + self.k**2 - (2*R)**(-2)))
            / (-self.k / (2*R) + np.sqrt(1 + self.k**2 - (2*R)**(-2)))
            ) / (2 * np.pi)
        theta_c = np.nan_to_num(theta_c)
        
        # Contribution from zero-frequency Fourier mode
        du = self.fs.a[0] * theta_c / (2 * self.k * R + 1)**2 * (
            1 + (8 * np.pi**2 * theta_c**2 * self.k * R) / (3 * (2 * self.k * R + 1)))
        #du = np.where(R>=0.5,du,u0)      
        
        # Initialize gradient and calculate zero-frequency modes
        if gradient == True:
            grad = np.zeros((len(self.layout_x),2))

            # Change in theta_c wrt radius
            dtdr = (-1 / (4 * np.pi * R**2 * np.sqrt(self.k**2 - (2*R)**(-2) + 1)))
            dtdr = np.nan_to_num(dtdr)

            # Zero-frequency mode of change in power deficit wrt radius
            dpdr = (-4 * self.fs.a[0] * self.k * theta_c * (3 + 6 * self.k * R + 2 * np.pi**2 * (4 * self.k * R - 1) * theta_c**2) + 
                    3 * self.fs.a[0] * (1 + 2 * self.k * R) * (1 + 2 * self.k * R + 8 * np.pi**2 * self.k * R * theta_c**2) * dtdr) / (
                3 * (1 + 2*self.k*R)**4)



        # Reshape variables for vectorized calculations
        m = np.arange(1, len(self.fs.b))
        a = self.fs.a[None, None,1:] 
        b = self.fs.b[None, None,1:] 
        R = R[:, :, None]
        THETA = THETA[:, :, None] 
        theta_c = theta_c[:, :, None] 

        # Auxilaries (pretty rudimentary)
        pi_m = np.pi * m
        t_pi_m = 2*pi_m
        t_pi_m_T = t_pi_m * THETA
        t_pi_m_tc = t_pi_m * theta_c
        s_t_pi_m_tc = np.sin(t_pi_m_tc)
                # Set up mask for rotor swept area

        # Vectorized contribution of higher Fourier modes
        du += np.sum((1 / (pi_m * (2 * self.k * R + 1)**2) * (
            a * np.cos(t_pi_m_T) + b * np.sin(t_pi_m_T)) * (
                s_t_pi_m_tc + 2 * self.k * R / (m**2 * (2 * self.k * R + 1)) * (
                    ((t_pi_m_tc)**2 - 2) * s_t_pi_m_tc + 2*t_pi_m_tc*np.cos(t_pi_m_tc)))), axis=2)

        if gradient==True:
            dtdr = np.tile(np.expand_dims(dtdr, axis=2),len(m))
            
            # Higher Fourier modes of change in power deficit wrt angle
            dpdt = np.sum((2 / (2 * self.k * R + 1)**2 * (
                b * np.cos(2 * np.pi * m * THETA) - a * np.sin(2 * np.pi * m * THETA)) * (
                    np.sin(2 * np.pi * m * theta_c) + 2 * self.k * R / (m**2 * (2 * self.k * R + 1)) * (
                        ((2 * np.pi * theta_c * m)**2 - 2) * np.sin(2 * np.pi * m * theta_c) + 4*np.pi*m*theta_c*np.cos(2 * np.pi * m * theta_c)))), axis=2)

            # Higher Fourier modes of change in power deficit wrt radius
            dpdr += np.sum(((a * np.cos(2 * np.pi * m * THETA) + b * np.sin(2 * np.pi * m * THETA)) / (np.pi * m**3 * (2 * self.k * R + 1)**4) * (
                -4 * self.k * np.sin(2 * np.pi * m * theta_c) * (1 + m**2 + 2 * self.k * R * (m**2 - 2) + 2 * np.pi**2 * m**2 * (4 * self.k * R - 1) * theta_c**2) + 
                2 * np.pi * m * np.cos(2 * np.pi * m * theta_c) * (4 * self.k * (1 - 4 * self.k * R) * theta_c + m**2 * (2 * self.k * R + 1) * (
                1 + 2 * self.k * R + 8 * np.pi**2 * self.k * R * theta_c**2) * dtdr))), axis=2)
            
        # Apply mask for points within rotor radius
        du = du * (1 - mask_area) + mask_val * mask_area
        np.fill_diagonal(du, 0.)
        
        # Sum power for each turbine
        du = np.sum(du, axis=1) #superposistion sum
        aep = (u0 - du)**3 #turbine sum #np.sum((u0 - du)**3)
        aep *= np.pi / 8 * 1.225 * self.D**2 * self.U**3  # * 8760 all my measurements are in MW

        # Complete gradient calculation
        if gradient==True:
            dx = xx/np.sqrt(xx**2+yy**2)*dpdr + -yy/(2*np.pi*(xx**2+yy**2))*dpdt
            dy = yy/np.sqrt(xx**2+yy**2)*dpdr + xx/(2*np.pi*(xx**2+yy**2))*dpdt

            dx = np.nan_to_num(dx)
            dy = np.nan_to_num(dy)
            
            coeff = (u0 - du)**2
            for i in range(len(grad)):
                # Isolate gradient to turbine 'i'
                grad_mask = np.zeros_like(xx)
                grad_mask[i,:] = -1.
                grad_mask[:,i] = 1.

                grad[i,0] = np.sum(coeff*np.sum(dx*grad_mask,axis=1)) 
                grad[i,1] = np.sum(coeff*np.sum(dy*grad_mask,axis=1))

            grad *= -3 * np.pi / 8 * 1.225 * self.D * self.U**3 * 8760 

            return aep, grad
        
        else:
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
        self.theta_i = np.remainder(450 - self.theta_i, 360)

        # Get the indices that would sort theta_i
        sorted_indices = np.argsort(self.theta_i)

        # Reorder all arrays based on theta_i values
        self.theta_i[:] = self.theta_i[sorted_indices]
        self.U_i[:] = self.U_i[sorted_indices]
        self.P_i[:] = self.P_i[sorted_indices]

        # Assuming self.turb.Ct_f and self.turb.Cp_f are functions that take wind speeds (self.U_i) and return thrust and power coefficients
        ct = self.turb.Ct_f(self.U_i)
        cp = self.turb.Cp_f(self.U_i)

        # Normalize wind speed by cut-out speed
        nU_i = self.U_i/self.U #(new variable to allow multiple runnings without reinitalisation)

        # Average freestream term
        c = np.sum(cp**(1/3) * nU_i * self.P_i)

        # Fourier expansion of wake deficit term
        c1 = cp**(1/3) * (1 - np.sqrt(1 - ct)) * nU_i * self.P_i
        c1ft = 2 * np.fft.rfft(c1)
        a =  c1ft.real
        b = -c1ft.imag

        # Truncate Fourier series to specified number of modes
        if num_terms > 0 and num_terms <= len(a):
            a = a[0:num_terms+1] #dc is the 0 term
            b = b[0:num_terms+1]
        else:
            if num_terms > 0 :
                raise ValueError(f"number of terms is {num_terms}")
            else:
                raise ValueError(f"num_terms ({num_terms}) > max length Fourier ({len(a)})")

        # Compile Fourier coefficients
        self.fs = pd.DataFrame({'a': a, 'b': b, 'c': c})