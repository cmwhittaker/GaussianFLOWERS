#%%

import numpy as np
#a lot of fuctionality has been added post initial writing, so it's a bit messy - but very functional.
class wind_rose:

    def load_csv_data(self,filepath):
        data = np.loadtxt(open(filepath,"rb"), delimiter=",", skiprows=2)
        direction = data[:,5] #wind direction is 6th column (zero indexed)
        magnitude = data[:,6] #wind magntiude is 7th column (zero indexed)
        return direction, magnitude

    def binWindData_d(self,direction,magnitude,bin_centres,no_bins):
        #Sort the wind data into bins by direction, then calculate an average velocity magnitude for each bin.
        #direction in degrees,magnitude in m/s, bin_width in degrees
        #note: The function returns the direction bins in radians!

        #bins are left/right +0.5/0.5 a bin_width
        bins = bin_centres+bin_centres[1]/2
        
        #index of which bin data belongs to bin_number = [0,0,1,1,2 ...]
        bin_number = np.digitize(direction,bins)
        frequency = np.zeros((no_bins),dtype='float64')
        avMagnitude = np.zeros((no_bins),dtype='float64')
        #The first bin needs to combine first and last bin bc of 2pi periodicy
        frequency[0] = np.size(direction[bin_number==0]) + np.size(direction[bin_number==no_bins])
        avMagnitude[0] = (np.mean(magnitude[bin_number==0]) + np.mean(magnitude[bin_number==no_bins]))/2
        for i in range(1,no_bins): #2nd until last bins
            frequency[i] = np.size(direction[bin_number==i]) #count number of occurances in that bin
            avMagnitude[i] = np.mean(magnitude[bin_number==i]) #calc average wind speed in that bin
        #Finally, express the frequency as a fraction 
        frequency = frequency/np.sum(frequency)
        return frequency,avMagnitude

    def Fourier_series(self,fourier_coeffs):
        a_0,a_n,b_n = fourier_coeffs
        n = np.array((np.arange(1,a_n.size+1))) #halve open things
        def f(x):
            return a_0/2 + np.sum(a_n[:,None]*np.cos(n[:,None]*x)+b_n[:,None]*np.sin((n[:,None]*x)),axis=0)
        return f

    def fft(self,data):   
        #naively fit a Fourier series to data (no normalisation takes place (!))
        import scipy.fft
        c = scipy.fft.rfft(data)/np.size(data)
        length = np.size(c)-1 #because the a_0 term is included in c # !! 
        a_0 = 2*np.real(c[0])
        a_n = 2*np.real(c[-length:])
        b_n =-2*np.imag(c[-length:])
        Fourier_coeffs = a_0,a_n,b_n
        return Fourier_coeffs
    
    def CMPLX_fft(self,data):   
        #naively fit a Fourier series to data 
        #return complex coefficients
        import scipy.fft
        c_n = scipy.fft.fft(data)/np.size(data)
        c_n = scipy.fft.fftshift(c_n)
        return c_n
    
    def truncate(self,coeffs,terms):
        a1,a2,a3 = coeffs
        if terms > a2.size:
            raise ValueError("The number of Fourier terms requested is greater than the length of the Fourier series")
        else:
            a2 = a2[0:terms]
            a3 = a3[0:terms]
        truncated_Fourier_coeffs = a1,a2,a3
        return truncated_Fourier_coeffs

    def __init__(self,bin_no_bins=20,custom=None,site=1,filepath=None,a_0=1,Cp_f=None):
        #no_bins here is the number of bins used in BINNING
        self.custom = custom
        self.bin_no_bins = bin_no_bins
        self.deg_bin_centers = np.linspace(0,360,bin_no_bins,endpoint=False)
        if custom == 1:
            print("UNIFORM wind rose, {}ms^-1 with {} bins".format(a_0,bin_no_bins))
            self.frequency = np.full(bin_no_bins,1/bin_no_bins) #each is identically likely
            self.avMagnitude = np.full(bin_no_bins,a_0) #each has same wind velocity
        elif custom == 2:
            print("IMPULSE wind rose, {}ms^-1 with {} bins, Northerly".format(a_0,bin_no_bins))
            self.frequency = np.zeros(bin_no_bins) 
            self.frequency[0] = 1 #blows from ONE direction
            self.avMagnitude = np.full(bin_no_bins,a_0) #with the average strength
        elif custom == 3:
            print("Paper sanity check rose, 72 bins 15N/4E")
            self.avMagnitude = np.zeros((24))
            self.avMagnitude[0],self.avMagnitude[6] = 15,4
            self.frequency = np.zeros((24))
            self.frequency[0],self.frequency[6] = 0.5,0.5
            #this is the one I did on paper
            self.bin_no_bins = None #catch errors
        elif custom == 4:
            print("Spiral wind rose 72 bins")
            self.avMagnitude = np.linspace(0,10,72)
            #CAREFUL, don't go too wind or it cuts out!
            self.frequency = np.full_like(self.avMagnitude,1/72) #uniform likelyhood
            self.bin_no_bins = 72
        elif custom == 5:
            print("2x IMPULSE wind rose, {}ms^-1 with {} bins, p=0.75 Northerly,p=0.25 3d index".format(a_0,bin_no_bins))
            self.frequency = np.zeros(bin_no_bins) 
            self.frequency[0] = 0.75 #blows from ONE direction
            self.frequency[4] = 0.25 #blows from ONE direction
            self.avMagnitude = np.full(bin_no_bins,a_0) #with the average strength

        else: #custom rose NOT selected, select site
            if 360%bin_no_bins != 0:
                raise ValueError("When using site data, the number of bins must be a factor of 360")
            
            if filepath == None:
                filepath = r"WindRoseData_C\site"

            filepath = filepath + str(site)+".csv"
            
            print("site {} selected with {} bins".format(site,bin_no_bins))

             #bit messy
            direction, magnitude = self.load_csv_data(filepath)
            self.frequency,self.avMagnitude = self.binWindData_d(direction,magnitude,self.deg_bin_centers,bin_no_bins)

        if Cp_f is not None: #treat the power coefficent as a function of the wind speed
            self.Cp = Cp_f(self.avMagnitude) #use the interpolation to find the power coefficent as a function of the windspeed
        else:
            self.Cp = np.ones_like(self.avMagnitude) #otherwise ignore (all one)

        #LINEAR discrete distribution (Cp included)
        self.djd = self.Cp*self.frequency*self.avMagnitude 
        #CUBIC discrete distribution (Cp included)
        self.djd3 = self.Cp*self.frequency*(self.avMagnitude**3)   

        #fit the full-length LINEAR Fourier series (Cp NOT ! included)
        self.cjd_full_Fourier_coeffs_noCp = self.fft((self.frequency*self.avMagnitude*self.bin_no_bins)/(2*np.pi))
        self.cjd_full_f = self.Fourier_series(self.cjd_full_Fourier_coeffs_noCp)

        #fit the full-length LINEAR Fourier series (Cp INCLUDED)
        self.cjd_full_Fourier_coeffs = self.fft((np.cbrt(self.Cp)*self.frequency*self.avMagnitude*self.bin_no_bins)/(2*np.pi))
        self.cjd_full_f = self.Fourier_series(self.cjd_full_Fourier_coeffs_noCp)

        #fit the full-length CUBE Fourier series (Cp included!)
        self.cjd3_full_Fourier_coeffs = self.fft((self.Cp*self.frequency*(self.avMagnitude**3)*self.bin_no_bins)/(2*np.pi))
        self.cjd3_full_f = self.Fourier_series(self.cjd3_full_Fourier_coeffs)
        #Find the Phase-Amplitude coefficients
        a1, a2, a3 = self.cjd3_full_Fourier_coeffs
        A_n = np.sqrt(a2**2+a3**2)
        Phi_n = -np.arctan2(a3,a2)
        self.cjd3_PA_all_coeffs = a1,A_n,Phi_n

        #fit the full-length CUBE complex Fourier series (Cp included!)
        self.cjd3_full_Fourier_coeffs_CMPLX = self.CMPLX_fft((self.Cp*self.frequency*(self.avMagnitude**3)*self.bin_no_bins)/(2*np.pi))

        self.U_inf = np.size(self.frequency)*np.mean(self.avMagnitude*self.frequency) #free stream velocity #WRONG(?)!
                                

 









